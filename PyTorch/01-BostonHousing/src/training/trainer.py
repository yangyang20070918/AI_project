"""
訓練システム

モデルの訓練、検証、Early Stopping、モデルチェックポイントなどを管理します。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json
from tqdm import tqdm
import numpy as np


class EarlyStopping:
    """
    Early Stopping クラス

    検証lossが改善しない場合に訓練を早期終了します。
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.0001,
        mode: str = "min"
    ):
        """
        Args:
            patience: 改善が見られないエポック数の許容値
            min_delta: 改善とみなす最小の変化量
            mode: "min"（lossを最小化）or "max"（メトリクスを最大化）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Early Stoppingの判定

        Args:
            score: 現在のスコア（lossまたはメトリクス）
            epoch: 現在のエポック

        Returns:
            Early Stopすべきかどうか
        """
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        # 改善があるか判定
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class Trainer:
    """
    モデル訓練クラス

    訓練ループ、検証、Early Stopping、モデルチェックポイントなどを管理します。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        device: str = "cpu",
        experiment_dir: Optional[Path] = None,
        logger: Optional[Any] = None
    ):
        """
        Args:
            model: 訓練するモデル
            optimizer: 最適化アルゴリズム
            criterion: 損失関数
            device: デバイス ("cpu" or "cuda")
            experiment_dir: 実験結果を保存するディレクトリ
            logger: ロガーオブジェクト
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        self.logger = logger

        # 訓練履歴
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": []
        }

        # ベストモデル情報
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_scale: float = 1.0
    ) -> float:
        """
        1エポック分の訓練を実行

        Args:
            train_loader: 訓練データローダー
            loss_scale: 損失値のスケーリング係数

        Returns:
            平均訓練loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # 勾配をゼロ化
            self.optimizer.zero_grad()

            # 順伝播
            predictions = self.model(batch_X).squeeze()

            # 損失計算
            loss = self.criterion(predictions, batch_y) * loss_scale

            # 逆伝播
            loss.backward()

            # パラメータ更新
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(
        self,
        val_loader: DataLoader,
        loss_scale: float = 1.0
    ) -> float:
        """
        1エポック分の検証を実行

        Args:
            val_loader: 検証データローダー
            loss_scale: 損失値のスケーリング係数

        Returns:
            平均検証loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # 順伝播
                predictions = self.model(batch_X).squeeze()

                # 損失計算
                loss = self.criterion(predictions, batch_y) * loss_scale

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 1000,
        loss_scale: float = 1.0,
        early_stopping: Optional[EarlyStopping] = None,
        lr_scheduler: Optional[Any] = None,
        save_best_only: bool = True,
        save_last: bool = True,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        訓練を実行

        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            epochs: エポック数
            loss_scale: 損失値のスケーリング係数
            early_stopping: Early Stoppingオブジェクト
            lr_scheduler: 学習率スケジューラー
            save_best_only: ベストモデルのみを保存するか
            save_last: 最後のモデルを保存するか
            verbose: 進捗を表示するか

        Returns:
            訓練履歴の辞書
        """
        if verbose:
            pbar = tqdm(range(epochs), desc="Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            # 訓練
            train_loss = self.train_epoch(train_loader, loss_scale)

            # 検証
            val_loss = self.validate_epoch(val_loader, loss_scale)

            # 学習率を記録
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rates"].append(current_lr)

            # プログレスバー更新
            if verbose:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'lr': f'{current_lr:.6f}'
                })

            # ロガーに記録
            if self.logger:
                self.logger.log_metrics(
                    epoch + 1,
                    {"loss": train_loss},
                    phase="train"
                )
                self.logger.log_metrics(
                    epoch + 1,
                    {"loss": val_loss},
                    phase="val"
                )

            # ベストモデルの保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1

                if save_best_only and self.experiment_dir:
                    self.save_checkpoint(
                        epoch + 1,
                        val_loss,
                        filename="model_best.pth"
                    )

            # 学習率スケジューリング
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            # Early Stopping
            if early_stopping is not None:
                if early_stopping(val_loss, epoch + 1):
                    if self.logger:
                        self.logger.info(
                            f"Early Stopping at epoch {epoch + 1}. "
                            f"Best epoch: {early_stopping.best_epoch}"
                        )
                    break

        # 最後のモデルを保存
        if save_last and self.experiment_dir:
            self.save_checkpoint(
                epoch + 1,
                val_loss,
                filename="model_last.pth"
            )

        # 訓練履歴を保存
        if self.experiment_dir:
            self.save_history()

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        filename: str = "checkpoint.pth"
    ) -> None:
        """
        モデルチェックポイントを保存

        Args:
            epoch: エポック数
            val_loss: 検証loss
            filename: 保存ファイル名
        """
        if self.experiment_dir is None:
            return

        checkpoint_path = self.experiment_dir / filename

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        モデルチェックポイントを読み込み

        Args:
            checkpoint_path: チェックポイントファイルのパス

        Returns:
            チェックポイント情報の辞書
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def save_history(self) -> None:
        """訓練履歴をJSONファイルに保存"""
        if self.experiment_dir is None:
            return

        history_path = self.experiment_dir / "training_history.json"

        # NumPy配列を通常のlistに変換
        history_to_save = {
            key: [float(v) for v in values]
            for key, values in self.history.items()
        }

        history_to_save["best_epoch"] = self.best_epoch
        history_to_save["best_val_loss"] = self.best_val_loss

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=2, ensure_ascii=False)

    def load_history(self, history_path: Path) -> None:
        """訓練履歴をJSONファイルから読み込み"""
        with open(history_path, 'r', encoding='utf-8') as f:
            loaded_history = json.load(f)

        self.best_epoch = loaded_history.pop("best_epoch", 0)
        self.best_val_loss = loaded_history.pop("best_val_loss", float('inf'))
        self.history = loaded_history


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> Optimizer:
    """
    設定からオプティマイザを作成

    Args:
        model: モデル
        config: 設定辞書

    Returns:
        オプティマイザ
    """
    optimizer_type = config.get('type', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)

    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = config.get('sgd', {}).get('momentum', 0.9)
        nesterov = config.get('sgd', {}).get('nesterov', True)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    else:
        raise ValueError(f"未対応のoptimizer type: {optimizer_type}")


def create_lr_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any]
) -> Optional[Any]:
    """
    設定から学習率スケジューラを作成

    Args:
        optimizer: オプティマイザ
        config: 設定辞書

    Returns:
        学習率スケジューラ（Noneの場合もあり）
    """
    if not config.get('enabled', False):
        return None

    scheduler_type = config.get('type', 'reduce_on_plateau').lower()

    if scheduler_type == 'reduce_on_plateau':
        params = config.get('reduce_on_plateau', {})
        return ReduceLROnPlateau(
            optimizer,
            mode=params.get('mode', 'min'),
            factor=params.get('factor', 0.5),
            patience=params.get('patience', 20),
            min_lr=params.get('min_lr', 0.00001)
        )
    elif scheduler_type == 'step':
        params = config.get('step', {})
        return StepLR(
            optimizer,
            step_size=params.get('step_size', 100),
            gamma=params.get('gamma', 0.5)
        )
    else:
        raise ValueError(f"未対応のscheduler type: {scheduler_type}")
