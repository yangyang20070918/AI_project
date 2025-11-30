"""
評価モジュール

モデルの性能を評価するための各種指標を計算します。
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class RegressionEvaluator:
    """
    回帰モデルの評価クラス

    各種評価指標の計算を行います。
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: 評価するモデル
            device: デバイス ("cpu" or "cuda")
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測を実行

        Args:
            dataloader: データローダー

        Returns:
            (予測値, 実測値) のタプル
        """
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # 予測
                pred = self.model(batch_X).squeeze()

                predictions.append(pred.cpu().numpy())
                targets.append(batch_y.cpu().numpy())

        # 配列を結合
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        return predictions, targets

    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        MSE（平均二乗誤差）を計算

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            MSE値
        """
        return mean_squared_error(y_true, y_pred)

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        RMSE（二乗平均平方根誤差）を計算

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            RMSE値
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        MAE（平均絶対誤差）を計算

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            MAE値
        """
        return mean_absolute_error(y_true, y_pred)

    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R²スコア（決定係数）を計算

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            R²値
        """
        return r2_score(y_true, y_pred)

    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        MAPE（平均絶対パーセント誤差）を計算

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            MAPE値（パーセント）
        """
        # ゼロ除算を防ぐ
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def calculate_max_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        最大誤差を計算

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            最大誤差
        """
        return np.max(np.abs(y_true - y_pred))

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        全ての評価指標を計算

        Args:
            y_true: 実測値
            y_pred: 予測値
            metrics: 計算する指標のリスト（Noneの場合は全て計算）

        Returns:
            評価指標を含む辞書
        """
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'r2', 'mape', 'max_error']

        results = {}

        if 'mse' in metrics:
            results['mse'] = self.calculate_mse(y_true, y_pred)

        if 'rmse' in metrics:
            results['rmse'] = self.calculate_rmse(y_true, y_pred)

        if 'mae' in metrics:
            results['mae'] = self.calculate_mae(y_true, y_pred)

        if 'r2' in metrics:
            results['r2'] = self.calculate_r2(y_true, y_pred)

        if 'mape' in metrics:
            results['mape'] = self.calculate_mape(y_true, y_pred)

        if 'max_error' in metrics:
            results['max_error'] = self.calculate_max_error(y_true, y_pred)

        return results

    def evaluate(
        self,
        dataloader: DataLoader,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        データローダーを使用してモデルを評価

        Args:
            dataloader: 評価用データローダー
            metrics: 計算する指標のリスト

        Returns:
            評価結果を含む辞書
        """
        # 予測を実行
        predictions, targets = self.predict(dataloader)

        # 指標を計算
        metrics_dict = self.calculate_all_metrics(targets, predictions, metrics)

        # 予測値と実測値も返す
        return {
            **metrics_dict,
            'predictions': predictions,
            'targets': targets
        }

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        評価指標を表示

        Args:
            metrics: 評価指標の辞書
        """
        print("=" * 70)
        print("Evaluation Metrics")
        print("=" * 70)

        metric_names = {
            'mse': 'MSE',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'r2': 'R2 Score',
            'mape': 'MAPE',
            'max_error': 'Max Error'
        }

        for key, value in metrics.items():
            if key in metric_names:
                if key == 'mape':
                    print(f"{metric_names[key]:20s}: {value:10.2f}%")
                elif key == 'r2':
                    print(f"{metric_names[key]:20s}: {value:10.4f}")
                else:
                    print(f"{metric_names[key]:20s}: {value:10.4f}")

        print("=" * 70)

    def get_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        残差を計算

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            残差配列
        """
        return y_true - y_pred

    def get_residual_statistics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        残差の統計情報を取得

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            残差統計情報の辞書
        """
        residuals = self.get_residuals(y_true, y_pred)

        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'median': np.median(residuals),
            'q75': np.percentile(residuals, 75)
        }


def compare_models(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    device: str = 'cpu',
    metrics: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    複数のモデルを比較

    Args:
        models: モデル名とモデルの辞書
        dataloader: 評価用データローダー
        device: デバイス
        metrics: 計算する指標のリスト

    Returns:
        モデル名と評価指標の辞書

    Examples:
        >>> models = {'baseline': baseline_model, 'simple_nn': simple_model}
        >>> results = compare_models(models, test_loader)
        >>> for name, metrics in results.items():
        ...     print(f"{name}: R²={metrics['r2']:.4f}")
    """
    results = {}

    for model_name, model in models.items():
        evaluator = RegressionEvaluator(model, device)
        eval_results = evaluator.evaluate(dataloader, metrics)

        # predictions と targets は除外
        results[model_name] = {
            k: v for k, v in eval_results.items()
            if k not in ['predictions', 'targets']
        }

    return results
