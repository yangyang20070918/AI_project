"""
訓練スクリプト

モデルを訓練し、実験結果を保存します。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from src.utils.seed import set_seed
from src.utils.logger import ExperimentLogger
from src.utils.config_loader import load_config
from src.data.preprocessing import HousingDataPreprocessor
from src.data.dataset import create_dataloaders
from src.models.baseline import BaselineModel
from src.models.simple_nn import SimpleNN
from src.models.deep_nn import DeepNN
from src.training.trainer import Trainer, EarlyStopping, create_optimizer, create_lr_scheduler
from src.training.experiment import Experiment
import argparse


def get_model(model_config: dict, input_dim: int = 13, output_dim: int = 1) -> nn.Module:
    """
    設定からモデルを作成

    Args:
        model_config: モデル設定
        input_dim: 入力次元数
        output_dim: 出力次元数

    Returns:
        モデルインスタンス
    """
    model_type = model_config.get('type', 'simple_nn')

    if model_type == 'baseline':
        model = BaselineModel(input_dim=input_dim, output_dim=output_dim)

    elif model_type == 'simple_nn':
        params = model_config.get('simple_nn', {})
        model = SimpleNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=params.get('hidden_size', 100),
            dropout_rate=params.get('dropout_rate', 0.2),
            use_batch_norm=params.get('use_batch_norm', False)
        )

    elif model_type == 'deep_nn':
        params = model_config.get('deep_nn', {})
        model = DeepNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=params.get('hidden_sizes', [128, 64, 32]),
            dropout_rate=params.get('dropout_rate', 0.3),
            use_batch_norm=params.get('use_batch_norm', True),
            activation=params.get('activation', 'relu')
        )

    else:
        raise ValueError(f"未対応のモデルタイプ: {model_type}")

    return model


def get_criterion(loss_config: dict) -> nn.Module:
    """
    損失関数を作成

    Args:
        loss_config: 損失関数の設定

    Returns:
        損失関数
    """
    loss_type = loss_config.get('type', 'mse')

    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"未対応の損失関数: {loss_type}")


def get_device(config: dict) -> str:
    """
    デバイスを取得

    Args:
        config: システム設定

    Returns:
        デバイス名 ("cpu" or "cuda")
    """
    device_config = config.get('system', {}).get('device', 'auto')

    if device_config == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return device_config


def main(config_path: str = None, experiment_id: str = None):
    """
    メイン訓練処理

    Args:
        config_path: 設定ファイルのパス（Noneの場合はデフォルト設定を使用）
        experiment_id: 実験ID（Noneの場合は自動生成）
    """
    # 設定の読み込み
    if config_path is None:
        config_path = project_root / "config" / "config.yaml"

    config = load_config(config_path)

    # 乱数シードの固定
    seed = config.get('system', {}).get('seed', 42)
    set_seed(seed)

    # デバイスの取得
    device = get_device(config)
    print(f"使用デバイス: {device}")

    # 実験の作成
    experiment = Experiment(
        base_dir=project_root / config.get('experiment', {}).get('base_dir', 'experiments'),
        experiment_id=experiment_id,
        config=config
    )

    # ロガーの作成
    logger = ExperimentLogger(
        experiment_dir=experiment.experiment_dir,
        experiment_id=experiment.experiment_id
    )

    logger.info("=" * 70)
    logger.info(f"実験を開始: {experiment.experiment_id}")
    logger.info("=" * 70)
    logger.log_config(config)

    # データの前処理
    logger.info("データを読み込み中...")

    data_config = config.get('data', {})
    preprocessing_config = data_config.get('preprocessing', {})

    preprocessor = HousingDataPreprocessor(
        scaler_type=preprocessing_config.get('scaler_type', 'standard'),
        train_ratio=data_config.get('train_ratio', 0.7),
        val_ratio=data_config.get('val_ratio', 0.15),
        test_ratio=data_config.get('test_ratio', 0.15),
        random_state=seed
    )

    # データの前処理
    data_path = project_root / data_config.get('raw_data_path', 'data/raw/housing.data')
    preprocessed_data = preprocessor.preprocess(data_path, apply_scaling=True)

    # 前処理済みデータの保存
    processed_data_path = project_root / data_config.get('processed_data_path', 'data/processed/preprocessed.npz')
    preprocessor.save_preprocessed_data(preprocessed_data, processed_data_path)

    logger.info(f"訓練データ: {preprocessed_data['X_train'].shape}")
    logger.info(f"検証データ: {preprocessed_data['X_val'].shape}")
    logger.info(f"テストデータ: {preprocessed_data['X_test'].shape}")

    # DataLoaderの作成
    training_config = config.get('training', {})

    train_loader, val_loader, test_loader = create_dataloaders(
        preprocessed_data['X_train'],
        preprocessed_data['y_train'],
        preprocessed_data['X_val'],
        preprocessed_data['y_val'],
        preprocessed_data['X_test'],
        preprocessed_data['y_test'],
        batch_size=training_config.get('batch_size', 32),
        num_workers=config.get('system', {}).get('num_workers', 0)
    )

    # モデルの作成
    logger.info("モデルを作成中...")

    model = get_model(
        config.get('model', {}),
        input_dim=preprocessed_data['X_train'].shape[1],
        output_dim=1
    )

    model.print_summary()
    logger.info(f"モデル: {model.__class__.__name__}")
    logger.info(f"パラメータ数: {model.count_parameters():,}")

    # 損失関数の作成
    criterion = get_criterion(training_config.get('loss', {}))

    # オプティマイザの作成
    optimizer_config = training_config.get('optimizer', {})
    optimizer_config['learning_rate'] = training_config.get('learning_rate', 0.001)
    optimizer = create_optimizer(model, optimizer_config)

    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Learning Rate: {training_config.get('learning_rate', 0.001)}")

    # 学習率スケジューラの作成
    lr_scheduler = create_lr_scheduler(
        optimizer,
        training_config.get('lr_scheduler', {})
    )

    if lr_scheduler:
        logger.info(f"LR Scheduler: {lr_scheduler.__class__.__name__}")

    # Early Stoppingの作成
    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping = None

    if early_stopping_config.get('enabled', True):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 50),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            mode='min'
        )
        logger.info(f"Early Stopping: patience={early_stopping.patience}")

    # Trainerの作成
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_dir=experiment.experiment_dir,
        logger=logger
    )

    # 訓練の実行
    logger.info("訓練を開始...")

    checkpoint_config = training_config.get('checkpoint', {})

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.get('epochs', 1000),
        loss_scale=training_config.get('loss', {}).get('loss_scale', 1.0),
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        save_best_only=checkpoint_config.get('save_best_only', True),
        save_last=checkpoint_config.get('save_last', True),
        verbose=True
    )

    logger.info("訓練が完了しました")
    logger.info(f"Best Epoch: {trainer.best_epoch}")
    logger.info(f"Best Val Loss: {trainer.best_val_loss:.6f}")

    # 実験情報の保存
    summary = f"""
Boston Housing 価格予測 - 実験結果

実験ID: {experiment.experiment_id}
モデル: {model.__class__.__name__}
パラメータ数: {model.count_parameters():,}

訓練設定:
- Epochs: {training_config.get('epochs', 1000)}
- Batch Size: {training_config.get('batch_size', 32)}
- Learning Rate: {training_config.get('learning_rate', 0.001)}
- Optimizer: {optimizer.__class__.__name__}

結果:
- Best Epoch: {trainer.best_epoch}
- Best Val Loss: {trainer.best_val_loss:.6f}
- Final Train Loss: {history['train_loss'][-1]:.6f}
- Final Val Loss: {history['val_loss'][-1]:.6f}
"""

    experiment.save_summary(summary)
    logger.info("\n" + summary)

    print("\n" + "=" * 70)
    print(f"実験が完了しました: {experiment.experiment_id}")
    print(f"結果ディレクトリ: {experiment.experiment_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boston Housing 価格予測モデルの訓練")

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='設定ファイルのパス'
    )

    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        help='実験ID（指定しない場合は自動生成）'
    )

    args = parser.parse_args()

    main(config_path=args.config, experiment_id=args.experiment_id)
