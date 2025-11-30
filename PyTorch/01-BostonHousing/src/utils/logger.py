"""
ロギングシステム

プロジェクト全体で使用する統一されたロギング機能を提供します。
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "boston_housing",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    ロガーをセットアップ

    Args:
        name: ロガー名
        log_file: ログファイルのパス（Noneの場合はファイル出力なし）
        level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        console_output: コンソールへの出力を有効にするか

    Returns:
        設定済みのロガーオブジェクト

    Examples:
        >>> logger = setup_logger("my_experiment", log_file=Path("logs/experiment.log"))
        >>> logger.info("実験を開始します")
    """
    # ロガーの作成
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 既存のハンドラをクリア（重複を防ぐ）
    if logger.handlers:
        logger.handlers.clear()

    # フォーマッタの設定
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # コンソール出力の設定
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ファイル出力の設定
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "boston_housing") -> logging.Logger:
    """
    既存のロガーを取得

    Args:
        name: ロガー名

    Returns:
        ロガーオブジェクト

    Examples:
        >>> logger = get_logger()
        >>> logger.info("メッセージ")
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    実験用のロガークラス

    実験ごとに自動的にログファイルを作成し、
    実験情報を記録します。
    """

    def __init__(
        self,
        experiment_dir: Path,
        experiment_id: str,
        level: int = logging.INFO
    ):
        """
        Args:
            experiment_dir: 実験結果を保存するディレクトリ
            experiment_id: 実験ID
            level: ログレベル
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_id = experiment_id
        self.log_file = self.experiment_dir / "experiment.log"

        # ロガーのセットアップ
        self.logger = setup_logger(
            name=f"experiment_{experiment_id}",
            log_file=self.log_file,
            level=level,
            console_output=True
        )

    def info(self, message: str) -> None:
        """INFOレベルのログを記録"""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """DEBUGレベルのログを記録"""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """WARNINGレベルのログを記録"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """ERRORレベルのログを記録"""
        self.logger.error(message)

    def log_config(self, config: dict) -> None:
        """
        実験設定をログに記録

        Args:
            config: 設定辞書
        """
        self.info("=" * 50)
        self.info(f"実験ID: {self.experiment_id}")
        self.info("実験設定:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 50)

    def log_metrics(self, epoch: int, metrics: dict, phase: str = "train") -> None:
        """
        評価指標をログに記録

        Args:
            epoch: エポック数
            metrics: 評価指標の辞書
            phase: フェーズ（train, val, testなど）
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch:04d} [{phase}] - {metrics_str}")
