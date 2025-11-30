"""
実験管理システム

実験の作成、設定管理、結果の保存などを行います。
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
import shutil


class Experiment:
    """
    実験管理クラス

    実験ごとのディレクトリ作成、設定保存、結果管理を行います。
    """

    def __init__(
        self,
        base_dir: Path = Path("experiments"),
        experiment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            base_dir: 実験結果の基底ディレクトリ
            experiment_id: 実験ID（Noneの場合は自動生成）
            config: 実験設定
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 実験IDの生成
        if experiment_id is None:
            self.experiment_id = self.generate_experiment_id()
        else:
            self.experiment_id = experiment_id

        # 実験ディレクトリの作成
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # サブディレクトリの作成
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.models_dir = self.experiment_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # 設定の保存
        self.config = config
        if config is not None:
            self.save_config(config)

    @staticmethod
    def generate_experiment_id(id_format: str = "%Y%m%d_%H%M%S") -> str:
        """
        実験IDを生成

        Args:
            id_format: strftime形式の文字列

        Returns:
            生成された実験ID

        Examples:
            >>> exp_id = Experiment.generate_experiment_id()
            >>> print(exp_id)  # "20250130_143527"
        """
        return datetime.now().strftime(id_format)

    def save_config(self, config: Dict[str, Any], filename: str = "config.json") -> None:
        """
        実験設定を保存

        Args:
            config: 設定辞書
            filename: 保存ファイル名
        """
        config_path = self.experiment_dir / filename

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_config(self, filename: str = "config.json") -> Dict[str, Any]:
        """
        実験設定を読み込み

        Args:
            filename: 設定ファイル名

        Returns:
            設定辞書
        """
        config_path = self.experiment_dir / filename

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return config

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> None:
        """
        評価指標を保存

        Args:
            metrics: 評価指標の辞書
            filename: 保存ファイル名
        """
        metrics_path = self.experiment_dir / filename

        # floatに変換（NumPy配列などを通常のfloatに）
        metrics_to_save = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):  # PyTorch/NumPyテンソル
                metrics_to_save[key] = float(value.item())
            elif isinstance(value, (list, tuple)):
                metrics_to_save[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                metrics_to_save[key] = value

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

    def load_metrics(self, filename: str = "metrics.json") -> Dict[str, Any]:
        """
        評価指標を読み込み

        Args:
            filename: メトリクスファイル名

        Returns:
            評価指標の辞書
        """
        metrics_path = self.experiment_dir / filename

        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return metrics

    def save_summary(self, summary: str, filename: str = "summary.txt") -> None:
        """
        実験の要約を保存

        Args:
            summary: 要約テキスト
            filename: 保存ファイル名
        """
        summary_path = self.experiment_dir / filename

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

    def get_plot_path(self, plot_name: str) -> Path:
        """
        プロットの保存パスを取得

        Args:
            plot_name: プロット名（拡張子含む）

        Returns:
            保存パス
        """
        return self.plots_dir / plot_name

    def get_model_path(self, model_name: str) -> Path:
        """
        モデルの保存パスを取得

        Args:
            model_name: モデル名（拡張子含む）

        Returns:
            保存パス
        """
        return self.models_dir / model_name

    def list_experiments(self) -> list:
        """
        全ての実験をリスト

        Returns:
            実験IDのリスト
        """
        experiments = []
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                experiments.append(exp_dir.name)

        return sorted(experiments, reverse=True)  # 新しい順

    def delete_experiment(self, confirm: bool = False) -> None:
        """
        実験ディレクトリを削除

        Args:
            confirm: 削除確認（Trueの場合のみ削除実行）

        Raises:
            RuntimeError: confirmがFalseの場合
        """
        if not confirm:
            raise RuntimeError(
                "実験を削除するにはconfirm=Trueを指定してください。"
            )

        if self.experiment_dir.exists():
            shutil.rmtree(self.experiment_dir)

    def get_experiment_info(self) -> Dict[str, Any]:
        """
        実験の情報を取得

        Returns:
            実験情報の辞書
        """
        info = {
            "experiment_id": self.experiment_id,
            "experiment_dir": str(self.experiment_dir),
            "created_at": None,
            "config_exists": (self.experiment_dir / "config.json").exists(),
            "metrics_exists": (self.experiment_dir / "metrics.json").exists(),
            "plots_count": len(list(self.plots_dir.glob("*"))),
            "models_count": len(list(self.models_dir.glob("*")))
        }

        # 作成日時を取得（ディレクトリの作成時刻）
        if self.experiment_dir.exists():
            timestamp = self.experiment_dir.stat().st_ctime
            info["created_at"] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        return info

    def print_info(self) -> None:
        """実験情報を表示"""
        info = self.get_experiment_info()

        print("=" * 70)
        print(f"実験ID: {info['experiment_id']}")
        print("=" * 70)
        print(f"ディレクトリ: {info['experiment_dir']}")
        print(f"作成日時:     {info['created_at']}")
        print(f"設定ファイル: {'あり' if info['config_exists'] else 'なし'}")
        print(f"評価指標:     {'あり' if info['metrics_exists'] else 'なし'}")
        print(f"プロット数:   {info['plots_count']}")
        print(f"モデル数:     {info['models_count']}")
        print("=" * 70)


class ExperimentManager:
    """
    複数の実験を管理するクラス
    """

    def __init__(self, base_dir: Path = Path("experiments")):
        """
        Args:
            base_dir: 実験結果の基底ディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        experiment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        新しい実験を作成

        Args:
            experiment_id: 実験ID（Noneの場合は自動生成）
            config: 実験設定

        Returns:
            Experimentオブジェクト
        """
        return Experiment(
            base_dir=self.base_dir,
            experiment_id=experiment_id,
            config=config
        )

    def load_experiment(self, experiment_id: str) -> Experiment:
        """
        既存の実験を読み込み

        Args:
            experiment_id: 実験ID

        Returns:
            Experimentオブジェクト

        Raises:
            FileNotFoundError: 実験が見つからない場合
        """
        experiment_dir = self.base_dir / experiment_id

        if not experiment_dir.exists():
            raise FileNotFoundError(f"実験が見つかりません: {experiment_id}")

        return Experiment(
            base_dir=self.base_dir,
            experiment_id=experiment_id
        )

    def list_all_experiments(self) -> list:
        """
        全ての実験をリスト

        Returns:
            実験情報のリスト
        """
        experiments = []

        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                exp = Experiment(
                    base_dir=self.base_dir,
                    experiment_id=exp_dir.name
                )
                experiments.append(exp.get_experiment_info())

        # 作成日時の降順でソート
        experiments.sort(
            key=lambda x: x['created_at'] if x['created_at'] else '',
            reverse=True
        )

        return experiments

    def compare_experiments(self, experiment_ids: list) -> Dict[str, Any]:
        """
        複数の実験を比較

        Args:
            experiment_ids: 比較する実験IDのリスト

        Returns:
            比較結果の辞書
        """
        comparison = {}

        for exp_id in experiment_ids:
            try:
                exp = self.load_experiment(exp_id)
                metrics = exp.load_metrics()
                comparison[exp_id] = metrics
            except FileNotFoundError:
                comparison[exp_id] = None

        return comparison
