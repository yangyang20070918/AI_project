"""
データ前処理モジュール

住宅価格データの読み込み、前処理、分割を行います。
"""

import numpy as np
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class HousingDataPreprocessor:
    """
    Boston Housing データの前処理クラス

    データの読み込み、正規化/標準化、訓練/検証/テスト分割を行います。
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        """
        Args:
            scaler_type: スケーラーの種類 ("standard" or "minmax")
            train_ratio: 訓練データの比率
            val_ratio: 検証データの比率
            test_ratio: テストデータの比率
            random_state: 乱数シード

        Raises:
            ValueError: データ分割比率の合計が1.0でない場合
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"データ分割比率の合計は1.0である必要があります: "
                f"{train_ratio} + {val_ratio} + {test_ratio} = "
                f"{train_ratio + val_ratio + test_ratio}"
            )

        self.scaler_type = scaler_type
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # スケーラーの初期化
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"未対応のscaler_type: {scaler_type}")

        self.is_fitted = False

    def load_data(self, data_path: Path) -> np.ndarray:
        """
        housing.dataファイルを読み込む

        Args:
            data_path: データファイルのパス

        Returns:
            読み込んだデータ配列 (shape: [n_samples, n_features+1])

        Raises:
            FileNotFoundError: ファイルが見つからない場合
        """
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")

        with open(data_path, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            # 複数の空白を1つに変換
            cleaned = re.sub(r"\s{2,}", " ", line).strip()
            # 空白で分割
            values = cleaned.split(" ")
            data.append(values)

        # NumPy配列に変換
        data_array = np.array(data, dtype=np.float32)

        return data_array

    def split_features_target(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        データを特徴量とターゲットに分割

        Args:
            data: 元のデータ配列

        Returns:
            (特徴量, ターゲット) のタプル
        """
        X = data[:, :-1]  # 最後の列以外が特徴量
        y = data[:, -1]   # 最後の列がターゲット

        return X, y

    def fit_scaler(self, X_train: np.ndarray) -> None:
        """
        訓練データでスケーラーをフィット

        Args:
            X_train: 訓練データの特徴量
        """
        self.scaler.fit(X_train)
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        データを変換（正規化/標準化）

        Args:
            X: 変換する特徴量

        Returns:
            変換後の特徴量

        Raises:
            RuntimeError: スケーラーがフィットされていない場合
        """
        if not self.is_fitted:
            raise RuntimeError("スケーラーがフィットされていません。先にfit_scaler()を呼んでください。")

        return self.scaler.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        訓練データでフィットし、変換を適用

        Args:
            X_train: 訓練データの特徴量

        Returns:
            変換後の訓練データ
        """
        self.fit_scaler(X_train)
        return self.transform(X_train)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        データを訓練/検証/テストに分割

        Args:
            X: 特徴量
            y: ターゲット

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test) のタプル
        """
        # まず訓練データと一時データ（検証+テスト）に分割
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_state
        )

        # 一時データを検証とテストに分割
        val_size_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size_ratio),
            random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(
        self,
        data_path: Path,
        apply_scaling: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        データの読み込みから分割、スケーリングまでの全処理を実行

        Args:
            data_path: データファイルのパス
            apply_scaling: スケーリングを適用するか

        Returns:
            前処理済みデータを含む辞書
            - X_train, X_val, X_test: 特徴量
            - y_train, y_val, y_test: ターゲット
            - scaler_params: スケーラーのパラメータ
        """
        # データ読み込み
        data = self.load_data(data_path)

        # 特徴量とターゲットに分割
        X, y = self.split_features_target(data)

        # 訓練/検証/テストに分割
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # スケーリング適用
        if apply_scaling:
            X_train = self.fit_transform(X_train)
            X_val = self.transform(X_val)
            X_test = self.transform(X_test)

            # スケーラーのパラメータを保存
            if self.scaler_type == "standard":
                scaler_params = {
                    "mean": self.scaler.mean_,
                    "scale": self.scaler.scale_
                }
            else:  # minmax
                scaler_params = {
                    "min": self.scaler.min_,
                    "scale": self.scaler.scale_
                }
        else:
            scaler_params = None

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "scaler_params": scaler_params
        }

    def save_preprocessed_data(
        self,
        preprocessed_data: Dict[str, np.ndarray],
        save_path: Path
    ) -> None:
        """
        前処理済みデータをnpzファイルに保存

        Args:
            preprocessed_data: 前処理済みデータの辞書
            save_path: 保存先パス
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(save_path, **preprocessed_data)

    @staticmethod
    def load_preprocessed_data(load_path: Path) -> Dict[str, np.ndarray]:
        """
        前処理済みデータを読み込む

        Args:
            load_path: 読み込むnpzファイルのパス

        Returns:
            前処理済みデータの辞書

        Raises:
            FileNotFoundError: ファイルが見つからない場合
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {load_path}")

        loaded = np.load(load_path, allow_pickle=True)

        return {key: loaded[key] for key in loaded.files}


def get_data_statistics(data: np.ndarray) -> Dict[str, Any]:
    """
    データの統計情報を取得

    Args:
        data: データ配列

    Returns:
        統計情報を含む辞書
    """
    return {
        "shape": data.shape,
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
        "median": np.median(data, axis=0)
    }
