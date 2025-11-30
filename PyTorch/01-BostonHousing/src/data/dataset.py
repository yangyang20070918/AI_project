"""
PyTorch Dataset モジュール

PyTorchのDatasetとDataLoaderを使用した効率的なデータ読み込みを提供します。
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict


class HousingDataset(Dataset):
    """
    Boston Housing データセット

    PyTorchのDatasetインターフェースを実装します。
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Args:
            X: 特徴量データ (shape: [n_samples, n_features])
            y: ターゲットデータ (shape: [n_samples,])
            transform: データ変換関数（オプション）

        Examples:
            >>> X_train = np.random.rand(100, 13)
            >>> y_train = np.random.rand(100)
            >>> dataset = HousingDataset(X_train, y_train)
            >>> print(len(dataset))
            100
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform

        if len(self.X) != len(self.y):
            raise ValueError(
                f"XとYのサンプル数が一致しません: X={len(self.X)}, y={len(self.y)}"
            )

    def __len__(self) -> int:
        """
        データセットのサイズを返す

        Returns:
            サンプル数
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのデータを取得

        Args:
            idx: データのインデックス

        Returns:
            (特徴量, ターゲット) のタプル
        """
        x = self.X[idx]
        y = self.y[idx]

        # データ変換を適用（指定されている場合）
        if self.transform:
            x = self.transform(x)

        return x, y

    def get_feature_dim(self) -> int:
        """
        特徴量の次元数を返す

        Returns:
            特徴量の次元数
        """
        return self.X.shape[1]

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """
        データセットの統計情報を取得

        Returns:
            統計情報を含む辞書
        """
        return {
            "X_mean": torch.mean(self.X, dim=0),
            "X_std": torch.std(self.X, dim=0),
            "X_min": torch.min(self.X, dim=0).values,
            "X_max": torch.max(self.X, dim=0).values,
            "y_mean": torch.mean(self.y),
            "y_std": torch.std(self.y),
            "y_min": torch.min(self.y),
            "y_max": torch.max(self.y)
        }


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    訓練/検証/テスト用のDataLoaderを作成

    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データのターゲット
        X_val: 検証データの特徴量
        y_val: 検証データのターゲット
        X_test: テストデータの特徴量
        y_test: テストデータのターゲット
        batch_size: バッチサイズ
        num_workers: データローダーのワーカー数
        shuffle_train: 訓練データをシャッフルするか

    Returns:
        (train_loader, val_loader, test_loader) のタプル

    Examples:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
        ... )
        >>> for batch_X, batch_y in train_loader:
        ...     # 訓練ループ
        ...     pass
    """
    # Datasetの作成
    train_dataset = HousingDataset(X_train, y_train)
    val_dataset = HousingDataset(X_val, y_val)
    test_dataset = HousingDataset(X_test, y_test)

    # DataLoaderの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # GPU使用時の高速化
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def create_single_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    単一のDataLoaderを作成

    Args:
        X: 特徴量
        y: ターゲット
        batch_size: バッチサイズ
        shuffle: データをシャッフルするか
        num_workers: ワーカー数

    Returns:
        DataLoaderオブジェクト
    """
    dataset = HousingDataset(X, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


class InferenceDataset(Dataset):
    """
    推論用のデータセット（ターゲットなし）
    """

    def __init__(self, X: np.ndarray):
        """
        Args:
            X: 特徴量データ
        """
        self.X = torch.FloatTensor(X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


def create_inference_dataloader(
    X: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0
) -> DataLoader:
    """
    推論用のDataLoaderを作成

    Args:
        X: 特徴量
        batch_size: バッチサイズ
        num_workers: ワーカー数

    Returns:
        DataLoaderオブジェクト

    Examples:
        >>> X_new = np.random.rand(10, 13)
        >>> inference_loader = create_inference_dataloader(X_new)
        >>> for batch_X in inference_loader:
        ...     predictions = model(batch_X)
    """
    dataset = InferenceDataset(X)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
