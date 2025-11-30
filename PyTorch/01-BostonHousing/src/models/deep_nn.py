"""
深層ニューラルネットワーク

複数の隠れ層を持つ深層ニューラルネットワークを提供します。
"""

import torch
import torch.nn as nn
from typing import List
from .base_model import BaseModel


class DeepNN(BaseModel):
    """
    深層ニューラルネットワーク

    構造: Input -> Hidden1 -> Hidden2 -> ... -> Output
    活性化関数: ReLU / LeakyReLU / ELU
    正則化: Dropout、BatchNorm
    """

    def __init__(
        self,
        input_dim: int = 13,
        output_dim: int = 1,
        hidden_sizes: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        """
        Args:
            input_dim: 入力次元数
            output_dim: 出力次元数
            hidden_sizes: 各隠れ層のユニット数のリスト
            dropout_rate: Dropout率
            use_batch_norm: Batch Normalizationを使用するか
            activation: 活性化関数 ("relu", "leaky_relu", "elu")

        Examples:
            >>> model = DeepNN(hidden_sizes=[128, 64, 32], activation="relu")
            >>> model.print_summary()
        """
        super(DeepNN, self).__init__(input_dim, output_dim)

        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation

        # 隠れ層の構築
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            # 線形層
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # 活性化関数
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"未対応の活性化関数: {activation}")

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # 隠れ層をSequentialにまとめる
        self.hidden_layers = nn.Sequential(*layers)

        # 出力層
        self.output_layer = nn.Linear(prev_size, output_dim)

        # 重みの初期化
        if activation in ["relu", "leaky_relu"]:
            self.initialize_weights(method="kaiming")
        else:
            self.initialize_weights(method="xavier")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理

        Args:
            x: 入力テンソル (batch_size, input_dim)

        Returns:
            出力テンソル (batch_size, output_dim)

        Examples:
            >>> model = DeepNN()
            >>> x = torch.randn(32, 13)
            >>> output = model(x)
            >>> print(output.shape)  # torch.Size([32, 1])
        """
        # 隠れ層
        x = self.hidden_layers(x)

        # 出力層
        x = self.output_layer(x)

        return x

    def get_architecture_info(self) -> dict:
        """
        モデルアーキテクチャの詳細情報を取得

        Returns:
            アーキテクチャ情報の辞書
        """
        return {
            "model_type": "DeepNN",
            "input_dim": self.input_dim,
            "hidden_sizes": self.hidden_sizes,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "num_hidden_layers": len(self.hidden_sizes)
        }
