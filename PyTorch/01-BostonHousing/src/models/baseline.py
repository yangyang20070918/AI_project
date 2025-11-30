"""
ベースラインモデル（線形回帰）

シンプルな線形回帰モデルを提供します。
他のモデルと比較するための基準となります。
"""

import torch
import torch.nn as nn
from .base_model import BaseModel


class BaselineModel(BaseModel):
    """
    線形回帰モデル

    単純な線形変換のみで予測を行います。
    y = Wx + b
    """

    def __init__(self, input_dim: int = 13, output_dim: int = 1):
        """
        Args:
            input_dim: 入力次元数（デフォルト: 13）
            output_dim: 出力次元数（デフォルト: 1）

        Examples:
            >>> model = BaselineModel(input_dim=13, output_dim=1)
            >>> model.print_summary()
        """
        super(BaselineModel, self).__init__(input_dim, output_dim)

        # 単一の線形層
        self.linear = nn.Linear(input_dim, output_dim)

        # 重みの初期化
        self.initialize_weights(method="xavier")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理

        Args:
            x: 入力テンソル (batch_size, input_dim)

        Returns:
            出力テンソル (batch_size, output_dim)

        Examples:
            >>> model = BaselineModel()
            >>> x = torch.randn(32, 13)
            >>> output = model(x)
            >>> print(output.shape)  # torch.Size([32, 1])
        """
        return self.linear(x)

    def get_weights_and_bias(self) -> tuple:
        """
        線形層の重みとバイアスを取得

        Returns:
            (weights, bias) のタプル

        Examples:
            >>> model = BaselineModel()
            >>> weights, bias = model.get_weights_and_bias()
            >>> print(f"重み形状: {weights.shape}, バイアス形状: {bias.shape}")
        """
        return self.linear.weight.data, self.linear.bias.data
