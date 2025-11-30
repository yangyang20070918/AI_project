"""
シンプルニューラルネットワーク

1層の隠れ層を持つシンプルなニューラルネットワークを提供します。
"""

import torch
import torch.nn as nn
from .base_model import BaseModel


class SimpleNN(BaseModel):
    """
    シンプルなニューラルネットワーク

    構造: Input -> Hidden -> Output
    活性化関数: ReLU
    正則化: Dropout（オプション）、BatchNorm（オプション）
    """

    def __init__(
        self,
        input_dim: int = 13,
        output_dim: int = 1,
        hidden_size: int = 100,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = False
    ):
        """
        Args:
            input_dim: 入力次元数
            output_dim: 出力次元数
            hidden_size: 隠れ層のユニット数
            dropout_rate: Dropout率（0.0-1.0）
            use_batch_norm: Batch Normalizationを使用するか

        Examples:
            >>> model = SimpleNN(hidden_size=100, dropout_rate=0.2)
            >>> model.print_summary()
        """
        super(SimpleNN, self).__init__(input_dim, output_dim)

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # 隠れ層
        self.hidden = nn.Linear(input_dim, hidden_size)

        # Batch Normalization（オプション）
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)

        # ReLU活性化関数
        self.relu = nn.ReLU()

        # Dropout（オプション）
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        # 出力層
        self.output = nn.Linear(hidden_size, output_dim)

        # 重みの初期化
        self.initialize_weights(method="kaiming")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理

        Args:
            x: 入力テンソル (batch_size, input_dim)

        Returns:
            出力テンソル (batch_size, output_dim)

        Examples:
            >>> model = SimpleNN()
            >>> x = torch.randn(32, 13)
            >>> output = model(x)
            >>> print(output.shape)  # torch.Size([32, 1])
        """
        # 隠れ層
        x = self.hidden(x)

        # Batch Normalization
        if self.use_batch_norm:
            x = self.batch_norm(x)

        # 活性化関数
        x = self.relu(x)

        # Dropout
        if self.dropout_rate > 0:
            x = self.dropout(x)

        # 出力層
        x = self.output(x)

        return x
