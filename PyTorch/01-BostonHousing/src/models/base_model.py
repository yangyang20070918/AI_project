"""
モデル基底クラス

全てのモデルが継承する共通機能を提供します。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    全てのモデルの基底クラス

    共通の機能とインターフェースを提供します。
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: 入力次元数
            output_dim: 出力次元数
        """
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理（サブクラスで実装必須）

        Args:
            x: 入力テンソル (batch_size, input_dim)

        Returns:
            出力テンソル (batch_size, output_dim)
        """
        pass

    def count_parameters(self) -> int:
        """
        モデルの学習可能なパラメータ数をカウント

        Returns:
            パラメータ数

        Examples:
            >>> model = SomeModel(input_dim=13, output_dim=1)
            >>> print(f"パラメータ数: {model.count_parameters():,}")
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict[str, Any]:
        """
        モデルの概要情報を取得

        Returns:
            モデル情報を含む辞書
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.count_parameters()
        non_trainable_params = total_params - trainable_params

        return {
            "model_name": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "model_size_mb": total_params * 4 / (1024 ** 2)  # float32 = 4 bytes
        }

    def print_summary(self) -> None:
        """
        モデルの概要を表示

        Examples:
            >>> model = SomeModel(input_dim=13, output_dim=1)
            >>> model.print_summary()
        """
        summary = self.get_model_summary()

        print("=" * 70)
        print(f"モデル: {summary['model_name']}")
        print("=" * 70)
        print(f"入力次元:               {summary['input_dim']}")
        print(f"出力次元:               {summary['output_dim']}")
        print("-" * 70)
        print(f"総パラメータ数:         {summary['total_parameters']:,}")
        print(f"訓練可能パラメータ数:   {summary['trainable_parameters']:,}")
        print(f"訓練不可パラメータ数:   {summary['non_trainable_parameters']:,}")
        print(f"モデルサイズ:           {summary['model_size_mb']:.2f} MB")
        print("=" * 70)

    def initialize_weights(self, method: str = "xavier") -> None:
        """
        重みの初期化

        Args:
            method: 初期化方法 ("xavier", "kaiming", "normal", "uniform")

        Examples:
            >>> model = SomeModel(input_dim=13, output_dim=1)
            >>> model.initialize_weights(method="xavier")
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == "normal":
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                elif method == "uniform":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                else:
                    raise ValueError(f"未対応の初期化方法: {method}")

                # バイアスは0で初期化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                # BatchNormのパラメータ初期化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layer_info(self) -> list:
        """
        各層の情報を取得

        Returns:
            層情報のリスト

        Examples:
            >>> model = SomeModel(input_dim=13, output_dim=1)
            >>> for layer in model.get_layer_info():
            ...     print(layer)
        """
        layer_info = []

        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 末端層のみ
                num_params = sum(p.numel() for p in module.parameters())

                layer_info.append({
                    "name": name if name else "root",
                    "type": module.__class__.__name__,
                    "parameters": num_params
                })

        return layer_info

    def freeze_layers(self, layer_names: list = None) -> None:
        """
        指定した層のパラメータを凍結（訓練しない）

        Args:
            layer_names: 凍結する層の名前リスト（Noneの場合は全層凍結）

        Examples:
            >>> model = SomeModel(input_dim=13, output_dim=1)
            >>> model.freeze_layers(['layer1', 'layer2'])
        """
        if layer_names is None:
            # 全層凍結
            for param in self.parameters():
                param.requires_grad = False
        else:
            # 指定層のみ凍結
            for name, param in self.named_parameters():
                for layer_name in layer_names:
                    if layer_name in name:
                        param.requires_grad = False
                        break

    def unfreeze_layers(self, layer_names: list = None) -> None:
        """
        指定した層のパラメータを凍結解除（訓練する）

        Args:
            layer_names: 凍結解除する層の名前リスト（Noneの場合は全層解除）

        Examples:
            >>> model = SomeModel(input_dim=13, output_dim=1)
            >>> model.unfreeze_layers(['layer3', 'layer4'])
        """
        if layer_names is None:
            # 全層凍結解除
            for param in self.parameters():
                param.requires_grad = True
        else:
            # 指定層のみ凍結解除
            for name, param in self.named_parameters():
                for layer_name in layer_names:
                    if layer_name in name:
                        param.requires_grad = True
                        break

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        予測（評価モード）

        Args:
            x: 入力テンソル

        Returns:
            予測値

        Examples:
            >>> model = SomeModel(input_dim=13, output_dim=1)
            >>> x = torch.randn(10, 13)
            >>> predictions = model.predict(x)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions

    def save_model(self, path: str) -> None:
        """
        モデルを保存

        Args:
            path: 保存先パス

        Examples:
            >>> model.save_model("model/best_model.pth")
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_summary': self.get_model_summary()
        }, path)

    def load_model(self, path: str, device: str = 'cpu') -> None:
        """
        モデルを読み込み

        Args:
            path: 読み込むモデルのパス
            device: デバイス ('cpu' or 'cuda')

        Examples:
            >>> model = SomeModel(input_dim=13, output_dim=1)
            >>> model.load_model("model/best_model.pth", device='cuda')
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
