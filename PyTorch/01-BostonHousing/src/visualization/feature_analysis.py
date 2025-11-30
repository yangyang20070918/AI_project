"""
特徴量分析の可視化

特徴量の重要度や相関などを可視化します。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import torch
import torch.nn as nn

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
plt.rcParams['axes.unicode_minus'] = False


def plot_feature_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    style: str = 'seaborn-v0_8'
) -> plt.Figure:
    """
    特徴量間の相関マトリックスをプロット

    Args:
        X: 特徴量配列
        y: ターゲット配列
        feature_names: 特徴量名のリスト
        save_path: 保存先パス
        figsize: 図のサイズ
        style: matplotlibスタイル

    Returns:
        matplotlib Figure オブジェクト
    """
    plt.style.use(style)
    # スタイル適用後にフォント設定を再設定
    plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
    plt.rcParams['axes.unicode_minus'] = False

    # データフレームの作成
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y

    # 相関行列の計算
    corr = df.corr()

    # ヒートマップのプロット
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title('Feature Correlation Matrix（特徴量相関マトリックス）', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_feature_importance_from_weights(
    model: nn.Module,
    feature_names: List[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    style: str = 'seaborn-v0_8',
    top_k: int = None
) -> plt.Figure:
    """
    モデルの重みから特徴量重要度をプロット（線形モデルまたは最初の層）

    Args:
        model: 訓練済みモデル
        feature_names: 特徴量名のリスト
        save_path: 保存先パス
        figsize: 図のサイズ
        style: matplotlibスタイル
        top_k: 上位k個のみを表示（Noneの場合は全て）

    Returns:
        matplotlib Figure オブジェクト
    """
    plt.style.use(style)
    # スタイル適用後にフォント設定を再設定
    plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
    plt.rcParams['axes.unicode_minus'] = False

    # モデルの最初の線形層から重みを取得
    first_linear_layer = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            first_linear_layer = module
            break

    if first_linear_layer is None:
        raise ValueError("モデルに線形層が見つかりません")

    # 重みを取得（絶対値の平均を特徴量重要度として使用）
    weights = first_linear_layer.weight.data.cpu().numpy()
    importance = np.abs(weights).mean(axis=0)

    # 特徴量名
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(len(importance))]

    # ソート
    indices = np.argsort(importance)[::-1]

    if top_k is not None:
        indices = indices[:top_k]

    # プロット
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importance[indices], color='#2E86AB', edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance（重要度 - 絶対重み）', fontsize=12)
    ax.set_title('Feature Importance from Model Weights（モデル重みからの特徴量重要度）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_feature_distributions(
    X: np.ndarray,
    feature_names: List[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
    style: str = 'seaborn-v0_8'
) -> plt.Figure:
    """
    各特徴量の分布をヒストグラムでプロット

    Args:
        X: 特徴量配列
        feature_names: 特徴量名のリスト
        save_path: 保存先パス
        figsize: 図のサイズ
        style: matplotlibスタイル

    Returns:
        matplotlib Figure オブジェクト
    """
    plt.style.use(style)
    # スタイル適用後にフォント設定を再設定
    plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
    plt.rcParams['axes.unicode_minus'] = False

    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]

    # サブプロットのグリッドを計算
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_features):
        axes[i].hist(X[:, i], bins=30, alpha=0.7, color='#F18F01',
                     edgecolor='black', linewidth=0.5)
        axes[i].set_title(feature_names[i], fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Frequency（頻度）', fontsize=9)
        axes[i].grid(True, alpha=0.3)

    # 余分な軸を非表示
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Feature Distributions（特徴量の分布）', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# Boston Housing データセットの特徴量名
BOSTON_FEATURE_NAMES = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]


def create_all_feature_analysis_plots(
    X: np.ndarray,
    y: np.ndarray,
    model: Optional[nn.Module] = None,
    feature_names: List[str] = None,
    save_dir: Optional[Path] = None
) -> dict:
    """
    全ての特徴量分析プロットを作成

    Args:
        X: 特徴量配列
        y: ターゲット配列
        model: 訓練済みモデル（オプション）
        feature_names: 特徴量名のリスト
        save_dir: 保存先ディレクトリ

    Returns:
        プロット名とFigureオブジェクトの辞書
    """
    if feature_names is None:
        feature_names = BOSTON_FEATURE_NAMES

    figures = {}

    # 相関マトリックス
    save_path = save_dir / "feature_correlation.png" if save_dir else None
    figures['correlation'] = plot_feature_correlation(
        X, y, feature_names, save_path=save_path
    )

    # 特徴量の分布
    save_path = save_dir / "feature_distributions.png" if save_dir else None
    figures['distributions'] = plot_feature_distributions(
        X, feature_names, save_path=save_path
    )

    # モデルの重みからの重要度（モデルが提供されている場合）
    if model is not None:
        try:
            save_path = save_dir / "feature_importance.png" if save_dir else None
            figures['importance'] = plot_feature_importance_from_weights(
                model, feature_names, save_path=save_path
            )
        except Exception as e:
            print(f"特徴量重要度のプロットをスキップ: {e}")

    return figures
