"""
学習曲線の可視化

訓練と検証のlossの推移を可視化します。
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
plt.rcParams['axes.unicode_minus'] = False


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
    style: str = 'seaborn-v0_8',
    show_grid: bool = True,
    smoothing: float = 0.0
) -> plt.Figure:
    """
    学習曲線をプロット

    Args:
        history: 訓練履歴（train_loss, val_lossを含む辞書）
        save_path: 保存先パス（Noneの場合は保存しない）
        figsize: 図のサイズ
        style: matplotlibスタイル
        show_grid: グリッドを表示するか
        smoothing: スムージング係数（0.0-1.0、0はスムージングなし）

    Returns:
        matplotlib Figure オブジェクト

    Examples:
        >>> history = {'train_loss': [...], 'val_loss': [...]}
        >>> fig = plot_learning_curves(history, save_path=Path("plots/learning_curve.png"))
    """
    plt.style.use(style)
    # スタイル適用後にフォント設定を再設定
    plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # エポック数
    epochs = range(1, len(history['train_loss']) + 1)

    # スムージングを適用
    if smoothing > 0:
        train_loss = smooth_curve(history['train_loss'], smoothing)
        val_loss = smooth_curve(history['val_loss'], smoothing)
    else:
        train_loss = history['train_loss']
        val_loss = history['val_loss']

    # Loss曲線のプロット
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2, color='#2E86AB')
    ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2, color='#A23B72')

    # ベストエポックをマーク
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = min(val_loss)

    ax1.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5,
                label=f'Best (epoch {best_epoch})')
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.3)

    ax1.set_xlabel('Epoch（エポック）', fontsize=12)
    ax1.set_ylabel('Loss（損失）', fontsize=12)
    ax1.set_title('Learning Curves（学習曲線）', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)

    if show_grid:
        ax1.grid(True, alpha=0.3)

    # Loss曲線（対数スケール）
    ax2.plot(epochs, train_loss, label='Train Loss', linewidth=2, color='#2E86AB')
    ax2.plot(epochs, val_loss, label='Val Loss', linewidth=2, color='#A23B72')
    ax2.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.3)

    ax2.set_xlabel('Epoch（エポック）', fontsize=12)
    ax2.set_ylabel('Loss（損失 - 対数スケール）', fontsize=12)
    ax2.set_title('Learning Curves - Log Scale（学習曲線 - 対数表示）', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=10)

    if show_grid:
        ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # 保存
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_learning_rate_schedule(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 4),
    style: str = 'seaborn-v0_8'
) -> plt.Figure:
    """
    学習率のスケジュールをプロット

    Args:
        history: 訓練履歴（learning_ratesを含む辞書）
        save_path: 保存先パス
        figsize: 図のサイズ
        style: matplotlibスタイル

    Returns:
        matplotlib Figure オブジェクト
    """
    if 'learning_rates' not in history:
        raise ValueError("history に 'learning_rates' キーが必要です")

    plt.style.use(style)
    # スタイル適用後にフォント設定を再設定
    plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history['learning_rates']) + 1)

    ax.plot(epochs, history['learning_rates'], linewidth=2, color='#F18F01')
    ax.set_xlabel('Epoch（エポック）', fontsize=12)
    ax.set_ylabel('Learning Rate（学習率）', fontsize=12)
    ax.set_title('Learning Rate Schedule（学習率スケジュール）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_train_val_comparison(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    style: str = 'seaborn-v0_8'
) -> plt.Figure:
    """
    訓練lossと検証lossの差を可視化（過学習の検出）

    Args:
        history: 訓練履歴
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    epochs = range(1, len(history['train_loss']) + 1)

    # 訓練lossと検証loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss',
             linewidth=2, color='#2E86AB')
    ax1.plot(epochs, history['val_loss'], label='Val Loss',
             linewidth=2, color='#A23B72')

    # ベストエポック
    best_epoch = np.argmin(history['val_loss']) + 1
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
                label=f'Best Epoch ({best_epoch})')

    ax1.set_ylabel('Loss（損失）', fontsize=12)
    ax1.set_title('Training vs Validation Loss（訓練 vs 検証損失）', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # lossの差分（過学習の指標）
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])

    ax2.plot(epochs, loss_diff, linewidth=2, color='#C73E1D', label='Val - Train')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Epoch（エポック）', fontsize=12)
    ax2.set_ylabel('Loss Difference（損失の差）', fontsize=12)
    ax2.set_title('Overfitting Indicator（過学習指標）', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def smooth_curve(values: List[float], weight: float = 0.9) -> np.ndarray:
    """
    曲線をスムージング（指数移動平均）

    Args:
        values: 元の値のリスト
        weight: スムージング係数（0.0-1.0、大きいほど滑らか）

    Returns:
        スムージングされた値の配列
    """
    smoothed = []
    last = values[0]

    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)


def load_history_and_plot(
    history_path: Path,
    save_dir: Optional[Path] = None
) -> Dict[str, plt.Figure]:
    """
    訓練履歴を読み込んで全ての学習曲線をプロット

    Args:
        history_path: 訓練履歴JSONファイルのパス
        save_dir: 図の保存先ディレクトリ

    Returns:
        プロット名とFigureオブジェクトの辞書
    """
    # 履歴を読み込み
    with open(history_path, 'r') as f:
        history = json.load(f)

    figures = {}

    # 学習曲線
    save_path = save_dir / "learning_curves.png" if save_dir else None
    figures['learning_curves'] = plot_learning_curves(history, save_path=save_path)

    # 学習率スケジュール
    if 'learning_rates' in history:
        save_path = save_dir / "lr_schedule.png" if save_dir else None
        figures['lr_schedule'] = plot_learning_rate_schedule(history, save_path=save_path)

    # 訓練vs検証の比較
    save_path = save_dir / "train_val_comparison.png" if save_dir else None
    figures['train_val_comparison'] = plot_train_val_comparison(history, save_path=save_path)

    return figures
