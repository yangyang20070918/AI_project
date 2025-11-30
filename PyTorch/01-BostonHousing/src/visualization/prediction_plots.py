"""
予測結果の可視化

予測値vs実測値、残差プロットなどを作成します。
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy import stats

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
plt.rcParams['axes.unicode_minus'] = False


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 8),
    style: str = 'seaborn-v0_8',
    title: str = "Predictions vs Actual Values（予測値 vs 実測値）"
) -> plt.Figure:
    """
    予測値vs実測値の散布図をプロット

    Args:
        y_true: 実測値
        y_pred: 予測値
        save_path: 保存先パス
        figsize: 図のサイズ
        style: matplotlibスタイル
        title: タイトル

    Returns:
        matplotlib Figure オブジェクト
    """
    plt.style.use(style)
    # スタイル適用後にフォント設定を再設定
    plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=figsize)

    # 散布図
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black',
               linewidth=0.5, color='#2E86AB')

    # 完全予測ライン（y=x）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='Perfect Prediction')

    # R²スコアを計算して表示
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)

    ax.text(0.05, 0.95, f'R² = {r2:.4f}',
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Actual Values（実測値）', fontsize=12)
    ax.set_ylabel('Predicted Values（予測値）', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 軸を等しくする
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    style: str = 'seaborn-v0_8'
) -> plt.Figure:
    """
    残差プロットを作成

    Args:
        y_true: 実測値
        y_pred: 予測値
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

    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 残差 vs 予測値
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5, color='#A23B72')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)

    ax1.set_xlabel('Predicted Values（予測値）', fontsize=12)
    ax1.set_ylabel('Residuals（残差）', fontsize=12)
    ax1.set_title('Residuals vs Predicted Values（残差 vs 予測値）', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 残差のヒストグラム
    ax2.hist(residuals, bins=30, alpha=0.7, color='#F18F01',
             edgecolor='black', linewidth=0.5)

    # 正規分布の曲線を重ねる
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 30,
             'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')

    ax2.set_xlabel('Residuals（残差）', fontsize=12)
    ax2.set_ylabel('Frequency（頻度）', fontsize=12)
    ax2.set_title('Distribution of Residuals（残差の分布）', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_qq_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 8),
    style: str = 'seaborn-v0_8'
) -> plt.Figure:
    """
    残差のQQプロット（正規性の確認）

    Args:
        y_true: 実測値
        y_pred: 予測値
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

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    # QQプロット
    stats.probplot(residuals, dist="norm", plot=ax)

    ax.set_title('Q-Q Plot of Residuals（残差のQ-Qプロット）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
    style: str = 'seaborn-v0_8'
) -> plt.Figure:
    """
    誤差の分布を可視化

    Args:
        y_true: 実測値
        y_pred: 予測値
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

    absolute_errors = np.abs(y_true - y_pred)
    percentage_errors = (absolute_errors / (y_true + 1e-8)) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 絶対誤差のヒストグラム
    ax1.hist(absolute_errors, bins=30, alpha=0.7, color='#C73E1D',
             edgecolor='black', linewidth=0.5)
    ax1.axvline(x=absolute_errors.mean(), color='blue', linestyle='--',
                linewidth=2, label=f'Mean = {absolute_errors.mean():.2f}')
    ax1.axvline(x=np.median(absolute_errors), color='green', linestyle='--',
                linewidth=2, label=f'Median = {np.median(absolute_errors):.2f}')

    ax1.set_xlabel('Absolute Error（絶対誤差）', fontsize=12)
    ax1.set_ylabel('Frequency（頻度）', fontsize=12)
    ax1.set_title('Absolute Error Distribution（絶対誤差の分布）', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # パーセント誤差のヒストグラム
    ax2.hist(percentage_errors, bins=30, alpha=0.7, color='#577590',
             edgecolor='black', linewidth=0.5)
    ax2.axvline(x=percentage_errors.mean(), color='blue', linestyle='--',
                linewidth=2, label=f'Mean = {percentage_errors.mean():.2f}%')
    ax2.axvline(x=np.median(percentage_errors), color='green', linestyle='--',
                linewidth=2, label=f'Median = {np.median(percentage_errors):.2f}%')

    ax2.set_xlabel('Percentage Error（誤差率 %）', fontsize=12)
    ax2.set_ylabel('Frequency（頻度）', fontsize=12)
    ax2.set_title('Percentage Error Distribution（誤差率の分布）', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_all_prediction_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Optional[Path] = None,
    dataset_name: str = "Test"
) -> dict:
    """
    全ての予測プロットを一度に作成

    Args:
        y_true: 実測値
        y_pred: 予測値
        save_dir: 保存先ディレクトリ
        dataset_name: データセット名（タイトルに使用）

    Returns:
        プロット名とFigureオブジェクトの辞書
    """
    figures = {}

    # 予測vs実測値
    save_path = save_dir / f"predictions_vs_actual_{dataset_name.lower()}.png" if save_dir else None
    figures['predictions_vs_actual'] = plot_predictions_vs_actual(
        y_true, y_pred, save_path=save_path,
        title=f"Predictions vs Actual Values ({dataset_name} Set)"
    )

    # 残差プロット
    save_path = save_dir / f"residuals_{dataset_name.lower()}.png" if save_dir else None
    figures['residuals'] = plot_residuals(y_true, y_pred, save_path=save_path)

    # QQプロット
    save_path = save_dir / f"qq_plot_{dataset_name.lower()}.png" if save_dir else None
    figures['qq_plot'] = plot_qq_plot(y_true, y_pred, save_path=save_path)

    # 誤差分布
    save_path = save_dir / f"error_distribution_{dataset_name.lower()}.png" if save_dir else None
    figures['error_distribution'] = plot_error_distribution(y_true, y_pred, save_path=save_path)

    return figures
