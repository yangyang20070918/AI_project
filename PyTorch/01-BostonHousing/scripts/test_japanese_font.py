"""
日本語フォントのテスト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

# 可視化モジュールと同じフォント設定を適用
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'BIZ UDGothic']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("日本語フォントテスト")
print("=" * 70)
print(f"\n使用フォント設定: {plt.rcParams['font.sans-serif']}")

# テスト画像を作成
fig, ax = plt.subplots(figsize=(12, 8))

# タイトルと軸ラベル（実際の可視化と同じ）
ax.set_title('Learning Curves（学習曲線）', fontsize=16, fontweight='bold')
ax.set_xlabel('Epoch（エポック）', fontsize=14)
ax.set_ylabel('Loss（損失）', fontsize=14)

# ダミーデータ
x = np.linspace(0, 100, 100)
y1 = np.exp(-x/30) + 0.1
y2 = np.exp(-x/50) + 0.15

ax.plot(x, y1, label='Train Loss', linewidth=2, color='#2E86AB')
ax.plot(x, y2, label='Val Loss', linewidth=2, color='#A23B72')

ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# 日本語テキストを追加
ax.text(50, 0.6, '日本語テスト: これが正しく表示されれば成功です',
        fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存
output_path = project_root / "test_japanese_plot.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nテスト画像を保存しました: {output_path}")
print("\nこのファイルを開いて、日本語が□□□ではなく正しく表示されているか確認してください。")
print("=" * 70)

plt.close()
