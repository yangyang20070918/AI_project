"""
システムで利用可能な日本語フォントを確認するスクリプト
"""

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

print("=" * 70)
print("利用可能なフォント一覧（日本語フォントを探す）")
print("=" * 70)

# 全フォントを取得
fonts = fm.findSystemFonts()

# 日本語フォントの候補
japanese_font_keywords = ['Gothic', 'Mincho', 'gothic', 'mincho', 'Yu', 'MS', 'Meiryo', 'メイリオ', 'ゴシック', '明朝']

print("\n日本語フォントの候補:")
japanese_fonts = []

for font in fonts:
    font_name = fm.FontProperties(fname=font).get_name()
    for keyword in japanese_font_keywords:
        if keyword in font_name:
            if font_name not in japanese_fonts:
                japanese_fonts.append(font_name)
                print(f"  - {font_name}")
            break

print("\n" + "=" * 70)
print(f"見つかった日本語フォント: {len(japanese_fonts)}個")
print("=" * 70)

# 推奨設定を表示
if japanese_fonts:
    print("\n推奨設定:")
    print("=" * 70)
    print("以下を可視化モジュールの先頭に追加してください：")
    print()
    print("import matplotlib.pyplot as plt")
    print("plt.rcParams['font.sans-serif'] = [")
    for i, font in enumerate(japanese_fonts[:5]):  # 上位5個
        if i < len(japanese_fonts[:5]) - 1:
            print(f"    '{font}',")
        else:
            print(f"    '{font}'")
    print("]")
    print("plt.rcParams['axes.unicode_minus'] = False")
    print("=" * 70)
else:
    print("\n警告: 日本語フォントが見つかりませんでした。")
    print("フォントをインストールする必要があります。")

# テスト画像を生成
print("\nテスト画像を生成中...")

fig, ax = plt.subplots(figsize=(8, 4))
ax.text(0.5, 0.5, '日本語テスト - Japanese Test',
        ha='center', va='center', fontsize=20)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

test_image_path = "font_test.png"
plt.savefig(test_image_path, dpi=100, bbox_inches='tight')
print(f"テスト画像を保存: {test_image_path}")
print("このファイルを開いて日本語が正しく表示されるか確認してください。")

plt.close()
