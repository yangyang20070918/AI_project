# CycleGAN画像変換結果可視化スクリプト
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from glob import glob

# 日本語フォント設定（Windows環境）
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# resultsディレクトリの作成
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("CycleGAN 画像変換結果可視化")
print("=" * 60)

# ========================================
# 1. outputsディレクトリから変換結果を収集
# ========================================
print("\n[1/2] 変換結果の画像を収集しています...")

# outputsディレクトリの画像を取得
output_dirs = {
    'A': 'outputs/A',
    'B': 'outputs/B'
}

images_found = False

for domain, path in output_dirs.items():
    if os.path.exists(path):
        image_files = glob(os.path.join(path, '*.png'))  + glob(os.path.join(path, '*.jpg'))
        if image_files:
            images_found = True
            print(f"✅ {domain}ドメイン: {len(image_files)}枚の画像を発見")

if not images_found:
    print("⚠️  outputsディレクトリに画像が見つかりません。")
    print("   先にtrain.pyまたはtest.pyを実行してください。")
    print("\n代わりに、サンプルの可視化レイアウトを作成します...")

    # サンプルレイアウトの作成
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('CycleGAN 画像変換結果（サンプルレイアウト）',
                fontsize=16, fontweight='bold')

    sample_texts = [
        ['元画像A\n(リンゴ)', '変換結果A→B\n(リンゴ→オレンジ)', '再構成A\n(サイクル)'],
        ['元画像B\n(オレンジ)', '変換結果B→A\n(オレンジ→リンゴ)', '再構成B\n(サイクル)'],
        ['元画像A', '変換結果A→B', '再構成A']
    ]

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.text(0.5, 0.5, sample_texts[i][j],
                   ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('results/sample_layout.png', dpi=150, bbox_inches='tight')
    print("✅ サンプルレイアウトを保存しました: results/sample_layout.png")
    plt.close()

else:
    # ========================================
    # 2. 変換結果の可視化
    # ========================================
    print("\n[2/2] 変換結果を可視化しています...")

    # 各ドメインから最大6枚ずつ画像を取得
    samples = {}
    for domain, path in output_dirs.items():
        image_files = glob(os.path.join(path, '*.png')) + glob(os.path.join(path, '*.jpg'))
        # 最新の6枚を取得
        image_files = sorted(image_files, key=os.path.getmtime, reverse=True)[:6]
        samples[domain] = image_files

    # ドメインAの変換結果を可視化
    if samples['A']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('ドメインA → ドメインB 変換結果',
                    fontsize=16, fontweight='bold')

        for idx, (ax, img_path) in enumerate(zip(axes.flat, samples['A'])):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f'サンプル {idx+1}', fontsize=12)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'エラー:\n{str(e)}',
                       ha='center', va='center')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('results/transformation_A2B.png', dpi=150, bbox_inches='tight')
        print("✅ A→B変換結果を保存しました: results/transformation_A2B.png")
        plt.close()

    # ドメインBの変換結果を可視化
    if samples['B']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('ドメインB → ドメインA 変換結果',
                    fontsize=16, fontweight='bold')

        for idx, (ax, img_path) in enumerate(zip(axes.flat, samples['B'])):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f'サンプル {idx+1}', fontsize=12)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'エラー:\n{str(e)}',
                       ha='center', va='center')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('results/transformation_B2A.png', dpi=150, bbox_inches='tight')
        print("✅ B→A変換結果を保存しました: results/transformation_B2A.png")
        plt.close()

    # 対比表示（A→B→A と B→A→B）
    if samples['A'] and samples['B']:
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle('CycleGAN 変換結果の比較', fontsize=16, fontweight='bold')

        # A→Bの例
        ax1 = plt.subplot(1, 2, 1)
        try:
            img_a = Image.open(samples['A'][0])
            ax1.imshow(img_a)
            ax1.set_title('A → B（例: リンゴ → オレンジ）', fontsize=14, fontweight='bold')
            ax1.axis('off')
        except:
            pass

        # B→Aの例
        ax2 = plt.subplot(1, 2, 2)
        try:
            img_b = Image.open(samples['B'][0])
            ax2.imshow(img_b)
            ax2.set_title('B → A（例: オレンジ → リンゴ）', fontsize=14, fontweight='bold')
            ax2.axis('off')
        except:
            pass

        plt.tight_layout()
        plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
        print("✅ 変換結果の比較を保存しました: results/comparison.png")
        plt.close()

print("\n" + "=" * 60)
print("✅ 可視化が完了しました！")
print("=" * 60)
print("\n📊 生成されたファイル:")
if os.path.exists('results/transformation_A2B.png'):
    print("  - results/transformation_A2B.png  : A→B変換結果")
if os.path.exists('results/transformation_B2A.png'):
    print("  - results/transformation_B2A.png  : B→A変換結果")
if os.path.exists('results/comparison.png'):
    print("  - results/comparison.png          : 変換結果の比較")
if os.path.exists('results/sample_layout.png'):
    print("  - results/sample_layout.png       : サンプルレイアウト")
print("\n次のステップ: これらの画像をREADMEに追加しましょう！")

# ========================================
# 補足情報の表示
# ========================================
print("\n📝 CycleGANについて:")
print("  CycleGANは、ペア画像データなしで異なるドメイン間の")
print("  画像変換を学習できる生成モデルです。")
print("\n  主な特徴:")
print("  - Unpaired Image-to-Image Translation")
print("  - Cycle Consistency Loss（循環一貫性損失）")
print("  - 2つのGenerator + 2つのDiscriminator")
print("\n  応用例:")
print("  - 写真 ↔ 絵画のスタイル変換")
print("  - 夏 ↔ 冬の景色変換")
print("  - 馬 ↔ シマウマの変換")
print("  - 医療画像のモダリティ変換")
