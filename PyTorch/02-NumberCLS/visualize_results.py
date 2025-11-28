# MNIST訓練結果可視化スクリプト
import json
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import numpy as np
import os

# 日本語フォント設定（Windows環境）
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# resultsディレクトリの作成
os.makedirs("results", exist_ok=True)

print("可視化を開始します...")

# ========================================
# 1. 訓練履歴の可視化
# ========================================
print("\n[1/3] 訓練履歴を読み込んでいます...")
try:
    with open("results/training_history.json", "r") as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # 損失の推移グラフ
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['test_loss'], 'r-s', label='Test Loss', linewidth=2)
    plt.title('損失の推移', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 精度の推移グラフ
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['test_accuracy'], 'g-^', label='Test Accuracy', linewidth=2)
    plt.title('テスト精度の推移', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    print("✅ 訓練曲線を保存しました: results/training_curves.png")
    plt.close()

except FileNotFoundError:
    print("⚠️  training_history.jsonが見つかりません。")
    print("   先にdemo_cls_with_logging.pyを実行してください。")

# ========================================
# 2. 予測結果の可視化
# ========================================
print("\n[2/3] モデルを読み込んでいます...")
try:
    # モデルの読み込み
    model = torch.load("model/mnist_model.pkl", map_location=torch.device('cpu'))
    model.eval()

    # テストデータの読み込み
    test_data = dataset.MNIST(root="mnisst",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)

    print("[3/3] 予測結果を生成しています...")

    # ランダムに20枚選択
    indices = np.random.choice(len(test_data), 20, replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('MNIST手書き数字認識 - 予測結果', fontsize=16, fontweight='bold')

    with torch.no_grad():
        for idx, ax in zip(indices, axes.flat):
            image, true_label = test_data[idx]

            # 予測
            output = model(image.unsqueeze(0))
            _, predicted = output.max(1)
            pred_label = predicted.item()

            # 画像表示
            ax.imshow(image.squeeze(), cmap='gray')

            # タイトル（正解か不正解かで色を変える）
            if pred_label == true_label:
                color = 'green'
                status = '✓'
            else:
                color = 'red'
                status = '✗'

            ax.set_title(f'{status} 予測: {pred_label} / 正解: {true_label}',
                        fontsize=11, color=color, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('results/predictions.png', dpi=150, bbox_inches='tight')
    print("✅ 予測結果を保存しました: results/predictions.png")
    plt.close()

    # ========================================
    # 3. 混同行列（Confusion Matrix）
    # ========================================
    print("\n[ボーナス] 混同行列を作成しています...")
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    all_preds = []
    all_labels = []

    # 全テストデータで予測
    with torch.no_grad():
        for i in range(len(test_data)):
            image, label = test_data[i]
            output = model(image.unsqueeze(0))
            _, pred = output.max(1)
            all_preds.append(pred.item())
            all_labels.append(label)

    # 混同行列の計算
    cm = confusion_matrix(all_labels, all_preds)

    # 可視化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('混同行列（Confusion Matrix）', fontsize=14, fontweight='bold')
    plt.ylabel('真のラベル', fontsize=12)
    plt.xlabel('予測ラベル', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✅ 混同行列を保存しました: results/confusion_matrix.png")
    plt.close()

    print("\n" + "="*50)
    print("✅ すべての可視化が完了しました！")
    print("="*50)
    print("\n📊 生成されたファイル:")
    print("  - results/training_curves.png    : 訓練曲線")
    print("  - results/predictions.png        : 予測結果")
    print("  - results/confusion_matrix.png   : 混同行列")
    print("\n次のステップ: これらの画像をREADMEに追加しましょう！")

except FileNotFoundError as e:
    print(f"⚠️  ファイルが見つかりません: {e}")
    print("   先にdemo_cls_with_logging.pyを実行してください。")
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
