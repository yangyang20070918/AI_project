# CIFAR-10訓練結果可視化スクリプト
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorboard.backend.event_processing import event_accumulator

# 日本語フォント設定（Windows環境）
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# CIFAR-10クラス名
classes = ('飛行機', '自動車', '鳥', '猫', '鹿',
           '犬', 'カエル', '馬', '船', 'トラック')

# resultsディレクトリの作成
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("CIFAR-10 訓練結果可視化")
print("=" * 60)

# ========================================
# 1. TensorBoardログから訓練履歴を読み込む
# ========================================
print("\n[1/3] TensorBoardログを読み込んでいます...")

def read_tensorboard_logs(log_dir):
    """TensorBoardログファイルから訓練履歴を抽出"""
    try:
        # イベントファイルを検索
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if 'events.out.tfevents' in file:
                    event_files.append(os.path.join(root, file))

        if not event_files:
            print(f"⚠️  {log_dir}にイベントファイルが見つかりません")
            return None

        # 最初のイベントファイルを読み込む
        ea = event_accumulator.EventAccumulator(event_files[0])
        ea.Reload()

        history = {}

        # スカラー値を取得
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            history[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }

        return history
    except Exception as e:
        print(f"⚠️  ログ読み込みエラー: {e}")
        return None

# ログディレクトリを検索
log_dirs = []
if os.path.exists("logs"):
    for item in os.listdir("logs"):
        item_path = os.path.join("logs", item)
        if os.path.isdir(item_path):
            log_dirs.append(item_path)

if log_dirs:
    print(f"✅ {len(log_dirs)}個のログディレクトリを発見")

    # 各ログディレクトリの訓練履歴を可視化
    plt.figure(figsize=(15, 5))

    for idx, log_dir in enumerate(log_dirs):
        model_name = os.path.basename(log_dir)
        print(f"   処理中: {model_name}")
        history = read_tensorboard_logs(log_dir)

        if history and 'test loss' in history:
            plt.subplot(1, 2, 1)
            plt.plot(history['test loss']['steps'],
                    history['test loss']['values'],
                    label=model_name, linewidth=2)

        if history and 'test correct' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['test correct']['steps'],
                    history['test correct']['values'],
                    label=model_name, linewidth=2)

    plt.subplot(1, 2, 1)
    plt.title('テスト損失の推移', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.title('テスト精度の推移', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    print("✅ 訓練曲線を保存しました: results/training_curves.png")
    plt.close()
else:
    print("⚠️  logsディレクトリが見つかりません")

# ========================================
# 2. 予測結果の可視化
# ========================================
print("\n[2/3] 予測結果を生成しています...")

try:
    # テストデータの読み込み
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=True, num_workers=0)

    # 最新のモデルを読み込む
    # まず、利用可能なモデルを確認
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
    if model_files:
        # 最も大きいエポック番号のモデルを選択
        latest_model = sorted(model_files, key=lambda x: int(x.replace('.pth', '')))[-1]
        model_path = os.path.join('models', latest_model)

        # モデルの読み込みには、モデルアーキテクチャの定義が必要
        # ここでは、pre_resnetを使用していると仮定
        from pre_resnet import pytorch_resnet18

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = pytorch_resnet18().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        print(f"✅ モデルを読み込みました: {model_path}")

        # ランダムな画像で予測
        dataiter = iter(testloader)

        # 複数バッチを取得して20枚の画像を用意
        images_list = []
        labels_list = []
        for _ in range(5):
            imgs, lbls = next(dataiter)
            images_list.append(imgs)
            labels_list.append(lbls)

        images = torch.cat(images_list, dim=0)[:20]
        labels = torch.cat(labels_list, dim=0)[:20]

        # 予測
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        # 可視化
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle('CIFAR-10画像分類 - 予測結果', fontsize=16, fontweight='bold')

        for idx, ax in enumerate(axes.flat):
            # 画像を正規化解除
            img = images[idx].cpu() / 2 + 0.5
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            ax.imshow(img)

            true_label = labels[idx].item()
            pred_label = predicted[idx].item()

            # タイトル（正解か不正解かで色を変える）
            if pred_label == true_label:
                color = 'green'
                status = '✓'
            else:
                color = 'red'
                status = '✗'

            ax.set_title(f'{status} 予測: {classes[pred_label]}\n正解: {classes[true_label]}',
                        fontsize=10, color=color, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('results/predictions.png', dpi=150, bbox_inches='tight')
        print("✅ 予測結果を保存しました: results/predictions.png")
        plt.close()

        # ========================================
        # 3. クラス別精度の分析
        # ========================================
        print("\n[3/3] クラス別精度を分析しています...")

        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        # クラス別精度を可視化
        accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                     for i in range(10)]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(10), accuracies, color='skyblue', edgecolor='navy')

        # 最高精度と最低精度のバーを色分け
        max_idx = np.argmax(accuracies)
        min_idx = np.argmin(accuracies)
        bars[max_idx].set_color('green')
        bars[min_idx].set_color('red')

        plt.xlabel('クラス', fontsize=12)
        plt.ylabel('精度 (%)', fontsize=12)
        plt.title('CIFAR-10 クラス別精度', fontsize=14, fontweight='bold')
        plt.xticks(range(10), classes, rotation=45, ha='right')
        plt.ylim([0, 100])
        plt.grid(True, alpha=0.3, axis='y')

        # 精度値を表示
        for i, v in enumerate(accuracies):
            plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('results/class_accuracy.png', dpi=150, bbox_inches='tight')
        print("✅ クラス別精度を保存しました: results/class_accuracy.png")
        plt.close()

        # 全体精度を計算
        overall_accuracy = 100 * sum(class_correct) / sum(class_total)
        print(f"\n📊 全体テスト精度: {overall_accuracy:.2f}%")

    else:
        print("⚠️  訓練済みモデルが見つかりません")

except FileNotFoundError:
    print("⚠️  必要なファイルが見つかりません")
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ 可視化が完了しました！")
print("=" * 60)
print("\n📊 生成されたファイル:")
if os.path.exists('results/training_curves.png'):
    print("  - results/training_curves.png    : 訓練曲線")
if os.path.exists('results/predictions.png'):
    print("  - results/predictions.png        : 予測結果")
if os.path.exists('results/class_accuracy.png'):
    print("  - results/class_accuracy.png     : クラス別精度")
print("\n次のステップ: これらの画像をREADMEに追加しましょう！")
