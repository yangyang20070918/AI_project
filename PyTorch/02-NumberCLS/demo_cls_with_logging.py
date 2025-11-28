# 手書き数字認識（分類問題）- 訓練履歴記録版
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
import json
import os

# （1）data データ
# 手書き数字のデータセット
train_data = dataset.MNIST(root="mnisst",
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

test_data = dataset.MNIST(root="mnisst",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)

# データローダー
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)

test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle=True)

# （2）net ネットワークモデル
cnn = CNN()

# （3）loss 損失関数
loss_func = torch.nn.CrossEntropyLoss()

# （4）optimizer 最適化方法
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# 訓練履歴を保存するリスト
history = {
    'train_loss': [],
    'test_loss': [],
    'test_accuracy': []
}

# （5）training 訓練
print("訓練を開始します...")
for epoch in range(10):
    # 訓練フェーズ
    cnn.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 平均訓練損失
    avg_train_loss = train_loss / len(train_loader)

    # （6）test/eval テスト
    cnn.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = cnn(images)
            loss = loss_func(outputs, labels)
            test_loss += loss.item()

            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()

    # 平均テスト損失と精度
    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / len(test_data)

    # 履歴に記録
    history['train_loss'].append(avg_train_loss)
    history['test_loss'].append(avg_test_loss)
    history['test_accuracy'].append(accuracy)

    print(f"Epoch {epoch + 1}/10 - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, "
          f"Test Accuracy: {accuracy:.4f}")

# （7）save モデルと履歴の保存
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

torch.save(cnn, "model/mnist_model.pkl")
print("\nモデルを保存しました: model/mnist_model.pkl")

# 履歴をJSONファイルに保存
with open("results/training_history.json", "w") as f:
    json.dump(history, f, indent=4)
print("訓練履歴を保存しました: results/training_history.json")

print("\n訓練が完了しました！")
print(f"最終テスト精度: {history['test_accuracy'][-1]:.4f}")
