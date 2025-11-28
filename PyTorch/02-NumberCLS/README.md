# MNIST手書き数字認識プロジェクト

## 📋 プロジェクト概要

畳み込みニューラルネットワーク（CNN）を使用して、手書き数字（0-9）を認識する画像分類モデルを実装したプロジェクトです。機械学習の代表的なベンチマークデータセットであるMNISTを使用しています。

## 🎯 目的

- CNNの基礎を理解し、画像分類問題への適用方法を習得する
- PyTorchでの画像データの扱い方を学ぶ
- バッチ処理とDataLoaderの使い方を理解する
- 訓練と評価のプロセスを実装する

## 🛠️ 使用技術

- **フレームワーク**: PyTorch, torchvision
- **言語**: Python 3.x
- **ニューラルネットワーク**: CNN (Convolutional Neural Network)

## 📊 データセット

**MNIST Dataset**
- **訓練データ**: 60,000枚
- **テストデータ**: 10,000枚
- **画像サイズ**: 28×28ピクセル（グレースケール）
- **クラス数**: 10クラス（数字0-9）
- **出典**: Yann LeCun et al.

データは自動的にダウンロードされます。

## 🏗️ モデル構造

### CNN アーキテクチャ

```python
CNN(
  (conv): Sequential(
    Conv2d(1, 32, kernel_size=5, padding=2)   # 28x28 -> 28x28
    BatchNorm2d(32)
    ReLU()
    MaxPool2d(2)                               # 28x28 -> 14x14
  )
  (fc): Linear(14*14*32=6272, 10)              # 全結合層
)
```

### 層の説明
1. **畳み込み層**:
   - 入力チャンネル: 1（グレースケール）
   - 出力チャンネル: 32
   - カーネルサイズ: 5×5
   - パディング: 2（画像サイズを維持）

2. **バッチ正規化層**: 訓練の安定化と高速化

3. **活性化関数**: ReLU（非線形性の導入）

4. **プーリング層**: MaxPool2d(2) - 画像サイズを半分に削減

5. **全結合層**: 6272次元 → 10次元（クラス数）

### ハイパーパラメータ
- **損失関数**: CrossEntropyLoss（交差エントロピー損失）
- **最適化アルゴリズム**: Adam
- **学習率**: 0.01
- **バッチサイズ**: 64
- **エポック数**: 10

## 📁 ファイル構成

```
02-NumberCLS/
├── README.md                           # プロジェクト説明（本ファイル）
├── requirements.txt                    # 依存ライブラリ
├── CNN.py                              # CNNモデル定義
├── demo_cls.py                         # 訓練スクリプト（オリジナル）
├── demo_cls_with_logging.py            # 訓練スクリプト（履歴記録版）
├── demo_cls_inference.py               # 推論スクリプト
├── visualize_results.py                # 可視化スクリプト
├── mnisst/                             # MNISTデータセット（自動ダウンロード）
├── model/
│   └── mnist_model.pkl                 # 保存された訓練済みモデル
└── results/                            # 訓練結果（可視化）
    ├── training_history.json           # 訓練履歴データ
    ├── training_curves.png             # 損失・精度グラフ
    ├── predictions.png                 # 予測結果
    └── confusion_matrix.png            # 混同行列
```

## 🚀 実行方法

### 1. 環境構築

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 2. モデルの訓練

**通常の訓練:**
```bash
python demo_cls.py
```

**訓練履歴を記録する場合（推奨）:**
```bash
python demo_cls_with_logging.py
```

初回実行時はMNISTデータセットが自動的にダウンロードされます。

### 3. 訓練結果の可視化

```bash
python visualize_results.py
```

以下の画像が `results/` ディレクトリに生成されます：
- 損失・精度の推移グラフ
- 予測結果のサンプル
- 混同行列

### 4. 推論の実行

```bash
python demo_cls_inference.py
```

訓練済みモデルを使用して、新しい画像に対する予測を行います。

## 📈 実行結果

### 訓練時の出力例

```
epoch is 1, accuracy is 0.95, loss_test is 0.1234
epoch is 2, accuracy is 0.97, loss_test is 0.0856
...
epoch is 10, accuracy is 0.99, loss_test is 0.0321
```

### 期待される性能
- **最終精度**: 約98-99%
- **訓練時間**: CPU: 約10-15分 / GPU: 約2-3分（エポック数10の場合）

## 📊 訓練結果の可視化

### 損失と精度の推移

訓練の進行に伴い、損失が減少し精度が向上していることが確認できます。

![訓練曲線](results/training_curves.png)

**グラフの見方:**
- **左**: 訓練損失（青）とテスト損失（赤）の推移
  - 両方とも減少 → 良好な学習
  - テスト損失が上昇 → 過学習の兆候
- **右**: テスト精度の推移
  - 最終的に約98-99%に到達

### 予測結果のサンプル

モデルの予測結果を視覚的に確認できます。

![予測結果](results/predictions.png)

- ✅ 緑色: 正しく分類された例
- ❌ 赤色: 誤分類された例

### 混同行列（Confusion Matrix）

各数字の分類精度を詳細に分析できます。

![混同行列](results/confusion_matrix.png)

**混同行列の見方:**
- 対角線上の値が大きい → 高精度
- 対角線外の値 → 誤分類（どの数字をどの数字と間違えたか）

## 💡 学習したポイント

### 技術面
1. **CNNの構造理解**:
   - 畳み込み層による特徴抽出
   - プーリング層による次元削減
   - 全結合層による分類

2. **画像データの前処理**:
   - ToTensor()によるテンソル変換
   - 正規化処理

3. **バッチ処理**:
   - DataLoaderを使用した効率的なデータ読み込み
   - バッチサイズの設定

4. **評価指標**:
   - 分類精度（Accuracy）の計算
   - 訓練損失とテスト損失の比較

5. **モデルの保存と読み込み**:
   - torch.save()とtorch.load()の使用

### CNNの利点
- 従来のMLPと比較して、画像の空間的構造を保持
- パラメータ数が少なく、過学習しにくい
- 平行移動不変性を持つ

## 🔧 改善案と今後の課題

- [ ] より深いCNNアーキテクチャの実装（LeNet、AlexNet風）
- [ ] データ拡張（Data Augmentation）の追加
- [ ] Dropoutによる過学習対策
- [ ] 学習率スケジューリングの導入
- [ ] 訓練・検証・テストデータの適切な分割
- [ ] 混同行列（Confusion Matrix）の作成と分析
- [ ] 誤分類された画像の可視化
- [ ] GPU対応のコード追加（.cuda()）
- [ ] TensorBoardによる学習過程の可視化
- [ ] Webアプリケーション化（手書き数字をリアルタイム認識）

## 📊 モデルの拡張可能性

このモデルは以下の用途に拡張できます：
- リアルタイム手書き数字認識アプリ
- 郵便番号自動読み取りシステム
- 数式認識システムの基礎モデル
- Transfer Learningのベースモデル

## 📝 参考資料

- [MNIST Dataset - Yann LeCun's Website](http://yann.lecun.com/exdb/mnist/)
- [PyTorch公式チュートリアル - MNIST](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

## 🏆 実装の特徴

- ✅ シンプルで理解しやすいCNN実装
- ✅ 詳細な日本語コメント
- ✅ バッチ処理とエポックの適切な管理
- ✅ 訓練とテストの明確な分離
- ✅ モデルの保存機能

## 👤 作成者

楊様 (Youyo)

## 📅 作成日

2024年

---

**注**: このプロジェクトは機械学習・深層学習の学習目的で作成されました。
