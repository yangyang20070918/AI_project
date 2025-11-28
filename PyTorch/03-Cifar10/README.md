# CIFAR-10画像分類プロジェクト

## 📋 プロジェクト概要

CIFAR-10データセットを使用した画像分類プロジェクトです。複数の最先端CNNアーキテクチャ（VGGNet、ResNet、MobileNet、Inception、事前訓練済みResNet18）を実装し、それぞれの性能を比較検証しました。TensorBoardを使用した訓練過程の可視化も実装しています。

## 🎯 目的

- 複数の代表的なCNNアーキテクチャを理解し実装する
- モデルアーキテクチャの違いが性能に与える影響を比較する
- 学習率スケジューリングとハイパーパラメータ調整の経験を積む
- TensorBoardを使用した実験管理手法を習得する
- Transfer Learning（転移学習）の実践

## 🛠️ 使用技術

- **フレームワーク**: PyTorch, torchvision
- **言語**: Python 3.x
- **可視化ツール**: TensorBoardX
- **実装モデル**:
  - VGGNet（自作実装）
  - ResNet（自作実装）
  - MobileNetV1（自作実装）
  - Inception Module（自作実装）
  - ResNet18（PyTorch事前訓練済みモデル）

## 📊 データセット

**CIFAR-10 Dataset**
- **訓練データ**: 50,000枚
- **テストデータ**: 10,000枚
- **画像サイズ**: 32×32ピクセル（RGB）
- **クラス数**: 10クラス
  - airplane（飛行機）
  - automobile（自動車）
  - bird（鳥）
  - cat（猫）
  - deer（鹿）
  - dog（犬）
  - frog（カエル）
  - horse（馬）
  - ship（船）
  - truck（トラック）

## 🏗️ 実装されたモデルアーキテクチャ

### 1. VGGNet (`vggnet.py`)
- VGG風の深い畳み込みネットワーク
- 3×3の小さなフィルタを重ねることで深いネットワークを実現
- バッチ正規化を使用

### 2. ResNet (`resnet.py`)
- **残差接続（Residual Connection）**を実装
- 深いネットワークでも勾配消失問題を回避
- ResBlockを積み重ねた構造
- 4つの層（64→128→256→512チャンネル）

```python
ResNet(
  conv1: Conv2d(3, 32) + BatchNorm + ReLU
  layer1: ResBlock(32→64) × 2
  layer2: ResBlock(64→128) × 2
  layer3: ResBlock(128→256) × 2
  layer4: ResBlock(256→512) × 2
  fc: Linear(512, 10)
)
```

### 3. MobileNetV1 (`mobilenetvl.py`)
- **Depthwise Separable Convolution**を使用
- 計算量とパラメータ数を大幅に削減
- モバイルデバイス向けの軽量モデル

### 4. Inception Module (`inceptionMolule.py`)
- 複数のフィルタサイズを並列に適用
- マルチスケールな特徴抽出
- GoogLeNet（Inception v1）風の実装

### 5. 事前訓練済みResNet18 (`pre_resnet.py`)
- PyTorchが提供する事前訓練済みモデルを使用
- ImageNetで学習した重みを利用（Transfer Learning）
- CIFAR-10用に最終層のみ変更

## 📁 ファイル構成

```
03-Cifar10/
├── README.md                    # プロジェクト説明（本ファイル）
├── requirements.txt             # 依存ライブラリ
├── train.py                     # 訓練スクリプト（メイン）
├── test.py                      # テストスクリプト
├── load_cifar10.py              # データローダー
├── readcifar10.py               # データ読み込みユーティリティ
├── vggnet.py                    # VGGNetモデル定義
├── resnet.py                    # ResNetモデル定義
├── mobilenetvl.py               # MobileNetモデル定義
├── inceptionMolule.py           # Inceptionモデル定義
├── pre_resnet.py                # 事前訓練済みResNet18
├── models/                      # 訓練済みモデル保存ディレクトリ
└── logs/                        # TensorBoardログディレクトリ
```

## 🚀 実行方法

### 1. 環境構築

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 2. モデルの訓練

`train.py`の中でモデルを選択して実行：

```python
# train.py の中で使用したいモデルのコメントを外す
from pre_resnet import pytorch_resnet18  # 例: 事前訓練済みResNet18
net = pytorch_resnet18().to(device)
```

```bash
# 訓練の実行
python train.py
```

### 3. TensorBoardで訓練過程を確認

```bash
tensorboard --logdir=logs
```

ブラウザで `http://localhost:6006` にアクセスして、以下を確認できます：
- 訓練損失とテスト損失の推移
- 訓練精度とテスト精度の推移
- 学習率の変化
- 訓練画像とテスト画像のサンプル

### 4. テストの実行

```bash
python test.py
```

## ⚙️ ハイパーパラメータ

```python
epoch_num = 200              # エポック数
lr = 0.01                    # 初期学習率
batch_size = 6               # バッチサイズ
optimizer = Adam             # 最適化アルゴリズム
loss_func = CrossEntropyLoss # 損失関数

# 学習率スケジューリング
scheduler = StepLR(
    step_size=5,   # 5エポックごとに学習率を減衰
    gamma=0.9      # 減衰率90%
)
```

## 📈 期待される実行結果

### 訓練時の出力例

```
train epoch is 0 lr is 0.01
epoch is 1, loss is: 1.234, test correct is: 45.6%

train epoch is 5 lr is 0.009  # 学習率が減衰
epoch is 6, loss is: 0.856, test correct is: 62.3%

...

train epoch is 199 lr is 0.00123
epoch is 200, loss is: 0.234, test correct is: 85.7%
```

### モデル別の期待精度（参考値）

| モデル | パラメータ数 | テスト精度 | 訓練時間 |
|--------|------------|-----------|---------|
| VGGNet | 〜15M | 80-85% | 中 |
| ResNet（自作） | 〜11M | 85-90% | 中 |
| MobileNetV1 | 〜3M | 75-80% | 速 |
| Inception | 〜6M | 80-85% | 中 |
| ResNet18（事前訓練） | 〜11M | **90-93%** | 速 |

※ 実際の精度は訓練条件により変動します

## 💡 学習したポイント

### アーキテクチャ設計
1. **ResNet - 残差接続の重要性**:
   - 深いネットワークでも勾配消失を防ぐ
   - skip connectionにより学習の安定性が向上

2. **MobileNet - 効率的なモデル設計**:
   - Depthwise Separable Convolutionで計算コストを削減
   - モバイル・エッジデバイス向けの実用的アプローチ

3. **Inception - マルチスケール特徴抽出**:
   - 複数のフィルタサイズを並列処理
   - より豊かな特徴表現

### 訓練テクニック
1. **学習率スケジューリング**:
   - StepLRによる段階的な学習率減衰
   - 訓練後半での精度向上に貢献

2. **TensorBoardによる可視化**:
   - リアルタイムでの訓練監視
   - 損失・精度・画像の可視化
   - ハイパーパラメータ調整の効率化

3. **Transfer Learning**:
   - 事前訓練済みモデルの活用
   - 少ないデータでも高精度を実現
   - 訓練時間の短縮

4. **バッチ正規化とInstanceNorm**:
   - 訓練の安定化
   - 収束速度の向上

## 🔧 改善案と今後の課題

### 実装済み ✅
- [x] 複数のCNNアーキテクチャの実装
- [x] TensorBoardによる訓練可視化
- [x] 学習率スケジューリング
- [x] バッチ正規化
- [x] Transfer Learningの実装

### 今後の改善案 📝
- [ ] データ拡張（Data Augmentation）の追加
  - RandomCrop、ColorJitter、Cutout等
- [ ] より高度な学習率スケジューリング（CosineAnnealing、ReduceLROnPlateau）
- [ ] Mixed Precision Training（AMP）による高速化
- [ ] Grad-CAMによるモデルの可視化・解釈
- [ ] アンサンブル学習の実装
- [ ] モデルの軽量化（量子化、プルーニング）
- [ ] ハイパーパラメータ自動調整（Optuna等）
- [ ] K-Fold Cross Validationの実装
- [ ] 各モデルの性能比較レポート作成

## 📊 このプロジェクトの実用性

このプロジェクトで習得した技術は以下の実務に応用可能：
- 製造業での不良品検出システム
- 医療画像診断支援システム
- 農業でのクロップ分類
- セキュリティカメラでの物体認識
- ECサイトでの商品カテゴリ自動分類

## 📝 参考資料

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Deep Residual Learning for Image Recognition (ResNet論文)](https://arxiv.org/abs/1512.03385)
- [MobileNets: Efficient Convolutional Neural Networks (MobileNet論文)](https://arxiv.org/abs/1704.04861)
- [Going Deeper with Convolutions (Inception論文)](https://arxiv.org/abs/1409.4842)
- [PyTorch公式ドキュメント](https://pytorch.org/docs/)

## 🏆 このプロジェクトの強み

- ✅ **複数の最先端アーキテクチャを実装** - 実務での選択肢を理解
- ✅ **詳細な日本語コメント** - 理解しやすく保守性が高い
- ✅ **実験管理の仕組み** - TensorBoardによる体系的な管理
- ✅ **Transfer Learningの実践** - 実務で最も使われる手法
- ✅ **モジュール化された設計** - モデルの切り替えが容易

## 👤 作成者

楊様 (Youyo)

## 📅 作成日

2024年

---

**注**: このプロジェクトは、深層学習における複数のアーキテクチャを実践的に学ぶために作成されました。実務レベルの実装パターンを含んでいます。
