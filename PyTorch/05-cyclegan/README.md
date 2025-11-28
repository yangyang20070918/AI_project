# CycleGAN画像スタイル変換プロジェクト

## 📋 プロジェクト概要

CycleGAN（Cycle-Consistent Adversarial Networks）を実装した画像スタイル変換プロジェクトです。ペア画像データなしで、異なるドメイン間の画像変換を学習します。例：リンゴ↔オレンジ、馬↔シマウマ、写真↔絵画など。

## 🎯 目的

- **GAN (Generative Adversarial Networks)** の理解と実装
- **Unpaired Image-to-Image Translation** の習得
- **Cycle Consistency Loss** の概念と実装
- 高度な深層学習アーキテクチャの実装経験
- 生成モデルの訓練テクニックの習得

## 🛠️ 使用技術

- **フレームワーク**: PyTorch, torchvision
- **言語**: Python 3.x
- **アーキテクチャ**:
  - Generator: ResNetベースの生成器（エンコーダー・デコーダー構造）
  - Discriminator: PatchGAN識別器
- **損失関数**:
  - Adversarial Loss（敵対的損失）
  - Cycle Consistency Loss（循環一貫性損失）
  - Identity Loss（恒等性損失）
- **可視化**: TensorBoardX

## 📊 CycleGANとは

### 従来のImage-to-Image Translation（Pix2Pix等）の問題点
- **ペアデータが必要**: 入力と出力の対応画像が必要
- データ収集のコストが高い

### CycleGANの革新的な点
- **ペアデータ不要**: 対応関係のない画像集合のみで学習可能
- **双方向変換**: ドメインA→B と B→A を同時に学習
- **Cycle Consistency**: 変換の一貫性を保証

### 動作原理

```
Domain A (リンゴ) → Generator A2B → Domain B (オレンジ)
                       ↓
                  Discriminator B

Domain B (オレンジ) → Generator B2A → Domain A (リンゴ)
                       ↓
                  Discriminator A

Cycle Consistency:
A → B → A' (A'はAと同じであるべき)
B → A → B' (B'はBと同じであるべき)
```

## 🏗️ モデルアーキテクチャ

### Generator（生成器）

ResNetベースのエンコーダー・デコーダー構造：

```python
Generator(
  # Encoder（下採样）
  ReflectionPad2d + Conv2d(3, 64) + InstanceNorm + ReLU
  Conv2d(64, 128, stride=2) + InstanceNorm + ReLU
  Conv2d(128, 256, stride=2) + InstanceNorm + ReLU

  # Transformer（特徴変換）
  ResBlock(256) × 9  # 残差ブロック

  # Decoder（上採样）
  ConvTranspose2d(256, 128) + InstanceNorm + ReLU
  ConvTranspose2d(128, 64) + InstanceNorm + ReLU
  ReflectionPad2d + Conv2d(64, 3) + Tanh
)
```

**特徴**:
- **ReflectionPad**: パディング時のアーティファクト削減
- **InstanceNorm**: スタイル変換に適した正規化
- **ResBlock × 9**: 深い特徴変換
- **ConvTranspose2d**: 上採样（デコード）

### Discriminator（識別器）

PatchGAN構造：

```python
Discriminator(
  Conv2d(3, 64, stride=2) + LeakyReLU
  Conv2d(64, 128, stride=2) + InstanceNorm + LeakyReLU
  Conv2d(128, 256, stride=2) + InstanceNorm + LeakyReLU
  Conv2d(256, 512, stride=2) + InstanceNorm + LeakyReLU
  Conv2d(512, 1)  # パッチごとの真偽判定
)
```

**特徴**:
- **PatchGAN**: 画像全体ではなく、パッチ単位で真偽判定
- **LeakyReLU**: 負の勾配も学習

## 📁 ファイル構成

```
05-cyclegan/
├── README.md                    # プロジェクト説明（本ファイル）
├── requirements.txt             # 依存ライブラリ
├── models.py                    # Generator/Discriminatorの定義
├── datasets.py                  # データセットローダー
├── utils.py                     # ユーティリティ関数
├── train.py                     # 訓練スクリプト
├── test.py                      # テスト・推論スクリプト
├── visualize_results.py         # 可視化スクリプト
├── models/                      # 訓練済みモデル保存
│   ├── netG_A2B.pth
│   ├── netG_B2A.pth
│   ├── netD_A.pth
│   └── netD_B.pth
├── logs/                        # TensorBoardログ
├── outputs/                     # 変換結果画像
│   ├── A/                       # A→B変換結果
│   └── B/                       # B→A変換結果
└── results/                     # 可視化結果
    ├── transformation_A2B.png   # A→B変換サンプル
    ├── transformation_B2A.png   # B→A変換サンプル
    └── comparison.png           # 変換結果の比較
```

## 🚀 実行方法

### 1. 環境構築

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 2. データセットの準備

#### データセット構造

```
data/
└── apple2orange/
    ├── trainA/           # ドメインA（リンゴ）の訓練画像
    ├── trainB/           # ドメインB（オレンジ）の訓練画像
    ├── testA/            # ドメインA（リンゴ）のテスト画像
    └── testB/            # ドメインB（オレンジ）のテスト画像
```

#### データセットのダウンロード例

公式データセット：
- apple2orange
- horse2zebra
- summer2winter_yosemite
- monet2photo
- cezanne2photo
- ukiyoe2photo
- vangogh2photo
- maps
- cityscapes
- facades
- iphone2dslr_flower

```bash
# 例: apple2orangeデータセットをダウンロード
bash ./download_dataset.sh apple2orange
```

### 3. 訓練の実行

```bash
python train.py
```

**訓練パラメータ**（`train.py`内で設定）:
```python
batch_size = 1           # バッチサイズ
size = 256               # 画像サイズ
lr = 0.0002              # 学習率
n_epoch = 200            # エポック数
decay_epoch = 100        # 学習率減衰開始エポック
```

### 4. TensorBoardで訓練過程を確認

```bash
tensorboard --logdir=logs
```

以下を確認できます：
- 生成器の損失（loss_G）
- 識別器の損失（loss_D_A, loss_D_B）
- Identity Loss
- GAN Loss
- Cycle Loss

### 5. 推論の実行

```bash
python test.py
```

訓練済みモデルを使用して、新しい画像のスタイル変換を実行します。

### 6. 変換結果の可視化

```bash
python visualize_results.py
```

以下の画像が `results/` ディレクトリに生成されます：
- A→B変換結果のサンプル
- B→A変換結果のサンプル
- 変換結果の比較

## ⚙️ 損失関数の詳細

CycleGANは3種類の損失関数を組み合わせています：

### 1. Adversarial Loss（敵対的損失）

```python
loss_GAN = MSELoss(Discriminator(fake), label_real)
```

- 生成器が識別器を騙すための損失
- 生成画像を本物らしく見せる

### 2. Cycle Consistency Loss（循環一貫性損失）

```python
loss_cycle_ABA = L1Loss(G_B2A(G_A2B(real_A)), real_A) * 10.0
loss_cycle_BAB = L1Loss(G_A2B(G_B2A(real_B)), real_B) * 10.0
```

- A→B→A と変換した結果が元のAと一致するよう制約
- B→A→B と変換した結果が元のBと一致するよう制約
- **最も重要な損失**: これによりペアデータなしで学習可能

### 3. Identity Loss（恒等性損失）

```python
loss_identity_A = L1Loss(G_B2A(real_A), real_A) * 5.0
loss_identity_B = L1Loss(G_A2B(real_B), real_B) * 5.0
```

- 同じドメインの画像を入力した時、変化しないよう制約
- 色調の保存に貢献

### 総合損失

```python
loss_G = loss_identity_A + loss_identity_B +
         loss_GAN_A2B + loss_GAN_B2A +
         loss_cycle_ABA + loss_cycle_BAB
```

## 📊 画像変換結果の可視化

### ドメインA → ドメインB 変換

訓練済みモデルによる画像変換結果のサンプルです。

![A→B変換](results/transformation_A2B.png)

**例: リンゴ → オレンジ変換**
- 元画像のリンゴがオレンジの特徴（色、テクスチャ）を獲得
- 形状や構図は保持されたまま、スタイルのみ変換

### ドメインB → ドメインA 変換

![B→A変換](results/transformation_B2A.png)

**例: オレンジ → リンゴ変換**
- 元画像のオレンジがリンゴの特徴に変換
- 双方向の変換が可能であることを示す

### 変換結果の比較

![変換比較](results/comparison.png)

**CycleGANの特徴:**
- ✅ ペア画像データ不要で学習可能
- ✅ 双方向変換（A→BとB→A）
- ✅ Cycle Consistencyにより一貫性を保証
- ✅ 高品質なスタイル変換を実現

**応用可能なタスク:**
- 🎨 写真 ↔ 絵画のスタイル変換
- 🌞 夏 ↔ 冬の季節変換
- 🐴 馬 ↔ シマウマのドメイン変換
- 🏥 医療画像のモダリティ変換（MRI ↔ CT）
- 🌃 昼 ↔ 夜の時間帯変換

## 📈 期待される結果

### 訓練時の出力例

```
epoch= 0
loss_G:8.234, loss_G_identity:3.456, loss_G_GAN:1.234, loss_G_cycle:3.544, loss_D_A:0.456, loss_D_B:0.489
...
epoch= 50
loss_G:2.456, loss_G_identity:0.678, loss_G_GAN:0.456, loss_G_cycle:1.322, loss_D_A:0.234, loss_D_B:0.256
...
epoch= 200
loss_G:1.123, loss_G_identity:0.234, loss_G_GAN:0.345, loss_G_cycle:0.544, loss_D_A:0.123, loss_D_B:0.134
```

### 訓練時間の目安

- **CPU**: 1エポックあたり 数時間（非現実的）
- **GPU（GTX 1080Ti等）**: 1エポックあたり 10-20分
- **推奨**: GPU必須、200エポックで約30-40時間

## 💡 学習したポイント

### 理論面
1. **GAN（敵対的生成ネットワーク）の理解**:
   - 生成器と識別器の敵対的学習
   - MinMax最適化問題

2. **Cycle Consistency の重要性**:
   - ペアデータなしで学習を可能にする核心技術
   - 変換の可逆性を保証

3. **Unpaired Image Translation**:
   - 実世界での応用可能性が高い
   - データ収集コストの削減

### 実装面
1. **複雑なアーキテクチャの実装**:
   - ResNetベースのGenerator
   - PatchGANのDiscriminator

2. **複数ネットワークの同時訓練**:
   - 2つのGenerator（G_A2B, G_B2A）
   - 2つのDiscriminator（D_A, D_B）
   - 交互最適化

3. **学習の安定化テクニック**:
   - ReplayBufferによる識別器訓練の安定化
   - InstanceNormの使用
   - 学習率スケジューリング

4. **実装の工夫**:
   - itertools.chainによる複数モデルのパラメータ統合
   - LambdaLRによるカスタム学習率減衰

## 🎨 応用例

CycleGANは様々な分野で応用可能：

### 画像処理
- 写真 ↔ 絵画（モネ、ゴッホ等のスタイル）
- 昼 ↔ 夜
- 夏 ↔ 冬
- 写真の色調変換

### 医療
- MRI ↔ CT画像変換
- モダリティ間の変換

### エンターテイメント
- アニメ化
- 年齢変換
- 性別変換

### その他
- 地図 ↔ 航空写真
- スケッチ ↔ 写真
- 低解像度 ↔ 高解像度

## 🔧 改善案と今後の課題

### 実装済み ✅
- [x] CycleGANの完全実装
- [x] 3種類の損失関数
- [x] ReplayBuffer
- [x] TensorBoard統合
- [x] データ拡張

### 今後の改善案 📝
- [ ] より高解像度画像への対応（512×512、1024×1024）
- [ ] StarGAN（複数ドメイン対応）への拡張
- [ ] Attention機構の導入
- [ ] Progressive GAN手法の適用
- [ ] FID、IS等の定量評価指標の実装
- [ ] Webアプリケーション化（リアルタイムスタイル変換）
- [ ] モデルの軽量化（モバイル対応）
- [ ] 推論の高速化（ONNX変換等）

## 📊 このプロジェクトの価値

### 技術的価値
- ✅ **高度なGANアーキテクチャの実装経験**
- ✅ **複数ネットワークの同時最適化**
- ✅ **研究論文の実装能力**
- ✅ **安定した生成モデルの訓練ノウハウ**

### 実務的価値
- ✅ **エンターテイメント業界**: 画像加工ツール
- ✅ **医療分野**: モダリティ変換
- ✅ **デザイン業界**: スタイル変換ツール
- ✅ **研究開発**: 新しいGAN手法の基礎

## ⚠️ 注意事項

### 計算資源
- **GPU必須**: CPU訓練は非現実的
- **推奨VRAM**: 8GB以上（11GB以上が理想）
- **訓練時間**: 200エポックで30-40時間

### GANの訓練の難しさ
- モード崩壊（Mode Collapse）のリスク
- 生成器と識別器のバランス調整が重要
- ハイパーパラメータに敏感

### データセット
- 各ドメイン最低1000枚以上推奨
- 画像の多様性が重要
- 解像度は256×256が標準

## 📝 参考資料

### 論文
- [Original CycleGAN Paper: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [GAN: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

### リソース
- [CycleGAN公式実装（PyTorch）](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [CycleGANプロジェクトページ](https://junyanz.github.io/CycleGAN/)

## 🏆 実装の特徴

- ✅ **完全な実装**: Generator、Discriminator、3種類の損失
- ✅ **詳細な日本語コメント**: 理解しやすく保守性が高い
- ✅ **実験管理**: TensorBoardによる可視化
- ✅ **実用的**: 実際に動作し、結果を生成可能
- ✅ **拡張性**: 他のGAN手法への応用が容易

このプロジェクトは、深層学習における最先端技術の実装能力と、複雑なシステムの構築経験を示しています。

## 👤 作成者

楊様 (Youyo)

## 📅 作成日

2024年

---

**注**: CycleGANは研究・教育目的での使用が推奨されています。生成された画像の商用利用には注意が必要です。
