# 深層学習・機械学習 学習プロジェクト集

## 👤 自己紹介

楊様（Youyo）
**目標**: 機械学習・深層学習エンジニアとしてフリーランス案件に携わる

このレポジトリは、PyTorchを使用した機械学習・深層学習の学習過程で実装したプロジェクトをまとめたものです。基礎的な回帰問題から、最先端のGANまで、段階的に難易度を上げながら実装しました。

## 🎯 学習目標

- PyTorchを使用した深層学習の基礎から応用まで習得
- 実務で使用される複数のCNNアーキテクチャの理解と実装
- 大規模データセットの扱い方を学ぶ
- 最先端技術（GAN等）への挑戦
- 実務で通用するコーディング能力の向上

## 📚 プロジェクト一覧

### 1. [Boston住宅価格予測（回帰）](PyTorch/01-BostonHousing/)
**難易度**: ⭐
**キーワード**: 回帰、全結合ネットワーク、PyTorch基礎

<details>
<summary>詳細を見る</summary>

#### 概要
ボストン地域の住宅価格を予測する回帰モデル。PyTorchの基本的なワークフローを習得。

#### 使用技術
- PyTorch
- 全結合ニューラルネットワーク（隠れ層付き）
- MSE損失、Adam最適化

#### 学習ポイント
- PyTorchの基本ワークフロー（データ準備→モデル定義→訓練→評価）
- 回帰問題の扱い方
- 隠れ層の効果
- 最適化アルゴリズムの比較（SGD vs Adam）

#### 成果
✅ 訓練損失の収束を確認
✅ モデルの保存・読み込み機能実装

</details>

---

### 2. [MNIST手書き数字認識（分類）](PyTorch/02-NumberCLS/)
**難易度**: ⭐⭐
**キーワード**: CNN、画像分類、畳み込み層

<details>
<summary>詳細を見る</summary>

#### 概要
畳み込みニューラルネットワーク（CNN）を使用した手書き数字認識。画像分類の基礎を学習。

#### 使用技術
- CNN（Conv2d、BatchNorm、MaxPool）
- DataLoaderによるバッチ処理
- CrossEntropyLoss

#### モデル構造
```
Conv2d(1→32) → BatchNorm → ReLU → MaxPool → FC(6272→10)
```

#### 学習ポイント
- CNNの基本構造（畳み込み層、プーリング層）
- バッチ正規化の効果
- DataLoaderの使い方
- 画像データの前処理

#### 成果
✅ テスト精度: 約98-99%
✅ 訓練・テストの明確な分離

</details>

---

### 3. [CIFAR-10画像分類（複数アーキテクチャ比較）](PyTorch/03-Cifar10/)
**難易度**: ⭐⭐⭐⭐
**キーワード**: ResNet、MobileNet、Inception、Transfer Learning、TensorBoard

<details>
<summary>詳細を見る</summary>

#### 概要
CIFAR-10データセットを使用し、複数の最先端CNNアーキテクチャを実装・比較。実務レベルの実装。

#### 実装したモデル
1. **VGGNet** - 深い畳み込みネットワーク
2. **ResNet** - 残差接続による深層化
3. **MobileNetV1** - 軽量モデル（Depthwise Separable Conv）
4. **Inception** - マルチスケール特徴抽出
5. **事前訓練済みResNet18** - Transfer Learning

#### 使用技術
- TensorBoardXによる訓練可視化
- 学習率スケジューリング（StepLR）
- バッチ正規化
- データローダーの最適化

#### 学習ポイント
- 複数のCNNアーキテクチャの理解と実装
- 残差接続（ResNet）の重要性
- 軽量化手法（MobileNet）
- Transfer Learningの実践
- 実験管理手法

#### 成果
✅ 複数モデルの実装（5種類）
✅ TensorBoardによる可視化
✅ テスト精度: 85-93%（モデルにより変動）
✅ モジュール化された設計

</details>

---

### 4. [COCOデータセット処理](PyTorch/04-coco/)
**難易度**: ⭐⭐⭐
**キーワード**: 大規模データセット、物体検出準備、pycocotools

<details>
<summary>詳細を見る</summary>

#### 概要
物体検出・セグメンテーションで使用される大規模データセットCOCOの処理。データのダウンロード、読み込み、可視化を実装。

#### 使用技術
- pycocotools（COCOデータセットAPI）
- scikit-image、matplotlib（可視化）
- 大規模データのダウンロード・管理

#### データセット規模
- 画像数: 330,000枚以上
- 物体カテゴリ: 80クラス
- アノテーション: バウンディングボックス、セグメンテーションマスク、キーポイント

#### 学習ポイント
- 大規模データセットの扱い方
- COCOフォーマットの理解
- アノテーションデータの処理
- データ可視化

#### 今後の展開
物体検出モデル（YOLO、Faster R-CNN）の訓練に使用予定

</details>

---

### 5. [CycleGAN（画像スタイル変換）](PyTorch/05-cyclegan/)
**難易度**: ⭐⭐⭐⭐⭐
**キーワード**: GAN、生成モデル、Unpaired Image Translation、高度なアーキテクチャ

<details>
<summary>詳細を見る</summary>

#### 概要
CycleGANを完全実装。ペア画像データなしで異なるドメイン間の画像変換を学習（例: リンゴ↔オレンジ）。

#### 使用技術
- **Generator**: ResNetベースのエンコーダー・デコーダー（ResBlock×9）
- **Discriminator**: PatchGAN
- **3種類の損失関数**:
  - Adversarial Loss（敵対的損失）
  - Cycle Consistency Loss（循環一貫性損失）
  - Identity Loss（恒等性損失）
- TensorBoardXによる訓練監視

#### モデル構成
- 2つのGenerator（G_A2B、G_B2A）
- 2つのDiscriminator（D_A、D_B）
- 交互最適化

#### 学習ポイント
- GANの理論と実装
- Unpaired Image-to-Image Translation
- Cycle Consistencyの重要性
- 複数ネットワークの同時訓練
- 学習の安定化テクニック（ReplayBuffer等）

#### 成果
✅ 完全な実装（Generator、Discriminator、3種類の損失）
✅ 複雑なアーキテクチャの実装経験
✅ 最先端技術への挑戦
✅ 実用的な画像変換が可能

#### 応用可能性
- 写真↔絵画のスタイル変換
- 医療画像のモダリティ変換
- デザイン業界での活用
- エンターテイメント（アニメ化等）

</details>

---

## 🛠️ 技術スタック

### フレームワーク・ライブラリ
- **PyTorch** - 深層学習フレームワーク
- **torchvision** - 画像処理・モデル
- **NumPy** - 数値計算
- **Matplotlib** - 可視化
- **TensorBoardX** - 訓練監視
- **pycocotools** - COCOデータセット処理
- **scikit-image** - 画像処理

### 実装した技術
#### 基礎
- 全結合ニューラルネットワーク
- 回帰問題と分類問題
- 損失関数（MSE、CrossEntropy）
- 最適化アルゴリズム（SGD、Adam）

#### CNN
- 畳み込み層（Conv2d）
- プーリング層（MaxPool、AvgPool）
- バッチ正規化、Instance正規化
- Dropout

#### アーキテクチャ
- VGGNet
- ResNet（残差接続）
- MobileNet（Depthwise Separable Conv）
- Inception（マルチスケール）
- GAN（Generator & Discriminator）

#### 訓練テクニック
- 学習率スケジューリング（StepLR、LambdaLR）
- Transfer Learning（事前訓練済みモデル）
- データ拡張
- TensorBoardによる実験管理
- ReplayBuffer（GAN訓練の安定化）

#### データ処理
- DataLoaderによるバッチ処理
- 大規模データセットの管理
- 画像の前処理・正規化
- アノテーションデータの処理

## 📊 プロジェクトの進行

```
難易度の推移:
回帰 → CNN基礎 → 複数CNN比較 → 大規模データ → GAN
  ⭐    ⭐⭐      ⭐⭐⭐⭐        ⭐⭐⭐      ⭐⭐⭐⭐⭐
```

段階的に難易度を上げながら、体系的に学習を進めました。

## 💪 強みと特徴

### 実装能力
✅ 基礎から高度な技術まで体系的に習得
✅ 複数の最先端アーキテクチャを実装
✅ 研究論文の実装能力（CycleGAN）
✅ 大規模データセットの扱い経験

### コードの品質
✅ 詳細な日本語コメント - 理解しやすく保守性が高い
✅ モジュール化された設計 - 再利用性が高い
✅ 各プロジェクトに詳細なREADME
✅ requirements.txtによる依存管理

### 実験管理
✅ TensorBoardによる訓練可視化
✅ モデルの保存・読み込み
✅ 体系的なプロジェクト構成

### 実務への応用可能性
✅ Transfer Learningの実践
✅ 実際に動作するコード
✅ 実用的なユースケースへの理解

## 🎯 今後の学習・開発計画

### 短期（1-2ヶ月）
- [ ] 物体検出モデルの実装（YOLO、Faster R-CNN）
- [ ] セマンティックセグメンテーションの実装
- [ ] Transformerベースのモデル（ViT等）
- [ ] これらのプロジェクトのWebアプリ化（FastAPI + Streamlit）

### 中期（3-6ヶ月）
- [ ] 自然言語処理（BERT、GPT）への拡張
- [ ] 強化学習の学習
- [ ] MLOps（MLflow、Docker、Kubernetes）
- [ ] クラウド（AWS SageMaker、GCP Vertex AI）

### 実践
- [ ] Kaggleコンペティションへの参加
- [ ] 実際のビジネス課題を解決するプロジェクト
- [ ] オープンソースへの貢献

## 📈 実務での応用可能性

これらのプロジェクトで習得した技術は、以下の実務に応用可能：

### 製造業
- 不良品検出（画像分類・物体検出）
- 外観検査の自動化

### 医療
- 医療画像診断支援
- モダリティ変換（CycleGAN）

### ECサイト
- 商品カテゴリ自動分類
- 商品推薦システム

### セキュリティ
- 監視カメラでの物体・人物検出
- 異常検知

### デザイン・エンターテイメント
- 画像スタイル変換ツール
- コンテンツ生成

## 🔗 関連リンク

- [PyTorch公式ドキュメント](https://pytorch.org/docs/)
- [Papers with Code](https://paperswithcode.com/)
- [Kaggle](https://www.kaggle.com/)

## 📞 お問い合わせ

機械学習・深層学習案件のご相談、技術的な質問等ございましたら、お気軽にご連絡ください。

**対応可能な業務**:
- 画像分類・物体検出モデルの開発
- 既存モデルのファインチューニング
- データ前処理・パイプライン構築
- モデルの推論API開発
- 技術調査・PoC開発

---

## 📝 各プロジェクトへのリンク

1. [Boston住宅価格予測](PyTorch/01-BostonHousing/README.md)
2. [MNIST手書き数字認識](PyTorch/02-NumberCLS/README.md)
3. [CIFAR-10画像分類](PyTorch/03-Cifar10/README.md)
4. [COCOデータセット処理](PyTorch/04-coco/README.md)
5. [CycleGAN画像スタイル変換](PyTorch/05-cyclegan/README.md)

---

**最終更新**: 2024年
**作成者**: 楊様 (Youyo)

このレポジトリは継続的に更新され、新しいプロジェクトが追加される予定です。
