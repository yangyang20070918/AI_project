# Boston住宅価格予測プロジェクト (Professional Edition)

**実務レベルの機械学習プロジェクトテンプレート実装**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 プロジェクト概要

ボストン地域の住宅価格を予測する回帰モデルを実装したプロジェクトです。**学習用のサンプルコードから実務レベルのポートフォリオプロジェクトへと拡張**し、産業界のベストプラクティスを適用しています。

### 主な特徴

✅ **モジュール化されたコード構造** - 再利用可能なコンポーネント設計
✅ **複数のモデルアーキテクチャ** - Baseline、SimpleNN、DeepNN を比較可能
✅ **自動化された実験管理** - 設定、モデル、結果を体系的に管理
✅ **包括的な評価システム** - 6種類の評価指標 + 詳細な可視化
✅ **プロフェッショナルなレポート** - HTML形式の自動生成レポート
✅ **完全な再現性** - 乱数シード固定、設定ファイル管理

---

## 🎯 目的

### 技術的目標
- 実務レベルの機械学習プロジェクト構造の理解
- PyTorchを使用したEnd-to-Endのパイプライン構築
- MLOpsのベストプラクティスの実践
- モデルの評価と可視化の体系的な実装

### ビジネス価値
- ポートフォリオとしての技術力証明
- 採用担当者・クライアントへのアピール材料
- 他のプロジェクトへの応用可能なテンプレート

---

## 🛠️ 技術スタック

### コア技術
- **深層学習**: PyTorch 2.0+
- **数値計算**: NumPy, pandas, scikit-learn
- **可視化**: Matplotlib, Seaborn
- **設定管理**: YAML

### プロジェクト機能
- **データ処理**: 前処理、正規化、データ分割（70/15/15）
- **モデル訓練**: Early Stopping、Model Checkpoint、LR Scheduling
- **評価**: MSE、RMSE、MAE、R²、MAPE、最大誤差
- **可視化**: 学習曲線、予測プロット、残差分析、特徴量分析
- **実験管理**: 設定保存、結果追跡、比較機能

---

## 📊 データセット

**Boston Housing Dataset**
- **サンプル数**: 506件
- **特徴量数**: 13個（犯罪率、部屋数、税率など）
- **目標変数**: 住宅価格（千ドル単位）
- **データ分割**: 訓練70% / 検証15% / テスト15%

### 特徴量一覧
1. **CRIM**: 犯罪率
2. **ZN**: 25,000平方フィート以上の住宅地の割合
3. **INDUS**: 非小売業の土地面積の割合
4. **CHAS**: チャールズ川沿いかどうか
5. **NOX**: 窒素酸化物濃度
6. **RM**: 住宅あたりの平均部屋数
7. **AGE**: 1940年以前建築物件の割合
8. **DIS**: 雇用センターまでの距離
9. **RAD**: 高速道路へのアクセス指数
10. **TAX**: 固定資産税率
11. **PTRATIO**: 生徒と教師の比率
12. **B**: 黒人居住者の割合
13. **LSTAT**: 低所得者の割合

---

## 📁 プロジェクト構成

```
01-BostonHousing/
├── config/                      # 設定ファイル
│   ├── config.yaml              # デフォルト設定
│   └── experiments/             # 実験ごとの設定
│
├── data/                        # データディレクトリ
│   ├── raw/                     # 生データ
│   │   └── housing.data
│   └── processed/               # 前処理済みデータ
│
├── src/                         # ソースコード
│   ├── data/                    # データ処理
│   │   ├── preprocessing.py     # データ前処理
│   │   └── dataset.py           # PyTorch Dataset
│   │
│   ├── models/                  # モデル定義
│   │   ├── base_model.py        # 基底クラス
│   │   ├── baseline.py          # 線形回帰
│   │   ├── simple_nn.py         # シンプルNN
│   │   └── deep_nn.py           # 深層NN
│   │
│   ├── training/                # 訓練システム
│   │   ├── trainer.py           # Trainerクラス
│   │   └── experiment.py        # 実験管理
│   │
│   ├── evaluation/              # 評価システム
│   │   └── evaluator.py         # 評価指標計算
│   │
│   ├── visualization/           # 可視化
│   │   ├── learning_curves.py   # 学習曲線
│   │   ├── prediction_plots.py  # 予測可視化
│   │   ├── feature_analysis.py  # 特徴量分析
│   │   └── report_generator.py  # レポート生成
│   │
│   └── utils/                   # ユーティリティ
│       ├── config_loader.py     # 設定読み込み
│       ├── logger.py            # ロギング
│       └── seed.py              # 乱数シード固定
│
├── scripts/                     # 実行スクリプト
│   ├── train.py                 # 訓練スクリプト
│   └── evaluate.py              # 評価スクリプト
│
├── experiments/                 # 実験結果（自動生成）
│   └── {experiment_id}/
│       ├── config.json          # 使用した設定
│       ├── model_best.pth       # ベストモデル
│       ├── training_history.json # 訓練履歴
│       ├── metrics_test.json    # 評価指標
│       ├── plots/               # 図表
│       └── report.html          # HTMLレポート
│
├── requirements.txt             # 依存関係
├── README.md                    # プロジェクト説明（本ファイル）
└── IMPROVEMENT_PLAN.md          # 改善計画書
```

---

## 🚀 クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
cd PyTorch/01-BostonHousing

# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 2. モデルの訓練

```bash
# デフォルト設定で訓練
python scripts/train.py

# カスタム設定で訓練
python scripts/train.py --config config/experiments/deep_nn.yaml

# 特定の実験IDで訓練
python scripts/train.py --experiment-id my_experiment_001
```

### 3. モデルの評価

```bash
# 実験IDを指定して評価（テストデータ）
python scripts/evaluate.py 20250130_143527

# 検証データで評価
python scripts/evaluate.py 20250130_143527 --split val

# 訓練データで評価
python scripts/evaluate.py 20250130_143527 --split train
```

### 4. 結果の確認

```bash
# 実験ディレクトリを開く
cd experiments/20250130_143527

# HTMLレポートをブラウザで開く
# Windows:
start report.html

# Mac:
open report.html

# Linux:
xdg-open report.html
```

---

## 📈 実装された機能

### Phase 1: 基盤構築 ✅
- [x] モジュール化されたプロジェクト構造
- [x] データ前処理パイプライン（標準化・正規化）
- [x] PyTorch Dataset & DataLoader
- [x] YAML設定ファイル管理
- [x] ロギングシステム
- [x] 乱数シード固定（再現性確保）

### Phase 2: モデルと訓練システム ✅
- [x] 3種類のモデルアーキテクチャ
  - BaselineModel（線形回帰）
  - SimpleNN（1層隠れ層）
  - DeepNN（多層NN、BatchNorm、Dropout）
- [x] Trainerクラス
  - Early Stopping
  - Model Checkpoint（ベストモデル保存）
  - 学習率スケジューリング（ReduceLROnPlateau、StepLR）
- [x] 実験管理システム
  - 自動的な実験ID生成
  - 設定・モデル・結果の一元管理
- [x] 訓練スクリプト（コマンドライン対応）

### Phase 3: 評価と可視化 ✅
- [x] 評価システム
  - MSE、RMSE、MAE、R²、MAPE、最大誤差
- [x] 学習曲線の可視化
  - 訓練loss vs 検証loss
  - 対数スケール表示
  - 学習率スケジュール
  - 過学習指標
- [x] 予測結果の可視化
  - 予測値 vs 実測値（散布図、R²表示）
  - 残差プロット
  - QQプロット（正規性確認）
  - 誤差分布
- [x] 特徴量分析
  - 相関マトリックス
  - 特徴量分布
  - モデル重みからの重要度
- [x] HTMLレポート自動生成
- [x] 評価スクリプト

---

## 💡 使用例

### 設定ファイルのカスタマイズ

`config/config.yaml` を編集して、モデルやハイパーパラメータを変更できます：

```yaml
# モデルの変更
model:
  type: "deep_nn"  # "baseline", "simple_nn", "deep_nn"

  deep_nn:
    hidden_sizes: [256, 128, 64]
    dropout_rate: 0.3
    use_batch_norm: true
    activation: "relu"

# 訓練設定
training:
  epochs: 2000
  batch_size: 32
  learning_rate: 0.001

  optimizer:
    type: "adam"
    weight_decay: 0.0001

  early_stopping:
    enabled: true
    patience: 100
```

### Python APIとしての使用

```python
from pathlib import Path
from src.data.preprocessing import HousingDataPreprocessor
from src.models.deep_nn import DeepNN
from src.training.trainer import Trainer, EarlyStopping

# データの前処理
preprocessor = HousingDataPreprocessor(scaler_type="standard")
data = preprocessor.preprocess(Path("data/raw/housing.data"))

# モデルの作成
model = DeepNN(
    input_dim=13,
    output_dim=1,
    hidden_sizes=[128, 64, 32],
    dropout_rate=0.3
)

# 訓練
trainer = Trainer(model, optimizer, criterion, device="cuda")
history = trainer.train(
    train_loader,
    val_loader,
    epochs=1000,
    early_stopping=EarlyStopping(patience=50)
)
```

---

## 📊 期待される性能

| モデル | R² Score | RMSE | MAE |
|--------|----------|------|-----|
| Baseline (線形回帰) | ~0.70 | ~5.5 | ~3.8 |
| SimpleNN | ~0.80 | ~4.2 | ~3.0 |
| DeepNN | **~0.85+** | **~3.5** | **~2.5** |

*実際の性能はハイパーパラメータ調整により変動します*

---

## 🔧 開発とカスタマイズ

### 新しいモデルの追加

1. `src/models/` に新しいモデルクラスを作成
2. `BaseModel` を継承
3. `scripts/train.py` の `get_model()` 関数に追加
4. `config/config.yaml` に設定を追加

### 新しい評価指標の追加

1. `src/evaluation/evaluator.py` にメソッドを追加
2. `config/config.yaml` の `evaluation.metrics` に指標名を追加

---

## 📝 実装の詳細

詳細な改善計画と実装仕様については、[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) を参照してください。

---

## 🎓 学習ポイント

このプロジェクトを通じて習得できるスキル：

### データサイエンス
- 探索的データ分析（EDA）
- データ前処理とスケーリング
- モデルの評価と比較

### 機械学習エンジニアリング
- PyTorchを使用したニューラルネットワーク構築
- 訓練パイプラインの自動化
- ハイパーパラメータチューニング
- 過学習対策（Early Stopping、Dropout、BatchNorm）

### ソフトウェアエンジニアリング
- モジュール化されたコード設計
- 設定ファイルによる管理
- ロギングとエラーハンドリング
- コマンドラインインターフェース

### MLOps
- 実験管理と再現性の確保
- モデルのバージョニング
- 自動化されたレポート生成

---

## 📚 参考資料

- [PyTorch公式ドキュメント](https://pytorch.org/docs/stable/index.html)
- [Boston Housing Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/housing)
- [MLOps ベストプラクティス](https://ml-ops.org/)

---

## 👤 作成者

**楊 洋**

---

## 📅 更新履歴

- **v2.0.0** (2025-01-30): プロフェッショナル版への大規模リファクタリング
  - モジュール化されたコード構造
  - 複数モデルのサポート
  - 包括的な評価と可視化
  - 実験管理システムの導入

- **v1.0.0** (2024): 初版リリース
  - 基本的な訓練・推論機能

---

## 📄 ライセンス

このプロジェクトは学習目的で作成されました。

---

**🌟 このプロジェクトがポートフォリオとして、または学習の参考として役立つことを願っています！**
