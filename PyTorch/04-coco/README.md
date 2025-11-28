# COCOデータセット処理プロジェクト

## 📋 プロジェクト概要

COCO (Common Objects in Context) データセットのダウンロード、読み込み、可視化を行うプロジェクトです。COCOは物体検出、セグメンテーション、キーポイント検出などのタスクで広く使用される大規模データセットです。

## 🎯 目的

- 大規模データセットの扱い方を習得する
- COCOデータセットのフォーマットと構造を理解する
- pycocotoolsライブラリの使用方法を学ぶ
- データのダウンロードと前処理の実装
- アノテーションデータの可視化

## 🛠️ 使用技術

- **言語**: Python 3.x
- **主要ライブラリ**:
  - pycocotools - COCOデータセットAPI
  - scikit-image - 画像処理
  - matplotlib - 可視化
  - requests - データダウンロード

## 📊 COCOデータセットについて

**COCO Dataset (Common Objects in Context)**

### データセット規模
- **画像数**: 330,000枚以上
- **物体インスタンス**: 250万個以上
- **カテゴリ数**: 80クラス（人、車、動物、家具など）
- **セグメンテーション**: インスタンスレベルのマスク
- **キーポイント**: 人物の関節点（17点）

### データセット種類
- **train2017**: 訓練用画像（約118,000枚）
- **val2017**: 検証用画像（約5,000枚）
- **test2017**: テスト用画像（約41,000枚）

### アノテーション情報
- バウンディングボックス（物体の位置）
- セグメンテーションマスク（ピクセル単位の領域）
- カテゴリラベル（80クラス）
- キーポイント（人物の骨格情報）
- キャプション（画像の説明文）

## 📁 ファイル構成

```
04-coco/
├── README.md                           # プロジェクト説明（本ファイル）
├── requirements.txt                    # 依存ライブラリ
├── download_coco_data.py               # データダウンロードスクリプト（小規模）
├── download_coco_BIGdata.py            # データダウンロードスクリプト（大規模）
├── read_coco.py                        # データ読み込みと可視化
└── DATA/                               # データ保存ディレクトリ（自動作成）
    └── coco/
        ├── annotations/                # アノテーションファイル
        │   ├── instances_train2017.json
        │   ├── instances_val2017.json
        │   └── ...
        ├── train2017/                  # 訓練画像
        ├── val2017/                    # 検証画像
        └── test2017/                   # テスト画像
```

## 🚀 実行方法

### 1. 環境構築

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 2. データのダウンロード

**注意**: COCOデータセットは非常に大きいです（全体で約25GB）。ディスク容量と通信環境を確認してください。

#### 小規模ダウンロード（サンプル用）
```bash
python download_coco_data.py
```
- 検証用データセット（val2017）のみダウンロード
- 約1GB

#### 大規模ダウンロード（フルデータセット）
```bash
python download_coco_BIGdata.py
```
- 訓練データ、検証データ、アノテーションすべてをダウンロード
- 約25GB
- ダウンロードに時間がかかります（環境により数時間）

### 3. データの読み込みと可視化

```bash
python read_coco.py
```

このスクリプトは以下を実行します：
- アノテーションファイルの読み込み
- 特定カテゴリ（例：person）の画像を取得
- 画像とアノテーションを可視化表示

## 💻 コード例

### 基本的な使い方

```python
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

# アノテーションファイルの読み込み
anno_file = "DATA/coco/annotations/instances_val2017.json"
coco = COCO(anno_file)

# カテゴリIDの取得（例：person）
catIds = coco.getCatIds(catNms=['person'])
print(f"Category IDs for 'person': {catIds}")

# 該当カテゴリの画像IDを取得
imgIds = coco.getImgIds(catIds=catIds)
print(f"Found {len(imgIds)} images")

# 画像とアノテーションの表示
for i in range(5):  # 最初の5枚を表示
    # 画像情報の読み込み
    image = coco.loadImgs(imgIds[i])[0]

    # 画像の読み込み
    I = io.imread(image["coco_url"])

    # 画像の表示
    plt.imshow(I)

    # アノテーションの取得と表示
    anno_id = coco.getAnnIds(imgIds=image["id"], catIds=catIds)
    annotation = coco.loadAnns(anno_id)
    coco.showAnns(annotation)

    plt.show()
```

### 利用可能なカテゴリ例

COCOデータセットの80カテゴリの一部：
- 人物: person
- 乗り物: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- 動物: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- 家具: chair, couch, bed, dining table, toilet
- 電子機器: tv, laptop, mouse, keyboard, cell phone
- その他: bottle, cup, fork, knife, spoon, bowl, banana, apple, etc.

## 📈 データセットの特徴

### COCOの強み
1. **大規模**: 実用的な深層学習モデルの訓練に十分
2. **高品質アノテーション**: 正確なセグメンテーションマスク
3. **文脈情報**: 自然なシーンでの物体配置
4. **多様性**: 様々な環境、角度、スケールの画像
5. **標準化**: コンペティションでの標準ベンチマーク

### 適用可能なタスク
- 物体検出 (Object Detection)
- インスタンスセグメンテーション (Instance Segmentation)
- 意味的セグメンテーション (Semantic Segmentation)
- キーポイント検出 (Keypoint Detection)
- パノプティックセグメンテーション (Panoptic Segmentation)
- 画像キャプショニング (Image Captioning)

## 💡 学習したポイント

### データ処理
1. **大規模データの効率的な扱い**:
   - 段階的なダウンロード
   - ディスク容量の管理

2. **COCOフォーマットの理解**:
   - JSONベースのアノテーション構造
   - 画像、カテゴリ、アノテーションの関係

3. **pycocotoolsの使用**:
   - データの読み込みAPI
   - アノテーションの可視化機能

### 実装上の工夫
- データダウンロードの自動化
- エラーハンドリング
- 可視化による確認

## 🔧 今後の拡張案

### 実装予定 📝
- [ ] データローダーの実装（PyTorch Dataset/DataLoader）
- [ ] データ拡張（Augmentation）の実装
- [ ] 物体検出モデルの訓練（Faster R-CNN、YOLO等）
- [ ] セグメンテーションモデルの訓練（Mask R-CNN等）
- [ ] 評価指標の実装（mAP計算）
- [ ] カスタムデータセットの作成
- [ ] アノテーションツールとの連携
- [ ] 推論パイプラインの構築

### 実用化への道筋
このプロジェクトを基に、以下のような実務アプリケーションに発展可能：
- 監視カメラでの人物検出システム
- 自動運転での物体認識
- 小売業での商品認識
- 医療画像での病変検出
- ロボットビジョン

## ⚠️ 注意事項

### ディスク容量
- val2017のみ: 約1GB
- train2017 + val2017: 約20GB
- 全データセット: 約25GB以上

### ダウンロード時間
- 高速回線: 30分〜2時間
- 通常回線: 数時間〜半日

### システム要件
- **メモリ**: 8GB以上推奨（データ処理時）
- **GPU**: 訓練時は必須（推論のみならCPU可）
- **Python**: 3.6以上

## 📊 COCOデータセットの実績

COCOは以下の著名な研究・手法で使用されています：
- Faster R-CNN (物体検出)
- Mask R-CNN (インスタンスセグメンテーション)
- YOLO (リアルタイム物体検出)
- RetinaNet (Focal Loss)
- EfficientDet
- その他多数の最先端モデル

## 📝 参考資料

- [COCO公式サイト](https://cocodataset.org/)
- [COCO APIドキュメント](https://github.com/cocodataset/cocoapi)
- [COCO論文: Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
- [pycocotoolsドキュメント](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI)

## 🏆 プロジェクトの価値

このプロジェクトは以下を示しています：

- ✅ 大規模データセットの扱い方の理解
- ✅ 業界標準データセットへの対応
- ✅ 実務で使用される技術スタックの経験
- ✅ 物体検出・セグメンテーションへの発展可能性
- ✅ データ前処理パイプラインの構築能力

**次のステップ**: このデータセットを使用して、実際の物体検出モデル（YOLOやFaster R-CNN）を訓練する予定です。

## 👤 作成者

楊様 (Youyo)

## 📅 作成日

2024年

---

**注**: COCOデータセットは研究・教育目的での使用が推奨されています。商用利用の際はライセンスを確認してください。
