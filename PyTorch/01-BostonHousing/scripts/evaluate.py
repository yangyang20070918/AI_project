"""
評価スクリプト

訓練済みモデルを評価し、結果を可視化します。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from src.data.preprocessing import HousingDataPreprocessor
from src.data.dataset import create_dataloaders
from src.evaluation.evaluator import RegressionEvaluator
from src.visualization.learning_curves import load_history_and_plot
from src.visualization.prediction_plots import create_all_prediction_plots
from src.visualization.feature_analysis import create_all_feature_analysis_plots
from src.visualization.report_generator import generate_html_report
from src.training.experiment import Experiment
from src.utils.logger import setup_logger
import matplotlib.pyplot as plt


def main(experiment_id: str, split: str = 'test'):
    """
    メイン評価処理

    Args:
        experiment_id: 評価する実験のID
        split: 評価するデータ分割 ('train', 'val', 'test')
    """
    # 実験をロード
    experiment = Experiment(
        base_dir=project_root / "experiments",
        experiment_id=experiment_id
    )

    print("=" * 70)
    print(f"実験を評価: {experiment_id}")
    print("=" * 70)

    # ロガーのセットアップ
    logger = setup_logger(
        name=f"eval_{experiment_id}",
        log_file=experiment.experiment_dir / "evaluation.log"
    )

    logger.info(f"実験 {experiment_id} の評価を開始")

    # 設定を読み込み
    try:
        config = experiment.load_config()
    except FileNotFoundError:
        print(f"エラー: 設定ファイルが見つかりません")
        return

    # デバイスの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用デバイス: {device}")

    # データの読み込み
    logger.info("データを読み込み中...")

    processed_data_path = project_root / config.get('data', {}).get('processed_data_path', 'data/processed/preprocessed.npz')

    try:
        preprocessed_data = HousingDataPreprocessor.load_preprocessed_data(processed_data_path)
    except FileNotFoundError:
        logger.error(f"前処理済みデータが見つかりません: {processed_data_path}")
        print(f"エラー: 前処理済みデータが見つかりません")
        print("先に train.py を実行してください")
        return

    # DataLoaderの作成
    train_loader, val_loader, test_loader = create_dataloaders(
        preprocessed_data['X_train'],
        preprocessed_data['y_train'],
        preprocessed_data['X_val'],
        preprocessed_data['y_val'],
        preprocessed_data['X_test'],
        preprocessed_data['y_test'],
        batch_size=config.get('evaluation', {}).get('batch_size', 64),
        num_workers=config.get('system', {}).get('num_workers', 0)
    )

    # データローダーの選択
    if split == 'train':
        dataloader = train_loader
        X_data = preprocessed_data['X_train']
        y_data = preprocessed_data['y_train']
    elif split == 'val':
        dataloader = val_loader
        X_data = preprocessed_data['X_val']
        y_data = preprocessed_data['y_val']
    else:  # test
        dataloader = test_loader
        X_data = preprocessed_data['X_test']
        y_data = preprocessed_data['y_test']

    logger.info(f"評価データ ({split}): {X_data.shape}")

    # モデルの読み込み
    logger.info("モデルを読み込み中...")

    model_path = experiment.experiment_dir / "model_best.pth"

    if not model_path.exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        print(f"エラー: モデルファイルが見つかりません")
        return

    checkpoint = torch.load(model_path, map_location=device)

    # モデルの再構築（train.pyと同じロジック）
    from scripts.train import get_model

    model = get_model(
        config.get('model', {}),
        input_dim=X_data.shape[1],
        output_dim=1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    logger.info(f"モデルをロード: {model.__class__.__name__}")
    logger.info(f"エポック: {checkpoint.get('epoch', 'N/A')}")

    # 評価の実行
    logger.info("モデルを評価中...")

    evaluator = RegressionEvaluator(model, device=device)

    metrics_list = config.get('evaluation', {}).get('metrics', ['mse', 'rmse', 'mae', 'r2', 'mape'])
    results = evaluator.evaluate(dataloader, metrics=metrics_list)

    # 評価指標を表示
    evaluator.print_metrics(results)

    # 評価指標を保存（予測値とターゲットは除外）
    metrics_to_save = {k: v for k, v in results.items() if k not in ['predictions', 'targets']}
    experiment.save_metrics(metrics_to_save, filename=f"metrics_{split}.json")

    logger.info(f"評価指標を保存: metrics_{split}.json")

    # 可視化の作成
    logger.info("可視化を作成中...")

    # 学習曲線
    try:
        history_path = experiment.experiment_dir / "training_history.json"
        if history_path.exists():
            logger.info("学習曲線をプロット中...")
            load_history_and_plot(history_path, save_dir=experiment.plots_dir)
    except Exception as e:
        logger.warning(f"学習曲線のプロット失敗: {e}")

    # 予測プロット
    logger.info("予測プロットを作成中...")
    create_all_prediction_plots(
        results['targets'],
        results['predictions'],
        save_dir=experiment.plots_dir,
        dataset_name=split.capitalize()
    )

    # 特徴量分析
    logger.info("特徴量分析を作成中...")
    create_all_feature_analysis_plots(
        X_data,
        y_data,
        model=model,
        save_dir=experiment.plots_dir
    )

    # HTMLレポートの生成
    logger.info("HTMLレポートを生成中...")
    generate_html_report(
        experiment_dir=experiment.experiment_dir,
        metrics=metrics_to_save,
        config=config
    )

    logger.info(f"HTMLレポートを保存: report.html")

    # 完了メッセージ
    print("\n" + "=" * 70)
    print("評価が完了しました")
    print("=" * 70)
    print(f"実験ディレクトリ: {experiment.experiment_dir}")
    print(f"HTMLレポート: {experiment.experiment_dir / 'report.html'}")
    print("=" * 70)

    # matplotlibの図を閉じる
    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boston Housing 価格予測モデルの評価")

    parser.add_argument(
        'experiment_id',
        type=str,
        help='評価する実験のID'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='評価するデータ分割 (デフォルト: test)'
    )

    args = parser.parse_args()

    main(experiment_id=args.experiment_id, split=args.split)
