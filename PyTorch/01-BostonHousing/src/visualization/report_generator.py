"""
レポート生成モジュール

実験結果をHTML形式のレポートとして生成します。
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


def generate_html_report(
    experiment_dir: Path,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    save_path: Optional[Path] = None
) -> str:
    """
    HTML形式のレポートを生成

    Args:
        experiment_dir: 実験ディレクトリ
        metrics: 評価指標
        config: 実験設定
        save_path: 保存先パス（Noneの場合は実験ディレクトリ内に保存）

    Returns:
        生成されたHTMLの文字列
    """
    experiment_dir = Path(experiment_dir)

    if save_path is None:
        save_path = experiment_dir / "report.html"

    # プロット画像のパスを取得
    plots_dir = experiment_dir / "plots"
    plot_files = list(plots_dir.glob("*.png")) if plots_dir.exists() else []

    # HTMLテンプレート
    html = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Boston Housing 価格予測 - 実験レポート</title>
        <style>
            * {{
                font-family: 'Yu Gothic', 'Meiryo', 'MS Gothic', 'Hiragino Kaku Gothic ProN', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            body {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metric-card h3 {{
                margin: 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .metric-card .value {{
                font-size: 32px;
                font-weight: bold;
                margin-top: 10px;
            }}
            .config-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .config-table th, .config-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .config-table th {{
                background-color: #3498db;
                color: white;
            }}
            .config-table tr:hover {{
                background-color: #f5f5f5;
            }}
            .plot-container {{
                margin: 30px 0;
                text-align: center;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 14px;
                margin-top: 30px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Boston Housing 価格予測 - 実験レポート</h1>

            <p><strong>実験ID:</strong> {experiment_dir.name}</p>
            <p><strong>生成日時:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>

            <h2>評価指標</h2>
            <div class="metric-grid">
                {generate_metric_cards(metrics)}
            </div>

            <h2>実験設定</h2>
            <table class="config-table">
                <tr>
                    <th>項目</th>
                    <th>値</th>
                </tr>
                {generate_config_rows(config)}
            </table>

            <h2>可視化結果</h2>
            {generate_plot_sections(plot_files)}

            <div class="timestamp">
                <p>このレポートは自動生成されました。</p>
            </div>
        </div>
    </body>
    </html>
    """

    # HTMLを保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return html


def generate_metric_cards(metrics: Dict[str, float]) -> str:
    """評価指標のカードHTMLを生成"""
    metric_names = {
        'mse': 'MSE',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'r2': 'R² Score',
        'mape': 'MAPE (%)',
        'max_error': '最大誤差'
    }

    cards = []
    for key, value in metrics.items():
        if key in metric_names:
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            cards.append(f"""
                <div class="metric-card">
                    <h3>{metric_names[key]}</h3>
                    <div class="value">{formatted_value}</div>
                </div>
            """)

    return '\n'.join(cards)


def generate_config_rows(config: Dict[str, Any], prefix: str = '') -> str:
    """設定テーブルの行HTMLを生成"""
    rows = []

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            rows.append(generate_config_rows(value, full_key))
        else:
            rows.append(f"""
                <tr>
                    <td><strong>{full_key}</strong></td>
                    <td>{value}</td>
                </tr>
            """)

    return '\n'.join(rows)


def generate_plot_sections(plot_files: list) -> str:
    """プロット画像のセクションHTMLを生成"""
    if not plot_files:
        return "<p>プロット画像がありません。</p>"

    sections = []

    for plot_file in sorted(plot_files):
        plot_name = plot_file.stem.replace('_', ' ').title()
        sections.append(f"""
            <div class="plot-container">
                <h3>{plot_name}</h3>
                <img src="plots/{plot_file.name}" alt="{plot_name}">
            </div>
        """)

    return '\n'.join(sections)
