"""
Matplotlibのフォントキャッシュをクリアするスクリプト
"""

import matplotlib
import matplotlib.font_manager as fm
from pathlib import Path
import os

print("=" * 70)
print("Matplotlibフォントキャッシュのクリア")
print("=" * 70)

try:
    # フォントキャッシュの場所を取得
    cache_dir = Path(matplotlib.get_cachedir())
    print(f"\nキャッシュディレクトリ: {cache_dir}")

    # キャッシュファイルをリスト
    cache_files = []
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.cache")) + list(cache_dir.glob("*.json"))

    if cache_files:
        print(f"\n見つかったキャッシュファイル: {len(cache_files)}個")
        for cache_file in cache_files:
            print(f"  - {cache_file.name}")
            try:
                cache_file.unlink()
                print(f"    削除しました")
            except Exception as e:
                print(f"    削除失敗: {e}")
    else:
        print("\nキャッシュファイルが見つかりませんでした")

    print("\n" + "=" * 70)
    print("フォントキャッシュのクリアが完了しました")
    print("次回のmatplotlib使用時に、フォントが再読み込みされます")
    print("=" * 70)

except Exception as e:
    print(f"\nエラー: {e}")
    print("\n代替方法: 手動でキャッシュをクリアしてください")
    print(f"ユーザーディレクトリ下の .matplotlib フォルダを探して削除してください")

print("\n現在のフォント設定:")
print(f"デフォルトフォント: {matplotlib.rcParams['font.sans-serif']}")
print("\n完了！")
