"""
設定ファイル読み込みモジュール

YAMLファイルから設定を読み込み、Pythonの辞書として提供します。
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    YAML設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定内容を含む辞書

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        yaml.YAMLError: YAML解析エラーが発生した場合

    Examples:
        >>> config = load_config(Path("config/config.yaml"))
        >>> print(config['model']['hidden_size'])
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML解析エラー: {e}")


def save_config(config: Dict[str, Any], save_path: Path) -> None:
    """
    設定を YAMLファイルに保存

    Args:
        config: 保存する設定辞書
        save_path: 保存先パス

    Examples:
        >>> config = {'model': {'hidden_size': 100}}
        >>> save_config(config, Path("experiments/exp001/config.yaml"))
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    2つの設定辞書をマージ（override_configが優先）

    Args:
        base_config: ベース設定
        override_config: 上書きする設定

    Returns:
        マージされた設定辞書

    Examples:
        >>> base = {'model': {'hidden_size': 100, 'dropout': 0.5}}
        >>> override = {'model': {'hidden_size': 200}}
        >>> merged = merge_configs(base, override)
        >>> print(merged['model']['hidden_size'])  # 200
        >>> print(merged['model']['dropout'])       # 0.5
    """
    result = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


class Config:
    """
    設定管理クラス

    辞書のように属性アクセス可能な設定オブジェクトを提供します。
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Args:
            config_dict: 設定辞書
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """辞書形式のアクセスをサポート"""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """辞書形式の代入をサポート"""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """in演算子のサポート"""
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """
        通常の辞書に変換

        Returns:
            設定辞書

        Examples:
            >>> config = Config({'model': {'hidden_size': 100}})
            >>> config_dict = config.to_dict()
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """文字列表現"""
        return f"Config({self.to_dict()})"


def load_config_as_object(config_path: Path) -> Config:
    """
    YAML設定ファイルを読み込み、Configオブジェクトとして返す

    Args:
        config_path: 設定ファイルのパス

    Returns:
        Config オブジェクト

    Examples:
        >>> config = load_config_as_object(Path("config/config.yaml"))
        >>> print(config.model.hidden_size)  # 属性アクセス
    """
    config_dict = load_config(config_path)
    return Config(config_dict)
