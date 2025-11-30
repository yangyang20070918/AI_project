"""
乱数シード固定モジュール

実験の再現性を確保するため、全ての乱数生成器のシードを固定します。
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    全ての乱数生成器のシードを固定

    Args:
        seed: 乱数シード値（デフォルト: 42）

    Examples:
        >>> set_seed(42)
        >>> # これ以降、全ての乱数生成が再現可能になります
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA使用時のシード固定
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuDNNの決定的動作を有効化（パフォーマンスは若干低下）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_state() -> dict:
    """
    現在の乱数生成器の状態を取得

    Returns:
        各乱数生成器の状態を含む辞書

    Examples:
        >>> state = get_random_state()
        >>> # 後で状態を復元できます
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_random_state(state: dict) -> None:
    """
    乱数生成器の状態を復元

    Args:
        state: get_random_state()で取得した状態辞書

    Examples:
        >>> state = get_random_state()
        >>> # ... 何か処理 ...
        >>> set_random_state(state)  # 状態を復元
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
