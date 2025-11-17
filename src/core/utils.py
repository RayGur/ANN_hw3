"""
基礎工具模組
包含最基本的工具函數
"""

import numpy as np


def activation_tanh(u, beta=100):
    """
    Hyperbolic tangent 激活函數

    公式: g(u) = tanh(β·u)

    Args:
        u: 輸入值(可以是標量或向量)
        beta: 斜率參數，默認100

    Returns:
        激活後的值，範圍(-1, 1)
    """
    return np.tanh(beta * u)


def binarize(values):
    """
    將連續值二值化為{-1, +1}
    處理 sign(0) = 0 的情況，將0視為+1

    Args:
        values: 輸入值（標量或向量）

    Returns:
        二值化結果 {-1, +1}
    """
    result = np.sign(values)
    result[result == 0] = 1  # 將0視為+1
    return result.astype(int)


def add_noise(pattern, noise_level, seed=None):
    """
    對圖像添加隨機噪聲

    Args:
        pattern: 原始圖像向量 (45維)
        noise_level: 噪聲比例 (0.0-1.0)
        seed: 隨機種子

    Returns:
        噪聲圖像向量
    """
    if seed is not None:
        np.random.seed(seed)

    noisy_pattern = pattern.copy()
    n_pixels = len(pattern)
    n_noise = int(n_pixels * noise_level)

    # 隨機選擇要翻轉的像素
    noise_indices = np.random.choice(n_pixels, n_noise, replace=False)

    # 翻轉選中的像素
    noisy_pattern[noise_indices] = -noisy_pattern[noise_indices]

    return noisy_pattern
