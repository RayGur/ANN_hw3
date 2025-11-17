"""
評估指標模組
包含準確率計算和多次測試運行
"""

import numpy as np
from utils import binarize


def calculate_pixel_accuracy(original, recovered):
    """
    計算像素準確率（Pixel Accuracy）
    衡量有多少像素被正確恢復

    Args:
        original: 原始圖像向量
        recovered: 復原圖像向量

    Returns:
        像素準確率 (0-100%)
    """
    return np.mean(original == recovered) * 100


def calculate_pattern_accuracy(original_idx, recovered, all_patterns):
    """
    計算分類準確率（Pattern Recognition Accuracy）
    判斷recovered是否被正確辨識為原始pattern

    Args:
        original_idx: 原始pattern的索引 (0-3)
        recovered: 復原圖像向量
        all_patterns: 所有stored patterns的列表

    Returns:
        分類準確率 (0 or 100%)
    """
    # 計算recovered與每個stored pattern的相似度
    similarities = [np.sum(recovered == pattern) for pattern in all_patterns]
    predicted_idx = np.argmax(similarities)

    # 只有預測正確才返回100%，否則返回0%
    return 100.0 if predicted_idx == original_idx else 0.0


def get_predicted_pattern(recovered, all_patterns):
    """
    獲取recovered最接近的pattern索引

    Args:
        recovered: 復原圖像向量
        all_patterns: 所有stored patterns的列表

    Returns:
        最相似的pattern索引 (0-3)
    """
    similarities = [np.sum(recovered == pattern) for pattern in all_patterns]
    return np.argmax(similarities)


def run_single_test(
    pattern, pattern_idx, all_patterns, hopfield, noise_level, noise_seed, recall_seed
):
    """
    運行單次測試

    Args:
        pattern: 原始pattern向量
        pattern_idx: pattern索引
        all_patterns: 所有patterns列表
        hopfield: 訓練好的Hopfield網絡
        noise_level: 噪聲比例
        noise_seed: 噪聲隨機種子
        recall_seed: recall隨機種子

    Returns:
        dict: 包含所有測試結果的字典
    """
    from utils import add_noise

    # 添加噪聲
    noisy = add_noise(pattern, noise_level, seed=noise_seed)

    # 復原
    recovered, n_iter, converged = hopfield.recall(noisy, seed=recall_seed)
    recovered_binary = binarize(recovered)

    # 計算準確率
    pixel_acc = calculate_pixel_accuracy(pattern, recovered_binary)
    pattern_acc = calculate_pattern_accuracy(
        pattern_idx, recovered_binary, all_patterns
    )
    predicted_idx = get_predicted_pattern(recovered_binary, all_patterns)

    return {
        "noisy": noisy,
        "recovered": recovered_binary,
        "pixel_accuracy": pixel_acc,
        "pattern_accuracy": pattern_acc,
        "predicted_idx": predicted_idx,
        "n_iterations": n_iter,
        "converged": converged,
    }


def run_multiple_tests(
    pattern, pattern_idx, all_patterns, hopfield, noise_level, n_trials, start_seed=0
):
    """
    運行多次測試並收集結果

    Args:
        pattern: 原始pattern向量
        pattern_idx: pattern索引
        all_patterns: 所有patterns列表
        hopfield: 訓練好的Hopfield網絡
        noise_level: 噪聲比例
        n_trials: 測試次數
        start_seed: 起始隨機種子

    Returns:
        dict: 包含所有測試結果的字典
            {
                'pixel_accuracies': list,
                'pattern_accuracies': list,
                'predictions': list,
                'n_iterations': list
            }
    """
    results = {
        "pixel_accuracies": [],
        "pattern_accuracies": [],
        "predictions": [],
        "n_iterations": [],
    }

    for i in range(n_trials):
        seed = start_seed + i
        result = run_single_test(
            pattern, pattern_idx, all_patterns, hopfield, noise_level, seed, seed
        )

        results["pixel_accuracies"].append(result["pixel_accuracy"])
        results["pattern_accuracies"].append(result["pattern_accuracy"])
        results["predictions"].append(result["predicted_idx"])
        results["n_iterations"].append(result["n_iterations"])

    return results


if __name__ == "__main__":
    # 測試評估函數
    print("=== 評估模組測試 ===\n")

    from patterns import get_patterns_as_vectors
    from hopfield import HopfieldNetwork

    patterns = get_patterns_as_vectors()
    hopfield = HopfieldNetwork()
    hopfield.train(patterns)

    # 測試單次
    print("1. 測試單次評估")
    result = run_single_test(
        patterns[0],
        0,
        patterns,
        hopfield,
        noise_level=0.2,
        noise_seed=42,
        recall_seed=42,
    )
    print(f"   Pixel Acc: {result['pixel_accuracy']:.1f}%")
    print(f"   Pattern Acc: {result['pattern_accuracy']:.0f}%")
    print(f"   迭代: {result['n_iterations']}次")

    # 測試多次
    print("\n2. 測試多次評估 (Pattern 2, 10次)")
    results = run_multiple_tests(
        patterns[1], 1, patterns, hopfield, noise_level=0.2, n_trials=10, start_seed=0
    )
    print(f"   Pixel Acc 平均: {np.mean(results['pixel_accuracies']):.1f}%")
    print(
        f"   Pattern Success: {sum(1 for p in results['pattern_accuracies'] if p == 100.0)}/10"
    )

    print("\n✓ 評估模組測試完成")
