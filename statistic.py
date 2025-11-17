"""
統計分析模組
處理實驗數據的統計計算
"""

import numpy as np


def calculate_statistics(results):
    """
    計算測試結果的統計指標

    Args:
        results: run_multiple_tests() 返回的結果字典

    Returns:
        dict: 統計指標
            {
                'pixel_mean': float,
                'pixel_std': float,
                'pixel_min': float,
                'pixel_max': float,
                'pattern_success_rate': float (0-100),
                'pattern_success_count': int,
                'total_count': int
            }
    """
    pixel_accs = results["pixel_accuracies"]
    pattern_accs = results["pattern_accuracies"]

    success_count = sum(1 for p in pattern_accs if p == 100.0)
    total_count = len(pattern_accs)

    return {
        "pixel_mean": np.mean(pixel_accs),
        "pixel_std": np.std(pixel_accs),
        "pixel_min": np.min(pixel_accs),
        "pixel_max": np.max(pixel_accs),
        "pattern_success_rate": (success_count / total_count) * 100,
        "pattern_success_count": success_count,
        "total_count": total_count,
    }


def print_statistics_summary(stats, pattern_name=""):
    """
    打印統計摘要

    Args:
        stats: calculate_statistics() 返回的統計字典
        pattern_name: pattern名稱（可選）
    """
    if pattern_name:
        print(f"\n{pattern_name} 統計摘要:")
    else:
        print("\n統計摘要:")

    print(f"  Pixel Accuracy:")
    print(f"    平均: {stats['pixel_mean']:.2f}%")
    print(f"    標準差: {stats['pixel_std']:.2f}%")
    print(f"    範圍: [{stats['pixel_min']:.1f}% - {stats['pixel_max']:.1f}%]")
    print(f"  Pattern Recognition:")
    print(
        f"    成功率: {stats['pattern_success_rate']:.1f}% ({stats['pattern_success_count']}/{stats['total_count']})"
    )


def create_statistics_table(all_results, pattern_names=None):
    """
    創建多個patterns的統計表格

    Args:
        all_results: list of results dicts
        pattern_names: list of pattern names (optional)

    Returns:
        list of dicts: 每個pattern的統計結果
    """
    if pattern_names is None:
        pattern_names = [f"Pattern {i+1}" for i in range(len(all_results))]

    table = []
    for i, results in enumerate(all_results):
        stats = calculate_statistics(results)
        stats["pattern_name"] = pattern_names[i]
        table.append(stats)

    return table


def print_statistics_table(table):
    """
    打印統計表格

    Args:
        table: create_statistics_table() 返回的表格
    """
    print("\n" + "=" * 80)
    print("統計摘要表")
    print("=" * 80)
    print(
        f"{'Pattern':<12} | {'Pixel Acc (Avg)':<15} | {'Pixel Range':<18} | {'Pattern Success':<15}"
    )
    print("-" * 80)

    for row in table:
        print(
            f"{row['pattern_name']:<12} | "
            f"{row['pixel_mean']:6.2f}% ± {row['pixel_std']:5.2f} | "
            f"[{row['pixel_min']:5.1f}% - {row['pixel_max']:5.1f}%] | "
            f"{row['pattern_success_count']}/{row['total_count']} = {row['pattern_success_rate']:5.1f}%"
        )

    print("=" * 80)


if __name__ == "__main__":
    # 測試統計函數
    print("=== 統計模組測試 ===\n")

    from patterns import get_patterns_as_vectors
    from hopfield import HopfieldNetwork
    from evaluation import run_multiple_tests

    patterns = get_patterns_as_vectors()
    hopfield = HopfieldNetwork()
    hopfield.train(patterns)

    # 運行測試
    print("1. 運行多次測試...")
    all_results = []
    for i, pattern in enumerate(patterns):
        results = run_multiple_tests(
            pattern, i, patterns, hopfield, noise_level=0.2, n_trials=10, start_seed=0
        )
        all_results.append(results)

    # 計算統計
    print("\n2. 計算統計指標...")
    table = create_statistics_table(all_results)

    # 打印表格
    print_statistics_table(table)

    # 打印單個pattern的詳細統計
    stats = calculate_statistics(all_results[1])  # Pattern 2
    print_statistics_summary(stats, "Pattern 2")

    print("\n✓ 統計模組測試完成")
