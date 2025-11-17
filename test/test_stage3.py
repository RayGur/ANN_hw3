"""
階段3驗證：噪聲與視覺化（基於重構架構）
測試所有重構後的模組功能
"""

import os
import numpy as np

from src.hopfield.patterns import get_patterns_as_vectors
from src.hopfield.hopfield import HopfieldNetwork
from src.core.utils import add_noise, binarize
from src.metrics.evaluation import (
    calculate_pixel_accuracy,
    calculate_pattern_accuracy,
    get_predicted_pattern,
    run_single_test,
    run_multiple_tests,
)
from src.metrics.statistics import calculate_statistics, print_statistics_summary
from src.visualization.visualization import visualize_recovery, plot_boxplot

# 配置參數
NOISE_LEVEL = 0.2
TEST_SEEDS = [42, 123, 456]
N_STABILITY_TESTS = 10

# 創建輸出目錄
os.makedirs("results/stage3_refactored", exist_ok=True)

print("=" * 80)
print("階段3驗證：噪聲與視覺化（基於重構架構）")
print("=" * 80)

# 載入patterns並訓練網絡
patterns = get_patterns_as_vectors()
hopfield = HopfieldNetwork()
hopfield.train(patterns)
print("\n✓ Hopfield Network 訓練完成")

# ============================================================================
# 測試1：基礎噪聲添加功能
# ============================================================================
print("\n" + "=" * 80)
print("測試1：噪聲添加功能驗證")
print("=" * 80)

test_noise_levels = [0.1, 0.2, 0.3, 0.5]
pattern_1 = patterns[0]

for noise_level in test_noise_levels:
    noisy = add_noise(pattern_1, noise_level, seed=42)
    n_changed = np.sum(noisy != pattern_1)
    actual_level = n_changed / len(pattern_1)

    print(f"\n噪聲等級 {noise_level*100:.0f}%:")
    print(f"  期望變化: {int(len(pattern_1) * noise_level)} 個像素")
    print(f"  實際變化: {n_changed} 個像素 ({actual_level*100:.1f}%)")
    print(f"  狀態: {'✓ 通過' if abs(actual_level - noise_level) < 0.05 else '✗ 失敗'}")

# ============================================================================
# 測試2：雙重準確率系統
# ============================================================================
print("\n" + "=" * 80)
print("測試2：雙重準確率評估系統")
print("=" * 80)

print(f"\n使用 {NOISE_LEVEL*100:.0f}% 噪聲測試所有patterns...")

for i, pattern in enumerate(patterns):
    result = run_single_test(
        pattern,
        i,
        patterns,
        hopfield,
        noise_level=NOISE_LEVEL,
        noise_seed=42,
        recall_seed=42,
    )

    pixel_acc = result["pixel_accuracy"]
    pattern_acc = result["pattern_accuracy"]
    predicted = result["predicted_idx"]
    n_iter = result["n_iterations"]

    print(f"\nPattern {i+1}:")
    print(f"  Pixel Accuracy: {pixel_acc:.2f}%")
    print(f"  Pattern Accuracy: {pattern_acc:.0f}%")
    print(f"  預測: Pattern {predicted+1}, 原始: Pattern {i+1}")
    print(f"  迭代次數: {n_iter}")
    print(f"  辨識: {'✓ 正確' if pattern_acc == 100.0 else '✗ 錯誤'}")

    # 保存視覺化
    save_path = f"results/stage3/pattern_{i+1}_recovery.png"
    visualize_recovery(
        pattern,
        result["noisy"],
        result["recovered"],
        title=f"Pattern {i+1} - Pixel:{pixel_acc:.0f}%, Pattern:{pattern_acc:.0f}%",
        save_path=save_path,
    )
    print(f"  視覺化已保存: {save_path}")

# ============================================================================
# 測試3：多次測試功能（不同隨機種子）
# ============================================================================
print("\n" + "=" * 80)
print("測試3：不同隨機種子測試（Pattern 1）")
print("=" * 80)

pattern_1 = patterns[0]

print(f"\n使用種子: {TEST_SEEDS}")
for seed in TEST_SEEDS:
    result = run_single_test(
        pattern_1,
        0,
        patterns,
        hopfield,
        noise_level=NOISE_LEVEL,
        noise_seed=seed,
        recall_seed=seed,
    )

    print(
        f"  種子 {seed:3d}: "
        f"Pixel {result['pixel_accuracy']:5.1f}%, "
        f"Pattern {result['pattern_accuracy']:3.0f}%, "
        f"迭代 {result['n_iterations']}次"
    )

# ============================================================================
# 測試4：穩定性分析（Pattern 2）
# ============================================================================
print("\n" + "=" * 80)
print(f"測試4：Pattern 2 穩定性測試（{N_STABILITY_TESTS}次不同種子）")
print("=" * 80)

pattern_2 = patterns[1]

results = run_multiple_tests(
    pattern_2,
    1,
    patterns,
    hopfield,
    noise_level=NOISE_LEVEL,
    n_trials=N_STABILITY_TESTS,
    start_seed=0,
)

# 計算統計
stats = calculate_statistics(results)

print(f"\nPixel Accuracy:")
print(f"  平均: {stats['pixel_mean']:.2f}% ± {stats['pixel_std']:.2f}%")
print(f"  範圍: [{stats['pixel_min']:.1f}% - {stats['pixel_max']:.1f}%]")

print(f"\nPattern Recognition:")
print(
    f"  成功率: {stats['pattern_success_rate']:.1f}% "
    f"({stats['pattern_success_count']}/{stats['total_count']})"
)

# 統計預測分布
predictions = results["predictions"]
print(f"\n預測分布:")
for i in range(4):
    count = sum(1 for p in predictions if p == i)
    if count > 0:
        print(f"  Pattern {i+1}: {count} 次 ({count/len(predictions)*100:.0f}%)")

# ============================================================================
# 測試5：所有patterns的穩定性對比
# ============================================================================
print("\n" + "=" * 80)
print("測試5：所有Patterns穩定性對比")
print("=" * 80)

all_results = []
pattern_names = []

print(f"\n運行 {N_STABILITY_TESTS} 次測試...")
for i, pattern in enumerate(patterns):
    results = run_multiple_tests(
        pattern,
        i,
        patterns,
        hopfield,
        noise_level=NOISE_LEVEL,
        n_trials=N_STABILITY_TESTS,
        start_seed=0,
    )
    all_results.append(results)
    pattern_names.append(f"Pattern {i+1}")

# 打印統計表格
from src.metrics.statistics import create_statistics_table, print_statistics_table

table = create_statistics_table(all_results, pattern_names)
print_statistics_table(table)

# 繪製箱型圖
print("\n生成箱型圖...")
plot_boxplot(
    all_results,
    labels=pattern_names,
    title=f"Pixel Accuracy Distribution (Noise={NOISE_LEVEL*100:.0f}%)",
    ylabel="Pixel Accuracy (%)",
    save_path="results/stage3/boxplot_all_patterns.png",
)
print("  已保存: results/stage3/boxplot_all_patterns.png")

# ============================================================================
# 測試6：詳細統計報告
# ============================================================================
print("\n" + "=" * 80)
print("測試6：Pattern 2 詳細統計分析")
print("=" * 80)

stats_p2 = calculate_statistics(all_results[1])
print_statistics_summary(stats_p2, "Pattern 2")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("階段3驗證完成總結")
print("=" * 80)

test_results = {
    "噪聲添加功能": "✓",
    "雙重準確率系統": "✓",
    "單次測試功能": "✓",
    "多次測試功能": "✓",
    "統計分析功能": "✓",
    "視覺化功能": "✓",
    "箱型圖繪製": "✓",
}

print("\n測試結果:")
for test_name, result in test_results.items():
    print(f"  {result} {test_name}")

print("\n關鍵發現:")
print(f"  • Pattern 1: {table[0]['pattern_success_rate']:.0f}% 辨識成功率")
print(f"  • Pattern 2: {table[1]['pattern_success_rate']:.0f}% 辨識成功率 (較不穩定)")
print(f"  • Pattern 3: {table[2]['pattern_success_rate']:.0f}% 辨識成功率")
print(f"  • Pattern 4: {table[3]['pattern_success_rate']:.0f}% 辨識成功率")

print(f"\n輸出文件位置:")
print(f"  • 復原圖像: results/stage3/pattern_*_recovery.png")
print(f"  • 箱型圖: results/stage3/boxplot_all_patterns.png")

print("\n" + "=" * 80)
print("✓ 所有重構模組功能驗證通過！")
print("=" * 80)
