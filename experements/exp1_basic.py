"""
實驗一：基本復原測試
測試Hopfield Network在20%噪聲下的復原能力

實驗設計:
- 4 patterns (數字 1, 2, 3, 4)
- 噪聲等級: 20%
- 隨機種子: 42, 123, 456
- 評估指標: Pixel Accuracy + Pattern Recognition Accuracy
"""

import os
import numpy as np
from datetime import datetime

from src.hopfield.patterns import get_patterns_as_vectors
from src.hopfield.hopfield import HopfieldNetwork
from src.core.utils import add_noise, binarize
from src.metrics.evaluation import run_single_test, run_multiple_tests
from src.metrics.statistics import (
    calculate_statistics,
    create_statistics_table,
    print_statistics_table,
)
from src.visualization.visualization import visualize_recovery, plot_boxplot

# ============================================================================
# 實驗配置
# ============================================================================
CONFIG = {
    "experiment_name": "exp1_basic",
    "noise_level": 0.2,
    "test_seeds": [42, 123, 456],
    "n_patterns": 4,
    "save_dir": "results/exp1_basic",
}

# ============================================================================
# 主實驗流程
# ============================================================================


def run_experiment_1():
    """
    執行實驗一：基本復原測試
    """
    print("=" * 80)
    print("實驗一：基本復原測試")
    print("=" * 80)
    print(f"\n實驗配置:")
    print(f"  噪聲等級: {CONFIG['noise_level']*100:.0f}%")
    print(f"  測試種子: {CONFIG['test_seeds']}")
    print(f"  Pattern數量: {CONFIG['n_patterns']}")
    print(f"  輸出目錄: {CONFIG['save_dir']}")

    # 創建輸出目錄
    os.makedirs(f"{CONFIG['save_dir']}/figures", exist_ok=True)

    # 載入patterns並訓練網絡
    print("\n" + "-" * 80)
    print("步驟1: 訓練Hopfield Network")
    print("-" * 80)
    patterns = get_patterns_as_vectors()
    hopfield = HopfieldNetwork()
    hopfield.train(patterns)
    print("✓ 訓練完成 (Outer Product Method)")

    # ========================================================================
    # 測試1: 每個pattern × 每個seed的復原結果
    # ========================================================================
    print("\n" + "-" * 80)
    print("步驟2: 基本復原測試 (4 patterns × 3 seeds = 12 tests)")
    print("-" * 80)

    all_results = []

    for i, pattern in enumerate(patterns):
        print(f"\n【Pattern {i+1}】")
        pattern_results = []

        for seed_idx, seed in enumerate(CONFIG["test_seeds"], 1):
            result = run_single_test(
                pattern,
                i,
                patterns,
                hopfield,
                noise_level=CONFIG["noise_level"],
                noise_seed=seed,
                recall_seed=seed,
            )

            pixel_acc = result["pixel_accuracy"]
            pattern_acc = result["pattern_accuracy"]
            predicted = result["predicted_idx"]
            n_iter = result["n_iterations"]

            # 記錄結果
            pattern_results.append(
                {
                    "seed": seed,
                    "pixel_acc": pixel_acc,
                    "pattern_acc": pattern_acc,
                    "predicted": predicted,
                    "n_iterations": n_iter,
                }
            )

            # 打印結果
            status = "✓" if pattern_acc == 100.0 else "✗"
            print(
                f"  種子 {seed}: "
                f"Pixel {pixel_acc:5.1f}%, "
                f"Pattern {pattern_acc:3.0f}%, "
                f"迭代 {n_iter}次 {status}"
            )

            # 保存視覺化
            fig_path = f"{CONFIG['save_dir']}/figures/pattern_{i+1}_seed_{seed}.png"
            visualize_recovery(
                pattern,
                result["noisy"],
                result["recovered"],
                title=f"Pattern {i+1} - Seed {seed} (Pixel: {pixel_acc:.1f}%, Pattern: {pattern_acc:.0f}%)",
                save_path=fig_path,
            )

        all_results.append(pattern_results)

    # ========================================================================
    # 測試2: 每個pattern的穩定性分析（多次測試）
    # ========================================================================
    print("\n" + "-" * 80)
    print("步驟3: 穩定性分析 (每個pattern測試10次)")
    print("-" * 80)

    stability_results = []

    for i, pattern in enumerate(patterns):
        results = run_multiple_tests(
            pattern,
            i,
            patterns,
            hopfield,
            noise_level=CONFIG["noise_level"],
            n_trials=10,
            start_seed=0,
        )
        stability_results.append(results)

        stats = calculate_statistics(results)
        print(f"\nPattern {i+1}:")
        print(
            f"  Pixel Accuracy: {stats['pixel_mean']:.2f}% ± {stats['pixel_std']:.2f}%"
        )
        print(
            f"  Pattern Success: {stats['pattern_success_count']}/10 = {stats['pattern_success_rate']:.0f}%"
        )

    # ========================================================================
    # 統計分析與視覺化
    # ========================================================================
    print("\n" + "-" * 80)
    print("步驟4: 生成統計表格與視覺化")
    print("-" * 80)

    # 統計表格
    pattern_names = [f"Pattern {i+1}" for i in range(CONFIG["n_patterns"])]
    stats_table = create_statistics_table(stability_results, pattern_names)
    print_statistics_table(stats_table)

    # 箱型圖
    boxplot_path = f"{CONFIG['save_dir']}/figures/boxplot_all_patterns.png"
    plot_boxplot(
        stability_results,
        labels=pattern_names,
        title=f"Pixel Accuracy Distribution (Noise={CONFIG['noise_level']*100:.0f}%)",
        ylabel="Pixel Accuracy (%)",
        save_path=boxplot_path,
    )
    print(f"\n✓ 箱型圖已保存: {boxplot_path}")

    # ========================================================================
    # 生成實驗報告
    # ========================================================================
    print("\n" + "-" * 80)
    print("步驟5: 生成實驗報告")
    print("-" * 80)

    report_path = f"{CONFIG['save_dir']}/experiment_report.txt"
    generate_report(all_results, stats_table, report_path)
    print(f"✓ 實驗報告已保存: {report_path}")

    # ========================================================================
    # 總結
    # ========================================================================
    print("\n" + "=" * 80)
    print("實驗一完成")
    print("=" * 80)
    print(f"\n輸出文件:")
    print(f"  • 復原圖像: {CONFIG['save_dir']}/figures/pattern_*_seed_*.png (12張)")
    print(f"  • 箱型圖: {boxplot_path}")
    print(f"  • 實驗報告: {report_path}")

    # 關鍵發現
    print(f"\n關鍵發現:")
    for i, stats in enumerate(stats_table, 1):
        success_rate = stats["pattern_success_rate"]
        status = "穩定" if success_rate == 100.0 else "不穩定"
        print(f"  • Pattern {i}: {success_rate:.0f}% 辨識成功率 ({status})")

    # 特別標注Pattern 2問題
    if stats_table[1]["pattern_success_rate"] < 100.0:
        print(f"\n⚠️  注意: Pattern 2 與 Pattern 3 相似度高達 82.2%，容易混淆")

    print("\n" + "=" * 80)


def generate_report(all_results, stats_table, save_path):
    """
    生成文字格式的實驗報告

    Args:
        all_results: 基本測試結果
        stats_table: 統計表格
        save_path: 保存路徑
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("實驗一：基本復原測試 - 實驗報告\n")
        f.write("=" * 80 + "\n\n")

        # 基本資訊
        f.write(f"實驗日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"實驗名稱: {CONFIG['experiment_name']}\n")
        f.write(f"噪聲等級: {CONFIG['noise_level']*100:.0f}%\n")
        f.write(f"測試種子: {CONFIG['test_seeds']}\n")
        f.write(f"Pattern數量: {CONFIG['n_patterns']}\n\n")

        # 詳細結果
        f.write("-" * 80 + "\n")
        f.write("一、詳細測試結果 (4 patterns × 3 seeds)\n")
        f.write("-" * 80 + "\n\n")

        for i, pattern_results in enumerate(all_results, 1):
            f.write(f"Pattern {i}:\n")
            for result in pattern_results:
                status = "✓ 成功" if result["pattern_acc"] == 100.0 else "✗ 失敗"
                f.write(
                    f"  種子 {result['seed']:3d}: "
                    f"Pixel {result['pixel_acc']:5.1f}%, "
                    f"Pattern {result['pattern_acc']:3.0f}%, "
                    f"迭代 {result['n_iterations']}次 - {status}\n"
                )
            f.write("\n")

        # 統計摘要
        f.write("-" * 80 + "\n")
        f.write("二、統計摘要 (基於10次測試)\n")
        f.write("-" * 80 + "\n\n")

        for stats in stats_table:
            f.write(f"{stats['pattern_name']}:\n")
            f.write(f"  Pixel Accuracy:\n")
            f.write(
                f"    平均: {stats['pixel_mean']:.2f}% ± {stats['pixel_std']:.2f}%\n"
            )
            f.write(
                f"    範圍: [{stats['pixel_min']:.1f}%, {stats['pixel_max']:.1f}%]\n"
            )
            f.write(f"  Pattern Recognition:\n")
            f.write(
                f"    成功率: {stats['pattern_success_rate']:.1f}% "
                f"({stats['pattern_success_count']}/{stats['total_count']})\n\n"
            )

        # 關鍵發現
        f.write("-" * 80 + "\n")
        f.write("三、關鍵發現與分析\n")
        f.write("-" * 80 + "\n\n")

        # 整體表現
        overall_success = sum(s["pattern_success_count"] for s in stats_table)
        overall_total = sum(s["total_count"] for s in stats_table)
        f.write(f"1. 整體表現:\n")
        f.write(
            f"   總體辨識成功率: {overall_success}/{overall_total} = "
            f"{overall_success/overall_total*100:.1f}%\n\n"
        )

        # 個別分析
        f.write(f"2. 個別Pattern分析:\n")
        for i, stats in enumerate(stats_table, 1):
            if stats["pattern_success_rate"] == 100.0:
                f.write(f"   Pattern {i}: 穩定，100% 辨識成功\n")
            else:
                f.write(
                    f"   Pattern {i}: 不穩定，僅 {stats['pattern_success_rate']:.0f}% 辨識成功\n"
                )
        f.write("\n")

        # Pattern 2 特殊說明
        if stats_table[1]["pattern_success_rate"] < 100.0:
            f.write(f"3. Pattern 2 問題分析:\n")
            f.write(f"   - Pattern 2 與 Pattern 3 的相似度高達 82.2%\n")
            f.write(f"   - 兩者僅在中間橫條（第5-6行）的8個像素不同\n")
            f.write(f"   - 20%噪聲可能翻轉關鍵區域，導致誤判\n")
            f.write(f"   - 這是Hopfield Network容量限制的典型案例\n\n")

        # 結論
        f.write("-" * 80 + "\n")
        f.write("四、結論\n")
        f.write("-" * 80 + "\n\n")
        f.write(
            f"本實驗驗證了Hopfield Network在{CONFIG['noise_level']*100:.0f}%噪聲下的復原能力。\n"
        )
        f.write(f"實驗發現Pattern相似度是影響復原成功率的關鍵因素。\n")
        f.write(
            f"建議在實驗二中測試更高噪聲等級，並考慮使用Pseudo-inverse方法改善。\n\n"
        )

        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment_1()
