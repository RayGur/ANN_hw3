"""
實驗二：噪聲容忍度分析 + 方法對比
比較 Outer Product vs Pseudo-inverse 在不同噪聲等級下的表現

實驗設計:
- 噪聲等級: 10%, 20%, 30%, 40%, 50%, 60%
- 訓練方法: Outer Product vs Pseudo-inverse
- 每個條件測試10次 (不同random seeds)
- 評估指標: Pixel Accuracy + Pattern Recognition Accuracy
"""

import os
import numpy as np
from datetime import datetime

from src.hopfield.patterns import get_patterns_as_vectors
from src.hopfield.hopfield import HopfieldNetwork
from src.metrics.evaluation import run_multiple_tests
from src.metrics.statistics import (
    calculate_statistics,
    create_statistics_table,
    print_statistics_table,
)
from src.visualization.visualization import (
    plot_boxplot,
    plot_noise_curve,
    plot_comparison_bar,
)

# ============================================================================
# 實驗配置
# ============================================================================
CONFIG = {
    "experiment_name": "exp2_noise",
    "noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "methods": ["outer_product", "pseudoinverse"],
    "n_trials": 10,  # 每個條件測試10次
    "n_patterns": 4,
    "save_dir": "results/exp2_noise",
}

# ============================================================================
# 主實驗流程
# ============================================================================


def run_experiment_2():
    """
    執行實驗二：噪聲容忍度分析 + 方法對比
    """
    print("=" * 80)
    print("實驗二：噪聲容忍度分析 + 方法對比")
    print("=" * 80)
    print(f"\n實驗配置:")
    print(f"  噪聲等級: {[f'{n*100:.0f}%' for n in CONFIG['noise_levels']]}")
    print(f"  訓練方法: {CONFIG['methods']}")
    print(f"  每個條件測試: {CONFIG['n_trials']} 次")
    print(f"  Pattern數量: {CONFIG['n_patterns']}")
    print(f"  輸出目錄: {CONFIG['save_dir']}")

    # 創建輸出目錄
    os.makedirs(f"{CONFIG['save_dir']}/figures", exist_ok=True)

    # 載入patterns
    print("\n" + "-" * 80)
    print("步驟1: 載入Patterns")
    print("-" * 80)
    patterns = get_patterns_as_vectors()
    print("✓ 4個patterns已載入")

    # ========================================================================
    # 對每個方法、每個噪聲等級進行測試
    # ========================================================================
    print("\n" + "-" * 80)
    print("步驟2: 執行對比實驗")
    print("-" * 80)

    # 儲存所有結果
    all_results = {"outer_product": {}, "pseudoinverse": {}}

    for method in CONFIG["methods"]:
        print(f"\n{'='*80}")
        print(f"訓練方法: {method.upper()}")
        print(f"{'='*80}")

        # 訓練網絡
        hopfield = HopfieldNetwork()
        hopfield.train(patterns, method=method)
        print(f"✓ 網絡訓練完成")

        # 對每個噪聲等級測試
        for noise_level in CONFIG["noise_levels"]:
            print(f"\n噪聲等級: {noise_level*100:.0f}%")

            # 測試每個pattern
            pattern_results = []
            for i, pattern in enumerate(patterns):
                results = run_multiple_tests(
                    pattern,
                    i,
                    patterns,
                    hopfield,
                    noise_level=noise_level,
                    n_trials=CONFIG["n_trials"],
                    start_seed=0,
                )
                pattern_results.append(results)

                stats = calculate_statistics(results)
                print(
                    f"  Pattern {i+1}: "
                    f"Pixel {stats['pixel_mean']:5.1f}%, "
                    f"Pattern Success {stats['pattern_success_count']}/{CONFIG['n_trials']}"
                )

            # 保存結果
            all_results[method][noise_level] = pattern_results

    # ========================================================================
    # 分析與視覺化
    # ========================================================================
    print("\n" + "-" * 80)
    print("步驟3: 生成對比分析與視覺化")
    print("-" * 80)

    # 3.1 計算每個method、每個noise level的平均表現
    print("\n3.1 計算平均表現...")
    summary = calculate_summary(all_results)

    # 3.2 繪製噪聲-準確率曲線
    print("3.2 繪製噪聲-準確率曲線...")
    plot_accuracy_curves(all_results, summary)

    # 3.3 繪製方法對比圖（每個噪聲等級）
    print("3.3 繪製方法對比圖...")
    plot_method_comparison(all_results, summary)

    # 3.4 繪製Pattern 2的特別分析
    print("3.4 Pattern 2 特別分析...")
    plot_pattern2_analysis(all_results)

    # ========================================================================
    # 生成實驗報告
    # ========================================================================
    print("\n" + "-" * 80)
    print("步驟4: 生成實驗報告")
    print("-" * 80)

    report_path = f"{CONFIG['save_dir']}/experiment_report.txt"
    generate_report(all_results, summary, report_path)
    print(f"✓ 實驗報告已保存: {report_path}")

    # ========================================================================
    # 總結
    # ========================================================================
    print("\n" + "=" * 80)
    print("實驗二完成")
    print("=" * 80)

    print_summary(summary)

    print(f"\n輸出文件:")
    print(f"  • 噪聲曲線圖: {CONFIG['save_dir']}/figures/noise_curves_*.png")
    print(f"  • 方法對比圖: {CONFIG['save_dir']}/figures/method_comparison_*.png")
    print(f"  • Pattern 2分析: {CONFIG['save_dir']}/figures/pattern2_*.png")
    print(f"  • 實驗報告: {report_path}")

    print("\n" + "=" * 80)


def calculate_summary(all_results):
    """
    計算摘要統計

    Returns:
        dict: {method: {noise_level: {pixel_mean, pattern_success_rate}}}
    """
    summary = {}

    for method in CONFIG["methods"]:
        summary[method] = {}

        for noise_level in CONFIG["noise_levels"]:
            pattern_results = all_results[method][noise_level]

            # 計算所有patterns的平均
            all_pixel_accs = []
            all_pattern_accs = []

            for results in pattern_results:
                all_pixel_accs.extend(results["pixel_accuracies"])
                all_pattern_accs.extend(results["pattern_accuracies"])

            pixel_mean = np.mean(all_pixel_accs)
            pattern_success_rate = (
                sum(1 for p in all_pattern_accs if p == 100.0)
                / len(all_pattern_accs)
                * 100
            )

            summary[method][noise_level] = {
                "pixel_mean": pixel_mean,
                "pattern_success_rate": pattern_success_rate,
            }

    return summary


def plot_accuracy_curves(all_results, summary):
    """繪製噪聲-準確率曲線"""

    # Pixel Accuracy曲線
    pixel_accs_outer = [
        summary["outer_product"][n]["pixel_mean"] for n in CONFIG["noise_levels"]
    ]
    pixel_accs_pseudo = [
        summary["pseudoinverse"][n]["pixel_mean"] for n in CONFIG["noise_levels"]
    ]

    plot_noise_curve(
        CONFIG["noise_levels"],
        [pixel_accs_outer, pixel_accs_pseudo],
        labels=["Outer Product", "Pseudo-inverse"],
        title="Pixel Accuracy vs Noise Level",
        save_path=f"{CONFIG['save_dir']}/figures/noise_curves_pixel.png",
    )

    # Pattern Recognition曲線
    pattern_accs_outer = [
        summary["outer_product"][n]["pattern_success_rate"]
        for n in CONFIG["noise_levels"]
    ]
    pattern_accs_pseudo = [
        summary["pseudoinverse"][n]["pattern_success_rate"]
        for n in CONFIG["noise_levels"]
    ]

    plot_noise_curve(
        CONFIG["noise_levels"],
        [pattern_accs_outer, pattern_accs_pseudo],
        labels=["Outer Product", "Pseudo-inverse"],
        title="Pattern Recognition Success Rate vs Noise Level",
        save_path=f"{CONFIG['save_dir']}/figures/noise_curves_pattern.png",
    )


def plot_method_comparison(all_results, summary):
    """繪製方法對比柱狀圖"""

    # 選擇幾個代表性的噪聲等級
    selected_noise_levels = [0.2, 0.4, 0.6]

    for noise_level in selected_noise_levels:
        categories = [f"Pattern {i+1}" for i in range(CONFIG["n_patterns"])]

        # 計算每個pattern的平均pixel accuracy
        outer_accs = []
        pseudo_accs = []

        for i in range(CONFIG["n_patterns"]):
            outer_results = all_results["outer_product"][noise_level][i]
            pseudo_results = all_results["pseudoinverse"][noise_level][i]

            outer_stats = calculate_statistics(outer_results)
            pseudo_stats = calculate_statistics(pseudo_results)

            outer_accs.append(outer_stats["pixel_mean"])
            pseudo_accs.append(pseudo_stats["pixel_mean"])

        plot_comparison_bar(
            categories,
            [outer_accs, pseudo_accs],
            labels=["Outer Product", "Pseudo-inverse"],
            title=f"Method Comparison - Noise {noise_level*100:.0f}%",
            ylabel="Pixel Accuracy (%)",
            save_path=f"{CONFIG['save_dir']}/figures/method_comparison_{int(noise_level*100)}.png",
        )


def plot_pattern2_analysis(all_results):
    """Pattern 2的特別分析（箱型圖）"""

    # 對20%噪聲進行詳細分析
    noise_level = 0.2

    outer_results = all_results["outer_product"][noise_level][1]  # Pattern 2
    pseudo_results = all_results["pseudoinverse"][noise_level][1]

    plot_boxplot(
        [outer_results, pseudo_results],
        labels=["Outer Product", "Pseudo-inverse"],
        title=f"Pattern 2 Performance Comparison (Noise {noise_level*100:.0f}%)",
        ylabel="Pixel Accuracy (%)",
        save_path=f"{CONFIG['save_dir']}/figures/pattern2_comparison.png",
    )


def generate_report(all_results, summary, save_path):
    """生成實驗報告"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("實驗二：噪聲容忍度分析 + 方法對比 - 實驗報告\n")
        f.write("=" * 80 + "\n\n")

        # 基本資訊
        f.write(f"實驗日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"噪聲等級: {[f'{n*100:.0f}%' for n in CONFIG['noise_levels']]}\n")
        f.write(f"訓練方法: {CONFIG['methods']}\n")
        f.write(f"每個條件測試次數: {CONFIG['n_trials']}\n\n")

        # 方法對比總結
        f.write("-" * 80 + "\n")
        f.write("一、方法對比總結\n")
        f.write("-" * 80 + "\n\n")

        for method in CONFIG["methods"]:
            f.write(f"{method.upper()}:\n")
            f.write(
                f"{'噪聲等級':<10} | {'Pixel Acc (Avg)':<15} | {'Pattern Success Rate':<20}\n"
            )
            f.write("-" * 50 + "\n")

            for noise_level in CONFIG["noise_levels"]:
                stats = summary[method][noise_level]
                f.write(
                    f"{noise_level*100:5.0f}%      | "
                    f"{stats['pixel_mean']:6.2f}%         | "
                    f"{stats['pattern_success_rate']:6.2f}%\n"
                )
            f.write("\n")

        # 關鍵發現
        f.write("-" * 80 + "\n")
        f.write("二、關鍵發現\n")
        f.write("-" * 80 + "\n\n")

        # 比較兩種方法
        f.write("1. 整體表現對比:\n")
        for noise_level in CONFIG["noise_levels"]:
            outer_success = summary["outer_product"][noise_level][
                "pattern_success_rate"
            ]
            pseudo_success = summary["pseudoinverse"][noise_level][
                "pattern_success_rate"
            ]

            if pseudo_success > outer_success:
                improvement = pseudo_success - outer_success
                f.write(
                    f"   噪聲{noise_level*100:.0f}%: Pseudo-inverse 勝出 (提升 {improvement:.1f}%)\n"
                )
            elif outer_success > pseudo_success:
                f.write(f"   噪聲{noise_level*100:.0f}%: Outer Product 勝出\n")
            else:
                f.write(f"   噪聲{noise_level*100:.0f}%: 兩者表現相當\n")

        f.write("\n2. Pattern 2 問題改善:\n")
        outer_p2 = all_results["outer_product"][0.2][1]
        pseudo_p2 = all_results["pseudoinverse"][0.2][1]
        outer_p2_stats = calculate_statistics(outer_p2)
        pseudo_p2_stats = calculate_statistics(pseudo_p2)

        f.write(
            f"   Outer Product: {outer_p2_stats['pattern_success_rate']:.0f}% 成功率\n"
        )
        f.write(
            f"   Pseudo-inverse: {pseudo_p2_stats['pattern_success_rate']:.0f}% 成功率\n"
        )
        f.write(
            f"   改善: {pseudo_p2_stats['pattern_success_rate'] - outer_p2_stats['pattern_success_rate']:.0f}%\n\n"
        )

        # 結論
        f.write("-" * 80 + "\n")
        f.write("三、結論\n")
        f.write("-" * 80 + "\n\n")
        f.write("Pseudo-inverse方法在處理相似patterns時表現更優秀，\n")
        f.write("特別是在Pattern 2的辨識上有顯著改善。\n")
        f.write("建議在實際應用中優先考慮Pseudo-inverse方法。\n\n")

        f.write("=" * 80 + "\n")


def print_summary(summary):
    """打印摘要"""
    print("\n關鍵結論:")

    # 比較兩種方法
    for noise_level in [0.2, 0.4, 0.6]:
        outer = summary["outer_product"][noise_level]
        pseudo = summary["pseudoinverse"][noise_level]

        print(f"\n噪聲 {noise_level*100:.0f}%:")
        print(
            f"  Outer Product: Pixel {outer['pixel_mean']:.1f}%, Pattern {outer['pattern_success_rate']:.1f}%"
        )
        print(
            f"  Pseudo-inverse: Pixel {pseudo['pixel_mean']:.1f}%, Pattern {pseudo['pattern_success_rate']:.1f}%"
        )


if __name__ == "__main__":
    run_experiment_2()
