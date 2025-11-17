"""
視覺化模組
包含所有繪圖和視覺化函數
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_pattern(pattern, title="Pattern", save_path=None):
    """
    顯示單個pattern

    Args:
        pattern: 圖像（45維向量或9x5矩陣）
        title: 圖像標題
        save_path: 保存路徑（可選）
    """
    if pattern.ndim == 1:
        pattern = pattern.reshape(9, 5)

    plt.figure(figsize=(3, 5))
    plt.imshow(pattern, cmap="gray_r", vmin=-1, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_recovery(
    original, noisy, recovered, title="Recovery Result", save_path=None
):
    """
    並排顯示原圖、噪聲圖、復原圖

    Args:
        original: 原始圖像 (45維向量或9x5矩陣)
        noisy: 噪聲圖像
        recovered: 復原圖像
        title: 圖像標題
        save_path: 保存路徑（可選）
    """
    # 轉換為矩陣
    if original.ndim == 1:
        original = original.reshape(9, 5)
    if noisy.ndim == 1:
        noisy = noisy.reshape(9, 5)
    if recovered.ndim == 1:
        recovered = recovered.reshape(9, 5)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(original, cmap="gray_r", vmin=-1, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray_r", vmin=-1, vmax=1)
    axes[1].set_title("Noisy")
    axes[1].axis("off")

    axes[2].imshow(recovered, cmap="gray_r", vmin=-1, vmax=1)
    axes[2].set_title("Recovered")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_boxplot(
    all_results,
    labels,
    title="Accuracy Distribution",
    ylabel="Accuracy (%)",
    save_path=None,
):
    """
    繪製箱型圖比較多個patterns或方法

    Args:
        all_results: list of results dicts (from run_multiple_tests)
        labels: list of labels for each result
        title: 圖表標題
        ylabel: y軸標籤
        save_path: 保存路徑（可選）
    """
    # 準備數據
    pixel_data = [results["pixel_accuracies"] for results in all_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 繪製箱型圖
    bp = ax.boxplot(pixel_data, labels=labels, patch_artist=True)

    # 美化
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_noise_curve(
    noise_levels,
    accuracies_list,
    labels,
    title="Accuracy vs Noise Level",
    save_path=None,
):
    """
    繪製噪聲-準確率曲線

    Args:
        noise_levels: list of noise levels (e.g., [0.1, 0.2, 0.3])
        accuracies_list: list of accuracy arrays (one per method/pattern)
        labels: list of labels for each curve
        title: 圖表標題
        save_path: 保存路徑（可選）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for accuracies, label in zip(accuracies_list, labels):
        ax.plot(noise_levels, accuracies, marker="o", label=label, linewidth=2)

    ax.set_xlabel("Noise Level", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison_bar(
    categories, values_list, labels, title="Comparison", ylabel="Value", save_path=None
):
    """
    繪製對比柱狀圖

    Args:
        categories: x軸類別（如pattern名稱）
        values_list: list of value arrays (one per method)
        labels: list of labels for each bar group
        title: 圖表標題
        ylabel: y軸標籤
        save_path: 保存路徑（可選）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.35 if len(values_list) == 2 else 0.25

    for i, (values, label) in enumerate(zip(values_list, labels)):
        offset = width * (i - len(values_list) / 2 + 0.5)
        ax.bar(x + offset, values, width, label=label)

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # 測試視覺化函數
    print("=== 視覺化模組測試 ===\n")

    import os
    from src.hopfield.patterns import get_patterns_as_vectors
    from src.hopfield.hopfield import HopfieldNetwork
    from src.metrics.evaluation import run_multiple_tests
    from src.core.utils import add_noise, binarize

    os.makedirs("results/visualization_test", exist_ok=True)

    patterns = get_patterns_as_vectors()
    hopfield = HopfieldNetwork()
    hopfield.train(patterns)

    # 測試1: 單圖顯示
    print("1. 測試單圖顯示...")
    visualize_pattern(
        patterns[0], "Pattern 1", save_path="results/visualization_test/test_single.png"
    )

    # 測試2: 復原過程
    print("2. 測試復原過程顯示...")
    noisy = add_noise(patterns[0], 0.2, seed=42)
    recovered, _, _ = hopfield.recall(noisy, seed=42)
    recovered_binary = binarize(recovered)
    visualize_recovery(
        patterns[0],
        noisy,
        recovered_binary,
        "Test Recovery",
        save_path="results/visualization_test/test_recovery.png",
    )

    # 測試3: 箱型圖
    print("3. 測試箱型圖...")
    all_results = []
    for i, pattern in enumerate(patterns):
        results = run_multiple_tests(
            pattern, i, patterns, hopfield, noise_level=0.2, n_trials=10, start_seed=0
        )
        all_results.append(results)

    plot_boxplot(
        all_results,
        [f"P{i+1}" for i in range(4)],
        title="Test Boxplot",
        save_path="results/visualization_test/test_boxplot.png",
    )

    print("\n✓ 視覺化模組測試完成")
    print("  結果保存在 results/visualization_test/")
