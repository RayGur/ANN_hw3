"""
輔助函數：激活函數、視覺化、評估等
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_pattern(pattern, title="Pattern"):
    """
    視覺化單個圖像
    注意: 白色(-1)顯示為白，深色(+1)顯示為黑/灰

    Args:
        pattern: 9x5矩陣或45維向量
        title: 圖像標題
    """
    if pattern.ndim == 1:
        pattern = pattern.reshape(9, 5)

    plt.figure(figsize=(3, 5))
    plt.imshow(pattern, cmap="gray_r", vmin=-1, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()


def activation_tanh(u, beta=100):
    """
    Hyperbolic tangent 激活函數

    公式: g(u) = (1 - e^(-β·u)) / (1 + e^(-β·u))

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


def calculate_accuracy(original, recovered):
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


if __name__ == "__main__":
    # 驗證檢查
    print("=== 階段1驗證：激活函數 ===\n")

    # 測試不同輸入值
    test_inputs = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]

    print("激活函數測試 (beta=100):")
    for u in test_inputs:
        output = activation_tanh(u, beta=100)
        print(f"  g({u:5.1f}) = {output:8.5f}")

    print("\n向量輸入測試:")
    u_vec = np.array([-1, -0.5, 0, 0.5, 1])
    output_vec = activation_tanh(u_vec, beta=100)
    print(f"  輸入: {u_vec}")
    print(f"  輸出: {output_vec}")

    # 視覺化激活函數
    print("\n生成激活函數圖形...")
    u_range = np.linspace(-0.1, 0.1, 1000)

    plt.figure(figsize=(10, 6))
    for beta in [1, 10, 100]:
        y = activation_tanh(u_range, beta=beta)
        plt.plot(u_range, y, label=f"β = {beta}")

    plt.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    plt.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
    plt.xlabel("u", fontsize=12)
    plt.ylabel("g(u)", fontsize=12)
    plt.title("Hyperbolic Tangent Activation Function", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("activation_function.png", dpi=150, bbox_inches="tight")
    print("圖形已保存至: activation_function.png")

    # 檢查單調性
    print("\n單調性檢查:")
    derivatives = np.diff(activation_tanh(u_range, beta=100))
    is_monotonic = np.all(derivatives >= 0)
    print(f"  函數是否單調遞增: {is_monotonic}")
    print(f"  導數最小值: {derivatives.min():.6f}")
    print(f"  導數最大值: {derivatives.max():.6f}")
