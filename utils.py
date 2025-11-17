"""
輔助函數：激活函數、視覺化、評估等
"""

import numpy as np
import matplotlib.pyplot as plt

import patterns


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
    plt.savefig(f"results/images/{title}.png", dpi=150, bbox_inches="tight")


def activation_tanh(u, beta=100):

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


if __name__ == "__main__":

    print("=== 階段1驗證：Plot ===\n")
    visualize_pattern(patterns.PATTERN_1, title="Pattern 1")
    visualize_pattern(patterns.PATTERN_2, title="Pattern 2")
    visualize_pattern(patterns.PATTERN_3, title="Pattern 3")
    visualize_pattern(patterns.PATTERN_4, title="Pattern 4")
    print("圖像已保存至: results/images/\n")
    # 驗證檢查

    print("=== 階段2驗證：激活函數 ===\n")

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
    plt.savefig("results/figures/activation_function.png", dpi=150, bbox_inches="tight")
    print("圖形已保存至: results/figures/activation_function.png")

    # 檢查單調性
    print("\n單調性檢查:")
    derivatives = np.diff(activation_tanh(u_range, beta=100))
    is_monotonic = np.all(derivatives >= 0)
    print(f"  函數是否單調遞增: {is_monotonic}")
    print(f"  導數最小值: {derivatives.min():.6f}")
    print(f"  導數最大值: {derivatives.max():.6f}")
