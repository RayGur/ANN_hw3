"""
測試 Pseudo-inverse 訓練方法
"""

import numpy as np
from src.hopfield.patterns import get_patterns_as_vectors
from src.hopfield.hopfield import HopfieldNetwork
from src.core.utils import add_noise, binarize
from src.metrics.evaluation import calculate_pixel_accuracy, calculate_pattern_accuracy

print("=" * 80)
print("測試 Pseudo-inverse 方法")
print("=" * 80)

patterns = get_patterns_as_vectors()

# 測試兩種方法
methods = ["outer_product", "pseudoinverse"]

for method in methods:
    print(f"\n{'='*80}")
    print(f"方法: {method.upper()}")
    print(f"{'='*80}")

    # 訓練網絡
    hopfield = HopfieldNetwork()
    hopfield.train(patterns, method=method)

    print(f"\n權重矩陣:")
    print(f"  形狀: {hopfield.W.shape}")
    print(f"  對角線和: {np.sum(np.diag(hopfield.W)):.6f}")
    print(f"  對稱性: {np.allclose(hopfield.W, hopfield.W.T)}")
    print(f"  權重範圍: [{hopfield.W.min():.4f}, {hopfield.W.max():.4f}]")

    # 測試復原能力（20%噪聲）
    print(f"\n測試復原能力 (20% 噪聲):")

    for i, pattern in enumerate(patterns, 1):
        # 添加噪聲
        noisy = add_noise(pattern, 0.2, seed=42)

        # 復原
        recovered, n_iter, converged = hopfield.recall(noisy, seed=42)
        recovered_binary = binarize(recovered)

        # 計算準確率
        pixel_acc = calculate_pixel_accuracy(pattern, recovered_binary)
        pattern_acc = calculate_pattern_accuracy(i - 1, recovered_binary, patterns)

        status = "✓" if pattern_acc == 100.0 else "✗"
        print(
            f"  Pattern {i}: Pixel {pixel_acc:5.1f}%, Pattern {pattern_acc:3.0f}%, 迭代 {n_iter}次 {status}"
        )

print("\n" + "=" * 80)
print("✓ Pseudo-inverse 方法測試完成")
print("=" * 80)
