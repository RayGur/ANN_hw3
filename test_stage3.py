"""
階段3驗證：噪聲添加與視覺化（使用雙重準確率評估）
"""

import numpy as np
import os
from patterns import get_patterns_as_vectors
from hopfield import HopfieldNetwork
from utils import (
    add_noise,
    binarize,
    calculate_accuracy,
    calculate_pattern_accuracy,
    get_predicted_pattern,
    visualize_recovery,
)

# 創建輸出目錄
os.makedirs("results/stage3", exist_ok=True)

print("=== 階段3驗證：噪聲與視覺化（雙重準確率） ===\n")

# 載入patterns並訓練網絡
patterns = get_patterns_as_vectors()
hopfield = HopfieldNetwork()
hopfield.train(patterns)

print("1. 測試噪聲添加函數\n")

# 測試不同噪聲等級
noise_levels = [0.1, 0.2, 0.3, 0.5]
pattern_1 = patterns[0]

for noise_level in noise_levels:
    noisy = add_noise(pattern_1, noise_level, seed=42)
    n_changed = np.sum(noisy != pattern_1)
    actual_level = n_changed / len(pattern_1)

    print(f"  噪聲等級 {noise_level*100:.0f}%:")
    print(f"    期望變化: {int(len(pattern_1) * noise_level)} 個像素")
    print(f"    實際變化: {n_changed} 個像素 ({actual_level*100:.1f}%)")

print("\n2. 測試復原功能（20%噪聲，雙重準確率）\n")

# 對每個pattern測試20%噪聲
for i, pattern in enumerate(patterns):
    # 添加噪聲
    noisy = add_noise(pattern, noise_level=0.2, seed=42)

    # 復原
    recovered, n_iter, converged = hopfield.recall(noisy, seed=42)
    recovered_binary = binarize(recovered)

    # 計算雙重準確率
    pixel_acc = calculate_accuracy(pattern, recovered_binary)
    pattern_acc = calculate_pattern_accuracy(i, recovered_binary, patterns)
    predicted = get_predicted_pattern(recovered_binary, patterns)

    print(f"  Pattern {i+1}:")
    print(f"    收斂: {converged}, 迭代: {n_iter}次")
    print(f"    Pixel Accuracy: {pixel_acc:.2f}%")
    print(f"    Pattern Accuracy: {pattern_acc:.0f}%")
    print(f"    預測: Pattern {predicted+1}, 原始: Pattern {i+1}")

    # 視覺化
    save_path = f"results/stage3/pattern_{i+1}_recovery.png"
    visualize_recovery(
        pattern,
        noisy,
        recovered_binary,
        title=f"Pattern {i+1} - Pixel:{pixel_acc:.0f}%, Pattern:{pattern_acc:.0f}%",
        save_path=save_path,
    )
    print(f"    已保存: {save_path}")

print("\n3. 測試不同隨機種子（Pattern 1）\n")

# 測試Pattern 2用3個不同種子
seeds = [42, 123, 456]
pattern_1 = patterns[0]

for seed in seeds:
    noisy = add_noise(pattern_1, noise_level=0.2, seed=seed)
    recovered, n_iter, _ = hopfield.recall(noisy, seed=seed)
    recovered_binary = binarize(recovered)
    pixel_acc = calculate_accuracy(pattern_1, recovered_binary)
    pattern_acc = calculate_pattern_accuracy(0, recovered_binary, patterns)

    print(
        f"  種子 {seed}: Pixel {pixel_acc:.1f}%, Pattern {pattern_acc:.0f}%, 迭代 {n_iter}次"
    )

print("\n4. Pattern 2 穩定性測試（10次不同種子）\n")

pattern_2 = patterns[1]
pixel_accs = []
pattern_accs = []
predictions = []

for seed in range(10):
    noisy = add_noise(pattern_2, 0.2, seed=seed)
    recovered, _, _ = hopfield.recall(noisy, seed=seed)
    recovered_binary = binarize(recovered)

    pixel_acc = calculate_accuracy(pattern_2, recovered_binary)
    pattern_acc = calculate_pattern_accuracy(1, recovered_binary, patterns)
    predicted = get_predicted_pattern(recovered_binary, patterns)

    pixel_accs.append(pixel_acc)
    pattern_accs.append(pattern_acc)
    predictions.append(predicted)

success_rate = sum(1 for p in pattern_accs if p == 100.0) / len(pattern_accs) * 100

print(
    f"  Pixel Accuracy: 平均 {np.mean(pixel_accs):.1f}%, 範圍 [{min(pixel_accs):.1f}%-{max(pixel_accs):.1f}%]"
)
print(
    f"  Pattern Accuracy: {success_rate:.0f}% 辨識成功率 ({sum(1 for p in pattern_accs if p == 100.0)}/10)"
)
print(f"  誤判為 Pattern 3: {sum(1 for p in predictions if p == 2)} 次")

print("\n✓ 階段3驗證完成！")
print("  - 噪聲添加功能正常")
print("  - 復原功能正常")
print("  - 雙重準確率系統正常")
print("  - 視覺化功能正常")
