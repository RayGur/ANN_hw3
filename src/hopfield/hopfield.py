"""
Hopfield Network 實作
使用 Outer Product Method 訓練
使用 異步更新 (Asynchronous) 進行 recall
"""

import numpy as np
from src.core.utils import activation_tanh


class HopfieldNetwork:
    def __init__(self, n_neurons=45):
        self.n = n_neurons
        self.W = None
        self.patterns = None

    def train(self, patterns):
        """
        使用 Outer Product Method 計算權重矩陣

        公式: W = (1/n)∑(z^(k)·(z^(k))^T) - (p/n)I
        - 對角線設為0 (無自連接)
        - bias設為0

        Args:
            patterns: list of np.array, 每個pattern是45維向量
        """
        self.patterns = patterns
        p = len(patterns)  # pattern數量
        n = self.n  # 神經元數量

        # 初始化權重矩陣
        self.W = np.zeros((n, n))

        # Outer Product: W = (1/n)∑(z^(k)·(z^(k))^T)
        for pattern in patterns:
            self.W += np.outer(pattern, pattern)

        self.W = self.W / n

        # 移除對角線項: W = W - (p/n)I
        self.W -= (p / n) * np.eye(n)

        # 確保對角線為0
        np.fill_diagonal(self.W, 0)

        # 驗證權重矩陣對稱性
        assert np.allclose(self.W, self.W.T), "權重矩陣必須對稱"

        return self.W

    def recall(self, noisy_pattern, max_iter=100, beta=100, verbose=False, seed=None):
        """
        同步更新 recall 過程

        Args:
            noisy_pattern: 噪聲圖像 (45維向量)
            max_iter: 最大迭代次數
            beta: tanh激活函數的斜率參數
            verbose: 是否打印迭代過程
            seed: 隨機種子（用於可重現性）

        Returns:
            tuple: (recovered_pattern, n_iterations, converged)
        """
        if self.W is None:
            raise ValueError("請先調用 train() 方法訓練網絡")

        # 設置隨機種子（如果提供）
        if seed is not None:
            np.random.seed(seed)

        # 初始化當前狀態
        current_state = noisy_pattern.copy()

        for iteration in range(max_iter):
            previous_state = current_state.copy()

            # 同步更新：計算所有神經元的新值
            u = np.dot(self.W, current_state)  # 所有神經元的potential
            current_state = activation_tanh(u, beta)  # 同時更新所有神經元

            # 檢查收斂條件：使用 np.allclose 處理浮點數精度問題
            if np.allclose(current_state, previous_state, rtol=1e-6, atol=1e-6):
                if verbose:
                    print(f"收斂於第 {iteration + 1} 次迭代")
                return current_state, iteration + 1, True

        if verbose:
            print(f"達到最大迭代次數 {max_iter}，未完全收斂")

        return current_state, max_iter, False

    def get_energy(self, state):
        """
        計算能量函數

        公式: E(v) = -0.5 * v^T * W * v
        註:i^b = 0

        Args:
            state: 當前狀態向量

        Returns:
            float: 能量值
        """
        if self.W is None:
            raise ValueError("請先調用 train() 方法訓練網絡")

        energy = -0.5 * np.dot(state.T, np.dot(self.W, state))
        return energy


if __name__ == "__main__":
    print("=== 階段2驗證：Hopfield Network ===\n")

    # 導入patterns和binarize
    from patterns import get_patterns_as_vectors
    from src.core.utils import binarize

    patterns = get_patterns_as_vectors()

    # 創建並訓練網絡
    print("1. 創建 Hopfield Network...")
    hopfield = HopfieldNetwork(n_neurons=45)

    print("2. 訓練網絡 (Outer Product Method)...")
    W = hopfield.train(patterns)

    print(f"   權重矩陣形狀: {W.shape}")
    print(f"   權重矩陣對稱: {np.allclose(W, W.T)}")
    print(f"   對角線元素和: {np.sum(np.diag(W)):.6f}")
    print(f"   權重範圍: [{W.min():.4f}, {W.max():.4f}]")
    print()

    # 測試：輸入原始圖像應該收斂到自己
    print("3. 測試收斂性：輸入原始圖像")
    for i, pattern in enumerate(patterns, 1):
        recovered, n_iter, converged = hopfield.recall(pattern, verbose=False, seed=42)

        # 二值化處理
        recovered_binary = binarize(recovered)

        # 計算準確率
        accuracy = np.mean(recovered_binary == pattern) * 100

        # 計算能量
        energy_original = hopfield.get_energy(pattern)
        energy_recovered = hopfield.get_energy(recovered)

        print(f"   Pattern {i}:")
        print(f"     收斂: {converged}, 迭代次數: {n_iter}")
        print(f"     準確率: {accuracy:.2f}%")
        print(f"     能量 (原始): {energy_original:.4f}")
        print(f"     能量 (復原): {energy_recovered:.4f}")

        # 檢查是否完全復原
        is_perfect = np.array_equal(recovered_binary, pattern)
        print(f"     完全復原: {is_perfect}")
        print()
