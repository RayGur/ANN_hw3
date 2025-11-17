"""
定義4個原始圖像 (9x5 = 45 bits)
編碼規則: 白色 = -1, 灰色 = +1
"""

import numpy as np

# 圖像1: 數字 "1"
PATTERN_1 = np.array(
    [
        [-1, -1, +1, +1, -1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, +1, -1],
        [-1, -1, +1, +1, -1],
        [-1, -1, +1, +1, -1],
        [-1, -1, +1, +1, -1],
        [-1, -1, +1, +1, -1],
        [-1, -1, +1, +1, -1],
        [-1, -1, +1, +1, -1],
    ]
)

# 圖像2: 數字 "2"
PATTERN_2 = np.array(
    [
        [+1, +1, +1, +1, +1],
        [+1, +1, +1, +1, +1],
        [-1, -1, -1, +1, +1],
        [-1, -1, -1, +1, +1],
        [+1, +1, +1, +1, +1],
        [+1, +1, -1, -1, -1],
        [+1, +1, -1, -1, -1],
        [+1, +1, +1, +1, +1],
        [+1, +1, +1, +1, +1],
    ]
)

# 圖像3: 數字 "3"
PATTERN_3 = np.array(
    [
        [+1, +1, +1, +1, +1],
        [+1, +1, +1, +1, +1],
        [-1, -1, -1, +1, +1],
        [-1, -1, -1, +1, +1],
        [+1, +1, +1, +1, +1],
        [-1, -1, -1, +1, +1],
        [-1, -1, -1, +1, +1],
        [+1, +1, +1, +1, +1],
        [+1, +1, +1, +1, +1],
    ]
)

# 圖像4: 數字 "4"
PATTERN_4 = np.array(
    [
        [+1, +1, -1, +1, +1],
        [+1, +1, -1, +1, +1],
        [+1, +1, -1, +1, +1],
        [+1, +1, +1, +1, +1],
        [+1, +1, +1, +1, +1],
        [-1, -1, -1, +1, +1],
        [-1, -1, -1, +1, +1],
        [-1, -1, -1, +1, +1],
        [-1, -1, -1, +1, +1],
    ]
)


def get_patterns_as_vectors():
    # 4個圖像轉換為45維向量
    patterns = [PATTERN_1, PATTERN_2, PATTERN_3, PATTERN_4]
    return [p.flatten() for p in patterns]


def get_patterns_as_matrices():
    return [PATTERN_1, PATTERN_2, PATTERN_3, PATTERN_4]


def vector_to_matrix(vector):
    # 將45維向量轉換回9x5矩陣
    return vector.reshape(9, 5)


if __name__ == "__main__":
    # 驗證檢查
    print("=== 階段1驗證：圖像定義 ===\n")

    vectors = get_patterns_as_vectors()
    matrices = get_patterns_as_matrices()

    for i, (vec, mat) in enumerate(zip(vectors, matrices), 1):
        print(f"圖像 {i}:")
        print(f"  向量形狀: {vec.shape}")
        print(f"  矩陣形狀: {mat.shape}")
        print(f"  向量前10個元素: {vec[:10]}")
        print(f"  值範圍: [{vec.min()}, {vec.max()}]")
        print(f"  白色(-1)數量: {np.sum(vec == -1)}")
        print(f"  灰色(+1)數量: {np.sum(vec == +1)}")
        print()

    # 測試轉換
    test_vec = vectors[0]
    test_mat = vector_to_matrix(test_vec)
    print("轉換測試:")
    print(f"  原始矩陣與轉換矩陣相等: {np.array_equal(matrices[0], test_mat)}")
