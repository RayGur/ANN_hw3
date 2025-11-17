# 快速命令參考卡

## 🚀 立即開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 運行實驗

#### 實驗一：基本復原測試
```bash
python -m experiments.exp1_basic
```
**時間：** ~10秒
**輸出：** 12張復原圖 + 統計報告

---

#### 實驗二：噪聲分析
```bash
python -m experiments.exp2_noise
```
**時間：** ~2分鐘
**輸出：** 6張對比圖 + 詳細報告

---

## 📂 結果位置

```
results/
├── exp1_basic/
│   ├── experiment_report.txt      ← 實驗一報告
│   └── figures/                   ← 12張復原圖 + 箱型圖
│
└── exp2_noise/
    ├── experiment_report.txt      ← 實驗二報告
    └── figures/                   ← 6張對比圖表
```

---

## 🔍 檢查結果

**查看報告：**
```bash
# Windows
type results\exp1_basic\experiment_report.txt
type results\exp2_noise\experiment_report.txt

# Linux/Mac
cat results/exp1_basic/experiment_report.txt
cat results/exp2_noise/experiment_report.txt
```

**開啟圖片目錄：**
```bash
# Windows
explorer results\exp1_basic\figures
explorer results\exp2_noise\figures

# Linux
nautilus results/exp1_basic/figures
nautilus results/exp2_noise/figures

# Mac
open results/exp1_basic/figures
open results/exp2_noise/figures
```

---

## 🧪 測試功能

```bash
# 測試重構模組
python -m test.test_stage3

# 測試Pseudo-inverse
python test_pseudoinverse.py
```

---

## ⚡ 快速實驗參數調整

### 修改實驗一參數
編輯 `experiments/exp1_basic.py`：

```python
CONFIG = {
    'noise_level': 0.2,           # 改變噪聲等級
    'test_seeds': [42, 123, 456], # 改變測試種子
}
```

### 修改實驗二參數
編輯 `experiments/exp2_noise.py`：

```python
CONFIG = {
    'noise_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 噪聲範圍
    'n_trials': 10,                                   # 每條件測試次數
}
```

---

## 🐛 問題排查

### 問題1: ModuleNotFoundError

**症狀：**
```
ModuleNotFoundError: No module named 'src'
```

**解決：**
```bash
# 使用模組方式運行（推薦）
python -m experiments.exp1_basic

# 或設置PYTHONPATH
export PYTHONPATH=$(pwd)  # Linux/Mac
set PYTHONPATH=%cd%       # Windows
```

---

### 問題2: 缺少依賴套件

**症狀：**
```
ModuleNotFoundError: No module named 'numpy'
```

**解決：**
```bash
pip install -r requirements.txt
```

---

### 問題3: 結果目錄不存在

**症狀：**
程式報錯找不到目錄

**解決：**
```bash
# 手動創建
mkdir -p results/exp1_basic/figures
mkdir -p results/exp2_noise/figures
```
（程式應該會自動創建，這是備用方案）

---

## 📊 預期結果摘要

### 實驗一
- ✅ Pattern 1, 3, 4: 100% 成功率
- ⚠️ Pattern 2: 80% 成功率（與Pattern 3混淆）

### 實驗二
| 噪聲 | Outer Product | Pseudo-inverse |
|-----|--------------|----------------|
| 20% | 95% | **100%** ✓ |
| 40% | 50% | **62.5%** ✓ |

**結論：** Pseudo-inverse 在中低噪聲顯著優於 Outer Product

---

## 💡 快速技巧

**只運行單個pattern測試：**
```python
from patterns import get_patterns_as_vectors
from hopfield import HopfieldNetwork
from utils import add_noise, binarize

patterns = get_patterns_as_vectors()
hopfield = HopfieldNetwork()
hopfield.train(patterns, method='pseudoinverse')

# 測試Pattern 1
noisy = add_noise(patterns[0], 0.2, seed=42)
recovered, _, _ = hopfield.recall(noisy)
```

**批次比較兩種方法：**
```python
for method in ['outer_product', 'pseudoinverse']:
    hopfield = HopfieldNetwork()
    hopfield.train(patterns, method=method)
    # ... 測試
```
