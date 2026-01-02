# NKUST Neural Network Final Project – YOLOv8 Object Detection

本專案為國立高雄科技大學（NKUST）類神經網路課程期末專題，採用 Ultralytics YOLOv8 目標偵測模型，針對特定影像資料集進行資料前處理、模型訓練與推論實驗。研究重點在於資料品質控管、負樣本比例平衡，以及多階段訓練策略對模型效能之影響。

---

##  專案目錄結構
    AI_CUP_2025/
    ├── data/
    │ ├── images/
    │ │ ├── train/
    │ │ └── val/
    │ ├── labels/
    │ │ ├── train/
    │ │ └── val/
    │ └── dataset.yaml
    ├── train.py
    ├── predict.py
    ├── README.md
    └── requirements.txt


---

##  專案方法概述

### 1. 資料前處理（Data Preprocessing）

原始資料集中包含大量未標註影像，若直接用於訓練將導致正負樣本比例嚴重失衡，使模型過度學習背景特徵。本研究在訓練前先進行系統化資料前處理，流程如下：

- 遞迴掃描所有影像與標註檔案
- 檢查影像是否存在對應 `.txt` 標註檔
- 判斷標註檔是否為空檔
- 將「無標註或空標註影像」視為負樣本
- 控制負樣本數量為正樣本的 **2 倍**
- 多餘負樣本以隨機方式移動至 `removed_negative/` 資料夾（不直接刪除）

此策略可有效改善資料分布不均問題，提升模型訓練穩定性與泛化能力。

---

### 2. 模型架構與訓練策略

- **模型架構**：YOLOv8m  
- **訓練方式**：遷移學習（Transfer Learning）  
- **類別設定**：單一類別（`single_cls=True`）

#### 兩階段訓練流程：

**Stage 1：預訓練模型微調**
- 初始權重：`yolov8m.pt`
- Epochs：30
- Batch size：16
- Image size：640 × 640

**Stage 2：模型再訓練**
- 初始權重：Stage 1 產生的 `best.pt`
- Epochs：100
- Batch size：32
- Image size：512 × 512

---

### 3. 訓練參數設定

| 參數 | 設定值 |
|----|----|
| Optimizer | AdamW |
| lr0 | 0.01 |
| lrf | 0.01 |
| Momentum | 0.937 |
| Warmup bias lr | 0.1 |
| AMP | Enabled |

#### 資料增強策略
- Mosaic Augmentation
- MixUp Augmentation
- Random Scale (`scale=0.5`)
- Horizontal Flip (`fliplr=0.5`)

---

##  執行環境

- **作業系統**：Windows 10 (64-bit)
- **Python**：3.12.12
- **主要套件**：
  - PyTorch
  - Ultralytics
  - os / random / shutil

---

##  訓練與推論

### 安裝環境
```bash
pip install -r requirements.txt

