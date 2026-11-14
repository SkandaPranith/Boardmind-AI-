# BoardMind AI: Dataset Architecture & Benchmark Splits

This directory houses the three core datasets utilized for training, fine-tuning, and evaluating the **BoardMind AI** computer vision and handwriting recognition pipelines.

---

## 1. Directory Structure

```
datasets/
├── handwriting/                 # IAM Handwriting Line Dataset (Teklia/IAM-line)
│   ├── train.parquet            # 6,482 line images (62.49%)
│   ├── validation.parquet       # 976 line images (9.41%)
│   └── test.parquet             # 2,915 line images (28.10%)
├── whiteboard/                  # University of Peshawar Whiteboard Dataset
│   ├── data.yaml                # YOLOv8 format config file
│   ├── train/                   # 1,608 annotated classroom images
│   ├── valid/                   # 179 annotated classroom images
│   └── test/                    # 177 annotated classroom images
└── campus_test_set/             # Real-world Zero-Shot Classroom Evaluation Set
    ├── board-srmap.jpg          # SRM AP classroom whiteboard
    ├── test_user_wb.jpg         # High-glare real test whiteboard
    └── user_test_board.jpg      # Extreme perspective test whiteboard
```

---

## 2. Dataset Specifications & Splits

### A. Handwriting Recognition Dataset (IAM-line)
- **Source**: Hugging Face `Teklia/IAM-line` (Aachen Writer-Independent Protocol)
- **Total Samples**: 10,373 handwritten line sequences
- **Data Splits**:
  - **Train**: 6,482 lines (62.49%)
  - **Validation**: 976 lines (9.41%)
  - **Test**: 2,915 lines (28.10%)
- **Is 2,915 Test Lines Too Low?**: **No.** 2,915 lines comprise over 25,000 words and ~100,000 characters from writers never seen in the training set. In statistical ML/CV, this sample size ensures tight 95% confidence intervals on Character Error Rate (CER) and Word Error Rate (WER).
- **Auto-Download**:
  ```bash
  python scripts/download_datasets.py
  ```

### B. Whiteboard Region Detection Dataset
- **Source**: University of Peshawar Whiteboard Dataset (Roboflow Universe)
- **Total Images**: 1,964 fully annotated whiteboard images
- **Data Splits**:
  - **Train**: 1,608 images
  - **Valid**: 179 images
  - **Test**: 177 images
- **Why this over BIIT (1,012 images)?**:
  1. **Scale**: Nearly 2× larger sample size (1,964 vs 1,012 images).
  2. **YOLOv8 Standard Annotations**: Clean bounding box labels for whiteboard boundaries.
  3. **Environmental Variety**: Diverse camera angles, fluorescent light glare, marker fading, and partial occlusions matching real university lecture halls.
- **Download Command**:
  ```bash
  pip install roboflow
  python -c "from roboflow import Roboflow; rf=Roboflow(); rf.workspace('university-of-peshawar-4qet5').project('whiteboard-dataset').version(1).download('yolov8', location='datasets/whiteboard')"
  ```

### C. Campus Zero-Shot Test Set
- **Source**: Real university lecture hall photographs (SRM AP campus).
- **Purpose**: Zero-shot out-of-distribution evaluation for perspective rectification, Otsu binary mask segmentation, and end-to-end OCR accuracy.
- **Evaluation**:
  ```bash
  python src/evaluate_campus_set.py
  ```

---

## 3. Evaluation Scripts

| Metric | Target Dataset | Execution Command | Output |
| :--- | :--- | :--- | :--- |
| **CER & WER** | IAM Test Split (`test.parquet`) | `python src/evaluate_htr.py` | Character & Word Error Rate table |
| **Zero-Shot Transfer** | Campus Test Set (`campus_test_set/`) | `python src/evaluate_campus_set.py` | Rectification & OCR accuracy |
| **YOLOv8 mAP50** | Whiteboard Test Split (`whiteboard/`) | `notebooks/10_colab_yolov8_training.ipynb` | Bounding box precision, recall, mAP50 |
