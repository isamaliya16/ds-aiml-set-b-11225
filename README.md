# 🚚 Delivery Risk Prediction — Set B

**Data Science & AI/ML Practical Exam** · End-to-end pipeline predicting whether a delivery will be *late*, plus operational-route segmentation, with a full model comparison (Baseline → Logistic Regression → ANN).

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-red)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Data](https://img.shields.io/badge/data-synthetic-lightgrey)

---

## 🎬 Video Walkthrough

<div align="center">

[![Watch the Project Walkthrough](https://img.shields.io/badge/▶%20Watch%20Full%20Walkthrough-Google%20Drive-4285F4?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1CZlv6WK_BTNo6FV4MVIxxmi502dkfA2v/view?usp=sharing)

> 🎥 **A complete end-to-end video explanation** of this Data Science & AI/ML project — covering the raw data audit and de-duplication, the stratified fit/validation/test split, descriptive & inferential statistics on delivery distance, leakage-safe preprocessing and feature engineering, the baseline vs. logistic regression comparison for predicting late deliveries, K-Means route segmentation, and the final ANN build with early stopping.

> 📌 *Click the button above or [open the video directly →](https://drive.google.com/file/d/1CZlv6WK_BTNo6FV4MVIxxmi502dkfA2v/view?usp=sharing)*

</div>

---

## 📑 Table of Contents

- [Video Walkthrough](#-video-walkthrough)
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Data Pipeline & Leakage Control](#-data-pipeline--leakage-control)
- [Results Summary](#-results-summary)
- [Key Findings](#-key-findings)
- [Getting Started](#-getting-started)
- [Outputs Generated](#-outputs-generated)
- [Reproducibility](#-reproducibility)
- [Author](#-author)

---

## 🔍 Overview

This project analyzes a synthetic logistics dataset (300 delivery records) to answer two operational questions:

1. **Will a delivery be late?** *(supervised classification)*
2. **What natural route segments** exist across distance, load, traffic, and staffing? *(unsupervised clustering)*

The notebook follows a strict **one-step-per-cell → output → interpretation** format, moving from raw data audit through statistics, preprocessing, modeling, clustering, and deep learning — with every step academically justified and leakage-checked.

## 🎯 Problem Statement

| Item | Detail |
|---|---|
| **Target variable** | `late` (1 = late delivery, 0 = on time) |
| **Predictors** | `distance`, `load`, `traffic`, `staff`, `group` (G1/G2) |
| **Dataset size** | 300 unique records (305 raw rows, 5 exact duplicates removed) |
| **Class balance** | ~51.7% on time / 48.3% late |
| **Data type** | Synthetic practice data (generated via a fixed-seed script) |

## 🗂 Project Structure

```
project-root/
├── notebooks/
│   └── exam_set_b.ipynb          # Main analysis notebook
├── src/
│   └── generate_data.py          # Synthetic data generator (auto-created)
├── data/
│   └── raw/
│       └── set_b.csv             # Raw generated dataset
├── models/   
│   ├── ann_model.keras           # Trained ANN model
│   └── ann_test_predictions.csv
├── outputs/
│   ├── figures/                  # Saved charts (histograms, confusion matrices, curves)
│   ├── splits.csv                # Record-level train/val/test assignment
│   ├── distance_summary.csv
│   ├── inference_results.csv
│   ├── preprocessing_audit.csv
│   ├── logreg_test_predictions.csv
│   ├── ann_test_predictions.csv
│   ├── k_selection_scores.csv
│   ├── cluster_profiles.csv
│   └── metrics_comparison.csv
└── README.md
```

## 🧭 Methodology

The notebook is organized into a **Setup phase** plus **five tasks**:

| Stage | Focus | Key Steps |
|---|---|---|
| **Setup** | Data audit & split | Load raw data → drop duplicates → stratified 80/20 train-pool/test split → 80/20 fit/validation split (all seeded, `random_state=42`) |
| **Task 1 — Statistics** | Descriptive & inferential stats | Mean/median/std of `distance`; Welch's t-test (G1 vs G2) with 95% CI; covariance matrix & eigen-decomposition of `distance`/`traffic` |
| **Task 2 — Preprocessing** | Feature engineering | Median imputation, one-hot encoding of `group`, engineered feature (`load / (staff + 1)`), standard scaling — all fitted on **fit data only** |
| **Task 3 — Supervised Learning** | Classification | `DummyClassifier` baseline vs. `LogisticRegression`; accuracy, precision, recall, F1; confusion matrix; cost-of-errors discussion |
| **Task 4 — Unsupervised Learning** | Segmentation | K-Means at k = 2, 3, 4; silhouette-based model selection; cluster profiling |
| **Task 5 — Deep Learning** | Neural network | ANN (7 → 16 → 8 → 1, ReLU/sigmoid) with early stopping on validation loss; final test evaluation |

## 🔐 Data Pipeline & Leakage Control

This project enforces a strict **no-leakage discipline** throughout:

- ✅ Duplicates are removed **before** splitting.
- ✅ All preprocessing objects (imputer, encoder, scaler) are **fit only on the 192 fit records**, then applied unchanged to validation and test.
- ✅ The engineered feature uses no target information.
- ✅ `record_id` and `late` are excluded from model inputs.
- ✅ Fit / validation / test record IDs are verified disjoint and saved to `outputs/splits.csv`.

| Split | Records | Late rate |
|---|---|---|
| Fit | 192 | — |
| Validation | 48 | — |
| Test | 60 | 48.3% |

## 📈 Results Summary

**Test set performance (60 records, threshold = 0.5):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (majority class) | 0.517 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.850 | 0.885 | 0.793 | 0.836 |
| ANN (7 → 16 → 8 → 1) | 0.833 | 0.852 | 0.793 | 0.821 |

**Confusion matrix — Logistic Regression:** TN 28 · FP 3 · FN 6 · TP 23

**Clustering (K-Means):**

| k | Silhouette |
|---|---|
| 2 | **0.246** ✅ chosen |
| 3 | 0.207 |
| 4 | 0.190 |

| Cluster | Size | Profile | Suggested Name | Action |
|---|---|---|---|---|
| 0 | 134 | Lighter load (47.1), more staff (54.0) | Well-staffed, light-load routes | Keep current staffing, routine monitoring |
| 1 | 58 | Heavier load (57.5), less staff (41.4) | High-load, understaffed routes | Add staff or rebalance loads |

## 💡 Key Findings

- **Logistic regression beats the naive baseline** by +0.333 accuracy and +0.836 F1 — the majority-class baseline never predicts a late delivery, so it is useless for this problem.
- **The ANN performs marginally worse than logistic regression** (−0.015 F1, equal to one test record: 4 false positives instead of 3), a gap within the margin of chance on only 60 test records.
- **Recommended model: Logistic Regression** — at least as accurate, with only 8 parameters (vs. 273 for the ANN), faster to train, more stable, and fully interpretable for operations staff.
- **False negatives (predicted on-time, actually late) are the costlier error** — they remove any chance to intervene and risk missed delivery promises, unlike false positives (unnecessary expediting).
- **Route segments are weakly separated** (silhouette ≈ 0.246), differing mainly in `load` and `staff` — useful as rough operational groupings rather than hard boundaries.
- **Distance is not significantly different between groups G1 and G2** (Welch's t-test, p = 0.481).

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook notebooks/exam_set_b.ipynb
```

The notebook auto-generates the dataset on first run (`src/generate_data.py`) and creates the `data/`, `outputs/`, and `models/` folders automatically — no manual setup required.

## 📦 Outputs Generated

Running the full notebook produces:

- 📉 **Figures** — distance histogram, ANN learning curves, confusion matrices (Logistic Regression & ANN)
- 📄 **CSV reports** — statistical summaries, preprocessing audit, per-record predictions, cluster profiles, k-selection scores, model comparison
- 🧠 **Saved model artifacts** — `ann_model.keras`

## ♻️ Reproducibility

- All random operations use a fixed **`SEED = 42`**, applied consistently to NumPy, TensorFlow, and scikit-learn.
- TensorFlow determinism is explicitly enabled (`enable_op_determinism()`).
- All stratified splits, model fits, and cluster assignments are fully reproducible from a clean run.

## 👤 Author

**Student:** Ayush Isamaliya  · **Student ID:** 11225 · **Exam Set:** B — Delivery Risk

---

*All data used in this project is synthetic and generated for academic/practice purposes only.*
