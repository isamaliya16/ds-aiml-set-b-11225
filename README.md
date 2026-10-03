# 📊 Campaign Response Prediction — Set D

**Data Science & AI/ML Practical Exam** · End-to-end pipeline predicting customer response to a marketing campaign, with audience segmentation and a full model comparison (Baseline → Logistic Regression → ANN).

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

This project analyzes a synthetic marketing dataset (300 customer records) to answer two questions:

1. **Will a customer respond** to a marketing campaign? *(supervised classification)*
2. **What natural segments** exist within the customer base? *(unsupervised clustering)*

The notebook follows a strict **one-step-per-cell → output → interpretation** format, moving from raw data audit through statistics, preprocessing, modeling, clustering, and deep learning — with every step academically justified and leakage-checked.

## 🎯 Problem Statement

| Item | Detail |
|---|---|
| **Target variable** | `response` (1 = responded, 0 = did not respond) |
| **Predictors** | `visits`, `recency`, `engagement`, `spend`, `group` (G1/G2) |
| **Dataset size** | 300 unique records (305 raw rows, 5 exact duplicates removed) |
| **Class balance** | ~57.7% responders / 42.3% non-responders |
| **Data type** | Synthetic practice data (generated via a fixed-seed script) |

## 🗂 Project Structure

```
project-root/
├── notebooks/
│   └── exam.ipynb                # Main analysis notebook
├── src/
│   └── generate_data.py          # Synthetic data generator (auto-created)
├── data/
│   └── raw/
│       └── set_d.csv             # Raw generated dataset
├── models/
│   ├── ann_model.keras           # Trained ANN model
│   └── ann_test_predictions.csv
├── outputs/
│   ├── figures/                  # Saved charts (histograms, confusion matrices, curves)
│   ├── splits.csv                # Record-level train/val/test assignment
│   ├── engagement_summary.csv
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
| **Task 1 — Statistics** | Descriptive & inferential stats | Mean/median/std of `engagement`; Welch's t-test (G1 vs G2) with 95% CI; covariance matrix & eigen-decomposition of `engagement`/`visits` |
| **Task 2 — Preprocessing** | Feature engineering | Median imputation, one-hot encoding of `group`, engineered feature (`engagement / (recency + 1)`), standard scaling — all fitted on **fit data only** |
| **Task 3 — Supervised Learning** | Classification | `DummyClassifier` baseline vs. `LogisticRegression`; accuracy, precision, recall, F1; confusion matrix |
| **Task 4 — Unsupervised Learning** | Segmentation | K-Means at k = 2, 3, 4; silhouette-based model selection; cluster profiling |
| **Task 5 — Deep Learning** | Neural network | ANN (7 → 16 → 8 → 1, ReLU/sigmoid) with early stopping on validation loss; final test evaluation |

## 🔐 Data Pipeline & Leakage Control

This project enforces a strict **no-leakage discipline** throughout:

- ✅ Duplicates are removed **before** splitting.
- ✅ All preprocessing objects (imputer, encoder, scaler) are **fit only on the 192 fit records**, then applied unchanged to validation and test.
- ✅ The engineered feature uses no target information.
- ✅ `record_id` and `response` are excluded from model inputs.
- ✅ Fit / validation / test record IDs are verified disjoint and saved to `outputs/splits.csv`.

| Split | Records | Response rate |
|---|---|---|
| Fit | 192 | 57.3% |
| Validation | 48 | 58.3% |
| Test | 60 | 58.3% |

## 📈 Results Summary

**Test set performance (60 records, threshold = 0.5):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (majority class) | 0.583 | 0.583 | 1.000 | 0.737 |
| Logistic Regression | 0.817 | 0.833 | 0.857 | 0.845 |
| ANN (7 → 16 → 8 → 1) | 0.833 | 0.838 | 0.886 | 0.861 |

**Clustering (K-Means):**

| k | Inertia | Silhouette |
|---|---|---|
| 2 | 713 | **0.226** ✅ chosen |
| 3 | 614 | 0.187 |
| 4 | 542 | 0.191 |

| Cluster | Size | Profile | Suggested Name | Action |
|---|---|---|---|---|
| 0 | 109 | High recency, low engagement | Cooling / Low-engagement | Low-cost win-back campaign |
| 1 | 83 | Low recency, high engagement | Active & Engaged | Loyalty rewards / early access |

## 💡 Key Findings

- **Logistic regression beats the naive baseline** by +0.233 accuracy and +0.108 F1 — a clear, meaningful improvement.
- **The ANN performs marginally better than logistic regression** (+0.016 F1 ≈ one test record), which is within the margin of chance on only 60 test records.
- **Recommended model: Logistic Regression** — comparable performance, only 8 parameters (vs. 273 for the ANN), faster, more stable, and fully interpretable for a marketing audience.
- **Customer segments are weakly separated** (silhouette ≈ 0.226), so clusters are useful as rough groupings rather than hard boundaries.
- **Engagement is not significantly different between groups G1 and G2** (Welch's t-test, p = 0.47).

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib tensorflow joblib
```

### Run the notebook

```bash
jupyter notebook notebooks/exam_set_d.ipynb
```

The notebook auto-generates the dataset on first run (`src/generate_data.py`) and creates the `data/`, `outputs/`, and `models/` folders automatically — no manual setup required.

## 📦 Outputs Generated

Running the full notebook produces:

- 📉 **Figures** — engagement histogram, ANN learning curves, confusion matrices (Logistic Regression & ANN)
- 📄 **CSV reports** — statistical summaries, preprocessing audit, per-record predictions, cluster profiles, k-selection scores, model comparison


## ♻️ Reproducibility

- All random operations use a fixed **`SEED = 42`**, applied consistently to NumPy, TensorFlow, and scikit-learn.
- TensorFlow determinism is explicitly enabled (`enable_op_determinism()`).
- All stratified splits, model fits, and cluster assignments are fully reproducible from a clean run.

## 👤 Author

**Student:** Ayush Isamaliya · **Student ID:** 11225 · **Exam Set:** D — Campaign Response

---
