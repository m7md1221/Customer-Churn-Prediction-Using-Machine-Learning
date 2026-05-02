# Customer Churn Prediction Using Machine Learning

A machine learning pipeline that predicts customer churn on the **Telco Customer Churn** dataset using a **Decision Tree** and a **Support Vector Machine (SVM)** classifier.

---

## Project Structure

```
Customer Churn Prediction Using Machine Learning/
├── churn_ml_modeling.py                  # Main ML pipeline script
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset (Kaggle)
├── 1_correlation_matrix.png              # Feature correlation heatmap
├── 2_feature_scores.png                  # SelectKBest F-score bar chart
├── 3a_dt_feature_importance.png          # Decision Tree feature importances
├── 3b_decision_tree_structure.png        # Decision Tree visual (first 3 levels)
├── 4_cm_decision_tree.png                # Decision Tree confusion matrix
├── 4_cm_svm.png                          # SVM confusion matrix
├── 5_model_comparison.png                # Grouped bar chart — all metrics
└── 6_model_radar.png                     # Radar chart — model comparison
```

---

## Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Rows:** 7,043 (7,032 after dropping nulls)
- **Target column:** `Churn` → encoded as `0` (No) / `1` (Yes)
- **Class distribution:** ~73% No Churn · ~27% Churn

---

## Pipeline Overview

### 1. Data Loading & Preprocessing
- Load raw CSV with `pandas`
- Convert `TotalCharges` to numeric (blank entries coerced to NaN and dropped)
- Drop non-informative `customerID` column
- Encode `Churn` as binary (0 / 1)
- One-hot encode all remaining categorical columns (`pd.get_dummies`)

### 2. Feature Analysis
- Compute and visualise the **correlation matrix**
- Flag highly correlated feature pairs (`|r| > 0.75`)
- Run **SelectKBest** (ANOVA F-test) to rank all features
- Select the **top 10 features** for modeling

**Top 10 selected features:**

| Rank | Feature |
|------|---------|
| 1 | tenure |
| 2 | InternetService_Fiber optic |
| 3 | Contract_Two year |
| 4 | PaymentMethod_Electronic check |
| 5 | InternetService_No |
| 6 | OnlineSecurity_No internet service |
| 7 | DeviceProtection_No internet service |
| 8 | TechSupport_No internet service |
| 9 | StreamingMovies_No internet service |
| 10 | StreamingTV_No internet service |

### 3. Train-Test Split
- Split: **70% train / 30% test**
- Stratified by target to preserve class ratio
- Training samples: 4,922 · Testing samples: 2,110

### 4. Models

**Decision Tree**
- `max_depth = 5`
- `min_samples_leaf = 15`
- `criterion = gini`

**Support Vector Machine**
- `kernel = rbf`
- `C = 1.0`
- `gamma = scale`

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Decision Tree** | **78.67%** | **0.6242** | **0.4973** | **0.5536** |
| SVM (RBF) | 77.11% | 0.6089 | 0.3886 | 0.4744 |

> Decision Tree outperformed SVM on all metrics for this dataset and feature set.

---

## Requirements

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and preprocessing |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | Feature selection, models, evaluation |

---

## How to Run

Open PowerShell, then run:

```powershell
$env:PYTHONUTF8=1; Set-Location "C:\Users\dell\Downloads\Customer Churn Prediction Using Machine Learning"; py churn_ml_modeling.py
```

Or if you are already inside the project folder:

```powershell
$env:PYTHONUTF8=1; py churn_ml_modeling.py
```

> `PYTHONUTF8=1` is required on Windows to correctly render box-drawing characters in the console output.

---

## Output Files

| File | Description |
|---|---|
| `1_correlation_matrix.png` | Heatmap of feature correlations |
| `2_feature_scores.png` | ANOVA F-score bar chart for all features |
| `3a_dt_feature_importance.png` | Gini importance per feature (Decision Tree) |
| `3b_decision_tree_structure.png` | Tree diagram (first 3 levels) |
| `4_cm_decision_tree.png` | Confusion matrix — Decision Tree |
| `4_cm_svm.png` | Confusion matrix — SVM |
| `5_model_comparison.png` | Side-by-side metric comparison bar chart |
| `6_model_radar.png` | Radar chart comparing both models |

---

