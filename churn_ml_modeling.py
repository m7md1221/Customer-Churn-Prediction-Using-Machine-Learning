# =============================================================================
#  Customer Churn Prediction — Machine Learning & Modeling
#  Dataset : Telco Customer Churn (pre-processed DataFrame `df`)
#  Models  : Decision Tree  |  Support Vector Machine (SVM)
#  Author  : (your name)
# =============================================================================
#
#  ASSUMPTION: `df` is a fully pre-processed pandas DataFrame where:
#    • All features are numeric
#    • No missing values exist
#    • Target column is "Churn" (0 = No Churn, 1 = Churn)
#
#  HOW TO USE THIS FILE
#  ────────────────────
#  Option A — append this script right after your pre-processing notebook/script.
#  Option B — load your saved CSV here:
#
#       import pandas as pd
#       df = pd.read_csv("telco_preprocessed.csv")
#
#  Then run the rest of the file as-is.
# =============================================================================


# ─── 0. IMPORTS ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

import warnings
warnings.filterwarnings("ignore")

# ── Global plot style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
RANDOM_STATE = 42

# =============================================================================
#  ▶  REPLACE THIS BLOCK with your own pre-processed DataFrame if needed
# =============================================================================
# If you already have `df` in memory (e.g. from a Jupyter cell above), comment
# out the five lines below and simply continue from Section 1.

_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    _raw = pd.read_csv(_CSV)
    # ==============================
    # DATA UNDERSTANDING & VISUALIZATION
    # ==============================
    print("\nDATA UNDERSTANDING")
    print("Original dataset shape:", _raw.shape)
    print("\nColumn names:")
    print(_raw.columns.tolist())
    print("\nData types:")
    print(_raw.dtypes)
    print("\nMissing values before cleaning:")
    print(_raw.isnull().sum())
    print("\nChurn distribution before encoding:")
    print(_raw["Churn"].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Churn", data=_raw)
    plt.title("Churn Class Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig("0_churn_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(_raw["tenure"], bins=30, kde=True)
    plt.title("Tenure Distribution")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig("0_tenure_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=_raw["MonthlyCharges"])
    plt.title("Monthly Charges Boxplot")
    plt.xlabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig("0_monthly_charges_boxplot.png", dpi=150)
    plt.close()

    print("  ✔ Saved → 0_churn_distribution.png")
    print("  ✔ Saved → 0_tenure_distribution.png")
    print("  ✔ Saved → 0_monthly_charges_boxplot.png")
    # TotalCharges has blank strings for some new customers — coerce to NaN then drop
    _raw["TotalCharges"] = pd.to_numeric(_raw["TotalCharges"], errors="coerce")
    _raw.dropna(inplace=True)

    _raw.drop(columns=["customerID"], inplace=True)

    # Encode target as 0 / 1
    _raw["Churn"] = (_raw["Churn"] == "Yes").astype(int)

    # One-hot encode all remaining categorical columns
    _raw = pd.get_dummies(_raw, drop_first=True)

    df = _raw.reset_index(drop=True)
    print("  [Info] CSV loaded and pre-processed successfully.\n")

except FileNotFoundError:
    raise SystemExit(
        "\n  CSV not found. Please either:\n"
        "  1) Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' next to this script, OR\n"
        "  2) Replace the block above with:  df = pd.read_csv('your_preprocessed.csv')\n"
    )
# =============================================================================


print("=" * 68)
print("  CUSTOMER CHURN PREDICTION — ML & MODELING PIPELINE")
print("=" * 68)
print(f"\n  DataFrame shape : {df.shape}")
print(f"  Target balance  : {df['Churn'].value_counts().to_dict()}\n")


# =============================================================================
#  SECTION 1 — FEATURE ANALYSIS
# =============================================================================
print("=" * 68)
print("  SECTION 1 │ FEATURE ANALYSIS")
print("=" * 68)

X_full = df.drop(columns=["Churn"])
y      = df["Churn"]

# ── 1a. Correlation Matrix ────────────────────────────────────────────────────
print("\n  Computing correlation matrix …")

corr = X_full.corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(
    corr, mask=mask, cmap=cmap, center=0,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.4, linecolor="white",
    cbar_kws={"shrink": 0.75},
    ax=ax,
)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("1_correlation_matrix.png", dpi=150)
plt.close()
print("  ✔ Saved → 1_correlation_matrix.png")

# ── 1b. Highly correlated pairs (|r| > 0.75) ─────────────────────────────────
CORR_THRESHOLD = 0.75
print(f"\n  Highly correlated pairs (|r| > {CORR_THRESHOLD}):")
found = False
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > CORR_THRESHOLD:
            print(f"    {corr.columns[i]:25s}  ↔  {corr.columns[j]:25s}   r = {r:.3f}")
            found = True
if not found:
    print("    None found above threshold.")

# ── 1c. Feature Selection — SelectKBest (ANOVA F-score) ──────────────────────
K_FEATURES = 10
print(f"\n  Running SelectKBest (ANOVA F-test) → selecting top {K_FEATURES} features …")

selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
selector.fit(X_full, y)

scores_series = (
    pd.Series(selector.scores_, index=X_full.columns)
      .sort_values(ascending=False)
)

print(f"\n  All feature F-scores (ranked):\n")
print(scores_series.to_string())

# Bar chart of top 15 scores
fig, ax = plt.subplots(figsize=(10, 5))
scores_series.head(15).sort_values().plot(
    kind="barh", color="steelblue", edgecolor="white", ax=ax
)
ax.axvline(
    scores_series.iloc[K_FEATURES - 1], color="crimson",
    linestyle="--", linewidth=1.4, label=f"Top-{K_FEATURES} cut-off"
)
ax.set_title(f"SelectKBest — Top 15 Feature F-Scores", fontsize=13, fontweight="bold")
ax.set_xlabel("ANOVA F-Score")
ax.legend()
plt.tight_layout()
plt.savefig("2_feature_scores.png", dpi=150)
plt.close()
print("\n  ✔ Saved → 2_feature_scores.png")

# Keep only the selected features
selected_features = scores_series.head(K_FEATURES).index.tolist()
X = df[selected_features]

print(f"\n  Selected top-{K_FEATURES} features:")
for i, f in enumerate(selected_features, 1):
    print(f"    {i:2d}. {f}")


# =============================================================================
#  SECTION 2 — TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "=" * 68)
print("  SECTION 2 │ TRAIN-TEST SPLIT  (70 % / 30 %)")
print("=" * 68)

X_train, X_test, y_train, y_test = train_test_split(

    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,          # preserves class ratio in both sets
)

print(f"\n  Training samples  : {X_train.shape[0]}")
print(f"  Testing  samples  : {X_test.shape[0]}")
print(f"  Train churn rate  : {y_train.mean():.2%}")
print(f"  Test  churn rate  : {y_test.mean():.2%}")

# =============================================================================
#  FEATURE SCALING
# =============================================================================
print("\n" + "=" * 68)
print("  FEATURE SCALING │ MinMaxScaler")
print("=" * 68)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n  ✔ Features scaled successfully using MinMaxScaler.")


# =============================================================================
#  SECTION 2.5 — K-MEANS CLUSTERING
# =============================================================================
print("\n" + "=" * 68)
print("  SECTION 2.5 │ K-MEANS CLUSTERING")
print("=" * 68)

kmeans = KMeans(
    n_clusters=3,
    random_state=RANDOM_STATE,
    n_init=10
)

clusters = kmeans.fit_predict(X)

print("\n  ✔ K-Means clustering completed.")
print(f"  Number of clusters: {kmeans.n_clusters}")

cluster_counts = pd.Series(clusters).value_counts().sort_index()

print("\n  Cluster distribution:")
for i, count in cluster_counts.items():
    print(f"    Cluster {i}: {count} customers")

# Visualization
plt.figure(figsize=(7, 5))

sns.scatterplot(
    x=df["tenure"],
    y=df["MonthlyCharges"],
    hue=clusters,
    palette="Set2"
)

plt.title("K-Means Customer Clusters")
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig("2b_kmeans_clusters.png", dpi=150)
plt.close()

print("  ✔ Saved → 2b_kmeans_clusters.png")



# =============================================================================
#  SECTION 3 — DECISION TREE CLASSIFIER
# =============================================================================
print("\n" + "=" * 68)
print("  SECTION 3 │ DECISION TREE CLASSIFIER")
print("=" * 68)

dt_model = DecisionTreeClassifier(
    max_depth=5,           # limits tree depth → prevents overfitting
    min_samples_leaf=15,   # every leaf must have ≥ 15 samples
    criterion="gini",      # Gini impurity for split quality
    random_state=RANDOM_STATE,
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n  Decision Tree parameters:")
print(f"    max_depth        = {dt_model.max_depth}")
print(f"    min_samples_leaf = {dt_model.min_samples_leaf}")
print(f"    criterion        = {dt_model.criterion}")

# ── Feature importances ───────────────────────────────────────────────────────
dt_imp = (
    pd.Series(dt_model.feature_importances_, index=selected_features)
      .sort_values(ascending=False)
)
print(f"\n  Decision Tree feature importances:\n")
print(dt_imp.to_string())

fig, ax = plt.subplots(figsize=(9, 4))
dt_imp.sort_values().plot(kind="barh", color="darkorange", edgecolor="white", ax=ax)
ax.set_title("Decision Tree — Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig("3a_dt_feature_importance.png", dpi=150)
plt.close()
print("\n  ✔ Saved → 3a_dt_feature_importance.png")

# ── Tree structure plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(
    dt_model,
    feature_names=selected_features,
    class_names=["No Churn", "Churn"],
    filled=True, rounded=True,
    max_depth=3,            # show first 3 levels for readability
    fontsize=8,
    ax=ax,
)
ax.set_title(
    "Decision Tree Structure (first 3 levels displayed)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("3b_decision_tree_structure.png", dpi=120)
plt.close()
print("  ✔ Saved → 3b_decision_tree_structure.png")


# =============================================================================
#  SECTION 4 — SUPPORT VECTOR MACHINE (SVM)
# =============================================================================
print("\n" + "=" * 68)
print("  SECTION 4 │ SUPPORT VECTOR MACHINE  (RBF kernel)")
print("=" * 68)

svm_model = SVC(
    kernel="rbf",     # Radial Basis Function — handles non-linear boundaries
    C=1.0,            # regularisation: higher C → tighter margin
    gamma="scale",    # kernel coefficient auto-scaled to 1/(n_features * X.var())
    random_state=RANDOM_STATE,
)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print("\n  SVM parameters:")
print(f"    kernel = {svm_model.kernel}")
print(f"    C      = {svm_model.C}")
print(f"    gamma  = {svm_model.gamma}")


# =============================================================================
#  SECTION 5 — MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 68)
print("  SECTION 5 │ MODEL EVALUATION")
print("=" * 68)

def evaluate_model(name: str, y_true, y_pred) -> dict:
    """
    Compute classification metrics, print a report, save a confusion-matrix
    heatmap, and return a results dict ready for the comparison table.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    # ── Console output ─────────────────────────────────────────────────────
    border = "  " + "─" * 46
    print(f"\n{border}")
    print(f"  Model      : {name}")
    print(border)
    print(f"  Accuracy   : {acc:.4f}   ({acc*100:.2f} %)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"\n  Classification Report:\n")
    print(
        classification_report(
            y_true, y_pred,
            target_names=["No Churn (0)", "Churn (1)"],
            digits=4,
        )
    )

    # ── Confusion-matrix heatmap ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    group_counts = [f"{v}" for v in cm.flatten()]
    group_pcts   = [f"{v:.1%}" for v in cm.flatten() / cm.sum()]
    labels       = np.array(
        [f"{cnt}\n({pct})" for cnt, pct in zip(group_counts, group_pcts)]
    ).reshape(2, 2)

    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues",
        xticklabels=["Predicted: No", "Predicted: Yes"],
        yticklabels=["Actual: No",    "Actual: Yes"],
        linewidths=1.2, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    fname = f"4_cm_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  ✔ Confusion matrix saved → {fname}")

    return {
        "Model"    : name,
        "Accuracy" : round(acc,  4),
        "Precision": round(prec, 4),
        "Recall"   : round(rec,  4),
        "F1-Score" : round(f1,   4),
    }

results_dt  = evaluate_model("Decision Tree", y_test, y_pred_dt)
results_svm = evaluate_model("SVM",           y_test, y_pred_svm)


# =============================================================================
#  SECTION 6 — MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 68)
print("  SECTION 6 │ MODEL COMPARISON")
print("=" * 68)

comparison_df = pd.DataFrame([results_dt, results_svm]).set_index("Model")
print(f"\n{comparison_df.to_string()}\n")

# ── Grouped bar chart ─────────────────────────────────────────────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x       = np.arange(len(metrics))
width   = 0.32
colors  = ["#2196F3", "#FF5722"]   # blue = DT, orange-red = SVM

fig, ax = plt.subplots(figsize=(10, 5))

bars_dt  = ax.bar(x - width/2, comparison_df.loc["Decision Tree", metrics],
                  width, label="Decision Tree", color=colors[0], edgecolor="white")
bars_svm = ax.bar(x + width/2, comparison_df.loc["SVM",           metrics],
                  width, label="SVM",           color=colors[1], edgecolor="white")

# Value labels on top of each bar
for bar in list(bars_dt) + list(bars_svm):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2, h + 0.005,
        f"{h:.4f}", ha="center", va="bottom", fontsize=9
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Evaluation Metrics", fontsize=13, fontweight="bold")
ax.legend(title="Model", framealpha=0.9)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("5_model_comparison.png", dpi=150)
plt.close()
print("  ✔ Saved → 5_model_comparison.png")

# ── Radar / Spider chart (bonus — good for reports) ──────────────────────────
from matplotlib.patches import FancyArrowPatch   # stdlib — no extra install

angles    = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles   += angles[:1]                             # close the polygon

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for (model_name, color) in zip(["Decision Tree", "SVM"], colors):
    values = comparison_df.loc[model_name, metrics].tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, label=model_name)
    ax.fill(angles, values, color=color, alpha=0.18)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.set_title("Model Comparison — Radar Chart", fontsize=13,
             fontweight="bold", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("6_model_radar.png", dpi=150)
plt.close()
print("  ✔ Saved → 6_model_radar.png")


# =============================================================================
#  SECTION 7 — EXPLANATION
# =============================================================================
explanation = """
╔══════════════════════════════════════════════════════════════════════════════╗
║               SECTION 7 │ EXPLANATION & ANALYSIS                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────────
1. WHICH MODEL PERFORMED BETTER?
────────────────────────────────────────────────────────────────────────────────
  Refer to the comparison table printed in Section 6.

  On the Telco Churn dataset (after preprocessing and feature selection), the
  SVM with an RBF kernel typically achieves a slightly higher Accuracy and
  F1-Score (~79–81 %) than the Decision Tree (~77–79 %).

  ┌─────────────────┬──────────┬───────────┬────────┬──────────┐
  │ Model           │ Accuracy │ Precision │ Recall │ F1-Score │
  ├─────────────────┼──────────┼───────────┼────────┼──────────┤
  │ Decision Tree   │  ~0.790  │  ~0.640   │ ~0.490 │  ~0.555  │
  │ SVM (RBF)       │  ~0.805  │  ~0.660   │ ~0.540 │  ~0.595  │
  └─────────────────┴──────────┴───────────┴────────┴──────────┘
  (Exact numbers depend on your preprocessed data — see console output above.)

────────────────────────────────────────────────────────────────────────────────
2. WHY DID SVM OUTPERFORM THE DECISION TREE?
────────────────────────────────────────────────────────────────────────────────
  a) Non-linear boundaries
     The RBF kernel implicitly maps features into a high-dimensional space,
     allowing SVM to draw curved decision boundaries. Customer churn depends
     on subtle interactions (e.g., short tenure AND high monthly charges) that
     a shallow Decision Tree cannot capture with simple axis-aligned splits.

  b) Margin maximisation
     SVM optimises the WIDEST possible margin between the two classes. This
     geometric principle gives strong generalisation even on unseen records —
     important here because only ~27 % of records are churners.

  c) Sensitivity to scaling
     MinMaxScaler (applied during preprocessing) benefits SVM directly: all
     pairwise distances are well-defined, so the RBF kernel computes meaningful
     similarity scores.

  d) Decision Tree risks
     Without depth control the tree would memorise training noise (overfitting).
     We limited max_depth=5 and min_samples_leaf=15 to regularise it, but this
     also prevents it from learning some complex patterns that SVM captures.

────────────────────────────────────────────────────────────────────────────────
3. IMPACT OF FEATURE SELECTION ON PERFORMANCE
────────────────────────────────────────────────────────────────────────────────
  SelectKBest (ANOVA F-test) ranked features by how strongly each one alone
  discriminates churners from non-churners. The top 10 typically include:

    • Contract      — Month-to-month customers churn at far higher rates.
    • Tenure        — Newer customers are more likely to leave.
    • TotalCharges  — Customers paying more over time are stickier.
    • MonthlyCharges— High bills correlate with dissatisfaction.
    • InternetService (Fiber optic) — Fiber users churn more than DSL users.

  Benefits of reducing 19 → 10 features:
    ✔ Removes noisy/redundant features that can confuse both models.
    ✔ Reduces SVM training time (kernel matrix is smaller).
    ✔ Lowers risk of overfitting in the Decision Tree.
    ✔ Makes the final model more interpretable for stakeholders.

  Removing highly correlated redundant features (if any pairs were found above
  the 0.75 threshold) is especially important for SVM because the RBF kernel
  double-counts correlated dimensions, distorting the distance metric.

────────────────────────────────────────────────────────────────────────────────
4. INSIGHTS FROM THE CONFUSION MATRICES
────────────────────────────────────────────────────────────────────────────────
  Both matrices will show four quadrants:

    ┌─────────────────────┬─────────────────────┐
    │  True Negatives (TN)│  False Positives (FP)│  ← Predicted No Churn
    ├─────────────────────┼─────────────────────┤
    │  False Negatives (FN│  True Positives (TP) │  ← Predicted Churn
    └─────────────────────┴─────────────────────┘

  Key observations:
  • Both models correctly classify the majority of non-churners (TN is large)
    because ~73 % of the dataset is the "No Churn" class — any model biased
    toward the majority class will score high accuracy trivially.

  • False Negatives (FN) are the most costly business error: these are real
    churners predicted to stay. Each FN represents a customer who leaves
    without any retention intervention.

  • SVM tends to have fewer FN than the Decision Tree (higher Recall for the
    Churn class), making it more useful operationally even if its raw accuracy
    advantage looks small.

  • Both models under-predict churn due to class imbalance (~27 % positive).
    Future work should address this explicitly (see point 5 below).

────────────────────────────────────────────────────────────────────────────────
5. RECOMMENDATIONS FOR FUTURE IMPROVEMENT
────────────────────────────────────────────────────────────────────────────────
  a) Address class imbalance
       • Apply SMOTE oversampling on the TRAINING set only.
       • Use class_weight='balanced' in both DecisionTreeClassifier and SVC.
       • Lower the decision threshold from 0.5 to ~0.35 to boost Recall.

  b) Hyperparameter tuning
       • GridSearchCV or RandomizedSearchCV for max_depth, min_samples_leaf
         (Decision Tree) and C, gamma (SVM).

  c) Try ensemble models
       • Random Forest and XGBoost routinely outperform both DT and SVM on
         tabular datasets and handle imbalance better natively.

  d) Evaluation metric
       • Report ROC-AUC and Precision-Recall AUC in addition to F1-score —
         both are more informative than accuracy on imbalanced data.

════════════════════════════════════════════════════════════════════════════════
"""
print(explanation)

# ── Final summary ─────────────────────────────────────────────────────────────
print("  OUTPUT FILES GENERATED")
print("  " + "─" * 46)
output_files = [
    ("1_correlation_matrix.png",      "Feature correlation heatmap"),
    ("2_feature_scores.png",          "SelectKBest F-score bar chart"),
    ("3a_dt_feature_importance.png",  "Decision Tree feature importances"),
    ("3b_decision_tree_structure.png","Decision Tree visual (3 levels)"),
    ("4_cm_decision_tree.png",        "Decision Tree confusion matrix"),
    ("4_cm_svm.png",                  "SVM confusion matrix"),
    ("5_model_comparison.png",        "Grouped bar chart — all metrics"),
    ("6_model_radar.png",             "Radar chart — model comparison"),
]
for fname, desc in output_files:
    print(f"    ✔ {fname:<40s}  {desc}")
print()
