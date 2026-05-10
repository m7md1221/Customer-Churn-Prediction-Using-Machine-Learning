# Customer Churn Prediction — Machine Learning & Modeling
# Dataset: Telco Customer Churn
# Models: Decision Tree | SVM
#
# Usage: Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' next to this script,
#        or replace the load block below with: df = pd.read_csv('your_preprocessed.csv')

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

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
RANDOM_STATE = 42


# --- Load & Preprocess Data ---

_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    _raw = pd.read_csv(_CSV)

    print("\nDATA UNDERSTANDING")
    print("Shape:", _raw.shape)
    print("\nColumns:", _raw.columns.tolist())
    print("\nData types:\n", _raw.dtypes)
    print("\nMissing values:\n", _raw.isnull().sum())
    print("\nChurn distribution:\n", _raw["Churn"].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Churn", data=_raw)
    plt.title("Churn Class Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("0_churn_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(_raw["tenure"], bins=30, kde=True)
    plt.title("Tenure Distribution")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Count")
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

    print("Saved: 0_churn_distribution.png, 0_tenure_distribution.png, 0_monthly_charges_boxplot.png")

    # TotalCharges has blank strings for new customers — coerce to NaN then drop
    _raw["TotalCharges"] = pd.to_numeric(_raw["TotalCharges"], errors="coerce")
    _raw.dropna(inplace=True)
    _raw.drop(columns=["customerID"], inplace=True)
    _raw["Churn"] = (_raw["Churn"] == "Yes").astype(int)
    _raw = pd.get_dummies(_raw, drop_first=True)

    df = _raw.reset_index(drop=True)
    print("Preprocessed successfully. Shape:", df.shape)

except FileNotFoundError:
    raise SystemExit(
        "\nCSV not found. Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' next to this script, "
        "or replace the load block with: df = pd.read_csv('your_preprocessed.csv')\n"
    )


print("\n--- Customer Churn Prediction Pipeline ---")
print(f"Dataset shape : {df.shape}")
print(f"Churn balance : {df['Churn'].value_counts().to_dict()}\n")


# --- Section 1: Feature Analysis ---

print("--- Section 1: Feature Analysis ---")

X_full = df.drop(columns=["Churn"])
y = df["Churn"]

# Correlation matrix
corr = X_full.corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
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
print("Saved: 1_correlation_matrix.png")

# Highly correlated pairs
CORR_THRESHOLD = 0.75
print(f"\nHighly correlated pairs (|r| > {CORR_THRESHOLD}):")
found = False
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > CORR_THRESHOLD:
            print(f"  {corr.columns[i]} <-> {corr.columns[j]}   r = {r:.3f}")
            found = True
if not found:
    print("  None found above threshold.")

# Feature selection — SelectKBest with ANOVA F-test
K_FEATURES = 10
print(f"\nRunning SelectKBest (ANOVA F-test), selecting top {K_FEATURES} features...")

selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
selector.fit(X_full, y)

scores_series = (
    pd.Series(selector.scores_, index=X_full.columns)
      .sort_values(ascending=False)
)
print("\nFeature F-scores (ranked):\n")
print(scores_series.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
scores_series.head(15).sort_values().plot(
    kind="barh", color="steelblue", edgecolor="white", ax=ax
)
ax.axvline(
    scores_series.iloc[K_FEATURES - 1], color="crimson",
    linestyle="--", linewidth=1.4, label=f"Top-{K_FEATURES} cutoff"
)
ax.set_title("SelectKBest — Top 15 Feature F-Scores", fontsize=13, fontweight="bold")
ax.set_xlabel("ANOVA F-Score")
ax.legend()
plt.tight_layout()
plt.savefig("2_feature_scores.png", dpi=150)
plt.close()
print("Saved: 2_feature_scores.png")

selected_features = scores_series.head(K_FEATURES).index.tolist()
X = df[selected_features]

print(f"\nTop {K_FEATURES} selected features:")
for i, f in enumerate(selected_features, 1):
    print(f"  {i:2d}. {f}")


# --- Section 2: Train-Test Split ---

print("\n--- Section 2: Train-Test Split (70/30) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print(f"Train churn rate : {y_train.mean():.2%}")
print(f"Test  churn rate : {y_test.mean():.2%}")


# --- Feature Scaling ---

print("\n--- Feature Scaling (MinMaxScaler) ---")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Features scaled.")


# --- Section 2.5: K-Means Clustering ---

print("\n--- Section 2.5: K-Means Clustering ---")

kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
clusters = kmeans.fit_predict(X)

print(f"Clusters: {kmeans.n_clusters}")
cluster_counts = pd.Series(clusters).value_counts().sort_index()
print("Distribution:")
for i, count in cluster_counts.items():
    print(f"  Cluster {i}: {count} customers")

plt.figure(figsize=(7, 5))
sns.scatterplot(x=df["tenure"], y=df["MonthlyCharges"], hue=clusters, palette="Set2")
plt.title("K-Means Customer Clusters")
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("2b_kmeans_clusters.png", dpi=150)
plt.close()
print("Saved: 2b_kmeans_clusters.png")


# --- Section 3: Decision Tree ---

print("\n--- Section 3: Decision Tree Classifier ---")

dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=15,
    criterion="gini",
    random_state=RANDOM_STATE,
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print(f"Params: max_depth={dt_model.max_depth}, min_samples_leaf={dt_model.min_samples_leaf}, criterion={dt_model.criterion}")

# Feature importances
dt_imp = (
    pd.Series(dt_model.feature_importances_, index=selected_features)
      .sort_values(ascending=False)
)
print("\nFeature importances:\n")
print(dt_imp.to_string())

fig, ax = plt.subplots(figsize=(9, 4))
dt_imp.sort_values().plot(kind="barh", color="darkorange", edgecolor="white", ax=ax)
ax.set_title("Decision Tree — Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig("3a_dt_feature_importance.png", dpi=150)
plt.close()
print("Saved: 3a_dt_feature_importance.png")

fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(
    dt_model,
    feature_names=selected_features,
    class_names=["No Churn", "Churn"],
    filled=True, rounded=True,
    max_depth=3,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision Tree Structure (first 3 levels)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("3b_decision_tree_structure.png", dpi=120)
plt.close()
print("Saved: 3b_decision_tree_structure.png")


# --- Section 4: SVM ---

print("\n--- Section 4: Support Vector Machine (RBF kernel) ---")

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=RANDOM_STATE,
)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print(f"Params: kernel={svm_model.kernel}, C={svm_model.C}, gamma={svm_model.gamma}")


# --- Section 5: Model Evaluation ---

print("\n--- Section 5: Model Evaluation ---")

def evaluate_model(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"], digits=4))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    group_counts = [f"{v}" for v in cm.flatten()]
    group_pcts   = [f"{v:.1%}" for v in cm.flatten() / cm.sum()]
    labels = np.array(
        [f"{cnt}\n({pct})" for cnt, pct in zip(group_counts, group_pcts)]
    ).reshape(2, 2)

    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues",
        xticklabels=["Predicted: No", "Predicted: Yes"],
        yticklabels=["Actual: No", "Actual: Yes"],
        linewidths=1.2, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    fname = f"4_cm_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

    return {
        "Model"    : name,
        "Accuracy" : round(acc,  4),
        "Precision": round(prec, 4),
        "Recall"   : round(rec,  4),
        "F1-Score" : round(f1,   4),
    }

results_dt  = evaluate_model("Decision Tree", y_test, y_pred_dt)
results_svm = evaluate_model("SVM", y_test, y_pred_svm)


# --- Section 6: Model Comparison ---

print("\n--- Section 6: Model Comparison ---\n")

comparison_df = pd.DataFrame([results_dt, results_svm]).set_index("Model")
print(comparison_df.to_string())

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x       = np.arange(len(metrics))
width   = 0.32
colors  = ["#2196F3", "#FF5722"]

fig, ax = plt.subplots(figsize=(10, 5))
bars_dt  = ax.bar(x - width/2, comparison_df.loc["Decision Tree", metrics],
                  width, label="Decision Tree", color=colors[0], edgecolor="white")
bars_svm = ax.bar(x + width/2, comparison_df.loc["SVM", metrics],
                  width, label="SVM",           color=colors[1], edgecolor="white")

for bar in list(bars_dt) + list(bars_svm):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.4f}", ha="center", va="bottom", fontsize=9)

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
print("\nSaved: 5_model_comparison.png")

# Radar chart
angles  = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for model_name, color in zip(["Decision Tree", "SVM"], colors):
    values = comparison_df.loc[model_name, metrics].tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, label=model_name)
    ax.fill(angles, values, color=color, alpha=0.18)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.set_title("Model Comparison — Radar Chart", fontsize=13, fontweight="bold", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("6_model_radar.png", dpi=150)
plt.close()
print("Saved: 6_model_radar.png")


# --- Section 7: Analysis & Discussion ---

dt_f1  = results_dt["F1-Score"]
svm_f1 = results_svm["F1-Score"]
better = "SVM" if svm_f1 > dt_f1 else "Decision Tree"
f1_diff = abs(svm_f1 - dt_f1)

print("\n--- Section 7: Analysis & Discussion ---\n")

print("1. WHICH MODEL PERFORMED BETTER?")
print(f"   {better} achieved a higher F1-Score ({max(dt_f1, svm_f1):.4f} vs {min(dt_f1, svm_f1):.4f}, difference = {f1_diff:.4f}).")
print(f"\n   Full results:")
print(comparison_df.to_string())

print("""
2. WHY DID ONE MODEL OUTPERFORM THE OTHER?
   - SVM with an RBF kernel models non-linear boundaries, which helps when
     churn depends on feature interactions (e.g., short tenure + high charges).
   - Margin maximisation gives SVM better generalisation on unseen data,
     especially since only ~27% of customers churned (class imbalance).
   - MinMaxScaling directly benefits SVM since the RBF kernel is distance-based.
   - The Decision Tree was capped at max_depth=5 and min_samples_leaf=15 to
     avoid overfitting, which also limits its ability to capture complex patterns.
""")

print("""
3. IMPACT OF FEATURE SELECTION
   SelectKBest narrowed the features from the full set down to the top 10 most
   discriminative ones (by ANOVA F-score). This removes noisy/correlated columns,
   reduces SVM training time, and helps prevent Decision Tree overfitting.
   Commonly selected features include: Contract type, Tenure, TotalCharges,
   MonthlyCharges, and InternetService (Fiber optic).
""")

print("""
4. CONFUSION MATRIX INSIGHTS
   Both models correctly identify most non-churners (large True Negative count)
   because ~73% of customers didn't churn. The critical metric is False Negatives
   (churners predicted as staying) — these are customers lost without intervention.
   A model with higher churn Recall is more useful operationally even if its
   overall accuracy advantage looks small.
""")

print("""
5. RECOMMENDATIONS FOR IMPROVEMENT
   - Handle class imbalance: try SMOTE, class_weight='balanced', or lower threshold.
   - Hyperparameter tuning: GridSearchCV for max_depth/min_samples_leaf (DT) and C/gamma (SVM).
   - Ensemble methods: Random Forest or XGBoost typically outperform both on tabular data.
   - Additional metrics: report ROC-AUC and Precision-Recall AUC for imbalanced evaluation.
""")

print("--- Output Files ---")
output_files = [
    ("0_churn_distribution.png",       "Churn class distribution"),
    ("0_tenure_distribution.png",      "Tenure histogram"),
    ("0_monthly_charges_boxplot.png",  "Monthly charges boxplot"),
    ("1_correlation_matrix.png",       "Feature correlation heatmap"),
    ("2_feature_scores.png",           "SelectKBest F-score bar chart"),
    ("2b_kmeans_clusters.png",         "K-Means customer clusters"),
    ("3a_dt_feature_importance.png",   "Decision Tree feature importances"),
    ("3b_decision_tree_structure.png", "Decision Tree visual (3 levels)"),
    ("4_cm_decision_tree.png",         "Decision Tree confusion matrix"),
    ("4_cm_svm.png",                   "SVM confusion matrix"),
    ("5_model_comparison.png",         "Bar chart — model metrics"),
    ("6_model_radar.png",              "Radar chart — model comparison"),
]
for fname, desc in output_files:
    print(f"  {fname:<42s} {desc}")
