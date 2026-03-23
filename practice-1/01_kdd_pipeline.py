# 01_kdd_pipeline.py
# =============================================================================
# Full End-to-End KDD (Knowledge Discovery in Databases) Pipeline
#
# Stages implemented:
#   1. Raw Data Collection
#   2. Data Selection
#   3. Data Preprocessing  (Weka: Preprocess tab)
#   4. Data Transformation (Weka: Preprocess tab – filters)
#   5. Data Mining         (Weka: Classify + Cluster tabs)
#   6. Pattern Evaluation  (Weka: Classify tab – accuracy, confusion matrix)
#   7. Knowledge Representation (summary + visualisation)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import section, save_fig, print_classification_results, plot_confusion_matrix


# =============================================================================
# STAGE 1 — Raw Data Collection
# =============================================================================
section("STAGE 1 — Raw Data Collection")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"  Dataset : Iris (UCI / sklearn built-in)")
print(f"  Shape   : {df.shape[0]} samples × {df.shape[1]} columns")
print(f"  Classes : {list(iris.target_names)}")
print(f"\n  First 5 rows:")
print(df.head().to_string(index=False))


# =============================================================================
# STAGE 2 — Data Selection
# (Choose the target variable and the relevant feature subset)
# =============================================================================
section("STAGE 2 — Data Selection")

# Select all four numeric features + target class
selected_features = ["sepal length (cm)", "sepal width (cm)",
                      "petal length (cm)", "petal width (cm)"]
target_col = "species"

X = df[selected_features].copy()
y = df[target_col].copy()

print(f"  Selected features : {selected_features}")
print(f"  Target column     : {target_col}")
print(f"  Class distribution:")
print(y.value_counts().to_string())


# =============================================================================
# STAGE 3 — Data Preprocessing
# (Handle missing values, duplicates, noise — mirrors Weka Preprocess tab)
# =============================================================================
section("STAGE 3 — Data Preprocessing")

# 3a. Inject 5 artificial missing values for demonstration
rng = np.random.default_rng(42)
missing_idx = rng.integers(0, len(X), size=5)
X.iloc[missing_idx, 0] = np.nan

print(f"  Missing values before imputation:\n  {X.isnull().sum().to_dict()}")

# 3b. Impute missing values with column mean  (Weka: ReplaceMissingValues filter)
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"\n  Missing values after imputation:\n  {X_imputed.isnull().sum().to_dict()}")

# 3c. Remove duplicates
before = len(X_imputed)
X_imputed.drop_duplicates(inplace=True)
y = y.iloc[X_imputed.index].reset_index(drop=True)
X_imputed.reset_index(drop=True, inplace=True)
print(f"\n  Duplicates removed : {before - len(X_imputed)}")
print(f"  Dataset size after preprocessing: {X_imputed.shape}")


# =============================================================================
# STAGE 4 — Data Transformation
# (Normalise / standardise — mirrors Weka Normalize / Standardize filter)
# =============================================================================
section("STAGE 4 — Data Transformation")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

print("  Standardisation (z-score) applied: mean=0, std=1 per feature")
print(f"\n  Feature stats after transformation:")
print(X_scaled.describe().round(3).to_string())


# =============================================================================
# STAGE 5 — Data Mining
# 5a. Classification using Decision Tree (Weka: J48)
# 5b. Clustering using K-Means           (Weka: SimpleKMeans)
# =============================================================================
section("STAGE 5 — Data Mining  (Classification)")

X_arr = X_scaled.values
y_arr = iris.target[:len(X_scaled)]   # numeric labels for sklearn

X_train, X_test, y_train, y_test = train_test_split(
    X_arr, y_arr, test_size=0.3, random_state=42, stratify=y_arr)

# J48 ≡ DecisionTreeClassifier with gini/entropy criterion
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

print(f"\n  Classifier : Decision Tree (entropy, max_depth=4)  — Weka equivalent: J48")
print(f"  Train size : {len(X_train)} samples")
print(f"  Test  size : {len(X_test)}  samples\n")
print("  Decision tree rules:")
print(export_text(clf, feature_names=selected_features))

section("STAGE 5 — Data Mining  (Clustering)")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_arr)

print(f"  Algorithm  : K-Means (k=3)  — Weka equivalent: SimpleKMeans")
print(f"  Inertia    : {kmeans.inertia_:.4f}")
print(f"  Cluster sizes:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f"    Cluster {c}: {n} samples")


# =============================================================================
# STAGE 6 — Pattern Evaluation
# =============================================================================
section("STAGE 6 — Pattern Evaluation")

y_pred = clf.predict(X_test)
print_classification_results(y_test, y_pred, iris.target_names)

# 10-fold cross-validation
cv_scores = cross_val_score(clf, X_arr, y_arr, cv=10, scoring="accuracy")
print(f"  10-Fold CV Accuracy : {cv_scores.mean()*100:.2f}%  ± {cv_scores.std()*100:.2f}%")


# =============================================================================
# STAGE 7 — Knowledge Representation  (Visualisations)
# =============================================================================
section("STAGE 7 — Knowledge Representation (generating plots…)")

# ── Plot 1: Attribute distributions ──────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("KDD Stage 3 — Feature Distributions (after preprocessing)", fontsize=12)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for ax, col, color in zip(axes, selected_features, colors):
    ax.hist(X_imputed[col], bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(col.split(" (")[0], fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
plt.tight_layout()
save_fig("01_feature_distributions.png")
plt.show()

# ── Plot 2: Confusion matrix ──────────────────────────────────────────────────
plot_confusion_matrix(y_test, y_pred, iris.target_names,
                      title="KDD Stage 6 — Decision Tree Confusion Matrix",
                      filename="02_confusion_matrix.png")

# ── Plot 3: Decision tree visualisation ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(clf, feature_names=selected_features, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("KDD Stage 5 — J48-style Decision Tree (entropy, max_depth=4)", fontsize=12)
plt.tight_layout()
save_fig("03_decision_tree.png")
plt.show()

# ── Plot 4: K-Means clusters (PCA 2D projection) ─────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_arr)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("KDD Stage 5 — K-Means Clusters vs True Labels (PCA projection)", fontsize=12)

palette_clusters = ["#E05C5C", "#5C8FE0", "#5CBF5C"]
palette_true = ["#F4A261", "#2A9D8F", "#E76F51"]

for ax, labels, palette, title in [
    (axes[0], cluster_labels, palette_clusters, "K-Means clusters (k=3)"),
    (axes[1], y_arr, palette_true, "True class labels"),
]:
    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=palette[lbl], s=45, alpha=0.8, edgecolors="white", linewidths=0.4,
                   label=f"{'Cluster' if title.startswith('K') else iris.target_names[lbl]} {lbl}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=9)

plt.tight_layout()
save_fig("04_kmeans_clusters.png")
plt.show()

# ── Plot 5: Cross-validation scores ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, 11), cv_scores * 100, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axhline(cv_scores.mean() * 100, color="crimson", linestyle="--",
           linewidth=1.5, label=f"Mean = {cv_scores.mean()*100:.1f}%")
ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy (%)")
ax.set_title("KDD Stage 6 — 10-Fold Cross-Validation Accuracy", fontsize=12)
ax.set_xticks(range(1, 11))
ax.set_ylim(80, 105)
ax.legend()
plt.tight_layout()
save_fig("05_cross_validation.png")
plt.show()

section("COMPLETE — All plots saved to outputs/")
