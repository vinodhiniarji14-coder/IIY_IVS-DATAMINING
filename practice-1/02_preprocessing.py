# 02_preprocessing.py
# =============================================================================
# Weka Preprocess Tab — Data Preprocessing & Transformation
#
# Filters demonstrated (Weka equivalents in brackets):
#   1. ReplaceMissingValues  → SimpleImputer (mean / most_frequent)
#   2. Normalize             → MinMaxScaler  [0, 1]
#   3. Standardize           → StandardScaler (z-score)
#   4. Discretize            → KBinsDiscretizer (equal-width / equal-frequency)
#   5. Remove attribute      → DataFrame.drop()
#   6. PrincipalComponents   → PCA
#   7. RemoveDuplicates      → DataFrame.drop_duplicates()
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from utils import section, save_fig


# ── Load dataset ──────────────────────────────────────────────────────────────
iris = load_iris()
df_orig = pd.DataFrame(iris.data, columns=iris.feature_names)
df_orig["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
feature_cols = iris.feature_names

print("\n" + "=" * 62)
print("  Weka Preprocess Tab — Data Preprocessing Demo")
print("=" * 62)
print(f"\n  Dataset: Iris  |  {df_orig.shape[0]} instances  |  {df_orig.shape[1]} attributes")
print("\n  Attribute summary:")
print(df_orig.describe().round(3).to_string())


# =============================================================================
# FILTER 1 — ReplaceMissingValues
# Weka: unsupervised → attribute → ReplaceMissingValues
# =============================================================================
section("FILTER 1 — ReplaceMissingValues")

df = df_orig.copy()
# Inject artificial missing values
rng = np.random.default_rng(0)
for col in feature_cols[:2]:
    idx = rng.choice(len(df), size=6, replace=False)
    df.loc[idx, col] = np.nan

print(f"  Missing values per attribute (before):")
print(f"  {df[feature_cols].isnull().sum().to_dict()}")

# Numeric: replace with mean; nominal would use most_frequent
imputer_mean = SimpleImputer(strategy="mean")
df[feature_cols] = imputer_mean.fit_transform(df[feature_cols])

print(f"\n  Missing values per attribute (after — mean imputation):")
print(f"  {df[feature_cols].isnull().sum().to_dict()}")


# =============================================================================
# FILTER 2 — Normalize  [0, 1]
# Weka: unsupervised → attribute → Normalize
# =============================================================================
section("FILTER 2 — Normalize (scale to [0, 1])")

scaler_minmax = MinMaxScaler()
df_norm = pd.DataFrame(
    scaler_minmax.fit_transform(df[feature_cols]),
    columns=feature_cols
)
df_norm["species"] = df["species"].values

print("  Before normalization (first 3 rows):")
print(df[feature_cols].head(3).round(4).to_string(index=False))
print("\n  After normalization (first 3 rows):")
print(df_norm[feature_cols].head(3).round(4).to_string(index=False))
print(f"\n  Min across all features: {df_norm[feature_cols].min().round(3).to_dict()}")
print(f"  Max across all features: {df_norm[feature_cols].max().round(3).to_dict()}")


# =============================================================================
# FILTER 3 — Standardize  (mean=0, std=1)
# Weka: unsupervised → attribute → Standardize
# =============================================================================
section("FILTER 3 — Standardize (z-score: mean=0, std=1)")

scaler_std = StandardScaler()
df_std = pd.DataFrame(
    scaler_std.fit_transform(df[feature_cols]),
    columns=feature_cols
)
df_std["species"] = df["species"].values

print("  After standardization — mean and std:")
stats = pd.DataFrame({
    "mean": df_std[feature_cols].mean().round(4),
    "std":  df_std[feature_cols].std().round(4),
})
print(stats.to_string())


# =============================================================================
# FILTER 4 — Discretize  (continuous → nominal bins)
# Weka: unsupervised → attribute → Discretize
# =============================================================================
section("FILTER 4 — Discretize (equal-width binning, 3 bins)")

kbd = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
disc = kbd.fit_transform(df[feature_cols])
df_disc = pd.DataFrame(disc.astype(int), columns=feature_cols)
df_disc["species"] = df["species"].values

bin_labels = {0: "low", 1: "medium", 2: "high"}
for col in feature_cols:
    df_disc[col] = df_disc[col].map(bin_labels)

print("  Discretized values (first 5 rows):")
print(df_disc.head().to_string(index=False))


# =============================================================================
# FILTER 5 — Remove attribute
# Weka: unsupervised → attribute → Remove
# =============================================================================
section("FILTER 5 — Remove attribute (drop 'sepal width (cm)')")

df_removed = df_norm.drop(columns=["sepal width (cm)"])
print(f"  Attributes before: {list(df_norm.columns)}")
print(f"  Attributes after : {list(df_removed.columns)}")


# =============================================================================
# FILTER 6 — PrincipalComponents (dimensionality reduction)
# Weka: unsupervised → attribute → PrincipalComponents
# =============================================================================
section("FILTER 6 — PrincipalComponents (PCA — 4D → 2D)")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(df_std[feature_cols])
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["species"] = df["species"].values

print(f"  Explained variance ratio : {pca.explained_variance_ratio_.round(4)}")
print(f"  Total variance captured  : {pca.explained_variance_ratio_.sum()*100:.1f}%")
print("\n  PCA output (first 5 rows):")
print(df_pca.head().round(4).to_string(index=False))


# =============================================================================
# FILTER 7 — Remove duplicates
# =============================================================================
section("FILTER 7 — Remove Duplicate Instances")

df_dup = df_norm.copy()
df_dup = pd.concat([df_dup, df_dup.iloc[:3]], ignore_index=True)  # add 3 dupes
before = len(df_dup)
df_dup.drop_duplicates(inplace=True)
print(f"  Instances before : {before}")
print(f"  Instances after  : {len(df_dup)}")
print(f"  Duplicates removed: {before - len(df_dup)}")


# =============================================================================
# Visualisations
# =============================================================================
section("Generating preprocessing visualisations…")

# ── Plot 1: Box-plots before vs after normalization ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Weka Preprocess — Normalize filter effect", fontsize=13)

df[feature_cols].plot(kind="box", ax=axes[0], patch_artist=True,
                      boxprops=dict(facecolor="#AEC6E8"))
axes[0].set_title("Before normalization")
axes[0].set_xticklabels([c.split(" (")[0] for c in feature_cols], rotation=15, ha="right")

df_norm[feature_cols].plot(kind="box", ax=axes[1], patch_artist=True,
                           boxprops=dict(facecolor="#A8D8A8"))
axes[1].set_title("After normalization [0, 1]")
axes[1].set_xticklabels([c.split(" (")[0] for c in feature_cols], rotation=15, ha="right")
plt.tight_layout()
save_fig("preprocess_01_normalize.png")
plt.show()

# ── Plot 2: PCA scatter ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors_map = {"setosa": "#4C72B0", "versicolor": "#DD8452", "virginica": "#55A868"}
for sp, grp in df_pca.groupby("species"):
    ax.scatter(grp["PC1"], grp["PC2"], label=sp,
               c=colors_map[sp], s=55, alpha=0.8, edgecolors="white", linewidths=0.3)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Weka Preprocess — PrincipalComponents (4D → 2D)")
ax.legend()
plt.tight_layout()
save_fig("preprocess_02_pca_scatter.png")
plt.show()

# ── Plot 3: Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
short_names = [c.split(" (")[0] for c in feature_cols]
corr = df[feature_cols].rename(columns=dict(zip(feature_cols, short_names))).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Weka Preprocess — Feature Correlation Matrix")
plt.tight_layout()
save_fig("preprocess_03_correlation.png")
plt.show()

section("COMPLETE — All preprocessing plots saved to outputs/")
