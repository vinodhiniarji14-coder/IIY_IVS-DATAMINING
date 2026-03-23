# 04_clustering.py
# =============================================================================
# Weka Cluster Tab — Clustering Algorithms
#
# Algorithms demonstrated (Weka equivalents in brackets):
#   1. K-Means            [SimpleKMeans]
#   2. Gaussian Mixture   [EM — Expectation Maximisation]
#   3. DBSCAN             [DBSCAN]
#   4. Agglomerative      [HierarchicalClusterer]
#
# Evaluation:
#   • Within-cluster sum of squared errors (WCSS / Inertia)
#   • Silhouette Score
#   • Calinski-Harabasz Index
#   • Classes-to-clusters comparison (using known labels as reference)
#   • Elbow method to choose optimal k
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score

from utils import section, save_fig


# ── Load and prepare data ─────────────────────────────────────────────────────
iris = load_iris()
X_raw = iris.data
y_true = iris.target
class_names = list(iris.target_names)
feature_names = iris.feature_names

imputer = SimpleImputer(strategy="mean")
scaler  = StandardScaler()
X = scaler.fit_transform(imputer.fit_transform(X_raw))

# PCA for 2D visualisation
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

print("\n" + "=" * 62)
print("  Weka Cluster Tab — Clustering Demo")
print("=" * 62)
print(f"\n  Dataset : Iris  ({len(X)} samples, {X.shape[1]} features, 3 true classes)")
print(f"  Data standardised (z-score) before clustering")
print(f"  2D PCA projection used for visualisation")


# =============================================================================
# ALGORITHM 1 — K-Means  (Weka: SimpleKMeans)
# =============================================================================
section("ALGORITHM 1 — K-Means  (Weka: SimpleKMeans, k=3)")

km = KMeans(n_clusters=3, n_init=10, random_state=42)
km_labels = km.fit_predict(X)

print(f"  Inertia (WCSS)     : {km.inertia_:.4f}")
print(f"  Silhouette Score   : {silhouette_score(X, km_labels):.4f}")
print(f"  Calinski-Harabasz  : {calinski_harabasz_score(X, km_labels):.4f}")
print(f"  Adjusted Rand Index: {adjusted_rand_score(y_true, km_labels):.4f}")
print(f"\n  Cluster sizes:")
for c, n in zip(*np.unique(km_labels, return_counts=True)):
    print(f"    Cluster {c} : {n} samples")

print(f"\n  Centroids (standardised feature space):")
centroid_df = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_), columns=feature_names
)
print(centroid_df.round(3).to_string())

# Classes-to-clusters mapping
print("\n  Classes-to-clusters evaluation:")
for true_label, true_name in enumerate(class_names):
    mask = y_true == true_label
    assigned = km_labels[mask]
    dominant = np.bincount(assigned).argmax()
    correct = (assigned == dominant).sum()
    print(f"    {true_name:<18} → Cluster {dominant}  ({correct}/50 correct)")


# =============================================================================
# ALGORITHM 2 — Gaussian Mixture / EM  (Weka: EM)
# =============================================================================
section("ALGORITHM 2 — EM / Gaussian Mixture  (Weka: EM, k=3)")

gm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gm.fit(X)
gm_labels = gm.predict(X)
gm_proba  = gm.predict_proba(X)

print(f"  Log-Likelihood     : {gm.score(X)*len(X):.4f}")
print(f"  BIC (lower=better) : {gm.bic(X):.4f}")
print(f"  AIC (lower=better) : {gm.aic(X):.4f}")
print(f"  Silhouette Score   : {silhouette_score(X, gm_labels):.4f}")
print(f"  Adjusted Rand Index: {adjusted_rand_score(y_true, gm_labels):.4f}")
print(f"\n  Component weights  : {gm.weights_.round(4)}")
print(f"\n  Sample soft assignments (first 3 rows — probability per cluster):")
print(pd.DataFrame(gm_proba[:3], columns=[f"Cluster {i}" for i in range(3)]).round(4).to_string())


# =============================================================================
# ALGORITHM 3 — DBSCAN  (Weka: DBSCAN)
# =============================================================================
section("ALGORITHM 3 — DBSCAN  (eps=0.6, min_samples=5)")

db = DBSCAN(eps=0.6, min_samples=5)
db_labels = db.fit_predict(X)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = (db_labels == -1).sum()

print(f"  Clusters found     : {n_clusters}")
print(f"  Noise points       : {n_noise}")
if n_clusters > 1:
    mask_valid = db_labels != -1
    print(f"  Silhouette Score   : {silhouette_score(X[mask_valid], db_labels[mask_valid]):.4f}")
    print(f"  Adjusted Rand Index: {adjusted_rand_score(y_true[mask_valid], db_labels[mask_valid]):.4f}")
print(f"\n  Label distribution: {dict(zip(*np.unique(db_labels, return_counts=True)))}")
print(f"  (Label -1 = noise points not assigned to any cluster)")


# =============================================================================
# ALGORITHM 4 — Agglomerative / Hierarchical  (Weka: HierarchicalClusterer)
# =============================================================================
section("ALGORITHM 4 — Hierarchical / Agglomerative  (Weka: HierarchicalClusterer, k=3)")

agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
agg_labels = agg.fit_predict(X)

print(f"  Linkage method     : ward")
print(f"  Silhouette Score   : {silhouette_score(X, agg_labels):.4f}")
print(f"  Adjusted Rand Index: {adjusted_rand_score(y_true, agg_labels):.4f}")
print(f"\n  Cluster sizes:")
for c, n in zip(*np.unique(agg_labels, return_counts=True)):
    print(f"    Cluster {c} : {n} samples")


# =============================================================================
# Elbow Method — choosing optimal k for K-Means
# =============================================================================
section("ELBOW METHOD — Finding optimal k for K-Means")

inertias, silhouettes = [], []
K_range = range(2, 11)
for k in K_range:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km_k.fit_predict(X)
    inertias.append(km_k.inertia_)
    silhouettes.append(silhouette_score(X, labels_k))

print(f"  {'k':>3}  {'Inertia':>12}  {'Silhouette':>12}")
print("  " + "-" * 32)
for k, ine, sil in zip(K_range, inertias, silhouettes):
    print(f"  {k:>3}  {ine:>12.4f}  {sil:>12.4f}")


# =============================================================================
# Visualisations
# =============================================================================
section("Generating clustering visualisations…")

PALETTE = ["#E05C5C", "#5C8FE0", "#5CBF5C", "#E0A030", "#8055C0", "#30B0C0"]
TRUE_COLORS = ["#F4A261", "#2A9D8F", "#E76F51"]

# ── Plot 1: K-Means vs True Labels ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Weka Cluster — K-Means vs True Labels (PCA 2D)", fontsize=12)

for ax, labels, title, palette in [
    (axes[0], km_labels, "K-Means (k=3)", PALETTE),
    (axes[1], y_true,    "True labels",   TRUE_COLORS),
]:
    for lbl in np.unique(labels):
        mask = labels == lbl
        lname = f"Cluster {lbl}" if "K-Means" in title else class_names[lbl]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=palette[lbl], s=50, alpha=0.82,
                   edgecolors="white", linewidths=0.4, label=lname)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=9)

plt.tight_layout()
save_fig("cluster_01_kmeans_vs_true.png")
plt.show()

# ── Plot 2: All 4 algorithms side by side ────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Weka Cluster — All Algorithms Comparison (PCA 2D)", fontsize=12)

algo_list = [
    ("K-Means", km_labels),
    ("EM / GMM", gm_labels),
    ("DBSCAN",  db_labels),
    ("Hierarchical", agg_labels),
]
for ax, (title, labels) in zip(axes, algo_list):
    unique = np.unique(labels)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        color = "#aaaaaa" if lbl == -1 else PALETTE[i % len(PALETTE)]
        lname = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, s=35, alpha=0.80,
                   edgecolors="white", linewidths=0.3, label=lname)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)

plt.tight_layout()
save_fig("cluster_02_all_algorithms.png")
plt.show()

# ── Plot 3: Elbow method ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Weka Cluster — Elbow Method (optimal k)", fontsize=12)

axes[0].plot(K_range, inertias, "o-", color="#4C72B0", linewidth=1.8)
axes[0].set_xlabel("Number of clusters k")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow curve")
axes[0].axvline(x=3, color="crimson", linestyle="--", linewidth=1.2, label="k=3 (chosen)")
axes[0].legend()

axes[1].plot(K_range, silhouettes, "s-", color="#DD8452", linewidth=1.8)
axes[1].set_xlabel("Number of clusters k")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette score vs k")
axes[1].axvline(x=3, color="crimson", linestyle="--", linewidth=1.2, label="k=3 (chosen)")
axes[1].legend()

plt.tight_layout()
save_fig("cluster_03_elbow_method.png")
plt.show()

# ── Plot 4: Dendrogram  (Hierarchical clustering) ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
Z = linkage(X[:50], method="ward")   # use a subset of 50 for readability
dendrogram(Z, ax=ax, color_threshold=5,
           above_threshold_color="#aaaaaa",
           leaf_rotation=90, leaf_font_size=8)
ax.set_title("Weka Cluster — Hierarchical Clustering Dendrogram (ward, 50 samples)", fontsize=12)
ax.set_xlabel("Sample index")
ax.set_ylabel("Distance")
ax.axhline(y=5, color="crimson", linestyle="--", linewidth=1.2, label="Cut = 3 clusters")
ax.legend()
plt.tight_layout()
save_fig("cluster_04_dendrogram.png")
plt.show()

# ── Plot 5: EM soft probability heatmap (first 30 samples) ───────────────────
fig, ax = plt.subplots(figsize=(7, 6))
import seaborn as sns
sns.heatmap(gm_proba[:30], annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=[f"Cluster {i}" for i in range(3)],
            yticklabels=range(30), ax=ax, linewidths=0.2,
            cbar_kws={"shrink": 0.7})
ax.set_title("Weka Cluster — EM Soft Assignments (first 30 samples)", fontsize=12)
ax.set_xlabel("Cluster")
ax.set_ylabel("Sample index")
plt.tight_layout()
save_fig("cluster_05_em_soft_assignments.png")
plt.show()

section("COMPLETE — All clustering plots saved to outputs/")
