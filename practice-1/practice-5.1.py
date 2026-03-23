"""
Simple K-Means Clustering — From Scratch
=========================================
Dataset : Customer Segmentation (Age, Annual Income, Spending Score)
          30 customers, 3 features

Steps:
  1.  Dataset creation & exploration
  2.  Data normalisation (Z-score)
  3.  K-Means++ initialisation
  4.  Assignment step  (Euclidean distance to nearest centroid)
  5.  Update step      (recompute centroid = cluster mean)
  6.  Iteration loop with convergence check
  7.  WCSS trace per iteration
  8.  Elbow method  (find optimal K)
  9.  Final cluster profiles & interpretation
  10. Silhouette score  (cluster quality)
"""

import math
import random
from copy import deepcopy
from collections import defaultdict

random.seed(42)


# ══════════════════════════════════════════════════════════
# 1. DATASET — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════

FEATURES = ["Age", "AnnualIncome_k", "SpendingScore"]

# 30 customers: [Age, Annual Income (k$), Spending Score (1-100)]
raw = [
    (19, 15,  39), (21, 15,  81), (20, 16,  6),  (23, 16,  77),
    (31, 17,  40), (22, 17,  76), (35, 18,  6),  (23, 18,  94),
    (64, 19,  3),  (30, 19,  72), (67, 19,  14), (35, 19,  99),
    (58, 20,  15), (24, 20,  77), (37, 21,  13), (22, 21,  79),
    (35, 23,  35), (20, 24,  35), (52, 25,  13), (35, 25,  79),
    (35, 26,  35), (25, 26,  87), (46, 28,  17), (31, 28,  73),
    (54, 28,  14), (29, 29,  82), (45, 30,  14), (35, 30,  90),
    (40, 33,  16), (33, 33,  97),
]

data = [list(row) for row in raw]
n    = len(data)

print("=" * 65)
print("K-MEANS CLUSTERING — CUSTOMER SEGMENTATION DATASET")
print("=" * 65)
print(f"\nDataset: {n} customers, {len(FEATURES)} features")
print(f"\n{'#':<5}{'Age':>6}{'Income(k$)':>12}{'SpendScore':>12}")
print("─" * 36)
for i, row in enumerate(data, 1):
    print(f"{i:<5}{row[0]:>6}{row[1]:>12}{row[2]:>12}")


# ══════════════════════════════════════════════════════════
# 2. STATISTICS & NORMALISATION  (Z-score per feature)
# ══════════════════════════════════════════════════════════

def col_stats(data, col):
    vals = [r[col] for r in data]
    mean = sum(vals) / len(vals)
    var  = sum((v - mean)**2 for v in vals) / len(vals)
    std  = math.sqrt(var)
    return mean, std

def zscore_normalise(data):
    """Standardise each feature to mean=0, std=1."""
    stats  = [col_stats(data, c) for c in range(len(data[0]))]
    normed = []
    for row in data:
        normed.append([(row[c] - stats[c][0]) / (stats[c][1] or 1)
                       for c in range(len(row))])
    return normed, stats

normed_data, feat_stats = zscore_normalise(data)

print("\n" + "─" * 65)
print("STEP 2 — FEATURE STATISTICS & Z-SCORE NORMALISATION")
print("─" * 65)
print(f"\n{'Feature':<18}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")
print("─" * 52)
for i, feat in enumerate(FEATURES):
    vals = [r[i] for r in data]
    print(f"{feat:<18}{feat_stats[i][0]:>10.2f}{feat_stats[i][1]:>10.2f}"
          f"{min(vals):>10.2f}{max(vals):>10.2f}")

print(f"\nNormalised sample (first 5 rows):")
print(f"{'#':<5}{'Age_z':>10}{'Income_z':>10}{'Spend_z':>10}")
print("─" * 36)
for i, row in enumerate(normed_data[:5], 1):
    print(f"{i:<5}{row[0]:>10.4f}{row[1]:>10.4f}{row[2]:>10.4f}")


# ══════════════════════════════════════════════════════════
# 3. DISTANCE & CENTROID UTILITIES
# ══════════════════════════════════════════════════════════

def euclidean(a, b):
    """Euclidean distance between two points."""
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def centroid(points):
    """Mean of a list of points (element-wise)."""
    d = len(points[0])
    return [sum(p[c] for p in points) / len(points) for c in range(d)]

def wcss(clusters, centroids):
    """Within-Cluster Sum of Squares — the K-Means objective."""
    total = 0.0
    for k, pts in clusters.items():
        for p in pts:
            total += euclidean(p, centroids[k])**2
    return total


# ══════════════════════════════════════════════════════════
# 4. K-MEANS++ INITIALISATION
# ══════════════════════════════════════════════════════════

def kmeans_pp_init(data, K):
    """
    K-Means++ seeding:
    1. Pick first centroid uniformly at random.
    2. Each subsequent centroid chosen with probability ∝ D(x)²
       where D(x) = distance to nearest already-chosen centroid.
    Guarantees O(log K) approximation of optimal WCSS.
    """
    centroids = [random.choice(data)[:]]
    for _ in range(K - 1):
        dists = []
        for p in data:
            d = min(euclidean(p, c)**2 for c in centroids)
            dists.append(d)
        total = sum(dists)
        probs = [d / total for d in dists]
        # Weighted random selection
        r = random.random()
        cumulative = 0
        chosen = data[-1]
        for p, prob in zip(data, probs):
            cumulative += prob
            if r <= cumulative:
                chosen = p
                break
        centroids.append(chosen[:])
    return centroids


# ══════════════════════════════════════════════════════════
# 5. FULL K-MEANS ALGORITHM
# ══════════════════════════════════════════════════════════

def kmeans(data, K, max_iter=100, verbose=True):
    """
    Simple K-Means:
      - K-Means++ initialisation
      - Assignment step: each point → nearest centroid
      - Update step:     each centroid ← mean of its cluster
      - Converges when assignments stop changing
    Returns: (labels, centroids, wcss_history, n_iters)
    """
    n = len(data)
    centroids = kmeans_pp_init(data, K)

    if verbose:
        print(f"\n  Initial centroids (K-Means++ seeding):")
        for k, c in enumerate(centroids):
            print(f"    C{k+1}: {[round(v, 4) for v in c]}")

    labels       = [-1] * n
    wcss_history = []

    for iteration in range(1, max_iter + 1):
        # ── Assignment step ───────────────────────────────
        new_labels = []
        for p in data:
            dists     = [euclidean(p, c) for c in centroids]
            new_labels.append(dists.index(min(dists)))

        # ── Build clusters ────────────────────────────────
        clusters = defaultdict(list)
        for i, lbl in enumerate(new_labels):
            clusters[lbl].append(data[i])

        # ── WCSS ──────────────────────────────────────────
        w = wcss(clusters, centroids)
        wcss_history.append(w)

        # ── Convergence check ─────────────────────────────
        changed = sum(1 for a, b in zip(labels, new_labels) if a != b)
        labels  = new_labels

        if verbose:
            sizes = [len(clusters[k]) for k in range(K)]
            print(f"  Iter {iteration:>3}: WCSS={w:>9.4f}  "
                  f"reassigned={changed:>3}  sizes={sizes}")

        if changed == 0:
            if verbose:
                print(f"\n  Converged after {iteration} iterations.")
            break

        # ── Update step ───────────────────────────────────
        for k in range(K):
            if clusters[k]:
                centroids[k] = centroid(clusters[k])

    return labels, centroids, wcss_history, iteration


# ══════════════════════════════════════════════════════════
# 6. RUN K-MEANS WITH K=3
# ══════════════════════════════════════════════════════════

K = 3
print("\n" + "─" * 65)
print(f"STEP 3–6 — K-MEANS ALGORITHM  (K={K}, normalised data)")
print("─" * 65)

labels, centroids, wcss_hist, n_iters = kmeans(normed_data, K, verbose=True)


# ══════════════════════════════════════════════════════════
# 7. WCSS TRACE
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 7 — WCSS CONVERGENCE TRACE")
print("─" * 65)
print(f"\n{'Iter':>6}  {'WCSS':>12}  {'Change':>12}  Progress bar")
print("─" * 60)
prev = wcss_hist[0]
for i, w in enumerate(wcss_hist, 1):
    delta  = prev - w if i > 1 else 0.0
    bar_w  = int((w / wcss_hist[0]) * 30)
    bar    = "█" * bar_w + "░" * (30 - bar_w)
    print(f"{i:>6}  {w:>12.4f}  {delta:>+12.4f}  {bar}")
    prev = w


# ══════════════════════════════════════════════════════════
# 8. ELBOW METHOD  (K = 1 … 7)
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 8 — ELBOW METHOD  (optimal K selection)")
print("─" * 65)

elbow_results = {}
print(f"\n{'K':>4}  {'WCSS':>12}  {'Drop%':>8}  Bar")
print("─" * 55)
prev_w = None
for k in range(1, 8):
    lbl, ctr, hist, _ = kmeans(normed_data, k, verbose=False)
    clusters = defaultdict(list)
    for i, l in enumerate(lbl):
        clusters[l].append(normed_data[i])
    w = wcss(clusters, ctr)
    elbow_results[k] = w
    drop = (prev_w - w) / prev_w * 100 if prev_w else 0
    bar  = "█" * int(w / elbow_results.get(1, w) * 20)
    mark = "  ← elbow" if k == 3 else ""
    print(f"{k:>4}  {w:>12.4f}  {drop:>7.1f}%  {bar}{mark}")
    prev_w = w

print("\n  Elbow at K=3: largest relative drop, best balance of fit vs complexity.")


# ══════════════════════════════════════════════════════════
# 9. FINAL CLUSTER PROFILES
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 9 — FINAL CLUSTER PROFILES  (original scale)")
print("─" * 65)

# Rebuild clusters on original (un-normalised) data
clusters_orig = defaultdict(list)
clusters_idx  = defaultdict(list)
for i, lbl in enumerate(labels):
    clusters_orig[lbl].append(data[i])
    clusters_idx[lbl].append(i + 1)   # 1-based customer ID

CLUSTER_NAMES = {0: "Cluster A", 1: "Cluster B", 2: "Cluster C"}

print()
for k in sorted(clusters_orig):
    pts  = clusters_orig[k]
    name = CLUSTER_NAMES[k]
    ages      = [p[0] for p in pts]
    incomes   = [p[1] for p in pts]
    scores    = [p[2] for p in pts]
    print(f"  {name}  ({len(pts)} customers — IDs: {clusters_idx[k]})")
    print(f"  {'':4}{'Feature':<18}{'Mean':>8}{'Min':>8}{'Max':>8}{'Std':>8}")
    print("  " + "─" * 46)
    for feat_name, vals in [("Age", ages), ("Income(k$)", incomes),
                             ("SpendScore", scores)]:
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
        print(f"  {'':4}{feat_name:<18}{mean:>8.1f}{min(vals):>8.1f}"
              f"{max(vals):>8.1f}{std:>8.1f}")

    # Centroid in original scale
    mean_age = sum(ages) / len(ages)
    mean_inc = sum(incomes) / len(incomes)
    mean_sco = sum(scores) / len(scores)
    # Label by dominant characteristic
    if mean_sco > 60:
        tag = "High Spenders"
    elif mean_sco < 35:
        tag = "Low Spenders"
    else:
        tag = "Moderate Spenders"
    print(f"\n  Centroid: Age={mean_age:.1f}, Income={mean_inc:.1f}k, "
          f"Score={mean_sco:.1f}  → [{tag}]")
    print()


# ══════════════════════════════════════════════════════════
# 10. SILHOUETTE SCORE
# ══════════════════════════════════════════════════════════

def silhouette_score(data, labels, K):
    """
    Silhouette coefficient per point:
      a(i) = mean intra-cluster distance
      b(i) = mean distance to nearest OTHER cluster
      s(i) = (b(i) - a(i)) / max(a(i), b(i))
    Range [-1, 1]: 1=perfect, 0=border, <0=wrong cluster
    """
    n = len(data)
    scores = []
    for i in range(n):
        my_k    = labels[i]
        my_pts  = [data[j] for j in range(n) if labels[j] == my_k and j != i]
        if not my_pts:
            scores.append(0.0)
            continue
        a = sum(euclidean(data[i], p) for p in my_pts) / len(my_pts)

        b_vals = []
        for k in range(K):
            if k == my_k:
                continue
            other = [data[j] for j in range(n) if labels[j] == k]
            if other:
                b_vals.append(sum(euclidean(data[i], p)
                                  for p in other) / len(other))
        b = min(b_vals) if b_vals else 0.0
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(s)
    return scores

sil_scores = silhouette_score(normed_data, labels, K)
avg_sil    = sum(sil_scores) / len(sil_scores)

print("─" * 65)
print("STEP 10 — SILHOUETTE ANALYSIS")
print("─" * 65)
print(f"\n  Overall Silhouette Score: {avg_sil:.4f}")
print(f"  (1.0 = perfect separation, 0 = overlapping, <0 = misclassified)\n")
print(f"  {'Customer':<12}{'Cluster':>9}{'Silhouette':>12}  Quality")
print("  " + "─" * 42)
for i, (sil, lbl) in enumerate(zip(sil_scores, labels), 1):
    quality = ("Excellent" if sil > 0.7 else
               "Good"      if sil > 0.5 else
               "Fair"      if sil > 0.25 else
               "Poor")
    bar = "█" * int(max(sil, 0) * 12)
    print(f"  C{i:<10}{CLUSTER_NAMES[lbl]:>9}{sil:>12.4f}  {bar} {quality}")

# Per-cluster silhouette
print(f"\n  Per-cluster average silhouette:")
for k in range(K):
    k_sils = [sil_scores[i] for i, l in enumerate(labels) if l == k]
    avg_k  = sum(k_sils) / len(k_sils)
    print(f"    {CLUSTER_NAMES[k]}: {avg_k:.4f}  ({len(k_sils)} points)")


# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\n  Algorithm         : Simple K-Means")
print(f"  Initialisation    : K-Means++")
print(f"  K (clusters)      : {K}")
print(f"  Convergence       : {n_iters} iterations")
print(f"  Final WCSS        : {wcss_hist[-1]:.4f}")
print(f"  Silhouette score  : {avg_sil:.4f}")
print(f"\n  Cluster summary:")
for k in sorted(clusters_orig):
    pts  = clusters_orig[k]
    mean_score = sum(p[2] for p in pts) / len(pts)
    mean_inc   = sum(p[1] for p in pts) / len(pts)
    tag = ("High Spenders" if mean_score > 60 else
           "Low Spenders"  if mean_score < 35 else
           "Moderate Spenders")
    print(f"    {CLUSTER_NAMES[k]}: {len(pts):>2} customers | "
          f"Avg Income={mean_inc:.0f}k | Avg Score={mean_score:.0f} | {tag}")

print(f"\n  K-Means objective : minimise WCSS = sum of squared distances")
print(f"                      from each point to its cluster centroid")
print(f"  Distance metric   : Euclidean  d = sqrt(sum((xi-ci)^2))")
print(f"  Convergence test  : no reassignment OR max_iter reached")
print("=" * 65)
