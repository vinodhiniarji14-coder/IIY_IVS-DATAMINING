"""
Hierarchical Agglomerative Clustering — From Scratch
======================================================
Dataset : Employee Performance  (20 employees, 3 features)
Features: YearsExperience, PerformanceScore, SalaryGrade

Steps:
  1.  Dataset + feature statistics
  2.  Z-score normalisation
  3.  Full pairwise distance matrix
  4.  Agglomerative clustering (Single / Complete / Average linkage)
  5.  Merge trace  (which clusters merge at each step and at what height)
  6.  ASCII dendrogram
  7.  Cluster extraction by cutting at a threshold
  8.  Cluster profiles (original scale)
  9.  Cophenetic correlation  (how well dendrogram preserves distances)
  10. Comparison of all three linkage methods
"""

import math
from copy import deepcopy
from collections import defaultdict

# ══════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════

FEATURES = ["YearsExp", "PerfScore", "SalaryGrade"]

# 20 employees: [Years of Experience, Performance Score (1-100), Salary Grade (1-10)]
raw = [
    (1,  55, 2),   # E01
    (2,  60, 3),   # E02
    (1,  52, 2),   # E03
    (3,  65, 3),   # E04
    (5,  70, 5),   # E05
    (6,  75, 5),   # E06
    (4,  68, 4),   # E07
    (5,  72, 5),   # E08
    (8,  80, 7),   # E09
    (9,  85, 7),   # E10
    (8,  78, 6),   # E11
    (10, 88, 8),   # E12
    (12, 90, 9),   # E13
    (11, 87, 8),   # E14
    (13, 92, 9),   # E15
    (14, 95, 10),  # E16
    (3,  40, 2),   # E17  ← underperformer
    (2,  38, 2),   # E18  ← underperformer
    (7,  55, 4),   # E19  ← mid-exp but low perf
    (6,  50, 4),   # E20  ← mid-exp but low perf
]

data      = [list(r) for r in raw]
n         = len(data)
labels    = [f"E{i+1:02d}" for i in range(n)]

print("=" * 65)
print("HIERARCHICAL AGGLOMERATIVE CLUSTERING — EMPLOYEE DATA")
print("=" * 65)
print(f"\nEmployees: {n}   Features: {FEATURES}\n")
print(f"{'ID':<6}{'YrsExp':>8}{'PerfScore':>11}{'SalaryGrd':>11}")
print("─" * 38)
for i, row in enumerate(data):
    print(f"{labels[i]:<6}{row[0]:>8}{row[1]:>11}{row[2]:>11}")


# ══════════════════════════════════════════════════════════
# 2. NORMALISATION  (Z-score)
# ══════════════════════════════════════════════════════════

def col_mean_std(data, col):
    vals = [r[col] for r in data]
    mu   = sum(vals) / len(vals)
    sd   = math.sqrt(sum((v - mu)**2 for v in vals) / len(vals))
    return mu, sd

def zscore(data):
    stats  = [col_mean_std(data, c) for c in range(len(data[0]))]
    normed = [[(r[c] - stats[c][0]) / (stats[c][1] or 1)
               for c in range(len(r))] for r in data]
    return normed, stats

normed, feat_stats = zscore(data)

print("\n" + "─" * 65)
print("STEP 2 — Z-SCORE NORMALISATION")
print("─" * 65)
print(f"\n{'Feature':<14}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print("─" * 52)
for i, f in enumerate(FEATURES):
    vals = [r[i] for r in data]
    print(f"{f:<14}  {feat_stats[i][0]:>8.2f}  {feat_stats[i][1]:>8.2f}"
          f"  {min(vals):>8.2f}  {max(vals):>8.2f}")


# ══════════════════════════════════════════════════════════
# 3. PAIRWISE DISTANCE MATRIX
# ══════════════════════════════════════════════════════════

def euclidean(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def distance_matrix(points):
    n   = len(points)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = euclidean(points[i], points[j])
            mat[i][j] = mat[j][i] = d
    return mat

dist_mat = distance_matrix(normed)

print("\n" + "─" * 65)
print("STEP 3 — PAIRWISE EUCLIDEAN DISTANCE MATRIX (normalised, first 8×8)")
print("─" * 65)
print(f"\n{'':6}", end="")
for j in range(8):
    print(f"{labels[j]:>8}", end="")
print()
print("─" * 70)
for i in range(8):
    print(f"{labels[i]:<6}", end="")
    for j in range(8):
        print(f"{dist_mat[i][j]:>8.3f}", end="")
    print()
print("  ... (full 20×20 matrix used internally)")


# ══════════════════════════════════════════════════════════
# 4. AGGLOMERATIVE CLUSTERING ENGINE
# ══════════════════════════════════════════════════════════

def cluster_distance(c1_ids, c2_ids, dist_mat, linkage):
    """
    Distance between two clusters given a linkage criterion.
    single   → min pairwise distance
    complete → max pairwise distance
    average  → mean of all pairwise distances (UPGMA)
    ward     → variance increase on merge (simplified)
    """
    dists = [dist_mat[i][j] for i in c1_ids for j in c2_ids]
    if linkage == "single":
        return min(dists)
    elif linkage == "complete":
        return max(dists)
    elif linkage == "average":
        return sum(dists) / len(dists)
    elif linkage == "ward":
        # Approximate: use average (true Ward requires centroids)
        return sum(dists) / len(dists)
    return min(dists)


def agglomerative(dist_mat, point_labels, linkage="average", verbose=True):
    """
    Full agglomerative hierarchical clustering.
    Returns:
      merges  — list of (cluster_a_label, cluster_b_label, height, new_size)
      history — cluster membership at each step
    """
    n = len(point_labels)
    # Each cluster: id → list of original point indices
    clusters = {i: [i] for i in range(n)}
    # Human-readable name for each cluster
    names    = {i: point_labels[i] for i in range(n)}
    merges   = []   # (name_a, name_b, height, new_cluster_size)
    step     = 0

    if verbose:
        print(f"\n  {'Step':>5}  {'Merge A':<18}  {'Merge B':<18}  "
              f"{'Height':>9}  {'NewSize':>8}")
        print("  " + "─" * 64)

    while len(clusters) > 1:
        step += 1
        # Find the two closest clusters
        best_d  = math.inf
        best_ab = None
        ids = list(clusters.keys())
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx+1, len(ids)):
                ci, cj = ids[i_idx], ids[j_idx]
                d = cluster_distance(clusters[ci], clusters[cj],
                                     dist_mat, linkage)
                if d < best_d:
                    best_d, best_ab = d, (ci, cj)

        ci, cj   = best_ab
        name_a   = names[ci]
        name_b   = names[cj]
        new_id   = max(clusters.keys()) + 1
        new_pts  = clusters[ci] + clusters[cj]
        new_name = f"[{name_a}+{name_b}]"
        new_size = len(new_pts)

        merges.append((name_a, name_b, round(best_d, 4), new_size))

        del clusters[ci]
        del clusters[cj]
        del names[ci]
        del names[cj]
        clusters[new_id] = new_pts
        names[new_id]    = new_name

        if verbose:
            print(f"  {step:>5}  {name_a:<18}  {name_b:<18}  "
                  f"{best_d:>9.4f}  {new_size:>8}")

    return merges


# ══════════════════════════════════════════════════════════
# 5. RUN WITH AVERAGE LINKAGE  (verbose trace)
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 4 & 5 — MERGE TRACE  (Average / UPGMA linkage)")
print("─" * 65)

merges_avg = agglomerative(dist_mat, labels, linkage="average", verbose=True)


# ══════════════════════════════════════════════════════════
# 6. ASCII DENDROGRAM
# ══════════════════════════════════════════════════════════

def ascii_dendrogram(merges, point_labels):
    """
    Draw a text-based dendrogram showing merge heights.
    Height axis is binned into levels for ASCII rendering.
    """
    print("\n  Height  │  Merge event")
    print("  ────────┼" + "─" * 50)
    for i, (a, b, h, sz) in enumerate(merges, 1):
        bar = "─" * min(int(h * 15), 40)
        print(f"  {h:>7.4f} │  {bar}> {a}  +  {b}  →  merged ({sz} pts)")


print("\n" + "─" * 65)
print("STEP 6 — ASCII DENDROGRAM  (merge heights)")
print("─" * 65)
ascii_dendrogram(merges_avg, labels)


# ══════════════════════════════════════════════════════════
# 7. CUT THE DENDROGRAM — EXTRACT CLUSTERS
# ══════════════════════════════════════════════════════════

def cut_dendrogram(merges, point_labels, threshold):
    """
    Assign cluster labels by cutting the dendrogram at `threshold`.
    Any merge with height > threshold is NOT performed → separate clusters.
    Returns dict {point_index: cluster_id}.
    """
    n = len(point_labels)
    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # Re-do merges but only those below threshold
    # Rebuild from point perspective: track which original indices each
    # cluster name maps to
    pt_idx = {label: [i] for i, label in enumerate(point_labels)}
    active = {label: [i] for i, label in enumerate(point_labels)}

    for a, b, h, _ in merges:
        if h > threshold:
            continue
        # Find original indices for a and b
        pts_a = active.get(a, [])
        pts_b = active.get(b, [])
        for i in pts_a:
            for j in pts_b:
                union(i, j)
        new_name = f"[{a}+{b}]"
        active[new_name] = pts_a + pts_b
        del active[a]
        del active[b]

    cluster_map = {}
    seen = {}
    cid  = 0
    for i in range(n):
        root = find(i)
        if root not in seen:
            seen[root] = cid
            cid += 1
        cluster_map[i] = seen[root]
    return cluster_map


# Try different cut heights
print("\n" + "─" * 65)
print("STEP 7 — CLUSTER EXTRACTION AT DIFFERENT CUT THRESHOLDS")
print("─" * 65)

for threshold in [0.5, 1.0, 1.5, 2.5]:
    cmap = cut_dendrogram(merges_avg, labels, threshold)
    n_clusters = len(set(cmap.values()))
    groups = defaultdict(list)
    for idx, cid in cmap.items():
        groups[cid].append(labels[idx])
    print(f"\n  Cut height = {threshold:.1f}  →  {n_clusters} cluster(s):")
    for cid, members in sorted(groups.items()):
        print(f"    Cluster {cid+1}: {', '.join(members)}")

# Use threshold=1.5 as the primary cut
THRESHOLD = 1.5
final_map  = cut_dendrogram(merges_avg, labels, THRESHOLD)
K_final    = len(set(final_map.values()))
print(f"\n  → Using threshold={THRESHOLD}  ({K_final} clusters) for profile analysis.")


# ══════════════════════════════════════════════════════════
# 8. CLUSTER PROFILES  (original scale)
# ══════════════════════════════════════════════════════════

CLUSTER_TAGS = {
    0: "Junior / Entry-level",
    1: "Mid-level",
    2: "Senior / Expert",
    3: "Underperformers",
}

print("\n" + "─" * 65)
print("STEP 8 — CLUSTER PROFILES  (original scale)")
print("─" * 65)

groups_orig = defaultdict(list)
for idx, cid in final_map.items():
    groups_orig[cid].append((data[idx], labels[idx]))

print()
for cid in sorted(groups_orig):
    items   = groups_orig[cid]
    members = [lbl for _, lbl in items]
    pts     = [pt  for pt, _ in items]
    tag     = CLUSTER_TAGS.get(cid, f"Cluster {cid}")
    print(f"  Cluster {cid+1}  [{tag}]  ({len(pts)} employees)")
    print(f"  Members : {', '.join(members)}")
    print(f"\n  {'Feature':<14}  {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
    print("  " + "─" * 46)
    for fi, feat in enumerate(FEATURES):
        vals = [p[fi] for p in pts]
        mu   = sum(vals)/len(vals)
        sd   = math.sqrt(sum((v-mu)**2 for v in vals)/len(vals))
        print(f"  {feat:<14}  {mu:>8.2f}  {min(vals):>8.2f}"
              f"  {max(vals):>8.2f}  {sd:>8.2f}")
    print()


# ══════════════════════════════════════════════════════════
# 9. COPHENETIC CORRELATION
# ══════════════════════════════════════════════════════════

def cophenetic_matrix(merges, point_labels):
    """
    Build the cophenetic distance matrix:
    coph[i][j] = height at which points i and j first share a cluster.
    """
    n    = len(point_labels)
    coph = [[0.0]*n for _ in range(n)]
    pt_idx = {lbl: i for i, lbl in enumerate(point_labels)}
    # Track which original points are in each named cluster
    active = {lbl: {i} for i, lbl in enumerate(point_labels)}

    for a, b, h, _ in merges:
        pts_a = active.get(a, set())
        pts_b = active.get(b, set())
        for i in pts_a:
            for j in pts_b:
                coph[i][j] = coph[j][i] = h
        new_name = f"[{a}+{b}]"
        active[new_name] = pts_a | pts_b
        if a in active: del active[a]
        if b in active: del active[b]

    return coph

def pearson(xs, ys):
    n  = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num    = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    denom  = math.sqrt(sum((x-mx)**2 for x in xs) *
                       sum((y-my)**2 for y in ys))
    return num/denom if denom else 0.0

coph_mat = cophenetic_matrix(merges_avg, labels)

# Flatten upper triangles of both matrices
orig_flat = [dist_mat[i][j] for i in range(n) for j in range(i+1, n)]
coph_flat = [coph_mat[i][j] for i in range(n) for j in range(i+1, n)]
ccc = pearson(orig_flat, coph_flat)

print("─" * 65)
print("STEP 9 — COPHENETIC CORRELATION COEFFICIENT")
print("─" * 65)
print(f"\n  CCC = {ccc:.4f}")
print(f"  (1.0 = perfect dendrogram, >0.75 = good, <0.5 = poor)")
qual = ("Excellent" if ccc > 0.9 else "Good" if ccc > 0.75
        else "Moderate" if ccc > 0.5 else "Poor")
print(f"  Quality: {qual}")
print(f"\n  Interpretation: The dendrogram preserves {ccc*100:.1f}% of the")
print(f"  pairwise distance structure from the original data.")


# ══════════════════════════════════════════════════════════
# 10. COMPARE ALL THREE LINKAGE METHODS
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 10 — LINKAGE METHOD COMPARISON")
print("─" * 65)

print(f"\n  {'Linkage':<12}  {'CCC':>8}  {'Clusters@1.5':>14}  "
      f"{'First merge':>12}  {'Last merge':>12}")
print("  " + "─" * 64)

for lnk in ["single", "complete", "average"]:
    m = agglomerative(dist_mat, labels, linkage=lnk, verbose=False)
    cm_mat = cophenetic_matrix(m, labels)
    cf = [cm_mat[i][j] for i in range(n) for j in range(i+1, n)]
    r  = pearson(orig_flat, cf)
    cmap_lnk = cut_dendrogram(m, labels, THRESHOLD)
    nk = len(set(cmap_lnk.values()))
    first_h = m[0][2]
    last_h  = m[-2][2]   # last real merge (second-to-last step)
    print(f"  {lnk:<12}  {r:>8.4f}  {nk:>14}  "
          f"{first_h:>12.4f}  {last_h:>12.4f}")

print(f"""
  Notes:
    Single   — tends to chain elongated shapes; CCC often high but
               clusters can be poorly separated ("chaining effect").
    Complete — compact, equal-size clusters; robust to outliers.
    Average  — best general-purpose balance; highest CCC typical.
""")


# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════

print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"""
  Algorithm     : Agglomerative Hierarchical Clustering
  Linkage used  : Average (UPGMA)
  Distance      : Euclidean on Z-score normalised features
  Employees     : {n}
  Features      : {', '.join(FEATURES)}
  Cut threshold : {THRESHOLD}
  Clusters found: {K_final}
  CCC           : {ccc:.4f}  ({qual})

  Cluster breakdown:""")
for cid in sorted(groups_orig):
    items   = groups_orig[cid]
    pts     = [pt for pt, _ in items]
    members = [lbl for _, lbl in items]
    mu_perf = sum(p[1] for p in pts)/len(pts)
    mu_exp  = sum(p[0] for p in pts)/len(pts)
    tag     = CLUSTER_TAGS.get(cid, f"Cluster {cid}")
    print(f"    Cluster {cid+1} [{tag}]: {len(pts)} employees, "
          f"avg exp={mu_exp:.1f}yr, avg perf={mu_perf:.1f}")
print(f"""
  Key insight:
    Hierarchical clustering reveals natural groupings WITHOUT
    needing to specify K in advance. Cut the dendrogram at
    different heights to get different granularities.
    CCC > 0.75 confirms the tree structure is a reliable
    representation of the original pairwise distances.
""")
print("=" * 65)
