"""
Statistical Description + Measures of Similarity & Dissimilarity
=================================================================
Covers:
  Part 1 — Descriptive statistics (central tendency, spread, shape)
  Part 2 — Similarity measures  (cosine, Pearson, Spearman, Jaccard)
  Part 3 — Dissimilarity measures (Euclidean, Manhattan, Minkowski,
            Chebyshev, Hamming, Mahalanobis, KL divergence)
  Part 4 — Pairwise distance / similarity matrix on a dataset
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ────────────────────────────────────────────────────
# SAMPLE DATASET  (5 customers: age, income, spend)
# ────────────────────────────────────────────────────

data = {
    "customer": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":      [25,      45,    35,      28,     52],
    "income":   [30000,   75000, 52000,   34000,  90000],
    "spend":    [1200,    4800,  3100,    1500,   5600],
}
df = pd.DataFrame(data)
print("=" * 60)
print("SAMPLE DATASET")
print("=" * 60)
print(df.to_string(index=False))


# ════════════════════════════════════════════════════
# PART 1 — DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════

def descriptive_stats(series: pd.Series) -> dict:
    """Compute a full descriptive profile of a numeric column."""
    n = len(series)
    mean   = series.mean()
    median = series.median()
    mode   = series.mode().iloc[0]
    var    = series.var(ddof=1)        # sample variance
    std    = series.std(ddof=1)        # sample std deviation
    q1     = series.quantile(0.25)
    q3     = series.quantile(0.75)
    iqr    = q3 - q1
    skew   = stats.skew(series)        # >0 right-skewed, <0 left-skewed
    kurt   = stats.kurtosis(series)    # excess kurtosis (0 = normal)
    cv     = (std / mean) * 100        # coefficient of variation (%)

    return {
        "n":        n,
        "mean":     round(mean, 2),
        "median":   round(median, 2),
        "mode":     round(mode, 2),
        "min":      round(series.min(), 2),
        "max":      round(series.max(), 2),
        "range":    round(series.max() - series.min(), 2),
        "variance": round(var, 2),
        "std_dev":  round(std, 2),
        "Q1":       round(q1, 2),
        "Q3":       round(q3, 2),
        "IQR":      round(iqr, 2),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "CV_%":     round(cv, 2),
    }


print("\n" + "=" * 60)
print("PART 1 — DESCRIPTIVE STATISTICS")
print("=" * 60)

numeric_cols = ["age", "income", "spend"]
for col in numeric_cols:
    print(f"\n── {col.upper()} ──")
    profile = descriptive_stats(df[col])
    for stat, val in profile.items():
        print(f"  {stat:<12} {val}")

# pandas .describe() for a quick overview
print("\n── pandas describe() ──")
print(df[numeric_cols].describe().round(2))


# ════════════════════════════════════════════════════
# PART 2 — SIMILARITY MEASURES
# ════════════════════════════════════════════════════

# We compare Alice (row 0) vs Bob (row 1) across [age, income, spend]
alice = np.array(df[numeric_cols].iloc[0], dtype=float)
bob   = np.array(df[numeric_cols].iloc[1], dtype=float)

print("\n" + "=" * 60)
print("PART 2 — SIMILARITY MEASURES")
print(f"  Alice: {alice}")
print(f"  Bob  : {bob}")
print("=" * 60)


# ── 2a. Cosine Similarity ────────────────────────────
# Measures the angle between two vectors.
# Range: [0, 1] → 1 = identical direction, 0 = orthogonal

def cosine_sim(a, b):
    dot     = np.dot(a, b)
    norm_a  = np.linalg.norm(a)
    norm_b  = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

cos = cosine_sim(alice, bob)
print(f"\n2a. Cosine Similarity        = {cos:.4f}  (1 = same direction)")

# sklearn method (operates on 2D arrays)
cos_sk = cosine_similarity([alice], [bob])[0][0]
print(f"    sklearn cosine similarity = {cos_sk:.4f}")


# ── 2b. Pearson Correlation ──────────────────────────
# Linear correlation between two variables.
# Range: [-1, 1] → 1 = perfect positive, -1 = perfect negative

corr_matrix = df[numeric_cols].corr(method="pearson")
print("\n2b. Pearson Correlation Matrix:")
print(corr_matrix.round(4))

r_age_income, p_val = pearsonr(df["age"], df["income"])
print(f"\n    age vs income: r = {r_age_income:.4f}, p-value = {p_val:.4f}")


# ── 2c. Spearman Rank Correlation ───────────────────
# Non-parametric correlation based on ranks.
# Robust to outliers and non-linear monotonic relationships.

rho, p_rho = spearmanr(df["age"], df["income"])
print(f"\n2c. Spearman Rank Correlation")
print(f"    age vs income: ρ = {rho:.4f}, p-value = {p_rho:.4f}")


# ── 2d. Jaccard Similarity (binary / set data) ───────
# Intersection / Union → 1 = identical sets, 0 = no overlap

def jaccard_similarity(set_a, set_b):
    a, b = set(set_a), set(set_b)
    return len(a & b) / len(a | b) if (a | b) else 0.0

purchases_alice = {"phone", "laptop", "headphones", "tablet"}
purchases_bob   = {"laptop", "headphones", "monitor", "keyboard"}

jac = jaccard_similarity(purchases_alice, purchases_bob)
print(f"\n2d. Jaccard Similarity (purchase sets)")
print(f"    Alice: {purchases_alice}")
print(f"    Bob  : {purchases_bob}")
print(f"    Jaccard = {jac:.4f}  (shared items / total items)")


# ── 2e. Dice Coefficient ─────────────────────────────
# Similar to Jaccard but weights matches more heavily.
# Range: [0, 1]

def dice_coefficient(set_a, set_b):
    a, b = set(set_a), set(set_b)
    return (2 * len(a & b)) / (len(a) + len(b)) if (len(a) + len(b)) else 0.0

dice = dice_coefficient(purchases_alice, purchases_bob)
print(f"\n2e. Dice Coefficient          = {dice:.4f}")


# ════════════════════════════════════════════════════
# PART 3 — DISSIMILARITY / DISTANCE MEASURES
# ════════════════════════════════════════════════════

# Normalize first so large-scale features (income) don't dominate
scaler      = StandardScaler()
normalized  = scaler.fit_transform(df[numeric_cols])
alice_n     = normalized[0]
bob_n       = normalized[1]

print("\n" + "=" * 60)
print("PART 3 — DISSIMILARITY / DISTANCE MEASURES")
print(f"  Alice (normalized): {alice_n.round(4)}")
print(f"  Bob   (normalized): {bob_n.round(4)}")
print("=" * 60)


# ── 3a. Euclidean Distance ───────────────────────────
# Straight-line distance in n-dimensional space.
# d = sqrt(Σ(aᵢ - bᵢ)²)

euclidean = np.linalg.norm(alice_n - bob_n)
print(f"\n3a. Euclidean Distance        = {euclidean:.4f}")
print(f"    (scipy check)             = {distance.euclidean(alice_n, bob_n):.4f}")


# ── 3b. Manhattan Distance ───────────────────────────
# Sum of absolute differences — "city block" distance.
# d = Σ|aᵢ - bᵢ|

manhattan = np.sum(np.abs(alice_n - bob_n))
print(f"\n3b. Manhattan Distance        = {manhattan:.4f}")
print(f"    (scipy check)             = {distance.cityblock(alice_n, bob_n):.4f}")


# ── 3c. Minkowski Distance ───────────────────────────
# Generalization: p=1 → Manhattan, p=2 → Euclidean, p→∞ → Chebyshev

print("\n3c. Minkowski Distance (varying p):")
for p in [1, 2, 3, np.inf]:
    d = distance.minkowski(alice_n, bob_n, p if p != np.inf else 1e9)
    label = {1: "= Manhattan", 2: "= Euclidean"}.get(p, "")
    print(f"    p={p:<4}  d = {d:.4f}  {label}")


# ── 3d. Chebyshev Distance ───────────────────────────
# Maximum absolute difference across all dimensions.
# d = max(|aᵢ - bᵢ|)

cheby = distance.chebyshev(alice_n, bob_n)
print(f"\n3d. Chebyshev Distance        = {cheby:.4f}  (max single-axis diff)")


# ── 3e. Hamming Distance (categorical / binary) ──────
# Fraction of positions where values differ.
# Perfect for comparing strings or binary vectors.

def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Sequences must be same length")
    return sum(x != y for x, y in zip(a, b)) / len(a)

str_alice = "10110101"
str_bob   = "10011101"
hamming = hamming_distance(str_alice, str_bob)
print(f"\n3e. Hamming Distance")
print(f"    Alice: {str_alice}")
print(f"    Bob  : {str_bob}")
print(f"    Hamming = {hamming:.4f}  ({int(hamming * len(str_alice))} of {len(str_alice)} positions differ)")


# ── 3f. Mahalanobis Distance ─────────────────────────
# Accounts for correlations between dimensions.
# Uses the inverse covariance matrix.
# Scale-invariant unlike Euclidean.

cov_matrix = np.cov(normalized.T)
inv_cov    = np.linalg.inv(cov_matrix)
diff       = alice_n - bob_n
mahal      = np.sqrt(diff @ inv_cov @ diff.T)
print(f"\n3f. Mahalanobis Distance      = {mahal:.4f}  (accounts for correlations)")


# ── 3g. Cosine Distance ──────────────────────────────
# 1 - cosine similarity. Range: [0, 2]

cos_dist = distance.cosine(alice_n, bob_n)
print(f"\n3g. Cosine Distance           = {cos_dist:.4f}  (= 1 - cosine similarity)")


# ── 3h. KL Divergence ────────────────────────────────
# Measures how one probability distribution P differs from Q.
# KL(P||Q) ≥ 0; not symmetric (KL(P||Q) ≠ KL(Q||P))

def kl_divergence(p, q, epsilon=1e-10):
    """KL(P || Q) — add epsilon to avoid log(0)."""
    p = np.array(p, dtype=float) + epsilon
    q = np.array(q, dtype=float) + epsilon
    p /= p.sum()   # normalize to probability distributions
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

dist_alice = [0.05, 0.20, 0.30, 0.25, 0.20]  # spend distribution Alice
dist_bob   = [0.01, 0.10, 0.25, 0.40, 0.24]  # spend distribution Bob

kl_pq = kl_divergence(dist_alice, dist_bob)
kl_qp = kl_divergence(dist_bob, dist_alice)
print(f"\n3h. KL Divergence")
print(f"    KL(Alice || Bob) = {kl_pq:.4f}")
print(f"    KL(Bob || Alice) = {kl_qp:.4f}  (asymmetric!)")


# ════════════════════════════════════════════════════
# PART 4 — PAIRWISE DISTANCE MATRIX
# ════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4 — PAIRWISE EUCLIDEAN DISTANCE MATRIX (normalized data)")
print("=" * 60)

names = df["customer"].tolist()
n = len(names)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = np.linalg.norm(normalized[i] - normalized[j])

dist_df = pd.DataFrame(dist_matrix, index=names, columns=names)
print(dist_df.round(3))

print("\nMost similar pair  :", end=" ")
mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
min_idx = np.unravel_index(np.where(mask, dist_matrix, np.inf).argmin(), dist_matrix.shape)
print(f"{names[min_idx[0]]} & {names[min_idx[1]]}  (distance = {dist_matrix[min_idx]:.3f})")

print("Most dissimilar pair:", end=" ")
max_idx = np.unravel_index(np.where(mask, dist_matrix, 0).argmax(), dist_matrix.shape)
print(f"{names[max_idx[0]]} & {names[max_idx[1]]}  (distance = {dist_matrix[max_idx]:.3f})")


# ════════════════════════════════════════════════════
# PART 5 — CORRELATION HEATMAP (text-based)
# ════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5 — PEARSON CORRELATION HEATMAP (numeric encoding)")
print("=" * 60)

corr = df[numeric_cols].corr().round(3)
print(corr)

print("\n── Interpretation ──")
for i, c1 in enumerate(numeric_cols):
    for j, c2 in enumerate(numeric_cols):
        if j > i:
            r = corr.loc[c1, c2]
            strength = (
                "very strong" if abs(r) > 0.9 else
                "strong"      if abs(r) > 0.7 else
                "moderate"    if abs(r) > 0.4 else
                "weak"
            )
            direction = "positive" if r > 0 else "negative"
            print(f"  {c1} vs {c2}: r={r} → {strength} {direction} correlation")


print("\n" + "=" * 60)
print("DONE — all statistical measures computed successfully.")
print("=" * 60)
