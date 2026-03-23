"""
J48 / C4.5 Classification Algorithm — Student Academic Performance
===================================================================
J48 is WEKA's implementation of Quinlan's C4.5 algorithm.
It improves on ID3 in three key ways:

  1. Gain Ratio  = Info Gain / Split Info
                   (removes bias toward high-cardinality attributes)
  2. Continuous  features handled via binary threshold splits
  3. Pruning     via Reduced Error Pruning (REP) to reduce overfitting

Dataset: 20 student records with:
  - StudyHours   (numeric, hrs/week)
  - Attendance   (numeric, %)
  - Assignments  (categorical: Excellent / Good / Poor)
  - PrevGrade    (numeric, previous semester %)
  - SleepHours   (numeric, hrs/day)
  → Target: Performance  (Pass / Merit / Fail)

Steps:
  1.  Dataset generation
  2.  Entropy & Gain Ratio  (the J48 split criterion)
  3.  Continuous attribute threshold finding
  4.  Recursive J48 tree build
  5.  Reduced Error Pruning
  6.  Tree display
  7.  Rule extraction
  8.  Prediction on new students
  9.  Confusion matrix & accuracy
  10. Comparison: unpruned vs pruned tree
"""

import math
import random
from collections import Counter
from copy import deepcopy

random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. STUDENT DATASET
# ══════════════════════════════════════════════════════════

FEATURES = ["StudyHours", "Attendance", "Assignments", "PrevGrade", "SleepHours"]
CONTINUOUS = {"StudyHours", "Attendance", "PrevGrade", "SleepHours"}
CATEGORICAL = {"Assignments"}
TARGET = "Performance"
CLASSES = ["Fail", "Pass", "Merit"]

# 25 labelled student records with realistic overlap
raw = [
    # StudyHrs  Attend%  Assignments  PrevGrade  Sleep  Performance
    (2,         55,      "Poor",       42,        5,     "Fail"),
    (3,         60,      "Poor",       48,        6,     "Fail"),
    (5,         70,      "Good",       55,        7,     "Pass"),
    (6,         75,      "Good",       60,        7,     "Pass"),
    (4,         65,      "Poor",       50,        6,     "Fail"),
    (8,         85,      "Excellent",  72,        7,     "Merit"),
    (9,         90,      "Excellent",  80,        8,     "Merit"),
    (7,         80,      "Good",       68,        7,     "Pass"),
    (2,         50,      "Poor",       40,        5,     "Fail"),
    (6,         72,      "Good",       63,        7,     "Pass"),
    (10,        92,      "Excellent",  88,        8,     "Merit"),
    (5,         68,      "Good",       58,        7,     "Pass"),
    (3,         55,      "Poor",       45,        6,     "Fail"),
    (8,         82,      "Excellent",  75,        8,     "Merit"),
    (4,         63,      "Good",       52,        6,     "Pass"),
    (1,         45,      "Poor",       35,        5,     "Fail"),
    (7,         78,      "Excellent",  70,        7,     "Merit"),
    (6,         74,      "Good",       65,        7,     "Pass"),
    (9,         88,      "Excellent",  82,        8,     "Merit"),
    (3,         58,      "Poor",       44,        6,     "Fail"),
    # overlap: same Assignments, different outcome based on StudyHours/PrevGrade
    (8,         83,      "Good",       74,        8,     "Merit"),
    (3,         58,      "Good",       46,        6,     "Fail"),
    (6,         71,      "Excellent",  61,        7,     "Pass"),
    (5,         66,      "Good",       54,        6,     "Pass"),
    (4,         62,      "Excellent",  49,        6,     "Fail"),
]

data = [
    {FEATURES[i]: v for i, v in enumerate(row[:5])} | {TARGET: row[5]}
    for row in raw
]

# ── Print dataset ─────────────────────────────────────────
print("=" * 70)
print("J48 / C4.5 — STUDENT ACADEMIC PERFORMANCE CLASSIFICATION")
print("=" * 70)
print(f"\nDataset: {len(data)} students, {len(FEATURES)} features → {TARGET}")
print(f"Classes: {CLASSES}\n")
print(f"{'#':<4}{'Study':<8}{'Attend':<9}{'Assign':<13}{'PrevGrade':<12}{'Sleep':<8}{TARGET}")
print("─" * 62)
for i, r in enumerate(data, 1):
    print(f"{i:<4}{r['StudyHours']:<8}{r['Attendance']:<9}{r['Assignments']:<13}"
          f"{r['PrevGrade']:<12}{r['SleepHours']:<8}{r[TARGET]}")


# ══════════════════════════════════════════════════════════
# 2. ENTROPY & GAIN RATIO  (core J48 criterion)
# ══════════════════════════════════════════════════════════

def entropy(subset):
    """H(S) = -Σ p·log₂(p)"""
    if not subset:
        return 0.0
    n = len(subset)
    counts = Counter(r[TARGET] for r in subset)
    return -sum((c/n)*math.log2(c/n) for c in counts.values() if c > 0)


def split_info(partitions, n_total):
    """
    SplitInfo(S, A) = -Σ (|Sv|/|S|)·log₂(|Sv|/|S|)
    Penalises splits that create many small partitions.
    """
    si = 0.0
    for part in partitions:
        if part:
            p = len(part) / n_total
            si -= p * math.log2(p)
    return si if si != 0 else 1e-9   # avoid division by zero


def gain_ratio_categorical(subset, feature):
    """
    Gain Ratio for a categorical feature.
    GainRatio(S,A) = InfoGain(S,A) / SplitInfo(S,A)
    """
    n = len(subset)
    H = entropy(subset)
    values = sorted(set(r[feature] for r in subset))
    partitions = [[r for r in subset if r[feature] == v] for v in values]
    weighted_H = sum(len(p)/n * entropy(p) for p in partitions)
    info_gain   = H - weighted_H
    si          = split_info(partitions, n)
    return info_gain / si, None   # (gain_ratio, threshold=None)


def best_threshold_and_gain_ratio(subset, feature):
    """
    J48 continuous split: try every midpoint between adjacent sorted values.
    Returns (best_gain_ratio, best_threshold).
    Binary split: S ≤ t  vs  S > t
    """
    values = sorted(set(r[feature] for r in subset))
    if len(values) < 2:
        return 0.0, None
    H = entropy(subset)
    n = len(subset)
    best_gr, best_t = -1, None
    for i in range(len(values) - 1):
        t = (values[i] + values[i+1]) / 2
        left  = [r for r in subset if r[feature] <= t]
        right = [r for r in subset if r[feature] >  t]
        if not left or not right:
            continue
        weighted_H = len(left)/n*entropy(left) + len(right)/n*entropy(right)
        ig = H - weighted_H
        si = split_info([left, right], n)
        gr = ig / si
        if gr > best_gr:
            best_gr, best_t = gr, t
    return best_gr, best_t


def best_split(subset, features):
    """
    Evaluate all features and return the best (feature, threshold, gain_ratio).
    threshold=None means categorical split.
    """
    best_gr, best_feat, best_thresh = -1, None, None
    for feat in features:
        if feat in CONTINUOUS:
            gr, t = best_threshold_and_gain_ratio(subset, feat)
        else:
            gr, t = gain_ratio_categorical(subset, feat)
        if gr > best_gr:
            best_gr, best_feat, best_thresh = gr, feat, t
    return best_feat, best_thresh, best_gr


# ── Demo: gain ratio at root ──────────────────────────────
print("\n" + "─" * 70)
print("STEP 2 — GAIN RATIO FOR ALL FEATURES (root split)")
print("─" * 70)
print(f"\nRoot entropy H(S) = {entropy(data):.4f} bits")
print(f"\n{'Feature':<14}  {'GainRatio':>10}  {'Threshold':>12}  {'Type'}")
print("─" * 52)
for feat in FEATURES:
    if feat in CONTINUOUS:
        gr, t = best_threshold_and_gain_ratio(data, feat)
        ftype = "continuous"
        tstr  = f"{t:.1f}"
    else:
        gr, t = gain_ratio_categorical(data, feat)
        ftype = "categorical"
        tstr  = "—"
    print(f"{feat:<14}  {gr:>10.4f}  {tstr:>12}  {ftype}")


# ══════════════════════════════════════════════════════════
# 3. J48 TREE NODE
# ══════════════════════════════════════════════════════════

class J48Node:
    def __init__(self, feature=None, threshold=None, label=None,
                 samples=0, class_dist=None):
        self.feature    = feature      # split attribute
        self.threshold  = threshold    # float for continuous, None for categorical
        self.label      = label        # class label at leaf
        self.children   = {}           # {value_or_side: J48Node}
        self.samples    = samples      # training instances reaching this node
        self.class_dist = class_dist or {}
        self.is_pruned  = False

    def is_leaf(self):
        return self.label is not None


# ══════════════════════════════════════════════════════════
# 4. RECURSIVE J48 TREE BUILD
# ══════════════════════════════════════════════════════════

MIN_SAMPLES = 2   # J48 min_instances_per_leaf (like WEKA default)

def build_j48(subset, features, depth=0, verbose=True):
    """
    Recursive J48 tree construction.
    Stops when:
      - All instances same class  → pure leaf
      - No features left          → majority leaf
      - Subset too small          → majority leaf
    Splits on best gain-ratio feature; handles continuous attributes.
    """
    indent   = "  " * depth
    n        = len(subset)
    dist     = dict(Counter(r[TARGET] for r in subset))
    majority = max(dist, key=dist.get)

    def make_leaf(label, reason):
        if verbose:
            print(f"{indent}[LEAF] {reason} → '{label}' | dist={dist}")
        return J48Node(label=label, samples=n, class_dist=dist)

    if n == 0:
        return make_leaf(majority, "empty")
    if len(set(r[TARGET] for r in subset)) == 1:
        return make_leaf(majority, f"pure ({n})")
    if not features or n <= MIN_SAMPLES:
        return make_leaf(majority, f"min_samples/no_features ({n})")

    feat, thresh, gr = best_split(subset, features)
    if feat is None or gr <= 0:
        return make_leaf(majority, f"no gain ({n})")

    node = J48Node(feature=feat, threshold=thresh, samples=n, class_dist=dist)

    if thresh is not None:   # continuous → binary split
        if verbose:
            print(f"{indent}[NODE d={depth}] {feat} <= {thresh:.1f}  "
                  f"GR={gr:.4f}  n={n}  dist={dist}")
        left  = [r for r in subset if r[feat] <= thresh]
        right = [r for r in subset if r[feat] >  thresh]
        remaining = [f for f in features if f != feat]
        if verbose:
            print(f"{indent}  <=  branch: {len(left)} instances")
        node.children["<="] = build_j48(left,  remaining, depth+1, verbose)
        if verbose:
            print(f"{indent}  >   branch: {len(right)} instances")
        node.children[">"]  = build_j48(right, remaining, depth+1, verbose)

    else:                    # categorical → multi-way split
        if verbose:
            print(f"{indent}[NODE d={depth}] {feat} (categorical)  "
                  f"GR={gr:.4f}  n={n}  dist={dist}")
        all_vals  = sorted(set(r[feat] for r in data))
        remaining = [f for f in features if f != feat]
        for val in all_vals:
            branch = [r for r in subset if r[feat] == val]
            if verbose:
                print(f"{indent}  '{val}' branch: {len(branch)} instances")
            node.children[val] = build_j48(branch, remaining, depth+1, verbose)

    return node


print("\n" + "─" * 70)
print("STEP 3 & 4 — BUILDING UNPRUNED J48 TREE (verbose trace)")
print("─" * 70 + "\n")

tree_unpruned = build_j48(data, FEATURES, verbose=True)


# ══════════════════════════════════════════════════════════
# 5. REDUCED ERROR PRUNING (REP)
# ══════════════════════════════════════════════════════════

def predict_one(node, instance):
    """Traverse tree to classify one instance."""
    if node.is_leaf():
        return node.label
    feat   = node.feature
    thresh = node.threshold
    if thresh is not None:
        side = "<=" if instance[feat] <= thresh else ">"
        child = node.children.get(side)
    else:
        val   = instance.get(feat, "")
        child = node.children.get(val)
    if child is None:
        # unseen value → return majority from current node's distribution
        return max(node.class_dist, key=node.class_dist.get)
    return predict_one(child, instance)


def evaluate(node, dataset):
    """Return accuracy (0–1) on a dataset."""
    if not dataset:
        return 0.0
    correct = sum(1 for r in dataset if predict_one(node, r) == r[TARGET])
    return correct / len(dataset)


def rep_prune(node, val_set):
    """
    Reduced Error Pruning:
    Bottom-up: replace an internal node with a leaf if accuracy on
    validation set does not decrease.
    """
    if node.is_leaf():
        return node

    # recurse first (post-order)
    for key in list(node.children):
        node.children[key] = rep_prune(node.children[key], val_set)

    # try collapsing this node into a majority leaf
    majority = max(node.class_dist, key=node.class_dist.get)
    acc_before = evaluate(node, val_set)

    leaf_candidate = J48Node(label=majority, samples=node.samples,
                              class_dist=node.class_dist)
    acc_after = evaluate(leaf_candidate, val_set)

    if acc_after >= acc_before:
        return leaf_candidate   # prune: replace with leaf
    return node                  # keep internal node


# Split into 80% train / 20% validation for pruning
random.shuffle(data)
split_idx  = int(0.8 * len(data))
train_data = data[:split_idx]
val_data   = data[split_idx:]

print("\n" + "─" * 70)
print("STEP 5 — REDUCED ERROR PRUNING")
print("─" * 70)
print(f"  Train set : {len(train_data)} instances")
print(f"  Val set   : {len(val_data)} instances (used for pruning)\n")

tree_pruned_base = build_j48(train_data, FEATURES, verbose=False)
tree_pruned = rep_prune(deepcopy(tree_pruned_base), val_data)

acc_unpruned = evaluate(tree_unpruned, data)
acc_pruned   = evaluate(tree_pruned,   data)
print(f"  Accuracy unpruned (full data): {acc_unpruned*100:.1f}%")
print(f"  Accuracy pruned   (full data): {acc_pruned*100:.1f}%")


# ══════════════════════════════════════════════════════════
# 6. TREE DISPLAY
# ══════════════════════════════════════════════════════════

def count_nodes(node):
    if node.is_leaf():
        return 1
    return 1 + sum(count_nodes(c) for c in node.children.values())

def count_leaves(node):
    if node.is_leaf():
        return 1
    return sum(count_leaves(c) for c in node.children.values())

def print_j48(node, prefix="", branch="", is_last=True):
    """Pretty-print J48 tree in WEKA-style text format."""
    connector = "└── " if is_last else "├── "
    if node.is_leaf():
        print(f"{prefix}{connector if branch else ''}{branch}: [{node.label}]"
              f"  ({node.samples} samples, dist={node.class_dist})")
        return

    feat   = node.feature
    thresh = node.threshold
    label  = f"{feat} <= {thresh:.1f}" if thresh else feat
    header = f"{prefix}{connector if branch else ''}{branch}"
    if header.strip():
        print(f"{header} → split on [{label}]  (n={node.samples}, dist={node.class_dist})")
    else:
        print(f"[{label}]  (n={node.samples}, dist={node.class_dist})")

    children  = list(node.children.items())
    new_pfx   = prefix + ("    " if is_last else "│   ")
    for i, (val, child) in enumerate(children):
        last = (i == len(children) - 1)
        branch_label = (f"{feat} <= {thresh:.1f}" if val == "<=" else
                        f"{feat} > {thresh:.1f}"  if val == ">"  else
                        f"{feat} = {val}")
        print_j48(child, new_pfx, branch_label, last)


print("\n" + "─" * 70)
print("STEP 6 — PRUNED J48 TREE STRUCTURE")
print("─" * 70)
print_j48(tree_pruned)
print(f"\n  Nodes  : {count_nodes(tree_pruned)}")
print(f"  Leaves : {count_leaves(tree_pruned)}")


# ══════════════════════════════════════════════════════════
# 7. RULE EXTRACTION
# ══════════════════════════════════════════════════════════

def extract_rules(node, path=None):
    if path is None:
        path = []
    if node.is_leaf():
        return [(list(path), node.label, node.samples)]
    rules = []
    for val, child in node.children.items():
        feat   = node.feature
        thresh = node.threshold
        if thresh is not None:
            cond = (f"{feat} <= {thresh:.1f}" if val == "<="
                    else f"{feat} > {thresh:.1f}")
        else:
            cond = f"{feat} = {val}"
        rules.extend(extract_rules(child, path + [cond]))
    return rules


print("\n" + "─" * 70)
print("STEP 7 — EXTRACTED CLASSIFICATION RULES")
print("─" * 70)

rules = extract_rules(tree_pruned)
for i, (conditions, label, n) in enumerate(rules, 1):
    cond_str = "\n         AND ".join(conditions)
    print(f"\nRule {i:02d}: IF {cond_str}")
    print(f"         THEN Performance = {label}  (covers {n} training sample(s))")


# ══════════════════════════════════════════════════════════
# 8. PREDICTION ON NEW STUDENTS
# ══════════════════════════════════════════════════════════

new_students = [
    {"StudyHours": 8, "Attendance": 88, "Assignments": "Excellent", "PrevGrade": 78, "SleepHours": 8},
    {"StudyHours": 2, "Attendance": 52, "Assignments": "Poor",      "PrevGrade": 41, "SleepHours": 5},
    {"StudyHours": 5, "Attendance": 70, "Assignments": "Good",      "PrevGrade": 60, "SleepHours": 7},
    {"StudyHours": 9, "Attendance": 91, "Assignments": "Excellent", "PrevGrade": 85, "SleepHours": 8},
    {"StudyHours": 3, "Attendance": 61, "Assignments": "Poor",      "PrevGrade": 49, "SleepHours": 6},
]

print("\n" + "─" * 70)
print("STEP 8 — PREDICTING NEW STUDENT PERFORMANCE")
print("─" * 70)
print(f"\n{'#':<4}{'Study':<8}{'Attend':<9}{'Assign':<13}{'PrevGrade':<12}{'Sleep':<8}{'Predicted'}")
print("─" * 58)
for i, s in enumerate(new_students, 1):
    pred = predict_one(tree_pruned, s)
    print(f"{i:<4}{s['StudyHours']:<8}{s['Attendance']:<9}{s['Assignments']:<13}"
          f"{s['PrevGrade']:<12}{s['SleepHours']:<8}{pred}")


# ══════════════════════════════════════════════════════════
# 9. CONFUSION MATRIX & METRICS
# ══════════════════════════════════════════════════════════

def confusion_matrix_and_metrics(node, dataset, classes):
    """Return confusion matrix + per-class precision, recall, F1."""
    cm = {a: {p: 0 for p in classes} for a in classes}
    for r in dataset:
        actual = r[TARGET]
        pred   = predict_one(node, r)
        cm[actual][pred] += 1

    print("\n  Confusion Matrix (rows=Actual, cols=Predicted):")
    header = f"  {'':>8}" + "".join(f"{c:>8}" for c in classes)
    print(header)
    print("  " + "─" * (8 + 8 * len(classes)))
    for actual in classes:
        row = f"  {actual:>8}" + "".join(f"{cm[actual][pred]:>8}" for pred in classes)
        print(row)

    print("\n  Per-class metrics:")
    print(f"  {'Class':<10}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Support':>10}")
    print("  " + "─" * 46)
    total_correct = 0
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[a][c] for a in classes if a != c)
        fn = sum(cm[c][p] for p in classes if p != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2*precision*recall / (precision+recall)
                     if (precision+recall) > 0 else 0)
        support   = sum(cm[c].values())
        total_correct += tp
        print(f"  {c:<10}{precision:>12.3f}{recall:>10.3f}{f1:>10.3f}{support:>10}")

    accuracy = total_correct / len(dataset)
    print(f"\n  Overall Accuracy : {total_correct}/{len(dataset)} = {accuracy*100:.1f}%")
    return accuracy


print("\n" + "─" * 70)
print("STEP 9 — CONFUSION MATRIX & EVALUATION METRICS (pruned tree, full data)")
print("─" * 70)
confusion_matrix_and_metrics(tree_pruned, data, CLASSES)


# ══════════════════════════════════════════════════════════
# 10. UNPRUNED vs PRUNED COMPARISON
# ══════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("STEP 10 — UNPRUNED vs PRUNED TREE COMPARISON")
print("═" * 70)
print(f"\n{'Metric':<30}{'Unpruned':>12}{'Pruned':>12}")
print("─" * 54)
print(f"{'Total nodes':<30}{count_nodes(tree_unpruned):>12}{count_nodes(tree_pruned):>12}")
print(f"{'Leaf nodes':<30}{count_leaves(tree_unpruned):>12}{count_leaves(tree_pruned):>12}")
print(f"{'Rules extracted':<30}{len(extract_rules(tree_unpruned)):>12}{len(extract_rules(tree_pruned)):>12}")
print(f"{'Training accuracy':<30}{evaluate(tree_unpruned, data)*100:>11.1f}%"
      f"{evaluate(tree_pruned, data)*100:>11.1f}%")
print(f"{'Val accuracy':<30}{evaluate(tree_unpruned, val_data)*100:>11.1f}%"
      f"{evaluate(tree_pruned, val_data)*100:>11.1f}%")
print(f"\nGain Ratio formula : GainRatio(S,A) = InfoGain(S,A) / SplitInfo(S,A)")
print(f"SplitInfo(S,A)     = -Σ (|Sv|/|S|) · log₂(|Sv|/|S|)")
print(f"Continuous split   : best binary threshold t where S≤t vs S>t")
print(f"Pruning strategy   : Reduced Error Pruning (bottom-up, val set)")

print("\n" + "═" * 70)
print("J48 ALGORITHM COMPLETE")
print("═" * 70)
