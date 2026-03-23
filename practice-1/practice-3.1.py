"""
ID3 Classification Algorithm — From Scratch
============================================
Dataset : Classic "Play Tennis" (14 instances, 4 features)
Target  : PlayTennis  (Yes / No)
Features: Outlook, Temperature, Humidity, Wind

Steps demonstrated:
  1. Dataset setup
  2. Entropy calculation        H(S) = -Σ p·log₂(p)
  3. Information Gain           Gain(S,A) = H(S) - Σ(|Sv|/|S|)·H(Sv)
  4. Recursive tree building    ID3 algorithm
  5. Tree printing              ASCII representation
  6. Prediction on new samples
  7. Rule extraction            IF-THEN classification rules
  8. Accuracy evaluation
"""

import math
from collections import Counter


# ══════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════

FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]
TARGET   = "PlayTennis"

dataset = [
    # Outlook      Temp    Humidity  Wind     PlayTennis
    ["Sunny",    "Hot",    "High",   "Weak",   "No"],
    ["Sunny",    "Hot",    "High",   "Strong", "No"],
    ["Overcast", "Hot",    "High",   "Weak",   "Yes"],
    ["Rain",     "Mild",   "High",   "Weak",   "Yes"],
    ["Rain",     "Cool",   "Normal", "Weak",   "Yes"],
    ["Rain",     "Cool",   "Normal", "Strong", "No"],
    ["Overcast", "Cool",   "Normal", "Strong", "Yes"],
    ["Sunny",    "Mild",   "High",   "Weak",   "No"],
    ["Sunny",    "Cool",   "Normal", "Weak",   "Yes"],
    ["Rain",     "Mild",   "Normal", "Weak",   "Yes"],
    ["Sunny",    "Mild",   "Normal", "Strong", "Yes"],
    ["Overcast", "Mild",   "High",   "Strong", "Yes"],
    ["Overcast", "Hot",    "Normal", "Weak",   "Yes"],
    ["Rain",     "Mild",   "High",   "Strong", "No"],
]

# Convert to list of dicts for readability
data = [dict(zip(FEATURES + [TARGET], row)) for row in dataset]

print("=" * 65)
print("ID3 DECISION TREE ALGORITHM — PLAY TENNIS DATASET")
print("=" * 65)

print(f"\nDataset: {len(data)} instances, {len(FEATURES)} features")
print(f"Target  : {TARGET}  (classes: Yes / No)")
print(f"\n{'#':<4}", end="")
for f in FEATURES:
    print(f"{f:<14}", end="")
print(TARGET)
print("─" * 65)
for i, row in enumerate(data, 1):
    print(f"{i:<4}", end="")
    for f in FEATURES:
        print(f"{row[f]:<14}", end="")
    print(row[TARGET])


# ══════════════════════════════════════════════════════════
# 2. ENTROPY
# ══════════════════════════════════════════════════════════

def entropy(subset: list, verbose: bool = False) -> float:
    """
    H(S) = -Σ (count/n) * log₂(count/n)
    Returns 0 for empty or pure subsets.
    """
    if not subset:
        return 0.0
    n = len(subset)
    counts = Counter(row[TARGET] for row in subset)
    h = 0.0
    parts = []
    for label, count in counts.items():
        p = count / n
        h -= p * math.log2(p)
        parts.append(f"  {label}: {count}/{n} = {p:.4f}  →  -{p:.4f}·log₂({p:.4f}) = {-p*math.log2(p):.4f}")
    if verbose:
        for line in parts:
            print(line)
    return round(h, 6)


print("\n" + "─" * 65)
print("STEP 2 — ENTROPY OF FULL DATASET H(S)")
print("─" * 65)

label_counts = Counter(row[TARGET] for row in data)
print(f"Class distribution: {dict(label_counts)}")
H_S = entropy(data, verbose=True)
print(f"  H(S) = {H_S:.4f} bits")


# ══════════════════════════════════════════════════════════
# 3. INFORMATION GAIN
# ══════════════════════════════════════════════════════════

def information_gain(subset: list, feature: str, verbose: bool = False) -> float:
    """
    Gain(S, A) = H(S) - Σ_v (|S_v| / |S|) · H(S_v)
    """
    n = len(subset)
    H = entropy(subset)
    values = set(row[feature] for row in subset)
    weighted_entropy = 0.0
    if verbose:
        print(f"  Feature: {feature}")
        print(f"  H(S) = {H:.4f}")
    for v in sorted(values):
        S_v = [row for row in subset if row[feature] == v]
        weight = len(S_v) / n
        H_v = entropy(S_v)
        weighted_entropy += weight * H_v
        if verbose:
            labels_v = Counter(row[TARGET] for row in S_v)
            print(f"    {v}: {len(S_v)} rows  {dict(labels_v)}  H={H_v:.4f}  weight={weight:.4f}  contribution={weight*H_v:.4f}")
    gain = H - weighted_entropy
    if verbose:
        print(f"  Gain = {H:.4f} - {weighted_entropy:.4f} = {gain:.4f}")
    return round(gain, 6)


print("\n" + "─" * 65)
print("STEP 3 — INFORMATION GAIN FOR ALL FEATURES (root level)")
print("─" * 65)

gains = {}
for feature in FEATURES:
    print(f"\n── {feature} ──")
    g = information_gain(data, feature, verbose=True)
    gains[feature] = g

best_root = max(gains, key=gains.get)
print(f"\n{'Feature':<15} {'Gain':>8}")
print("─" * 25)
for f, g in sorted(gains.items(), key=lambda x: -x[1]):
    marker = " ← BEST (root)" if f == best_root else ""
    print(f"{f:<15} {g:>8.4f}{marker}")


# ══════════════════════════════════════════════════════════
# 4. ID3 TREE BUILDING
# ══════════════════════════════════════════════════════════

class Node:
    """A node in the ID3 decision tree."""
    def __init__(self, feature=None, label=None):
        self.feature  = feature    # split attribute (internal node)
        self.label    = label      # class label (leaf node)
        self.children = {}         # { attribute_value: Node }

    def is_leaf(self):
        return self.label is not None


def id3(subset: list, features: list, depth: int = 0, parent_label: str = None) -> Node:
    """
    Recursive ID3 algorithm.
    Base cases:
      (a) All instances same class → leaf
      (b) No features left         → leaf (majority class)
      (c) Empty subset             → leaf (parent majority)
    Recursive: split on best information gain feature.
    """
    indent = "  " * depth

    # Empty subset → use parent's majority label
    if not subset:
        print(f"{indent}[LEAF] Empty subset → inheriting label '{parent_label}'")
        return Node(label=parent_label)

    labels = [row[TARGET] for row in subset]
    majority = Counter(labels).most_common(1)[0][0]

    # All same class → pure leaf
    if len(set(labels)) == 1:
        print(f"{indent}[LEAF] Pure — all '{labels[0]}' ({len(subset)} instances)")
        return Node(label=labels[0])

    # No features left → majority leaf
    if not features:
        print(f"{indent}[LEAF] No features left → majority '{majority}' ({Counter(labels)})")
        return Node(label=majority)

    # Pick best feature
    gains = {f: information_gain(subset, f) for f in features}
    best  = max(gains, key=gains.get)
    print(f"{indent}[NODE] depth={depth}  best='{best}'  gain={gains[best]:.4f}  "
          f"({len(subset)} instances, H={entropy(subset):.4f})")
    print(f"{indent}       gains: { {f: round(g,4) for f,g in gains.items()} }")

    node = Node(feature=best)
    remaining = [f for f in features if f != best]
    values = sorted(set(row[best] for row in data))   # use all known values

    for v in values:
        S_v = [row for row in subset if row[best] == v]
        print(f"{indent}  Branch {best}='{v}'  ({len(S_v)} instances)")
        node.children[v] = id3(S_v, remaining, depth + 1, majority)

    return node


print("\n" + "─" * 65)
print("STEP 4 — BUILDING THE ID3 TREE (recursive trace)")
print("─" * 65 + "\n")

tree = id3(data, FEATURES)


# ══════════════════════════════════════════════════════════
# 5. PRINT TREE
# ══════════════════════════════════════════════════════════

def print_tree(node: Node, indent: str = "", branch: str = ""):
    """ASCII-art tree printer."""
    if node.is_leaf():
        print(f"{indent}{'└── ' if branch else ''}{branch} → [{node.label}]")
    else:
        label = f"{branch} → " if branch else ""
        print(f"{indent}{label}[{node.feature}]")
        children = list(node.children.items())
        for i, (val, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            new_indent = indent + ("    " if is_last else "│   ")
            connector  = "└── " if is_last else "├── "
            print(f"{indent}{connector}{val}", end="")
            if child.is_leaf():
                print(f" → [{child.label}]")
            else:
                print(f" → [{child.feature}]")
                sub_children = list(child.children.items())
                for j, (sv, sc) in enumerate(sub_children):
                    sub_last = (j == len(sub_children) - 1)
                    sub_conn = "    └── " if sub_last else "    ├── "
                    print(f"{indent}{sub_conn}{sv} → [{sc.label if sc.is_leaf() else sc.feature}]")
                    if not sc.is_leaf():
                        for k, (ssv, ssc) in enumerate(sc.children.items()):
                            print(f"{indent}        {'└── ' if k==len(sc.children)-1 else '├── '}{ssv} → [{ssc.label}]")


print("\n" + "─" * 65)
print("STEP 5 — DECISION TREE STRUCTURE")
print("─" * 65)
print_tree(tree)


# ══════════════════════════════════════════════════════════
# 6. PREDICTION
# ══════════════════════════════════════════════════════════

def predict(node: Node, instance: dict) -> str:
    """Traverse the tree to classify one instance."""
    if node.is_leaf():
        return node.label
    val = instance.get(node.feature)
    if val not in node.children:
        # unseen value → return most common child leaf label
        for child in node.children.values():
            if child.is_leaf():
                return child.label
        return list(node.children.values())[0].label
    return predict(node.children[val], instance)


print("\n" + "─" * 65)
print("STEP 6 — PREDICTIONS")
print("─" * 65)

test_cases = [
    {"Outlook": "Sunny",    "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong"},
    {"Outlook": "Rain",     "Temperature": "Mild", "Humidity": "High",   "Wind": "Strong"},
    {"Outlook": "Overcast", "Temperature": "Hot",  "Humidity": "High",   "Wind": "Weak"},
    {"Outlook": "Sunny",    "Temperature": "Hot",  "Humidity": "High",   "Wind": "Weak"},
    {"Outlook": "Rain",     "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak"},
]

print(f"\n{'#':<4}{'Outlook':<12}{'Temp':<10}{'Humidity':<11}{'Wind':<10}{'Prediction'}")
print("─" * 58)
for i, tc in enumerate(test_cases, 1):
    pred = predict(tree, tc)
    print(f"{i:<4}{tc['Outlook']:<12}{tc['Temperature']:<10}{tc['Humidity']:<11}{tc['Wind']:<10}{pred}")


# ══════════════════════════════════════════════════════════
# 7. RULE EXTRACTION
# ══════════════════════════════════════════════════════════

def extract_rules(node: Node, path: list = None) -> list:
    """
    Depth-first traversal to collect all root-to-leaf paths
    as IF-THEN classification rules.
    """
    if path is None:
        path = []
    if node.is_leaf():
        return [(path[:], node.label)]
    rules = []
    for val, child in node.children.items():
        rules.extend(extract_rules(child, path + [(node.feature, val)]))
    return rules


print("\n" + "─" * 65)
print("STEP 7 — EXTRACTED IF-THEN CLASSIFICATION RULES")
print("─" * 65)

rules = extract_rules(tree)
for i, (conditions, outcome) in enumerate(rules, 1):
    cond_str = " AND ".join(f"{feat} = {val}" for feat, val in conditions)
    print(f"\nRule {i:02d}: IF {cond_str}")
    print(f"         THEN PlayTennis = {outcome}")


# ══════════════════════════════════════════════════════════
# 8. ACCURACY ON TRAINING DATA
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 8 — ACCURACY ON TRAINING DATA")
print("─" * 65)

correct = 0
print(f"\n{'#':<4}{'Actual':<10}{'Predicted':<12}{'Match'}")
print("─" * 35)
for i, row in enumerate(data, 1):
    actual = row[TARGET]
    predicted = predict(tree, row)
    match = "✓" if actual == predicted else "✗"
    if actual == predicted:
        correct += 1
    print(f"{i:<4}{actual:<10}{predicted:<12}{match}")

accuracy = correct / len(data) * 100
print(f"\nCorrect : {correct}/{len(data)}")
print(f"Accuracy: {accuracy:.1f}%")
print(f"\nNote: ID3 on its own training data always achieves 100% — it memorises")
print(f"the dataset. Use cross-validation or a held-out test set for real evaluation.")


# ══════════════════════════════════════════════════════════
# 9. INFORMATION GAIN SUMMARY TABLE
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SUMMARY — INFORMATION GAIN AT EACH SPLIT LEVEL")
print("=" * 65)

print(f"\nRoot dataset entropy H(S) = {entropy(data):.4f} bits")
print(f"\n{'Feature':<15}  {'Gain (root)':<14}  Notes")
print("─" * 55)
for f in FEATURES:
    g = information_gain(data, f)
    note = " ← chosen as root" if f == max(FEATURES, key=lambda x: information_gain(data, x)) else ""
    print(f"{f:<15}  {g:<14.4f}  {note}")

print("\nUnder Sunny branch:")
sunny = [r for r in data if r["Outlook"] == "Sunny"]
for f in ["Temperature", "Humidity", "Wind"]:
    g = information_gain(sunny, f)
    print(f"  {f:<13}  {g:.4f}")

print("\nUnder Rain branch:")
rain = [r for r in data if r["Outlook"] == "Rain"]
for f in ["Temperature", "Humidity", "Wind"]:
    g = information_gain(rain, f)
    print(f"  {f:<13}  {g:.4f}")

print("\n" + "=" * 65)
print("ID3 ALGORITHM COMPLETE")
print("=" * 65)
