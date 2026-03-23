"""
FP-Growth Algorithm — Contact Lenses Dataset
==============================================
Dataset : contact-lenses.arff  (WEKA classic, 24 instances)
Attributes:
  age              : young | pre-presbyopic | presbyopic
  spectacle-prescrip : myope | hypermetrope
  astigmatism      : no | yes
  tear-prod-rate   : reduced | normal
  contact-lenses   : none | soft | hard   ← class label

Algorithm steps:
  1.  Parse .arff and encode as item transactions
  2.  Scan 1 — count item frequencies, prune below min_support
  3.  Scan 2 — build FP-Tree (compressed prefix tree)
  4.  Mine FP-Tree recursively via conditional pattern bases
  5.  Collect all frequent itemsets
  6.  Generate association rules (support, confidence, lift, conviction)
  7.  Filter & rank rules
  8.  Business/clinical insights
"""

from collections import defaultdict, OrderedDict
from itertools import combinations


# ══════════════════════════════════════════════════════════
# 1. CONTACT LENSES DATASET  (embedded — mirrors the .arff)
# ══════════════════════════════════════════════════════════

ARFF_HEADER = """
@relation contact-lenses

@attribute age              {young, pre-presbyopic, presbyopic}
@attribute spectacle-prescrip {myope, hypermetrope}
@attribute astigmatism      {no, yes}
@attribute tear-prod-rate   {reduced, normal}
@attribute contact-lenses   {soft, hard, none}

@data
"""

# All 24 instances from contact-lenses.arff
raw_data = [
    ("young",          "myope",        "no",  "reduced", "none"),
    ("young",          "myope",        "no",  "normal",  "soft"),
    ("young",          "myope",        "yes", "reduced", "none"),
    ("young",          "myope",        "yes", "normal",  "hard"),
    ("young",          "hypermetrope", "no",  "reduced", "none"),
    ("young",          "hypermetrope", "no",  "normal",  "soft"),
    ("young",          "hypermetrope", "yes", "reduced", "none"),
    ("young",          "hypermetrope", "yes", "normal",  "hard"),
    ("pre-presbyopic", "myope",        "no",  "reduced", "none"),
    ("pre-presbyopic", "myope",        "no",  "normal",  "soft"),
    ("pre-presbyopic", "myope",        "yes", "reduced", "none"),
    ("pre-presbyopic", "myope",        "yes", "normal",  "hard"),
    ("pre-presbyopic", "hypermetrope", "no",  "reduced", "none"),
    ("pre-presbyopic", "hypermetrope", "no",  "normal",  "soft"),
    ("pre-presbyopic", "hypermetrope", "yes", "reduced", "none"),
    ("pre-presbyopic", "hypermetrope", "yes", "normal",  "none"),
    ("presbyopic",     "myope",        "no",  "reduced", "none"),
    ("presbyopic",     "myope",        "no",  "normal",  "none"),
    ("presbyopic",     "myope",        "yes", "reduced", "none"),
    ("presbyopic",     "myope",        "yes", "normal",  "hard"),
    ("presbyopic",     "hypermetrope", "no",  "reduced", "none"),
    ("presbyopic",     "hypermetrope", "no",  "normal",  "soft"),
    ("presbyopic",     "hypermetrope", "yes", "reduced", "none"),
    ("presbyopic",     "hypermetrope", "yes", "normal",  "none"),
]

ATTRS = ["age", "prescription", "astigmatism", "tear_rate", "lenses"]
N = len(raw_data)

# ── Convert each row to a set of attribute=value items ────
# e.g. ("young","myope","no","reduced","none")
#   → {"age=young","presc=myope","astig=no","tear=reduced","lens=none"}
PREFIX = ["age", "presc", "astig", "tear", "lens"]

def encode_transaction(row):
    return frozenset(f"{PREFIX[i]}={v}" for i, v in enumerate(row))

transactions = [encode_transaction(r) for r in raw_data]

print("=" * 65)
print("FP-GROWTH ALGORITHM — CONTACT LENSES DATASET")
print("=" * 65)
print(f"\nInstances : {N}")
print(f"Attributes: age, spectacle-prescrip, astigmatism, "
      f"tear-prod-rate, contact-lenses\n")
print(f"{'#':<4} {'Age':<18} {'Prescrip':<14} {'Astig':<7} "
      f"{'Tear':<10} {'Lenses'}")
print("─" * 60)
for i, r in enumerate(raw_data, 1):
    print(f"{i:<4} {r[0]:<18} {r[1]:<14} {r[2]:<7} {r[3]:<10} {r[4]}")


# ══════════════════════════════════════════════════════════
# 2. SCAN 1 — ITEM FREQUENCY COUNT
# ══════════════════════════════════════════════════════════

MIN_SUPPORT    = 0.25   # item must appear in ≥ 25% of transactions = 6/24
MIN_CONFIDENCE = 0.70
MIN_LIFT       = 1.10
MIN_SUP_COUNT  = int(MIN_SUPPORT * N)   # absolute count threshold

def item_support(item, transactions):
    return sum(1 for t in transactions if item in t) / N

# Count all individual items
item_counts = defaultdict(int)
for t in transactions:
    for item in t:
        item_counts[item] += 1

# Keep only frequent items, sorted by frequency desc (FP-Tree ordering)
freq_items = {item: cnt for item, cnt in item_counts.items()
              if cnt >= MIN_SUP_COUNT}
ordered_items = sorted(freq_items, key=lambda x: -freq_items[x])

print(f"\n{'─'*65}")
print(f"SCAN 1 — ITEM FREQUENCY  (min_support={MIN_SUPPORT:.0%} = {MIN_SUP_COUNT}/{N})")
print(f"{'─'*65}")
print(f"\n{'Item':<28}  {'Count':>6}  {'Support':>9}  Bar")
print("─" * 58)
for item in ordered_items:
    cnt  = item_counts[item]
    sup  = cnt / N
    bar  = "█" * int(sup * 24)
    mark = "  ✓ frequent" if cnt >= MIN_SUP_COUNT else "  ✗ pruned"
    print(f"{item:<28}  {cnt:>6}  {sup:>9.3f}  {bar}{mark}")
pruned = [i for i, c in item_counts.items() if c < MIN_SUP_COUNT]
print(f"\nFrequent items: {len(freq_items)}"
      f"  |  Pruned (below threshold): {len(pruned)}")
if pruned:
    print(f"Pruned: {pruned}")


# ══════════════════════════════════════════════════════════
# 3. FP-TREE DATA STRUCTURE
# ══════════════════════════════════════════════════════════

class FPNode:
    """A single node in the FP-Tree."""
    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item, count=0, parent=None):
        self.item      = item       # item name (str) or None for root
        self.count     = count      # support count
        self.parent    = parent
        self.children  = {}         # { item: FPNode }
        self.node_link = None       # link to next node with same item (header table)

    def increment(self, count=1):
        self.count += count


class FPTree:
    """
    FP-Tree: compressed representation of the transaction database.
    Header table maps item → first node in the linked list of all
    nodes with that item (enables fast prefix-path retrieval).
    """
    def __init__(self, ordered_items, min_sup_count):
        self.root          = FPNode(None)
        self.header        = OrderedDict()   # item → first FPNode
        self.min_sup_count = min_sup_count
        # Initialise header table in frequency order (most frequent first)
        for item in ordered_items:
            self.header[item] = None

    def insert_transaction(self, ordered_transaction, count=1):
        """Insert one (sorted) transaction path into the tree."""
        node = self.root
        for item in ordered_transaction:
            if item not in self.header:
                continue                     # skip infrequent items
            if item in node.children:
                node.children[item].increment(count)
            else:
                new_node = FPNode(item, count, parent=node)
                node.children[item] = new_node
                # append to header linked list
                self._link_node(item, new_node)
            node = node.children[item]

    def _link_node(self, item, node):
        """Add node to end of header linked list for item."""
        if self.header[item] is None:
            self.header[item] = node
        else:
            curr = self.header[item]
            while curr.node_link is not None:
                curr = curr.node_link
            curr.node_link = node

    def nodes(self, item):
        """Yield all nodes with this item via the linked list."""
        node = self.header.get(item)
        while node is not None:
            yield node
            node = node.node_link

    def is_single_path(self):
        """True if the tree is a single chain (no branching)."""
        node = self.root
        while node.children:
            if len(node.children) > 1:
                return False
            node = next(iter(node.children.values()))
        return True

    def prefix_paths(self, item):
        """
        Return all prefix paths ending at nodes with this item.
        Each path is a list of (item, count) pairs, count = node.count.
        """
        paths = []
        for node in self.nodes(item):
            path  = []
            count = node.count
            curr  = node.parent
            while curr.item is not None:
                path.append(curr.item)
                curr = curr.parent
            if path:
                paths.append((list(reversed(path)), count))
        return paths


# ══════════════════════════════════════════════════════════
# 4. SCAN 2 — BUILD FP-TREE
# ══════════════════════════════════════════════════════════

def sort_transaction(transaction, ordered_items):
    """
    Keep only frequent items and sort by global frequency (desc).
    This canonical ordering ensures shared prefixes are maximised.
    """
    item_order = {item: i for i, item in enumerate(ordered_items)}
    return sorted(
        [i for i in transaction if i in item_order],
        key=lambda x: item_order[x]
    )

tree = FPTree(ordered_items, MIN_SUP_COUNT)
for t in transactions:
    sorted_t = sort_transaction(t, ordered_items)
    tree.insert_transaction(sorted_t)

print(f"\n{'─'*65}")
print("SCAN 2 — FP-TREE BUILT")
print(f"{'─'*65}")
print(f"\nHeader table (item → node count):")
print(f"  {'Item':<28}  {'Nodes':>6}  {'Total count':>12}")
print("─" * 52)
for item in ordered_items:
    nodes   = list(tree.nodes(item))
    tot_cnt = sum(nd.count for nd in nodes)
    print(f"  {item:<28}  {len(nodes):>6}  {tot_cnt:>12}")


def print_fptree(node, prefix="", is_last=True):
    """Recursively print the FP-Tree in ASCII art."""
    connector = "└── " if is_last else "├── "
    if node.item is not None:
        print(f"{prefix}{connector}[{node.item}: {node.count}]")
        prefix += "    " if is_last else "│   "
    children = list(node.children.values())
    for i, child in enumerate(children):
        print_fptree(child, prefix, i == len(children) - 1)

print("\nFP-Tree structure:")
print("  [root]")
root_children = list(tree.root.children.values())
for i, child in enumerate(root_children):
    print_fptree(child, "  ", i == len(root_children) - 1)


# ══════════════════════════════════════════════════════════
# 5. FP-GROWTH RECURSIVE MINING
# ══════════════════════════════════════════════════════════

def fp_growth(transactions, min_sup_count, prefix=frozenset(),
              frequent_itemsets=None, depth=0):
    """
    Recursive FP-Growth mining.
    For each frequent item (bottom of header table upward):
      1. Collect its conditional pattern base (prefix paths)
      2. Build conditional FP-Tree from those paths
      3. Recurse on conditional FP-Tree with extended prefix
      4. Emit frequent itemset = prefix ∪ {item}
    """
    if frequent_itemsets is None:
        frequent_itemsets = {}

    # Count items in these transactions
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[item] += 1

    # Keep only frequent items, sort by count desc
    local_freq = {k: v for k, v in item_counts.items() if v >= min_sup_count}
    if not local_freq:
        return frequent_itemsets

    local_order = sorted(local_freq, key=lambda x: -local_freq[x])

    # Process each item from least frequent to most frequent
    for item in reversed(local_order):
        new_itemset = prefix | frozenset([item])
        support_cnt = local_freq[item]
        frequent_itemsets[new_itemset] = support_cnt

        # Build conditional pattern base
        cond_transactions = []
        for t in transactions:
            if item in t:
                # prefix path = items before 'item' in this transaction
                sorted_t = sort_transaction(t, local_order)
                idx = sorted_t.index(item) if item in sorted_t else -1
                if idx > 0:
                    cond_transactions.append(frozenset(sorted_t[:idx]))

        # Recurse if conditional DB is non-empty
        if cond_transactions:
            fp_growth(cond_transactions, min_sup_count,
                      new_itemset, frequent_itemsets, depth + 1)

    return frequent_itemsets


print(f"\n{'─'*65}")
print("STEP 5 — FP-GROWTH RECURSIVE MINING")
print(f"{'─'*65}")

frequent_itemsets = fp_growth(transactions, MIN_SUP_COUNT)

# Convert counts to support ratios
fi_with_support = {fs: cnt / N for fs, cnt in frequent_itemsets.items()}

print(f"\nTotal frequent itemsets discovered: {len(fi_with_support)}")
print(f"\nAll frequent itemsets (sorted by size then support):")
print(f"  {'Itemset':<52}  {'Sup':>6}  {'Count':>6}")
print("─" * 68)
for fs, sup in sorted(fi_with_support.items(),
                      key=lambda x: (len(x[0]), -x[1])):
    label = "{" + ", ".join(sorted(fs)) + "}"
    cnt   = int(sup * N)
    print(f"  {label:<52}  {sup:>6.3f}  {cnt:>6}")


# ══════════════════════════════════════════════════════════
# 6. ASSOCIATION RULE GENERATION
# ══════════════════════════════════════════════════════════

def get_support(itemset):
    return fi_with_support.get(frozenset(itemset), 0.0)

def generate_rules(fi_with_support, min_confidence, min_lift=1.0):
    """Generate all valid association rules from frequent itemsets."""
    rules = []
    for itemset, sup_ab in fi_with_support.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for ant_tuple in combinations(sorted(itemset), size):
                ant = frozenset(ant_tuple)
                con = itemset - ant
                sup_a = get_support(ant)
                sup_b = get_support(con)
                if sup_a == 0 or sup_b == 0:
                    continue
                conf = sup_ab / sup_a
                lift = conf / sup_b
                conv = ((1 - sup_b) / (1 - conf)
                        if conf < 1.0 else float("inf"))
                leverage = round(sup_ab - sup_a * sup_b, 4)
                if conf >= min_confidence and lift >= min_lift:
                    rules.append({
                        "ant": ant, "con": con,
                        "support":    round(sup_ab, 4),
                        "confidence": round(conf, 4),
                        "lift":       round(lift, 4),
                        "conviction": round(conv, 4) if conv != float("inf") else float("inf"),
                        "leverage":   leverage,
                        "count":      int(sup_ab * N),
                    })
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    return rules

rules = generate_rules(fi_with_support, MIN_CONFIDENCE, MIN_LIFT)

def fmt(fs):
    return "{" + ", ".join(sorted(fs)) + "}"

print(f"\n{'─'*65}")
print("STEP 6 — ASSOCIATION RULES")
print(f"  min_confidence={MIN_CONFIDENCE:.0%}  min_lift={MIN_LIFT}  "
      f"→  {len(rules)} rules found")
print(f"{'─'*65}")
print(f"\n{'#':<4}  {'Rule':<52}  {'Sup':>6}  {'Conf':>7}  {'Lift':>6}")
print("─" * 80)
for i, r in enumerate(rules, 1):
    rstr = f"{fmt(r['ant'])} → {fmt(r['con'])}"
    print(f"{i:<4}  {rstr:<52}  {r['support']:>6.3f}  "
          f"{r['confidence']:>7.3f}  {r['lift']:>6.3f}")


# ══════════════════════════════════════════════════════════
# 7. RULES GROUPED BY CONSEQUENT (lens type)
# ══════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print("STEP 7 — RULES GROUPED BY LENS TYPE  (clinical view)")
print(f"{'─'*65}")

for lens_val in ["lens=hard", "lens=soft", "lens=none"]:
    subset = [r for r in rules if r["con"] == frozenset([lens_val])]
    if not subset:
        continue
    label = lens_val.replace("lens=", "").upper()
    print(f"\n  ── {label} LENSES ({len(subset)} rules) ──")
    print(f"  {'Conditions':<48}  {'Conf':>7}  {'Lift':>6}  {'n':>4}")
    print("  " + "─" * 64)
    for r in subset:
        conds = " ∧ ".join(sorted(r["ant"]))
        conv_str = (f"{r['conviction']:.2f}"
                    if r["conviction"] != float("inf") else "∞")
        print(f"  {conds:<48}  {r['confidence']:>7.3f}  "
              f"{r['lift']:>6.3f}  {r['count']:>4}")


# ══════════════════════════════════════════════════════════
# 8. DEEP-DIVE ON TOP 5 RULES
# ══════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print("STEP 8 — TOP 5 RULES  (metric deep-dive)")
print(f"{'─'*65}")

for i, r in enumerate(rules[:5], 1):
    sup_b = get_support(r["con"])
    print(f"\nRule {i}: {fmt(r['ant'])}  →  {fmt(r['con'])}")
    print(f"  Support    = {r['support']:.4f}  ({r['count']}/{N} patients)")
    print(f"  Confidence = {r['confidence']:.4f}  "
          f"({r['confidence']*100:.1f}% of matching patients get this outcome)")
    print(f"  Lift       = {r['lift']:.4f}  "
          f"({r['lift']:.2f}× better than random; baseline = {sup_b:.3f})")
    if r['conviction'] == float("inf"):
        print(f"  Conviction = ∞  (perfect rule, no counter-example)")
    else:
        print(f"  Conviction = {r['conviction']:.4f}")
    print(f"  Leverage   = {r['leverage']:+.4f}  "
          f"({'positive' if r['leverage']>0 else 'negative'} co-occurrence)")


# ══════════════════════════════════════════════════════════
# 9. CONDITIONAL PATTERN BASE TRACE (educational)
# ══════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print("STEP 9 — CONDITIONAL PATTERN BASE  (for 'lens=hard')")
print(f"{'─'*65}")
print("\nThis shows what FP-Growth mines when focusing on hard-lens patients.")

target_item = "lens=hard"
hard_tx = [t for t in transactions if target_item in t]
print(f"\nTransactions containing '{target_item}': {len(hard_tx)}")
print(f"\n{'#':<4}  Items (excluding '{target_item}')")
print("─" * 65)
for i, t in enumerate(hard_tx, 1):
    cond_items = sorted(t - {target_item})
    print(f"{i:<4}  {', '.join(cond_items)}")

# Conditional item counts
cond_counts = defaultdict(int)
for t in hard_tx:
    for item in t - {target_item}:
        cond_counts[item] += 1
print(f"\nConditional item frequencies (min_sup_count={MIN_SUP_COUNT}):")
print(f"  {'Item':<28}  {'Count':>6}  {'Frequent?'}")
print("─" * 48)
for item, cnt in sorted(cond_counts.items(), key=lambda x: -x[1]):
    tag = "✓" if cnt >= MIN_SUP_COUNT else "✗"
    print(f"  {item:<28}  {cnt:>6}  {tag}")


# ══════════════════════════════════════════════════════════
# 10. SUMMARY TABLE
# ══════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"\n  Dataset              : contact-lenses.arff  ({N} instances)")
print(f"  Min support          : {MIN_SUPPORT:.0%}  ({MIN_SUP_COUNT} instances)")
print(f"  Min confidence       : {MIN_CONFIDENCE:.0%}")
print(f"  Min lift             : {MIN_LIFT}")
print(f"\n  Frequent items       : {len(freq_items)} of {len(item_counts)}")
print(f"  Frequent itemsets    : {len(fi_with_support)}")
print(f"  Association rules    : {len(rules)}")
print(f"  Avg confidence       : "
      f"{sum(r['confidence'] for r in rules)/len(rules):.3f}")
print(f"  Avg lift             : "
      f"{sum(r['lift'] for r in rules)/len(rules):.3f}")

hard_rules = [r for r in rules if "lens=hard" in r["con"]]
soft_rules = [r for r in rules if "lens=soft" in r["con"]]
none_rules = [r for r in rules if "lens=none" in r["con"]]
print(f"\n  Rules → hard lenses  : {len(hard_rules)}")
print(f"  Rules → soft lenses  : {len(soft_rules)}")
print(f"  Rules → no lenses    : {len(none_rules)}")

print(f"\n  Key clinical findings:")
top = rules[0]
print(f"    Strongest rule: {fmt(top['ant'])} → {fmt(top['con'])}")
print(f"    (conf={top['confidence']:.2f}, lift={top['lift']:.2f})")

print(f"\n  FP-Growth advantages over Apriori:")
print(f"    • Only 2 database scans  (vs k scans in Apriori)")
print(f"    • No candidate generation  (avoids combinatorial explosion)")
print(f"    • Compressed FP-Tree fits in memory")
print(f"    • Same frequent itemsets as Apriori — just faster")
print(f"\n{'='*65}")
print("FP-GROWTH COMPLETE")
print(f"{'='*65}")
