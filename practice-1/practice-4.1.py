"""
Apriori Algorithm — Credit Card Transaction Association Rules
=============================================================
Finds spending patterns like:
  "Customers who buy Groceries + Fuel also buy Dining  (conf=78%, lift=2.3)"

Steps:
  1.  Credit card transaction dataset (20 customers, 10 spend categories)
  2.  Transaction encoding  (binary matrix)
  3.  Support counting  (item frequency scan)
  4.  Apriori candidate generation & pruning  (anti-monotone property)
  5.  Frequent itemset discovery  (all levels L1 → Lk)
  6.  Association rule generation  (support, confidence, lift, conviction)
  7.  Rule filtering & ranking
  8.  Business insights summary
"""

from itertools import combinations
from collections import defaultdict


# ══════════════════════════════════════════════════════════
# 1. CREDIT CARD TRANSACTION DATASET
# ══════════════════════════════════════════════════════════

# Spending categories a credit card customer uses in a month
CATEGORIES = [
    "Groceries", "Dining", "Fuel", "Travel", "Electronics",
    "Clothing", "Healthcare", "Streaming", "Shopping", "Utilities"
]

# 20 customer monthly transaction baskets (1 = used this category)
# Each row: [Groc, Dine, Fuel, Trav, Elec, Clot, Hlth, Strm, Shop, Util]
raw_transactions = [
    [1, 1, 1, 0, 0, 0, 1, 1, 0, 1],  # C01
    [1, 0, 1, 0, 0, 1, 0, 1, 1, 1],  # C02
    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],  # C03
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 1],  # C04
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # C05
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],  # C06
    [1, 1, 1, 0, 0, 0, 1, 1, 0, 1],  # C07
    [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],  # C08
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 0],  # C09
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],  # C10
    [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],  # C11
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],  # C12
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],  # C13
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],  # C14
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],  # C15
    [0, 1, 0, 0, 1, 0, 1, 1, 1, 0],  # C16
    [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],  # C17
    [1, 0, 0, 1, 1, 0, 1, 0, 1, 1],  # C18
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 0],  # C19
    [1, 1, 1, 0, 0, 0, 1, 1, 0, 1],  # C20
]

# Convert to list of frozensets
transactions = []
for row in raw_transactions:
    basket = frozenset(CATEGORIES[i] for i, v in enumerate(row) if v == 1)
    transactions.append(basket)

N = len(transactions)

print("=" * 68)
print("APRIORI ALGORITHM — CREDIT CARD TRANSACTION ANALYSIS")
print("=" * 68)
print(f"\nCustomers : {N}")
print(f"Categories: {len(CATEGORIES)}")
print(f"\nTransaction baskets:")
print(f"{'#':<6}{'Customer':<10}Items")
print("─" * 68)
for i, t in enumerate(transactions, 1):
    items = ", ".join(sorted(t))
    print(f"C{i:02d}   {'':6}{items}")


# ══════════════════════════════════════════════════════════
# 2. SUPPORT COUNTING UTILITY
# ══════════════════════════════════════════════════════════

def support_count(itemset, transactions):
    """Count how many transactions contain this itemset."""
    return sum(1 for t in transactions if itemset.issubset(t))

def support(itemset, transactions):
    """Support = count / N  (proportion of transactions)."""
    return support_count(itemset, transactions) / len(transactions)


# ══════════════════════════════════════════════════════════
# 3. APRIORI — CANDIDATE GENERATION (Fk-1 × Fk-1 join)
# ══════════════════════════════════════════════════════════

def generate_candidates(frequent_prev, k):
    """
    Join step: merge pairs of (k-1)-itemsets that share k-2 items.
    Then prune: discard any candidate whose (k-1)-subsets are not all frequent.
    Returns candidate k-itemsets.
    """
    freq_list = list(frequent_prev)
    candidates = set()

    for i in range(len(freq_list)):
        for j in range(i + 1, len(freq_list)):
            union = freq_list[i] | freq_list[j]
            if len(union) == k:
                # Anti-monotone pruning: all k-1 subsets must be frequent
                all_subsets_frequent = all(
                    frozenset(sub) in frequent_prev
                    for sub in combinations(union, k - 1)
                )
                if all_subsets_frequent:
                    candidates.add(union)
    return candidates


# ══════════════════════════════════════════════════════════
# 4. FULL APRIORI ALGORITHM
# ══════════════════════════════════════════════════════════

def apriori(transactions, min_support=0.3, verbose=True):
    """
    Complete Apriori algorithm.
    Returns: dict { frozenset → support_value } for all frequent itemsets.
    """
    N = len(transactions)
    all_frequent = {}    # { frozenset: support }
    all_items = set(item for t in transactions for item in t)

    if verbose:
        print(f"\n{'─'*68}")
        print(f"APRIORI  (min_support = {min_support:.0%}  = "
              f"{int(min_support*N)} of {N} transactions)")
        print(f"{'─'*68}")

    # ── Level 1: scan all single items ───────────────────
    L_prev = {}
    for item in sorted(all_items):
        fs = frozenset([item])
        sup = support(fs, transactions)
        if sup >= min_support:
            L_prev[fs] = sup

    all_frequent.update(L_prev)

    if verbose:
        print(f"\nL1 — frequent 1-itemsets ({len(L_prev)} of {len(all_items)}):")
        for fs, sup in sorted(L_prev.items(), key=lambda x: -x[1]):
            label = "{" + ", ".join(sorted(fs)) + "}"
            print(f"  {label:<35}  sup = {sup:.3f}  "
                  f"({int(sup*N)}/{N} customers)")

    k = 2
    while L_prev:
        # ── Generate candidates ───────────────────────────
        candidates = generate_candidates(set(L_prev.keys()), k)

        if verbose:
            print(f"\nL{k} — candidates generated: {len(candidates)}")

        # ── Scan DB and filter by min_support ─────────────
        L_curr = {}
        for cand in candidates:
            sup = support(cand, transactions)
            if sup >= min_support:
                L_curr[cand] = sup

        if verbose:
            print(f"L{k} — frequent {k}-itemsets after pruning: {len(L_curr)}")
            for fs, sup in sorted(L_curr.items(), key=lambda x: -x[1]):
                label = "{" + ", ".join(sorted(fs)) + "}"
                print(f"  {label:<45}  sup = {sup:.3f}  ({int(sup*N)}/{N})")

        all_frequent.update(L_curr)
        L_prev = L_curr
        k += 1

    if verbose:
        print(f"\nTotal frequent itemsets found: {len(all_frequent)}")

    return all_frequent


# ══════════════════════════════════════════════════════════
# 5. RUN APRIORI ON CREDIT CARD DATA
# ══════════════════════════════════════════════════════════

MIN_SUPPORT    = 0.40   # itemset must appear in ≥ 40% of transactions
MIN_CONFIDENCE = 0.65   # rule must be correct ≥ 65% of the time
MIN_LIFT       = 1.20   # rule must be at least 20% better than random

frequent_itemsets = apriori(transactions, min_support=MIN_SUPPORT, verbose=True)


# ══════════════════════════════════════════════════════════
# 6. ASSOCIATION RULE GENERATION
# ══════════════════════════════════════════════════════════

def generate_rules(frequent_itemsets, transactions, min_confidence, min_lift=1.0):
    """
    For every frequent itemset of size ≥ 2, generate all non-empty
    antecedent → consequent splits and compute:
      Confidence = sup(A∪B) / sup(A)
      Lift       = Confidence / sup(B)
      Conviction = (1 - sup(B)) / (1 - Confidence)
                   (∞ when confidence=1; > 1 means A implies B)
    """
    rules = []
    N = len(transactions)

    for itemset, sup_ab in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                antecedent  = frozenset(antecedent)
                consequent  = itemset - antecedent
                sup_a       = support(antecedent, transactions)
                sup_b       = support(consequent, transactions)
                confidence  = sup_ab / sup_a if sup_a > 0 else 0
                lift        = confidence / sup_b if sup_b > 0 else 0
                conviction  = ((1 - sup_b) / (1 - confidence)
                               if confidence < 1 else float("inf"))
                leverage    = sup_ab - sup_a * sup_b

                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedent":  antecedent,
                        "consequent":  consequent,
                        "support":     round(sup_ab, 4),
                        "confidence":  round(confidence, 4),
                        "lift":        round(lift, 4),
                        "conviction":  round(conviction, 4),
                        "leverage":    round(leverage, 4),
                        "count":       int(sup_ab * N),
                    })

    # Sort by lift desc, then confidence desc
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    return rules


print("\n" + "─" * 68)
print("STEP 6 — ASSOCIATION RULES GENERATION")
print(f"         min_confidence = {MIN_CONFIDENCE:.0%}  |  min_lift = {MIN_LIFT}")
print("─" * 68)

rules = generate_rules(frequent_itemsets, transactions,
                       min_confidence=MIN_CONFIDENCE, min_lift=MIN_LIFT)
print(f"\nTotal rules meeting thresholds: {len(rules)}")


# ══════════════════════════════════════════════════════════
# 7. DISPLAY RULES — FULL DETAIL
# ══════════════════════════════════════════════════════════

def fmt_set(fs):
    return "{" + ", ".join(sorted(fs)) + "}"

print("\n" + "─" * 68)
print("STEP 7 — ALL ASSOCIATION RULES (sorted by Lift ↓)")
print("─" * 68)
print(f"\n{'#':<4}  {'Rule':<48}  {'Sup':>6}  {'Conf':>7}  {'Lift':>6}  {'Conv':>7}")
print("─" * 80)

for i, r in enumerate(rules, 1):
    rule_str = f"{fmt_set(r['antecedent'])} → {fmt_set(r['consequent'])}"
    conv_str = (f"{r['conviction']:>7.3f}" if r["conviction"] != float("inf")
                else "    inf")
    print(f"{i:<4}  {rule_str:<48}  {r['support']:>6.3f}  "
          f"{r['confidence']:>7.3f}  {r['lift']:>6.3f}  {conv_str}")


# ══════════════════════════════════════════════════════════
# 8. METRIC DEEP-DIVE FOR TOP RULES
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 68)
print("STEP 8 — METRIC DEEP-DIVE: TOP 5 RULES BY LIFT")
print("─" * 68)

for i, r in enumerate(rules[:5], 1):
    ant, con = r["antecedent"], r["consequent"]
    print(f"\n{'─'*60}")
    print(f"Rule {i}:  {fmt_set(ant)}  →  {fmt_set(con)}")
    print(f"{'─'*60}")
    print(f"  Support    = {r['support']:.4f}  "
          f"→  appears in {r['count']} of {N} customers")
    print(f"  Confidence = {r['confidence']:.4f}  "
          f"→  {r['confidence']*100:.1f}% of customers with "
          f"{fmt_set(ant)} ALSO have {fmt_set(con)}")
    sup_con = support(con, transactions)
    print(f"  Lift       = {r['lift']:.4f}  "
          f"→  {r['lift']:.2f}× more likely than random "
          f"(baseline {sup_con:.3f})")
    if r["conviction"] == float("inf"):
        print(f"  Conviction = ∞  →  perfect implication (confidence = 1.0)")
    else:
        print(f"  Conviction = {r['conviction']:.4f}  "
              f"→  {'strong' if r['conviction']>2 else 'moderate'} implication")
    print(f"  Leverage   = {r['leverage']:+.4f}  "
          f"→  joint probability gain over independence")


# ══════════════════════════════════════════════════════════
# 9. SUPPORT × CONFIDENCE GRID (itemset frequency matrix)
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 68)
print("STEP 9 — ITEM FREQUENCY SUMMARY (single items)")
print("─" * 68)
print(f"\n{'Category':<14}  {'Count':>6}  {'Support':>9}  {'Frequency bar'}")
print("─" * 55)
item_supports = {}
for cat in CATEGORIES:
    sup = support(frozenset([cat]), transactions)
    item_supports[cat] = sup
    bar = "█" * int(sup * 30)
    print(f"{cat:<14}  {int(sup*N):>6}  {sup:>9.3f}  {bar}")


# ══════════════════════════════════════════════════════════
# 10. BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("STEP 10 — BUSINESS INSIGHTS FROM ASSOCIATION RULES")
print("=" * 68)

# Top rule by confidence
top_conf = max(rules, key=lambda r: r["confidence"])
print(f"\n  Highest-confidence rule:")
print(f"    {fmt_set(top_conf['antecedent'])} → {fmt_set(top_conf['consequent'])}")
print(f"    Confidence = {top_conf['confidence']:.1%}")
print(f"    Insight: Almost certain co-spend pattern — target for bundle offers.")

# Top rule by lift
top_lift = rules[0]
print(f"\n  Highest-lift rule (strongest non-random association):")
print(f"    {fmt_set(top_lift['antecedent'])} → {fmt_set(top_lift['consequent'])}")
print(f"    Lift = {top_lift['lift']:.3f}")
print(f"    Insight: Customers are {top_lift['lift']:.1f}× more likely to make "
      f"this combination than chance.")

# Rules involving Groceries (most common)
groc_rules = [r for r in rules if "Groceries" in r["antecedent"]
              or "Groceries" in r["consequent"]]
print(f"\n  Rules involving Groceries ({len(groc_rules)} found):")
for r in groc_rules[:3]:
    print(f"    {fmt_set(r['antecedent'])} → {fmt_set(r['consequent'])}"
          f"  (conf={r['confidence']:.2f}, lift={r['lift']:.2f})")

# High-value cross-sell opportunities (Lift > 1.5)
cross_sell = [r for r in rules if r["lift"] >= 1.5 and len(r["antecedent"]) == 1]
if cross_sell:
    print(f"\n  Strong single-item → item cross-sell opportunities (Lift ≥ 1.5):")
    for r in cross_sell:
        print(f"    {fmt_set(r['antecedent'])} → {fmt_set(r['consequent'])}"
              f"  (lift={r['lift']:.2f})")

print(f"\n  Summary statistics:")
print(f"    Frequent itemsets found  : {len(frequent_itemsets)}")
print(f"    Association rules found  : {len(rules)}")
print(f"    Avg confidence of rules  : {sum(r['confidence'] for r in rules)/len(rules):.3f}")
print(f"    Avg lift of rules        : {sum(r['lift'] for r in rules)/len(rules):.3f}")
print(f"    Rules with lift > 2.0    : {sum(1 for r in rules if r['lift'] > 2.0)}")

print("\n" + "=" * 68)
print("APRIORI ALGORITHM COMPLETE")
print("=" * 68)
print(f"\nThresholds used:")
print(f"  min_support    = {MIN_SUPPORT:.0%}  (itemset in ≥ {int(MIN_SUPPORT*N)}/{N} transactions)")
print(f"  min_confidence = {MIN_CONFIDENCE:.0%}  (rule correct ≥ {MIN_CONFIDENCE*100:.0f}% of time)")
print(f"  min_lift       = {MIN_LIFT}   (at least {(MIN_LIFT-1)*100:.0f}% better than random)")
