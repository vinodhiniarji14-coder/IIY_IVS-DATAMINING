# 03_classification.py
# =============================================================================
# Weka Classify Tab — Classification Algorithms
#
# Algorithms demonstrated (Weka equivalents in brackets):
#   1. Decision Tree   [J48]
#   2. Naive Bayes     [NaiveBayes]
#   3. k-Nearest Neighbour [IBk]
#   4. Random Forest   [RandomForest]
#   5. Support Vector Machine [SMO]
#   6. Logistic Regression [Logistic]
#   7. ZeroR Baseline  [ZeroR]
#
# Evaluation:
#   • Train/test split (70/30)
#   • 10-fold cross-validation
#   • Confusion matrix, accuracy, precision, recall, F1
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils import section, save_fig, print_classification_results, plot_confusion_matrix


# ── Load and prepare data ─────────────────────────────────────────────────────
iris = load_iris()
X_raw = iris.data
y = iris.target
class_names = iris.target_names
feature_names = iris.feature_names

# Preprocessing: impute → standardise
imputer = SimpleImputer(strategy="mean")
scaler  = StandardScaler()
X = scaler.fit_transform(imputer.fit_transform(X_raw))

# 70/30 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

print("\n" + "=" * 62)
print("  Weka Classify Tab — Classification Demo")
print("=" * 62)
print(f"\n  Dataset   : Iris  ({len(X)} samples, {X.shape[1]} features, {len(class_names)} classes)")
print(f"  Train set : {len(X_train)} samples")
print(f"  Test  set : {len(X_test)} samples")
print(f"  Evaluation: 70/30 split  +  10-fold cross-validation")


# =============================================================================
# Define all classifiers
# =============================================================================
classifiers = {
    "ZeroR (baseline)":     DummyClassifier(strategy="most_frequent"),
    "Naive Bayes":          GaussianNB(),
    "IBk — k-NN (k=3)":    KNeighborsClassifier(n_neighbors=3),
    "J48 — Decision Tree":  DecisionTreeClassifier(criterion="entropy",
                                                    max_depth=4, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
    "SMO — SVM (RBF)":     SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = {}


# =============================================================================
# Train, evaluate, cross-validate every classifier
# =============================================================================
section("CLASSIFIER COMPARISON — 10-fold CV Accuracy")

print(f"\n  {'Algorithm':<28} {'Test Acc':>9}  {'CV Mean':>9}  {'CV Std':>9}")
print("  " + "-" * 58)

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred) * 100
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy") * 100
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    f1 = f1_score(y_test, y_pred, average="weighted")
    results[name] = {"clf": clf, "y_pred": y_pred,
                     "test_acc": test_acc, "cv_mean": cv_mean,
                     "cv_std": cv_std, "f1": f1, "cv_scores": cv_scores}
    print(f"  {name:<28} {test_acc:>8.1f}%  {cv_mean:>8.1f}%  ±{cv_std:>6.1f}%")


# =============================================================================
# Detailed output for best classifier (highest CV mean)
# =============================================================================
best_name = max(results, key=lambda n: results[n]["cv_mean"] if n != "ZeroR (baseline)" else -1)
best = results[best_name]

section(f"DETAILED RESULTS — Best Classifier: {best_name}")
print_classification_results(y_test, best["y_pred"], class_names)

section(f"DETAILED RESULTS — Naive Bayes (Weka: NaiveBayes)")
nb_res = results["Naive Bayes"]
print_classification_results(y_test, nb_res["y_pred"], class_names)


# =============================================================================
# Visualisations
# =============================================================================
section("Generating classification visualisations…")

# ── Plot 1: Algorithm comparison bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
names = list(results.keys())
means = [results[n]["cv_mean"] for n in names]
stds  = [results[n]["cv_std"]  for n in names]
colors_bar = ["#cccccc" if "ZeroR" in n else "#4C72B0" for n in names]

bars = ax.bar(names, means, yerr=stds, capsize=5,
              color=colors_bar, edgecolor="white", alpha=0.88)
ax.set_ylabel("10-Fold CV Accuracy (%)")
ax.set_title("Weka Classify — Algorithm Comparison (10-fold CV)", fontsize=12)
ax.set_ylim(50, 108)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=18, ha="right", fontsize=10)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
save_fig("classify_01_comparison.png")
plt.show()

# ── Plot 2: Best classifier confusion matrix ──────────────────────────────────
plot_confusion_matrix(y_test, best["y_pred"], class_names,
                      title=f"Weka Classify — {best_name} Confusion Matrix",
                      filename="classify_02_confusion_matrix.png")

# ── Plot 3: Decision tree plot ────────────────────────────────────────────────
dt_clf = classifiers["J48 — Decision Tree"]
fig, ax = plt.subplots(figsize=(15, 6))
plot_tree(dt_clf, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Weka Classify — J48 Decision Tree (entropy, max_depth=4)", fontsize=12)
plt.tight_layout()
save_fig("classify_03_decision_tree.png")
plt.show()

# ── Plot 4: CV score distributions (box plot) ────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
cv_data = [results[n]["cv_scores"] for n in names]
bp = ax.boxplot(cv_data, patch_artist=True, notch=False,
                medianprops=dict(color="crimson", linewidth=1.8))
palette = cm.get_cmap("tab10")
for patch, i in zip(bp["boxes"], range(len(names))):
    patch.set_facecolor(palette(i % 10))
    patch.set_alpha(0.75)
ax.set_xticklabels(names, rotation=18, ha="right", fontsize=10)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Weka Classify — 10-Fold CV Score Distribution per Algorithm", fontsize=12)
plt.tight_layout()
save_fig("classify_04_cv_distributions.png")
plt.show()

# ── Plot 5: Feature importance (Random Forest) ───────────────────────────────
rf_clf = classifiers["Random Forest"]
importances = rf_clf.feature_importances_
short_names  = [f.split(" (")[0] for f in feature_names]
sorted_idx  = np.argsort(importances)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([short_names[i] for i in sorted_idx], importances[sorted_idx],
        color="#55A868", edgecolor="white", alpha=0.88)
ax.set_xlabel("Importance Score")
ax.set_title("Weka Classify — Random Forest Feature Importances", fontsize=12)
plt.tight_layout()
save_fig("classify_05_feature_importance.png")
plt.show()

section("COMPLETE — All classification plots saved to outputs/")
