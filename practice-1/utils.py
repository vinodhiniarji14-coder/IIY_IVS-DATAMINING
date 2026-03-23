# utils.py
# Shared helper functions used across all KDD practice scripts.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(filename: str) -> None:
    """Save the current matplotlib figure to outputs/."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [saved] {path}")


def section(title: str) -> None:
    """Print a visible section header to the console."""
    bar = "─" * 62
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_classification_results(y_true, y_pred, class_names) -> None:
    """Print a Weka-style summary: accuracy, report, confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    correct = int(np.trace(cm))
    total = len(y_true)
    acc = correct / total * 100
    print(f"\n  Correctly classified   : {correct}/{total}  ({acc:.1f}%)")
    print(f"  Incorrectly classified : {total - correct}/{total}  ({100 - acc:.1f}%)\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("  Confusion Matrix:")
    for row in cm:
        print("  ", row)


def plot_confusion_matrix(y_true, y_pred, class_names,
                           title="Confusion Matrix", filename=None) -> None:
    """Plot a labelled confusion matrix and optionally save it."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    if filename:
        save_fig(filename)
    plt.show()
