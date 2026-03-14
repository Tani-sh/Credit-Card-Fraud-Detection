"""
Visualisation: ROC curves, confusion matrices, and metric comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

sns.set_style("darkgrid")
PLOT_DIR = "./plots"


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_confusion_matrices(models: dict, X_test, y_test):
    """Plot confusion matrices for all models."""
    ensure_plot_dir()

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            ax=ax,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("[✓] Confusion matrices saved.")


def plot_roc_curves(models: dict, X_test, y_test):
    """Plot ROC curves for all models."""
    ensure_plot_dir()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#FF5722", "#2196F3", "#4CAF50", "#9C27B0", "#FF9800"]

    for (name, model), color in zip(models.items(), colors):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC = {roc_auc:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Fraud Detection Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    print("[✓] ROC curves saved.")


def plot_metric_comparison(results: dict):
    """Plot bar chart comparing precision, recall, F1 across models."""
    ensure_plot_dir()

    names = list(results.keys())
    precision = [results[n]["precision"] for n in names]
    recall = [results[n]["recall"] for n in names]
    f1 = [results[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", color="#FF5722", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-Score", color="#4CAF50", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "metric_comparison.png"), dpi=150)
    plt.close()
    print("[✓] Metric comparison saved.")


def plot_all(models: dict, X_test, y_test):
    """Generate all visualisations."""
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    print(f"[✓] All plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    # Quick demo — requires trained models
    print("[!] Run `python train.py` first to generate plots with real data.")
