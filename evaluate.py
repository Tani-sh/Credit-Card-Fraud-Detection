"""
Evaluation: classification report, confusion matrix, ROC-AUC.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, name="Model") -> dict:
    """
    Evaluate a trained model and return metrics dict.

    Returns:
        dict with keys: precision, recall, f1, auc, y_pred, y_proba
    """
    y_pred = model.predict(X_test)

    # Some models support predict_proba
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        auc = 0.0

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def print_classification_report(model, X_test, y_test, name="Model"):
    """Print a full sklearn classification report."""
    y_pred = model.predict(X_test)
    print(f"\n  Classification Report — {name}")
    print(f"  {'─' * 45}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))


def print_summary(results: dict):
    """Print a summary table of all model results."""
    print(f"\n{'═' * 65}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'═' * 65}")
    print(f"\n  {'Model':<30} {'Prec':>6} {'Recall':>7} {'F1':>6} {'AUC':>6}")
    print(f"  {'─' * 55}")

    for name, metrics in results.items():
        print(f"  {name:<30} "
              f"{metrics['precision']:>6.3f} "
              f"{metrics['recall']:>7.3f} "
              f"{metrics['f1']:>6.3f} "
              f"{metrics['auc']:>6.3f}")

    print(f"  {'─' * 55}")

    # Highlight best
    best_name = max(results, key=lambda k: results[k]["recall"])
    print(f"\n  🏆 Best recall: {best_name} ({results[best_name]['recall']:.3f})")
    print(f"{'═' * 65}")
