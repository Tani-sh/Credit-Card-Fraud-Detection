"""
Main training pipeline: train all models and tune GBM with GridSearchCV.

Usage:
    python train.py
"""

import os
import time
import numpy as np
from sklearn.model_selection import GridSearchCV

from preprocess import generate_synthetic_data, preprocess
from model import get_models, get_gbm_param_grid
from evaluate import evaluate_model, print_summary


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all baseline models and return results."""
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"  Training: {name}")
        print(f"{'─' * 50}")

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        metrics = evaluate_model(model, X_test, y_test, name)
        metrics["train_time"] = train_time
        results[name] = metrics

        print(f"  ├── Precision : {metrics['precision']:.4f}")
        print(f"  ├── Recall    : {metrics['recall']:.4f}")
        print(f"  ├── F1-Score  : {metrics['f1']:.4f}")
        print(f"  ├── AUC       : {metrics['auc']:.4f}")
        print(f"  └── Time      : {train_time:.2f}s")

    return results


def tune_gbm(X_train, y_train):
    """
    Tune Gradient Boosting with GridSearchCV.

    Uses 5-fold cross-validation optimising for recall.
    """
    print(f"\n{'═' * 50}")
    print(f"  GridSearchCV: Tuning Gradient Boosting")
    print(f"{'═' * 50}")

    from sklearn.ensemble import GradientBoostingClassifier

    grid = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=get_gbm_param_grid(),
        cv=5,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    grid.fit(X_train, y_train)
    tune_time = time.time() - start

    print(f"\n[✓] Best recall: {grid.best_score_:.4f}")
    print(f"[✓] Best params: {grid.best_params_}")
    print(f"[✓] Tuning time: {tune_time:.1f}s")

    return grid.best_estimator_


def main():
    print("=" * 60)
    print("  CREDIT CARD FRAUD DETECTION")
    print("=" * 60)

    # Load and preprocess data
    df = generate_synthetic_data(n_samples=50000, fraud_ratio=0.0017)
    X_train, X_test, y_train, y_test, _ = preprocess(df)

    # Train all baseline models
    results = train_all_models(X_train, X_test, y_train, y_test)

    # Tune GBM with GridSearchCV
    best_gbm = tune_gbm(X_train, y_train)
    tuned_metrics = evaluate_model(best_gbm, X_test, y_test, "Gradient Boosting (Tuned)")
    results["Gradient Boosting (Tuned)"] = tuned_metrics

    # Summary
    print_summary(results)

    # Generate visualisations
    try:
        from visualize import plot_all
        all_models = get_models()
        all_models["Gradient Boosting (Tuned)"] = best_gbm
        plot_all(all_models, X_test, y_test)
    except Exception as e:
        print(f"[!] Could not generate plots: {e}")

    print("\n[✓] Training pipeline complete!")


if __name__ == "__main__":
    main()
