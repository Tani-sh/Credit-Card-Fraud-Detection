"""
Model definitions for fraud detection: GBM, Logistic Regression, LDA, Random Forest.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_models() -> dict:
    """
    Return a dictionary of baseline models.

    Returns:
        dict of model_name → sklearn estimator
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
    }


def get_gbm_param_grid() -> dict:
    """
    Return the hyperparameter grid for GridSearchCV on GBM.
    """
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 5],
    }
