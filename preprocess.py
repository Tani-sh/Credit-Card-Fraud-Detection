"""
Data preprocessing: loading, Random Under-Sampling, and feature scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = "./data/creditcard.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    print(f"[*] Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"[*] Dataset: {df.shape[0]:,} transactions, {df.shape[1]} features")
    print(f"[*] Fraud ratio: {df['Class'].mean() * 100:.3f}% "
          f"({df['Class'].sum():,} fraud / {len(df):,} total)")
    return df


def generate_synthetic_data(n_samples=10000, n_features=30, fraud_ratio=0.0017):
    """
    Generate synthetic credit card transaction data for demonstration.

    In a real project, you would use the Kaggle creditcard.csv dataset.
    """
    print("[*] Generating synthetic data for demonstration...")
    np.random.seed(RANDOM_STATE)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions — centered around 0
    X_legit = np.random.randn(n_legit, n_features).astype(np.float32)
    y_legit = np.zeros(n_legit, dtype=np.int32)

    # Fraudulent transactions — shifted distribution
    X_fraud = np.random.randn(n_fraud, n_features).astype(np.float32) + 1.5
    y_fraud = np.ones(n_fraud, dtype=np.int32)

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([y_legit, y_fraud])

    feature_names = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=feature_names)
    df["Class"] = y

    print(f"[*] Synthetic data: {len(df):,} transactions, "
          f"{y.sum()} fraud ({y.mean() * 100:.2f}%)")
    return df


def preprocess(df: pd.DataFrame):
    """
    Preprocess: scale features, apply Random Under-Sampling, and split.

    Returns:
        X_train, X_test, y_train, y_test (after under-sampling on train)
    """
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
    )

    # Train/test split (BEFORE under-sampling to keep test set realistic)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[*] Before under-sampling — Train: {len(X_train):,} "
          f"(fraud: {y_train.sum():,})")

    # Apply Random Under-Sampling on training set only
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    print(f"[*] After under-sampling  — Train: {len(X_train_res):,} "
          f"(fraud: {y_train_res.sum():,}, legit: {(y_train_res == 0).sum():,})")
    print(f"[*] Test set (untouched)  — {len(X_test):,} "
          f"(fraud: {y_test.sum():,})")

    return X_train_res, X_test, y_train_res, y_test, scaler


if __name__ == "__main__":
    # Demo with synthetic data
    df = generate_synthetic_data()
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    print(f"\n[✓] Preprocessing complete.")
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
