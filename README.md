# 🔍 Credit Card Fraud Detection

Detecting fraudulent transactions in a highly imbalanced dataset (~0.17% fraud) using classical ML models. Applies **Random Under-Sampling** on the training set to handle class imbalance, then benchmarks four classifiers — tuning **Gradient Boosting** with GridSearchCV to maximise recall on the fraud class.

## ⚠️ The problem

Credit card fraud datasets are extremely skewed. Out of ~284K transactions, only ~490 are fraudulent. Training directly on this distribution causes models to simply predict "not fraud" for everything and still hit 99%+ accuracy — which is useless.

The fix here: Random Under-Sampling on the training set to balance the classes, while keeping the test set untouched (so evaluation reflects real-world distribution).

## 📊 Models benchmarked

| Model | Purpose |
|-------|---------|
| Logistic Regression | Linear baseline |
| LDA | Linear discriminant baseline |
| Random Forest | Ensemble baseline |
| **Gradient Boosting** | Primary model, tuned via GridSearchCV |

GBM is tuned with 5-fold CV optimising for **recall** (we care more about catching fraud than avoiding false positives). Best result: **92% recall** on the fraud class.

## 📁 Project structure

```
├── preprocess.py       # Loading, scaling, Random Under-Sampling
├── model.py            # Model definitions + GBM hyperparameter grid
├── train.py            # Full pipeline: train all → tune GBM → summary
├── evaluate.py         # Precision, recall, F1, AUC, confusion matrix
├── visualize.py        # ROC curves, confusion matrix heatmaps
├── requirements.txt
└── .gitignore
```

## 🚀 Usage

```bash
pip install -r requirements.txt

# Run the full pipeline
python train.py
```

This trains all four models, tunes GBM, and prints a comparison table. To use the real Kaggle dataset, download `creditcard.csv` into `./data/` and update `DATA_PATH` in `preprocess.py`.

## 🔧 Dependencies

`scikit-learn`, `imbalanced-learn` (for Random Under-Sampling), `numpy`, `pandas`, `matplotlib`, `seaborn`
