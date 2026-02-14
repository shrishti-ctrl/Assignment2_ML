#!/usr/bin/env python3
"""
Train and evaluate baseline models on the UCI Bank Marketing dataset.

- Downloads dataset from UCI (prefers bank-additional-full.csv; falls back to bank-full.csv).
- Preprocess: drop 'duration' (leakage), convert y: {'yes','no'} -> {1,0}.
- Train/test split (stratified).
- Models: Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes,
          Random Forest, XGBoost (if installed).
- Metrics: Accuracy, AUC, Precision, Recall, F1, Matthews Correlation Coefficient (MCC).
- Side effect: writes only 'test_split.csv' to the current working directory.
"""

from pathlib import Path
import tempfile
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# -----------------------------
# Data download & loading
# -----------------------------
DATA_URLS = [
    # Preferred: newer dataset (with 'bank-additional/bank-additional-full.csv')
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
    # Fallback: older dataset (with 'bank/bank-full.csv')
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
]

def download_and_load_uci() -> pd.DataFrame:
    """Download UCI dataset to a temp dir and return a DataFrame."""
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for url in DATA_URLS:
            try:
                print(f"Downloading: {url}")
                zip_path = tdir / "dataset.zip"
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tdir)

                # Preferred file names (first match wins)
                candidates = [
                    ("bank-additional/bank-additional-full.csv", ";"),
                    ("bank/bank-full.csv", ";"),
                ]
                for rel, sep in candidates:
                    f = tdir / rel
                    if f.exists():
                        df = pd.read_csv(f, sep=sep)
                        print(f"Loaded: {rel} with shape {df.shape}")
                        return df
            except Exception as e:
                print(f"Failed from {url}: {e}")

    raise FileNotFoundError(
        "Could not download the dataset from UCI. Please check your internet connection."
    )


# -----------------------------
# Preprocessing
# -----------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage column and normalize target."""
    df = df.copy()
    # Drop 'duration' (known leakage for this dataset)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Ensure target 'y' exists and convert to 0/1
    if "y" not in df.columns:
        raise ValueError("Target column 'y' not found in dataset.")
    if df["y"].dtype == object:
        df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)
    return df


def build_preprocessors(X: pd.DataFrame):
    """Create dense and tree-friendly preprocessors."""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    if "y" in num_cols: num_cols.remove("y")
    if "y" in cat_cols: cat_cols.remove("y")

    preproc_dense = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preproc_tree = ColumnTransformer(
        transformers=[
            ("passthrough_num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preproc_dense, preproc_tree


# -----------------------------
# Training & Evaluation
# -----------------------------
def evaluate_all(X_train, X_test, y_train, y_test, preproc_dense, preproc_tree) -> pd.DataFrame:
    results = []

    def eval_model(name, pipe, use_proba=True):
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # AUC: prefer predict_proba; fallback to decision_function; else NaN
        if use_proba and hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X_test)
                auc = roc_auc_score(y_test, scores)
            else:
                auc = np.nan

        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MCC": mcc,
        })

    # 1) Logistic Regression
    lr = Pipeline([
        ("pre", preproc_dense),
        ("clf", LogisticRegression(max_iter=1000, solver="saga", class_weight="balanced")),
    ])
    lr.fit(X_train, y_train)
    eval_model("Logistic Regression", lr)

    # 2) Decision Tree
    dt = Pipeline([
        ("pre", preproc_tree),
        ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
    ])
    dt.fit(X_train, y_train)
    eval_model("Decision Tree Classifier", dt)

    # 3) K-Nearest Neighbor
    knn = Pipeline([
        ("pre", preproc_dense),
        ("clf", KNeighborsClassifier(n_neighbors=15)),
    ])
    knn.fit(X_train, y_train)
    # KNN has predict_proba; keep use_proba=True
    eval_model("K-Nearest Neighbor Classifier", knn, use_proba=True)

    # 4) Naive Bayes (Gaussian)
    gnb = Pipeline([
        ("pre", preproc_dense),
        ("clf", GaussianNB()),
    ])
    gnb.fit(X_train, y_train)
    eval_model("Naive Bayes Classifier (Gaussian)", gnb)

    # 5) Random Forest (Ensemble)
    rf = Pipeline([
        ("pre", preproc_tree),
        ("clf", RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1
        )),
    ])
    rf.fit(X_train, y_train)
    eval_model("Ensemble Model - Random Forest", rf)

    # 6) XGBoost (Ensemble) — optional
    if XGB_AVAILABLE:
        pos_weight = (y_train.shape[0] - y_train.sum()) / max(1, y_train.sum())
        xgb = Pipeline([
            ("pre", preproc_tree),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42,
                scale_pos_weight=float(pos_weight),
                n_jobs=4,
            )),
        ])
        xgb.fit(X_train, y_train)
        eval_model("Ensemble Model - XGBoost", xgb)
    else:
        print("WARNING: xgboost not installed. Skipping XGBoost.")

    return pd.DataFrame(results)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Load data from UCI
    df = download_and_load_uci()

    # 2) Preprocess
    df = prepare(df)

    # 3) Split
    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 4) Export only the test split (no folders)
    test_df = X_test.copy()
    test_df["y"] = y_test.values
    out_path = Path.cwd() / "test_split.csv"
    test_df.to_csv(out_path, index=False)
    print(f"Saved test split to: {out_path.resolve()}")

    # 5) Build preprocessors & evaluate models
    preproc_dense, preproc_tree = build_preprocessors(X)
    metrics_df = evaluate_all(X_train, X_test, y_train, y_test, preproc_dense, preproc_tree)

    # 6) Print metrics
    print("\nEvaluation Metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
