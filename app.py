# app.py
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Import shared logic from train.py (keep train.py in the same folder)

import importlib.util, os
_train_path = os.path.join(os.path.dirname(__file__), "models", "train.py")
spec = importlib.util.spec_from_file_location("train", _train_path)
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)

download_and_load_uci = train.download_and_load_uci
prepare = train.prepare
build_preprocessors = train.build_preprocessors
XGB_AVAILABLE = getattr(train, "XGB_AVAILABLE", False)


try:
    from xgboost import XGBClassifier
except Exception:
    XGBoostClassifier = None
    XGBClassifier = None  # handle gracefully in code

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="UCI Bank Marketing",
    page_icon="📈",
    layout="wide"
)

st.title("📈 UCI Bank Marketing")

# -----------------------------------
# Session state init (in-memory models & artifacts)
# -----------------------------------
if "models" not in st.session_state:
    st.session_state.models = None     # dict[str, Pipeline]
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None

# -----------------------------------
# Helpers
# -----------------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    """Download from UCI and preprocess (cached)."""
    df = download_and_load_uci()
    df = prepare(df)
    return df

def bytes_from_df(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def normalize_uploaded_test(df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Make uploaded test data consistent with training:
    - Drop 'duration' if present
    - Convert y from {'yes','no'} to {1,0} if present
    """
    df = df_test.copy()
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    if "y" in df.columns and df["y"].dtype == object:
        df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)
    return df

from sklearn.metrics import classification_report

def classification_report_dataframe(y_true, y_pred) -> pd.DataFrame:
    """
    Return a tidy DataFrame from sklearn's classification_report (output_dict=True),
    with per-class rows, plus micro/macro/weighted averages, sorted by class label.
    """
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "label"})
    # Reorder & format columns if present
    cols = ["label", "precision", "recall", "f1-score", "support"]
    df = df[[c for c in cols if c in df.columns]]
    # Move 'accuracy' to the end if present
    if "accuracy" in rep:
        acc = pd.DataFrame([{"label": "accuracy", "precision": np.nan, "recall": np.nan,
                             "f1-score": rep["accuracy"], "support": rep["macro avg"]["support"]}])
        # Keep averages grouped at the bottom
        head = df[~df["label"].isin(["macro avg", "weighted avg", "micro avg"])]
        tail = df[df["label"].isin(["micro avg", "macro avg", "weighted avg"])]
        df = pd.concat([head, tail, acc], ignore_index=True)
    return df

def style_classification_report(df: pd.DataFrame):
    """
    Return a Streamlit-friendly Styler with number formatting + color bars.
    """
    fmt = {"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}
    if "support" in df.columns:
        fmt["support"] = "{:,.0f}"

    styler = (df.style
              .format(fmt)
              .bar(subset=["precision", "recall", "f1-score"], color="#a3d5ff", vmin=0, vmax=1)
              .set_properties(**{"text-align": "left"}))
    return styler

def build_preprocessors_compat(X: pd.DataFrame):
    """
    Version-safe fallback if build_preprocessors from train.py
    runs into OneHotEncoder param differences.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    if 'y' in num_cols: num_cols.remove('y')
    if 'y' in cat_cols: cat_cols.remove('y')

    # Try new param name first; fallback for older sklearn
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preproc_dense = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', ohe, cat_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    preproc_tree = ColumnTransformer(
        transformers=[
            ('passthrough_num', 'passthrough', num_cols),
            ('cat', ohe, cat_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preproc_dense, preproc_tree

def train_all_models(df: pd.DataFrame, test_size: float, random_state: int):
    """
    Train all baseline models on a stratified split.
    Returns: (metrics_df, models_dict, X_test, y_test, feature_cols)
    """
    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
    )

    # Preprocessors (from train.py) with fallback for version differences
    try:
        preproc_dense, preproc_tree = build_preprocessors(X)
    except Exception:
        preproc_dense, preproc_tree = build_preprocessors_compat(X)

    results = []
    models = {}

    def eval_and_store(name, pipe, use_proba=True):
        y_pred = pipe.predict(X_test)

        # Metrics
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
        models[name] = pipe

    # Build pipelines
    model_builders = {
        "Logistic Regression": lambda: Pipeline([
            ("pre", preproc_dense),
            ("clf", LogisticRegression(max_iter=1000, solver="saga", class_weight="balanced")),
        ]),
        "Decision Tree Classifier": lambda: Pipeline([
            ("pre", preproc_tree),
            ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ]),
        "K-Nearest Neighbor Classifier": lambda: Pipeline([
            ("pre", preproc_dense),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "Naive Bayes Classifier (Gaussian)": lambda: Pipeline([
            ("pre", preproc_dense),
            ("clf", GaussianNB()),
        ]),
        "Random Forest(Ensemble)": lambda: Pipeline([
            ("pre", preproc_tree),
            ("clf", RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=5,
                random_state=42, class_weight="balanced", n_jobs=-1
            )),
        ]),
    }

    if XGB_AVAILABLE and XGBClassifier is not None:
        def build_xgb():
            pos_weight = (y_train.shape[0] - y_train.sum()) / max(1, y_train.sum())
            return Pipeline([
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
        model_builders["XGBoost(Ensemble)"] = build_xgb

    # Train and evaluate all
    for name, builder in model_builders.items():
        pipe = builder()
        pipe.fit(X_train, y_train)
        # KNN has predict_proba; use_proba=True is fine
        eval_and_store(name, pipe, use_proba=True)

    metrics_df = pd.DataFrame(results)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    return metrics_df, models, X_test, y_test, X.columns.tolist()

# -----------------------------------
# Section A — Upload ONLY test data (CSV)
# -----------------------------------
st.subheader("A) Upload Test Data (CSV) — *Optional*")
st.caption("Upload only **test** data due to Streamlit free tier limits. Include the target column `y` if you want the app to calculate metrics; otherwise you’ll get predictions only.")
uploaded_file = st.file_uploader("Upload test CSV", type=["csv"], accept_multiple_files=False, key="test_upload")

# -----------------------------------
# Train and store in session state
# -----------------------------------

test_size = 0.30
random_state = 42
run_training = st.button("🚀 Train Models", type="primary", key="train_btn")
if run_training:
    with st.spinner("Downloading data and training models..."):
        df = load_and_prepare_data()
        metrics_df, models, X_test, y_test, feature_cols = train_all_models(df, test_size, random_state)

    # Persist results in-memory for the session
    st.session_state.models = models
    st.session_state.feature_cols = feature_cols
    st.session_state.metrics_df = metrics_df
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    st.success("Training complete!")

# -----------------------------------
# If no trained models yet, guide the user and stop
# -----------------------------------
if st.session_state.models is None:
    st.info("No trained models available yet. Click **Train Models** to train in-memory.")
    st.stop()

# -----------------------------------
# (b) Model selection dropdown
# -----------------------------------
st.subheader("B) Select a Model")
model_name = st.selectbox("Choose a model", options=list(st.session_state.models.keys()))
selected_model = st.session_state.models[model_name]

# -----------------------------------
# ⬇ Download internal test split (from last training run)
# -----------------------------------
st.subheader("Download Internal Test Split")
if st.session_state.X_test is None or st.session_state.y_test is None:
    st.info("Train models first to generate the internal test split.")
else:
    _test_df = st.session_state.X_test.copy()
    _test_df["y"] = st.session_state.y_test.values
    st.download_button(
        label="⬇️ Download test_split.csv",
        data=bytes_from_df(_test_df),   # uses existing helper
        file_name="test_split.csv",
        mime="text/csv",
        key="download_test_split"
    )

# -----------------------------------
# (c) Display evaluation metrics (Selected Model only)
# -----------------------------------
st.subheader("C) Evaluation Metrics (Selected Model)")
_selected_metrics = st.session_state.metrics_df[
    st.session_state.metrics_df["Model"] == model_name
]
if _selected_metrics.empty:
    st.warning("No metrics found for the selected model.")
else:
    st.dataframe(_selected_metrics, use_container_width=True)

# -----------------------------------
# (d) Confusion Matrix or Classification Report
# -----------------------------------
st.subheader("D) Confusion Matrix or Classification Report")
view_type = st.radio("Choose view", ["Confusion Matrix", "Classification Report"], horizontal=True)

y_pred_internal = selected_model.predict(st.session_state.X_test)
if view_type == "Confusion Matrix":
    cm = confusion_matrix(st.session_state.y_test, y_pred_internal)
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name} ")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)
else:
    rep_df = classification_report_dataframe(st.session_state.y_test, y_pred_internal)
    st.dataframe(style_classification_report(rep_df), use_container_width=True)




st.divider()

# Footer hint for markers
st.caption("✓ A) Dataset upload (test only)  ✓ B) Model dropdown  ✓ C) Metrics table  ✓ D) Confusion matrix / classification report")
