"""Predictive maintenance - training pipeline.

Run:  python train.py --data data/ai4i2020.csv

What it does (matches the steps in the project plan):
  1. Load + clean data
  2. Quick EDA plots
  3. Feature engineering (features.py)
  4. Train/test split (stratified, because failures are rare)
  5. Train several models and compare them
  6. Pick the best, tune the decision threshold
  7. Evaluate on the untouched test set
  8. Explain the model (permutation importance) + what-if sweeps (optimization)
  9. Save everything the Streamlit app needs
"""
import argparse
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    average_precision_score, classification_report, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, RAW_COLS, add_features, normalize_columns

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    needed = RAW_COLS + ["failure"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}. Found: {list(df.columns)}")
    # NOTE: we deliberately ignore TWF/HDF/PWF/OSF/RNF columns (failure-mode labels).
    # They are outcomes of the failure, so using them as inputs would be data leakage.
    df = df[needed].dropna().reset_index(drop=True)
    return df


def eda(df: pd.DataFrame, out_dir: str) -> None:
    print("\n=== EDA ===")
    print("Shape:", df.shape)
    print(df.describe().T.round(2))
    rate = df["failure"].mean() * 100
    print(f"Failure rate: {rate:.2f}%  (heavily imbalanced -> don't trust accuracy)")

    num = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), num):
        ax.hist(df.loc[df.failure == 0, col], bins=40, alpha=0.6, label="no failure", density=True)
        ax.hist(df.loc[df.failure == 1, col], bins=40, alpha=0.6, label="failure", density=True)
        ax.set_title(col)
        ax.legend()
    corr = df[num + ["failure"]].corr()
    axes.ravel()[-1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes.ravel()[-1].set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    axes.ravel()[-1].set_yticks(range(len(corr)), corr.columns, fontsize=7)
    axes.ravel()[-1].set_title("correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda.png"), dpi=120)
    plt.close()


def build_models() -> dict:
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }
    try:  # XGBoost is optional
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        print("xgboost not installed - skipping it (pip install xgboost)")
    return models


def main(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_data(args.data)
    eda(df, args.out)

    df = add_features(df)
    X, y = df[FEATURES], df["failure"]

    # Stratified split keeps the same failure rate in train and test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---- compare models ----
    print("\n=== Model comparison (test set, default 0.5 threshold) ===")
    results, fitted = [], {}
    for name, model in build_models().items():
        if name == "XGBoost":  # handle imbalance for XGBoost
            model.set_params(scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append({
            "model": name,
            "PR-AUC": average_precision_score(y_te, proba),
            "ROC-AUC": roc_auc_score(y_te, proba),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred),
            "F1": f1_score(y_te, pred),
        })
        fitted[name] = model
    res = pd.DataFrame(results).set_index("model").round(3)
    print(res)
    res.to_csv(os.path.join(args.out, "model_comparison.csv"))

    best_name = res["PR-AUC"].idxmax()  # PR-AUC suits rare-event problems
    best = fitted[best_name]
    print(f"\nBest model by PR-AUC: {best_name}")

    # ---- tune the decision threshold using CV on TRAIN data only ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_tr, oof)
    f1 = 2 * p * r / (p + r + 1e-9)
    threshold = float(t[np.argmax(f1[:-1])])
    print(f"Tuned threshold (max F1 in CV): {threshold:.3f}")

    # ---- final evaluation on untouched test set ----
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    print("\n=== Final test report ===")
    print(classification_report(y_te, pred, target_names=["no failure", "failure"], digits=3))
    metrics = {
        "best_model": best_name, "threshold": threshold,
        "PR-AUC": float(average_precision_score(y_te, proba)),
        "ROC-AUC": float(roc_auc_score(y_te, proba)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred)),
        "F1": float(f1_score(y_te, pred)),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(y_te, proba, ax=axes[1])
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_te, proba, ax=axes[2])
    axes[2].set_title("Precision-Recall curve")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "evaluation.png"), dpi=120)
    plt.close()

    # ---- explainability: which features drive failures? ----
    pi = permutation_importance(
        best, X_te, y_te, scoring="average_precision",
        n_repeats=5, random_state=RANDOM_STATE,
    )
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    imp.plot.barh(figsize=(7, 5), title="Permutation importance (drop in PR-AUC)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "feature_importance.png"), dpi=120)
    plt.close()
    print("\nTop drivers of failure:\n", imp.sort_values(ascending=False).round(4).head(5))

    # ---- process optimisation: what-if sweeps ----
    raw_num = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
    base = {c: float(df[c].median()) for c in raw_num}
    ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                  "median": base[c]} for c in raw_num}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, var in zip(axes, ["tool_wear", "torque", "rpm", "process_temp"]):
        grid = np.linspace(ranges[var]["min"], ranges[var]["max"], 60)
        sweep = pd.DataFrame([{**base, "type": "M", var: v} for v in grid])
        sweep_p = best.predict_proba(add_features(sweep)[FEATURES])[:, 1]
        ax.plot(grid, sweep_p)
        ax.axhline(threshold, color="red", ls="--", label="alert threshold")
        ax.set_xlabel(var)
        ax.set_ylabel("failure probability")
        ax.legend()
    plt.suptitle("What-if analysis: effect of one setting, all others at median")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "what_if.png"), dpi=120)
    plt.close()

    # ---- save for the app ----
    joblib.dump({
        "model": best, "model_name": best_name, "threshold": threshold,
        "features": FEATURES, "ranges": ranges, "metrics": metrics,
    }, "models/best_model.joblib")
    print("\nSaved: models/best_model.joblib  and plots/metrics in", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ai4i2020.csv")
    ap.add_argument("--out", default="outputs")
    main(ap.parse_args())
