from pathlib import Path
import os
import json
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Direct matplotlib cache to system temp directory to prevent permission warnings
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_cache"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, IsolationForest
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

TARGET = "Machine failure"
CAT = ["Type"]
NUM = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
FEATURES = CAT + NUM
LEAKAGE = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

def preprocessor(scale=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", num_pipe, NUM),
        ("categorical", cat_pipe, CAT)
    ])

def evaluate(name, estimator, X_test, y_test):
    pred = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    safe = name.lower().replace(" ", "_")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No failure", "Failure"],
                yticklabels=["No failure", "Failure"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / f"confusion_matrix_{safe}.png", dpi=160)
    plt.close()
    (OUT / f"classification_report_{safe}.txt").write_text(
        classification_report(y_test, pred, zero_division=0), encoding="utf-8"
    )
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "PR_AUC": average_precision_score(y_test, proba)
    }

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset missing: {DATA}")
    df = pd.read_csv(DATA)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # Prevent target leakage: only pre-failure operating variables are inputs.
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / max(positives, 1)

    models = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor(scale=True)),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=42
            ))
        ]),
        "Decision Tree": Pipeline([
            ("prep", preprocessor()),
            ("model", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=8,
                class_weight="balanced", random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42, n_jobs=-1
            ))
        ]),
        "Extra Trees": Pipeline([
            ("prep", preprocessor()),
            ("model", ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=2,
                class_weight="balanced",
                random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor()),
            ("model", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05,
                max_depth=2, random_state=42
            ))
        ]),
        "AdaBoost": Pipeline([
            ("prep", preprocessor()),
            ("model", AdaBoostClassifier(
                n_estimators=150, learning_rate=0.5, random_state=42
            ))
        ]),
        "XGBoost": Pipeline([
            ("prep", preprocessor()),
            ("model", XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", random_state=42,
                n_jobs=-1
            ))
        ]),
    }

    bundle = {"models": {}, "features": FEATURES, "target": TARGET}
    rows = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        bundle["models"][name] = model
        rows.append(evaluate(name, model, X_test, y_test))

    # Eighth model: unsupervised anomaly detector, not a direct failure classifier.
    iso_prep = preprocessor(scale=True)
    train_transformed = iso_prep.fit_transform(X_train)
    test_transformed = iso_prep.transform(X_test)
    contamination = min(max(float(y_train.mean()), 0.001), 0.5)
    iso = IsolationForest(
        n_estimators=300, contamination=contamination,
        random_state=42, n_jobs=-1
    )
    iso.fit(train_transformed)
    iso_pred = (iso.predict(test_transformed) == -1).astype(int)
    iso_score = -iso.score_samples(test_transformed)
    cm = confusion_matrix(y_test, iso_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No failure", "Failure"],
                yticklabels=["No failure", "Failure"])
    plt.title("Isolation Forest - Anomaly vs Failure Label")
    plt.xlabel("Predicted anomaly")
    plt.ylabel("Actual failure")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix_isolation_forest.png", dpi=160)
    plt.close()
    rows.append({
        "Model": "Isolation Forest",
        "Accuracy": accuracy_score(y_test, iso_pred),
        "Precision": precision_score(y_test, iso_pred, zero_division=0),
        "Recall": recall_score(y_test, iso_pred, zero_division=0),
        "F1": f1_score(y_test, iso_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, iso_score),
        "PR_AUC": average_precision_score(y_test, iso_score)
    })
    bundle["isolation_preprocessor"] = iso_prep
    bundle["isolation_model"] = iso
    bundle["isolation_contamination"] = contamination
    joblib.dump(bundle, MODEL_DIR / "failure_model.pkl")

    results = pd.DataFrame(rows).sort_values("Recall", ascending=False)
    results.to_csv(OUT / "model_metrics.csv", index=False)
    ax = results.set_index("Model")[["Precision", "Recall", "F1"]].plot(
        kind="bar", figsize=(12, 6), ylim=(0, 1)
    )
    ax.set_title("Eight-Model Comparison (Held-Out Test Set)")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=170)
    plt.close()

    summary = {
        "rows": int(len(df)),
        "features": FEATURES,
        "target_distribution": y.value_counts().to_dict(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "excluded_columns": LEAKAGE
    }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nMODEL RESULTS")
    print(results.round(4).to_string(index=False))
    print("\nSaved models and reports in models/ and reports/")

if __name__ == "__main__":
    main()
