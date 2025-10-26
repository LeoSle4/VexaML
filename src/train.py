import json, joblib, yaml, numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    recall_score,
    accuracy_score,
)
from xgboost import XGBClassifier

CFG = yaml.safe_load(Path("configs/config.yml").read_text())
SEED = CFG["ml"]["seed"]
TEST_SIZE = CFG["ml"]["test_size"]
TH_STRATEGY = CFG["ml"]["threshold_strategy"]
RECALL_TARGET = CFG["ml"]["recall_target"]

DATA_PATH = Path(CFG["data"]["processed_path"])
MODEL_PATH = Path("models/model.pkl")
THRESH_PATH = Path("models/threshold.json")


def choose_threshold(y_true, proba, strategy="max_f1", recall_target=0.75):
    p, r, th = precision_recall_curve(y_true, proba)
    f1 = (2 * p * r) / (p + r + 1e-9)
    if strategy == "max_f1":
        i = np.nanargmax(f1)
        return float(th[i - 1]) if i > 0 else 0.5
    # recall_first
    mask = r >= recall_target
    if mask.any():
        i = np.argmax(p[mask])
        idx = np.arange(len(r))[mask][i]
        return float(th[idx - 1]) if idx > 0 else 0.5
    return 0.5


def main():
    df = pd.read_parquet(DATA_PATH)
    y = df["loss_event"].astype(int)
    X = df.drop(columns=["loss_event", "event_id"], errors="ignore")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(
        include=["number", "int", "float", "bool"]
    ).columns.tolist()

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(with_mean=False), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                cat_cols,
            ),
        ]
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        random_state=SEED,
        eval_metric="logloss",
        tree_method="hist",
    )

    pipe = Pipeline([("pre", pre), ("clf", model)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    pipe.fit(X_tr, y_tr)

    proba_tr = pipe.predict_proba(X_tr)[:, 1]
    th = choose_threshold(y_tr, proba_tr, TH_STRATEGY, RECALL_TARGET)

    proba_te = pipe.predict_proba(X_te)[:, 1]
    y_pred = (proba_te >= th).astype(int)
    metrics = {
        "threshold": th,
        "recall": float(recall_score(y_te, y_pred)),
        "f1": float(f1_score(y_te, y_pred)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
    }
    print("[TEST]", metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    THRESH_PATH.write_text(json.dumps({"threshold": th}, indent=2), encoding="utf-8")
    print(f"[OK] Modelo -> {MODEL_PATH}")
    print(f"[OK] Umbral -> {THRESH_PATH}")


if __name__ == "__main__":
    main()
