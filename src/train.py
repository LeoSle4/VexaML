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

# ---------------- Config ----------------
CFG = yaml.safe_load(Path("configs/config.yml").read_text())
SEED = CFG["ml"]["seed"]
TEST_SIZE = CFG["ml"]["test_size"]
TH_STRATEGY = CFG["ml"]["threshold_strategy"]
RECALL_TARGET = CFG["ml"]["recall_target"]

DATA_PATH = Path(CFG["data"]["processed_path"])
MODEL_PATH = Path("models/model.pkl")
THRESH_PATH = Path("models/threshold.json")
SCHEMA_PATH = Path("models/feature_schema.json")


def choose_threshold(y_true, proba, strategy="max_f1", recall_target=0.75):
    """Selecciona el umbral según estrategia (máx F1 o Recall objetivo)."""
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
    # --------- Lee dataset procesado ---------
    df = pd.read_parquet(DATA_PATH)

    # y y X
    y = df["loss_event"].astype(int)
    X = df.drop(columns=["loss_event", "event_id"], errors="ignore")

    # Columnas por tipo
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(
        include=["number", "int", "float", "bool"]
    ).columns.tolist()

    # --------- NUEVO: guardar esquema de features ---------
    feature_schema = {
        "all_columns": X.columns.tolist(),
        "num_cols": [c for c in X.columns if c in num_cols],
        "cat_cols": [c for c in X.columns if c in cat_cols],
    }
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    print(f"[OK] Esquema de features -> {SCHEMA_PATH}")
    print(f"[INFO] num_cols={len(num_cols)} | cat_cols={len(cat_cols)}")

    # --------- Preprocesamiento dinámico ---------
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=False), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                cat_cols,
            )
        )

    if not transformers:
        raise RuntimeError("No se detectaron columnas numéricas ni categóricas en X.")

    pre = ColumnTransformer(transformers, remainder="drop")

    # --------- Modelo ---------
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

    # --------- Split, entrenamiento y umbral ---------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    pipe.fit(X_tr, y_tr)

    proba_tr = pipe.predict_proba(X_tr)[:, 1]
    th = choose_threshold(y_tr, proba_tr, TH_STRATEGY, RECALL_TARGET)

    # --------- Evaluación rápida en test ---------
    proba_te = pipe.predict_proba(X_te)[:, 1]
    y_pred = (proba_te >= th).astype(int)
    metrics = {
        "threshold": th,
        "recall": float(recall_score(y_te, y_pred)),
        "f1": float(f1_score(y_te, y_pred)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
    }
    print("[TEST]", metrics)

    # --------- Guardar artefactos ---------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    THRESH_PATH.write_text(json.dumps({"threshold": th}, indent=2), encoding="utf-8")
    print(f"[OK] Modelo -> {MODEL_PATH}")
    print(f"[OK] Umbral -> {THRESH_PATH}")


if __name__ == "__main__":
    main()
