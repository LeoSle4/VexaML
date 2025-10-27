# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json
import yaml
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Risk-ML API")

# ---------------- Config & artefactos ----------------
CFG = yaml.safe_load(Path("configs/config.yml").read_text())

MODEL_PATH = Path("models/model.pkl")
THRESH_PATH = Path("models/threshold.json")
SCHEMA_PATH = Path("models/feature_schema.json")
VAR_PARAMS_PATH = Path(
    CFG["data"]["var_params_path"]
)  # p.ej. data/processed/var_params.parquet

# Modelo y umbral
pipe = joblib.load(MODEL_PATH)
used_threshold = float(json.loads(THRESH_PATH.read_text())["threshold"])

# Esquema de features (guardado por train.py)
_feature_schema = json.loads(SCHEMA_PATH.read_text())
ALL_COLS: list[str] = _feature_schema.get("all_columns", [])
NUM_COLS: list[str] = _feature_schema.get("num_cols", [])
CAT_COLS: list[str] = _feature_schema.get("cat_cols", [])

# Parametría de VaR (opcional)
if VAR_PARAMS_PATH.exists():
    var_df = pd.read_parquet(VAR_PARAMS_PATH)
    # columnas mínimas
    for c in ["segment", "VA_seg", "O_seg", "Z", "T", "source"]:
        if c not in var_df.columns:
            var_df[c] = np.nan
else:
    var_df = None


# ---------------- Pydantic input ----------------
class PredictIn(BaseModel):
    # categóricas base
    unit_impacted: str
    product_service: str
    process_impacted: str
    event_type: str
    risk_assoc: str

    # derivadas de fecha
    year: int
    month: int
    quarter: int
    weekday: int
    is_month_end: int

    # recencia / frecuencia (por unidad impactada en tu UI)
    n_events_30d_unit_impacted: float = 0.0
    n_events_60d_unit_impacted: float = 0.0
    n_events_90d_unit_impacted: float = 0.0
    days_since_last_unit_impacted: float = 9999.0

    # Consecuencia para R̂ = C × P̂
    consequence: float = 3.0

    # Umbral opcional
    threshold_override: float | None = None

    class Config:
        extra = "ignore"  # Ignora campos adicionales sin romper


# ---------------- Helpers ----------------
def build_segment(p: PredictIn) -> str:
    """
    Debe coincidir con el builder de var_params:
    event_type | unit_impacted | product_service
    """
    return f"{p.event_type}|{p.unit_impacted}|{p.product_service}"


def lookup_var_params(segment: str):
    """
    Devuelve (VA_seg, O_seg, Z, T, source) o None si no hay.
    """
    if var_df is None or var_df.empty:
        return None
    row = var_df.loc[var_df["segment"] == segment]
    if row.empty:
        return None
    r = row.iloc[0]
    return (
        float(r.get("VA_seg", 0.0)),
        float(r.get("O_seg", 0.0)),
        float(r.get("Z", 1.0)),
        float(r.get("T", 1.0)),
        (str(r.get("source")) if pd.notna(r.get("source")) else "segment"),
    )


def to_model_df(p: PredictIn) -> pd.DataFrame:
    """
    Construye un DataFrame con **todas** las columnas que espera el pipeline,
    tipando correctamente: categóricas -> str, numéricas -> float.
    """
    # 1) payload base (sin threshold_override)
    base = {k: getattr(p, k) for k in p.__fields__.keys() if k != "threshold_override"}
    X = pd.DataFrame([base])

    # 2) Añadir columnas faltantes con el tipo correcto
    for c in ALL_COLS:
        if c not in X.columns:
            if c in CAT_COLS:
                X[c] = ""
            else:
                X[c] = 0.0

    # 3) Tipado estricto
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("")
    for c in NUM_COLS:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)

    # 4) Orden igual al entrenamiento
    X = X[ALL_COLS]
    return X


# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "threshold": used_threshold,
        "has_var_params": bool(var_df is not None and len(var_df) > 0),
        "n_var_segments": int(len(var_df)) if var_df is not None else 0,
        "n_features": len(ALL_COLS),
    }


@app.post("/predict")
def predict(p: PredictIn):
    # 1) Probabilidad de pérdida
    X = to_model_df(p)
    prob = float(pipe.predict_proba(X)[0, 1])

    # 2) Umbral efectivo
    thr = (
        float(p.threshold_override)
        if p.threshold_override is not None
        else used_threshold
    )
    label = int(prob >= thr)

    # 3) Índice de riesgo
    R_index = float(p.consequence) * prob

    # 4) VaR̂ del segmento (si existe)
    segment = build_segment(p)
    VA_seg = O_seg = Z = T = None
    var_source = None
    VaR_hat = 0.0

    params = lookup_var_params(segment)
    if params:
        VA_seg, O_seg, Z, T, var_source = params
        try:
            VaR_hat = float(Z * VA_seg * O_seg * np.sqrt(T))
        except Exception:
            VaR_hat = 0.0

    # 5) Respuesta
    return {
        "risk_proba": prob,
        "used_threshold": float(thr),
        "label": label,
        "R_index": R_index,
        "segment": segment,
        # VaR con desglose
        "VaR_hat": VaR_hat,
        "var_source": var_source,
        "VA_seg": VA_seg,
        "O_seg": O_seg,
        "Z": Z,
        "T": T,
    }
