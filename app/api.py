# app/api.py
from __future__ import annotations
import json
from math import sqrt
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]   # .../VexaML
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rc import get_rc  # << integra tu RC (BIA) banco/unidad

# ----------------- Carga de config y artefactos -----------------
CFG = yaml.safe_load(Path("configs/config.yml").read_text())

MODEL_PATH = Path("models/model.pkl")
THRESH_PATH = Path("models/threshold.json")
SCHEMA_PATH = Path("models/feature_schema.json")
VAR_PARAMS_PATH = Path(CFG["data"]["var_params_path"])
STAGE1_PATH = Path(CFG["data"]["stage1_path"])

SEG_KEYS = CFG["var"][
    "segment_keys"
]  # ["event_type","unit_impacted","product_service"]


def _load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No existe {MODEL_PATH}. Entrena el modelo primero.")
    return joblib.load(MODEL_PATH)


def _load_threshold() -> float:
    if THRESH_PATH.exists():
        return float(json.loads(THRESH_PATH.read_text())["threshold"])
    return 0.5


def _load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise RuntimeError(f"Falta {SCHEMA_PATH}. Corre src/train.py.")
    return json.loads(SCHEMA_PATH.read_text())


def _load_var_params() -> pd.DataFrame | None:
    if not VAR_PARAMS_PATH.exists():
        return None
    df = pd.read_parquet(VAR_PARAMS_PATH)
    # columnas esperadas: segment, VA_seg, O_seg, Z, T, (opcional) source
    for c in ["segment", "VA_seg", "O_seg"]:
        if c not in df.columns:
            return None
    if "Z" not in df.columns:
        df["Z"] = float(CFG["var"]["z_value"])
    if "T" not in df.columns:
        df["T"] = float(CFG["var"]["horizon_T"])
    if "source" not in df.columns:
        df["source"] = "segment"
    return df


PIPE = _load_model()
THRESH = _load_threshold()
SCHEMA = _load_schema()
VARP = _load_var_params()

# ----------------- FastAPI -----------------
app = FastAPI(title="Risk-ML API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Utils -----------------
def _segment_from_payload(p: Dict[str, Any]) -> str:
    parts = []
    for k in SEG_KEYS:
        parts.append(str(p.get(k, "MISSING")))
    return "|".join(parts)


def _make_feature_row(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea un DataFrame con el mismo orden/columnas que entrenaste.
    - num_cols -> float
    - cat_cols -> str
    - llena faltantes con 0 / 'MISSING'
    """
    all_cols = SCHEMA["all_columns"]
    num_cols = set(SCHEMA["num_cols"])
    cat_cols = set(SCHEMA["cat_cols"])

    data = {}
    for c in all_cols:
        v = payload.get(c, None)
        if c in num_cols:
            try:
                data[c] = float(v) if v is not None else 0.0
            except Exception:
                data[c] = 0.0
        elif c in cat_cols:
            data[c] = str(v) if v is not None and str(v) != "" else "MISSING"
        else:
            # por seguridad, si quedó algo fuera de num/cat, lo forcemos a string
            data[c] = str(v) if v is not None else "MISSING"

    df = pd.DataFrame([data], columns=all_cols)
    # Evita errores de OHE por NaN en categories_
    for c in cat_cols:
        df[c] = df[c].astype(str)
    return df


def _var_for_segment(segment: str) -> Dict[str, Any]:
    if VARP is None:
        return {"VaR_hat": 0.0, "var_source": "missing", "VA_seg": 0.0, "O_seg": 0.0}
    row = VARP.loc[VARP["segment"] == segment]
    if row.empty:
        return {"VaR_hat": 0.0, "var_source": "missing", "VA_seg": 0.0, "O_seg": 0.0}
    r = row.iloc[0]
    VaR_hat = (
        float(r["Z"]) * float(r["VA_seg"]) * float(r["O_seg"]) * sqrt(float(r["T"]))
    )
    return {
        "VaR_hat": float(VaR_hat),
        "var_source": str(r.get("source", "segment")),
        "VA_seg": float(r["VA_seg"]),
        "O_seg": float(r["O_seg"]),
    }


def _asof_year(payload: Dict[str, Any]) -> int:
    # intenta usar el año del payload (fecha de identificación en UI)
    y = payload.get("year", None)
    if y is not None:
        try:
            return int(y)
        except Exception:
            pass
    # si no, usa el último año en stage1
    if STAGE1_PATH.exists():
        try:
            di = pd.read_parquet(STAGE1_PATH, columns=[CFG["data"]["date_ident_col"]])
            di = pd.to_datetime(di[CFG["data"]["date_ident_col"]], errors="coerce")
            if di.notna().any():
                return int(di.dt.year.max())
        except Exception:
            pass
    # fallback: año actual
    from datetime import date

    return date.today().year


# ----------------- Endpoints -----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "threshold": THRESH,
        "has_var_params": VARP is not None,
        "has_stage1": STAGE1_PATH.exists(),
        "schema_cols": len(SCHEMA["all_columns"]),
    }


@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    # 1) DataFrame con el esquema entrenado
    X = _make_feature_row(payload)

    # 2) Probabilidad con el pipeline
    proba = float(PIPE.predict_proba(X)[0, 1])

    # 3) Umbral (con override si viene en payload)
    th = float(payload.get("threshold_override", THRESH))
    label = int(proba >= th)

    # 4) Índice de riesgo R = C × P̂ si enviaste "consequence"
    consequence = payload.get("consequence", None)
    R_index = None
    if consequence is not None:
        try:
            R_index = float(consequence) * proba
        except Exception:
            R_index = None

    # 5) VaR̂ por segmento
    seg = _segment_from_payload(payload)
    var_info = _var_for_segment(seg)

    # 6) RC (BIA). Si rc.by_unit=true en config, se calcula por unidad; si no, banco.
    asof_year = _asof_year(payload)
    unit = payload.get("unit_impacted", None)
    rc_info = get_rc(asof_year=asof_year, unit=unit)

    out = {
        "risk_proba": proba,
        "label": label,
        "used_threshold": th,
        "segment": seg,
        "R_index": R_index,
        # VaR̂
        "VaR_hat": var_info["VaR_hat"],
        "var_source": var_info["var_source"],
        "VA_seg": var_info["VA_seg"],
        "O_seg": var_info["O_seg"],
        # RC
        "rc": rc_info,  # incluye rc, avg_ib, years_used, method, level, (unit), category
    }
    return out
