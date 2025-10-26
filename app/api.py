from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib, json, yaml
import pandas as pd
from pathlib import Path

CFG = yaml.safe_load(Path("configs/config.yml").read_text())
MODEL_PATH = Path("models/model.pkl")
THRESH_PATH = Path("models/threshold.json")
VAR_PARAMS_PATH = Path(CFG["data"]["var_params_path"])

app = FastAPI(title="Risk-ML API")

pipe = joblib.load(MODEL_PATH)
threshold = json.loads(THRESH_PATH.read_text())["threshold"] if THRESH_PATH.exists() else 0.5
var_params = pd.read_parquet(VAR_PARAMS_PATH) if VAR_PARAMS_PATH.exists() else pd.DataFrame(columns=["segment","VA_seg","O_seg","Z","T"])

SEG_KEYS = CFG["var"]["segment_keys"]
Z_DEFAULT = CFG["var"]["z_value"]
T_DEFAULT = CFG["var"]["horizon_T"]

class EventIn(BaseModel):
    unit_impacted: str
    product_service: str
    process_impacted: str
    unit_origin: str | None = None
    area_origin: str | None = None
    process_origin: str | None = None
    source_origin: str | None = None
    event_type: str
    risk_assoc: str | None = None
    geo_zone: str | None = None
    cat_n1: str | None = None
    cat_n2: str | None = None
    cat_n3: str | None = None
    year: int
    month: int
    quarter: int
    weekday: int
    is_month_end: int
    n_events_30d_unit_impacted: float | None = None
    n_events_60d_unit_impacted: float | None = None
    n_events_90d_unit_impacted: float | None = None
    days_since_last_unit_impacted: float | None = None
    consequence: float | None = Field(default=None, description="Escala 1-5")

def seg_from_row(row: dict) -> str:
    return "|".join(str(row.get(k, "")) for k in SEG_KEYS)

@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold, "has_var_params": not var_params.empty}

@app.post("/predict")
def predict(e: EventIn):
    x = pd.DataFrame([e.model_dump()])
    proba = float(pipe.predict_proba(x)[0,1])
    label = int(proba >= threshold)

    R_hat = float(e.consequence) * proba if e.consequence is not None else None

    seg = seg_from_row(x.iloc[0].to_dict())
    row = var_params[var_params["segment"] == seg]
    if not row.empty:
        Z = float(row["Z"].iloc[0]); T = float(row["T"].iloc[0])
        VA = float(row["VA_seg"].iloc[0]); O = float(row["O_seg"].iloc[0])
    else:
        Z, T, VA, O = Z_DEFAULT, T_DEFAULT, 1.0, 0.0
    VaR_hat = float(Z * VA * O * (T ** 0.5))

    return {"risk_proba": proba, "label": label, "R_index": R_hat,
            "segment": seg, "VaR_hat": VaR_hat, "used_threshold": threshold}