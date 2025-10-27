# src/rc.py
from __future__ import annotations
import pandas as pd
import yaml
from pathlib import Path
from datetime import date

CFG = yaml.safe_load(Path("configs/config.yml").read_text())
RC_CFG = CFG.get("rc", {})
ALPHA = float(RC_CFG.get("alpha", 0.15))
N_YEARS = int(RC_CFG.get("n_years", 3))
INCOME_PATH = Path(RC_CFG.get("incomes_path", "data/finance/gross_income.csv"))
BY_UNIT = bool(RC_CFG.get("by_unit", False))

THR = CFG.get("rc_thresholds", {})
T_LOW = float(THR.get("low", 0))
T_MED = float(THR.get("medium", 1e12))


def _load_incomes() -> pd.DataFrame:
    if not INCOME_PATH.exists():
        raise FileNotFoundError(f"No existe {INCOME_PATH}")
    df = pd.read_csv(INCOME_PATH)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "unit_impacted" in df.columns:
        df["unit_impacted"] = df["unit_impacted"].astype(str).str.strip()
    df["ib"] = pd.to_numeric(df["ib"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["year"])


def _classify_rc(x: float) -> str:
    if x < T_LOW:
        return "Bajo"
    if x < T_MED:
        return "Moderado"
    return "Alto"


def _window(df: pd.DataFrame, asof_year: int) -> pd.DataFrame:
    years = list(range(asof_year - N_YEARS + 1, asof_year + 1))
    return df[df["year"].isin(years)].copy()


def rc_bia_bank(asof_year: int | None = None) -> dict:
    df = _load_incomes()
    if asof_year is None:
        asof_year = date.today().year
    w = _window(df, asof_year)
    if w.empty:
        return {
            "rc": 0.0,
            "avg_ib": 0.0,
            "years_used": [],
            "alpha": ALPHA,
            "method": "BIA",
            "level": "bank",
        }
    w["ib_pos"] = w["ib"].clip(lower=0.0)
    avg_ib = w["ib_pos"].mean()
    rc = ALPHA * avg_ib
    return {
        "rc": float(rc),
        "avg_ib": float(avg_ib),
        "years_used": [int(y) for y in w["year"].tolist()],
        "alpha": ALPHA,
        "method": "BIA",
        "level": "bank",
        "category": _classify_rc(float(rc)),
    }


def rc_bia_unit(asof_year: int | None, unit: str | None) -> dict:
    df = _load_incomes()
    if asof_year is None:
        asof_year = date.today().year
    if unit is None or "unit_impacted" not in df.columns:
        return rc_bia_bank(asof_year)

    dfu = df[df["unit_impacted"].str.lower() == str(unit).strip().lower()]
    if dfu.empty:
        # fallback banco si no hay datos para la unidad
        return rc_bia_bank(asof_year)

    w = _window(dfu, asof_year)
    if w.empty:
        return rc_bia_bank(asof_year)

    w["ib_pos"] = w["ib"].clip(lower=0.0)
    avg_ib = w["ib_pos"].mean()
    rc = ALPHA * avg_ib
    return {
        "rc": float(rc),
        "avg_ib": float(avg_ib),
        "years_used": [int(y) for y in w["year"].tolist()],
        "alpha": ALPHA,
        "method": "BIA",
        "level": "unit",
        "unit_impacted": unit,
        "category": _classify_rc(float(rc)),
    }


def get_rc(asof_year: int | None = None, unit: str | None = None) -> dict:
    if BY_UNIT:
        return rc_bia_unit(asof_year, unit)
    return rc_bia_bank(asof_year)


if __name__ == "__main__":
    # CLI rápido: imprime RC banco y, si hay columna unit_impacted, RC por unidad
    info_bank = rc_bia_bank()
    print("[BANK]", info_bank)
    df = _load_incomes()
    if "unit_impacted" in df.columns:
        for u in sorted(df["unit_impacted"].dropna().unique()):
            print("[UNIT]", u, rc_bia_unit(date.today().year, u))
