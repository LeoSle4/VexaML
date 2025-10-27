# src/impact.py
from __future__ import annotations
import argparse
import json
import yaml
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

from rc import get_rc  # calcula RC según config (BIA banco o por unidad)

CFG = yaml.safe_load(Path("configs/config.yml").read_text())


# ------------------------------ Utilidades ------------------------------
def simulate_impact(
    recall: float,
    precision: float,
    m_effectiveness: float = 0.5,
    review_cost: float = 0.0,
    loss_mean: float = 10_000.0,
    n_pos: int = 100,
    n_total: int = 1_000,
) -> dict:
    """
    Modelo simple de impacto económico:
      - TP = recall * n_pos
      - FN = (1 - recall) * n_pos
      - FP = TP * (1 - precision) / precision
      - L_post = FN*loss_mean + (1-m)*TP*loss_mean + review_cost*(TP+FP)
      - L_base = n_pos*loss_mean
    """
    p_safe = max(precision, 1e-9)
    TP = recall * n_pos
    FN = (1 - recall) * n_pos
    FP = TP * (1 - p_safe) / p_safe

    L_post = (
        FN * loss_mean
        + (1 - m_effectiveness) * TP * loss_mean
        + review_cost * (TP + FP)
    )
    L_base = n_pos * loss_mean
    delta = L_base - L_post

    return {
        "TP": float(TP),
        "FN": float(FN),
        "FP": float(FP),
        "L_base": float(L_base),
        "L_post": float(L_post),
        "ahorro": float(delta),
        "ahorro_pct": float(100 * delta / max(L_base, 1e-9)),
    }


def estimate_loss_mean_from_stage1() -> float | None:
    """
    Estima la pérdida media por evento a partir del stage1 (solo positivos),
    con winsorización 1–99% para robustez.
    Usa el nombre de columna definido en config: data.loss_net_usd_col
    """
    stage1_path = Path(
        CFG["data"].get("stage1_path", "data/processed/events_stage1.parquet")
    )
    col = CFG["data"]["loss_net_usd_col"]
    if not stage1_path.exists():
        return None

    df = pd.read_parquet(stage1_path)
    if col not in df.columns:
        return None

    s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    s = s[s > 0]
    if len(s) == 0:
        return None

    lo, hi = np.nanpercentile(s, [1, 99])
    s = s.clip(lo, hi)
    return float(s.mean())


def infer_asof_year_from_stage1(default_year: int | None = None) -> int:
    """
    Busca el último año disponible en date_ident del stage1 para usarlo
    como 'as-of' del RC. Si no puede, retorna el año actual o default_year.
    """
    stage1_path = Path(
        CFG["data"].get("stage1_path", "data/processed/events_stage1.parquet")
    )
    date_col = CFG["data"].get("date_ident_col", "date_ident")

    if stage1_path.exists():
        try:
            df = pd.read_parquet(stage1_path, columns=[date_col])
            di = pd.to_datetime(df[date_col], errors="coerce")
            if di.notna().any():
                return int(di.dt.year.max())
        except Exception:
            pass

    if default_year is not None:
        return int(default_year)
    return date.today().year


# ------------------------------ Main ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Simulación de impacto económico y RC."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Año as-of para RC (por defecto: último año en stage1 o año actual).",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default=None,
        help="Unidad para RC por unidad (si rc.by_unit=true).",
    )
    parser.add_argument(
        "--m", type=float, default=None, help="Efectividad de mitigación m (override)."
    )
    parser.add_argument(
        "--review-cost",
        type=float,
        default=None,
        help="Costo por alerta revisada (override).",
    )
    parser.add_argument(
        "--loss-mean",
        type=float,
        default=None,
        help="Pérdida media por evento (override).",
    )
    args = parser.parse_args()

    # 1) Cargar métricas del test
    rep_path = Path("reports/metrics_test.json")
    if not rep_path.exists():
        raise SystemExit(
            "Ejecuta primero src/evaluate.py para generar reports/metrics_test.json"
        )
    rep = json.loads(rep_path.read_text())

    rec = float(rep["metrics"]["recall"]["value"])
    pre = float(rep["metrics"]["precision"]["value"])
    n_test = int(rep["n_test"])
    prevalence = float(rep["prevalence"])
    n_pos = int(round(prevalence * n_test))

    # 2) Parámetros de impacto (con overrides si se pasan por CLI)
    m = float(args.m) if args.m is not None else float(CFG["impact"]["m_effectiveness"])
    review_cost = (
        float(args.review_cost)
        if args.review_cost is not None
        else float(CFG["impact"]["review_cost_per_alert"])
    )

    # 3) Pérdida media estimada desde stage1 (winsorizada); si se pasa por CLI, usa override
    loss_mean_auto = estimate_loss_mean_from_stage1()
    loss_mean = (
        float(args.loss_mean)
        if args.loss_mean is not None
        else (loss_mean_auto if loss_mean_auto is not None else 10_000.0)
    )

    # 4) Año y unidad para RC
    asof_year = (
        int(args.year) if args.year is not None else infer_asof_year_from_stage1()
    )
    unit_for_rc = (
        args.unit
    )  # None -> banco; si rc.by_unit=true y se pasa unidad, calculará por unidad

    # 5) Calcular RC dinámico por BIA (o fallback)
    try:
        rc_info = get_rc(asof_year=asof_year, unit=unit_for_rc)
        rc_pre = float(rc_info.get("rc", 0.0))
        rc_method = rc_info.get("method", "BIA")
    except Exception:
        # Fallback a rc_pre estático de config si rc.py falla o no hay datos
        rc_pre = float(CFG["impact"].get("rc_pre", 0.0))
        rc_info = {
            "rc": rc_pre,
            "method": "CONST",
            "alpha": None,
            "years_used": [],
            "level": "bank",
        }
        rc_method = "CONST"

    # 6) Simular impacto esperado con recall/precision del test
    sim = simulate_impact(
        recall=rec,
        precision=pre,
        m_effectiveness=m,
        review_cost=review_cost,
        loss_mean=loss_mean,
        n_pos=n_pos,
        n_total=n_test,
    )

    # 7) Reducir RC por la efectividad real (rho = m * recall)
    rho = m * rec
    rc_post = (1 - rho) * rc_pre

    out = {
        "input": {
            "recall": rec,
            "precision": pre,
            "m_effectiveness": m,
            "review_cost": review_cost,
            "loss_mean": loss_mean,
            "n_pos": n_pos,
            "n_total": n_test,
        },
        "impact": sim,
        "rc": {
            "rc_pre": rc_pre,
            "rc_post": float(rc_post),
            "reduction_rho": float(rho),
            "method": rc_method,
            "details": rc_info,  # incluye alpha, years_used, level, etc. si viene de rc.py
        },
        "sources": {
            "metrics_file": "reports/metrics_test.json",
            "stage1_file": CFG["data"].get(
                "stage1_path", "data/processed/events_stage1.parquet"
            ),
            "rc_incomes_file": CFG["rc"].get(
                "incomes_path", "data/finance/gross_income.csv"
            ),
            "asof_year": asof_year,
            "unit_for_rc": unit_for_rc,
        },
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/impact_sim.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
