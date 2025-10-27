import json, yaml
from pathlib import Path
import pandas as pd
import numpy as np

CFG = yaml.safe_load(Path("configs/config.yml").read_text())


def simulate_impact(
    recall,
    precision,
    m_effectiveness=0.5,
    review_cost=0.0,
    loss_mean=10000.0,
    n_pos=100,
    n_total=1000,
):
    TP = recall * n_pos
    FN = (1 - recall) * n_pos
    FP = TP * (1 - max(precision, 1e-9)) / max(precision, 1e-9)

    L_post = (
        FN * loss_mean
        + (1 - m_effectiveness) * TP * loss_mean
        + review_cost * (TP + FP)
    )
    L_base = n_pos * loss_mean
    delta = L_base - L_post
    return {
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "L_base": L_base,
        "L_post": L_post,
        "ahorro": delta,
        "ahorro_pct": 100 * delta / max(L_base, 1e-9),
    }


def estimate_loss_mean_from_stage1():
    """Usa pérdidas reales del stage1 para un loss medio robusto."""
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
    s = s[s > 0]  # solo positivos
    if len(s) == 0:
        return None
    # Media winsorizada (1–99%) para robustez
    lo, hi = np.nanpercentile(s, [1, 99])
    s = s.clip(lo, hi)
    return float(s.mean())


def main():
    rep_path = Path("reports/metrics_test.json")
    if not rep_path.exists():
        raise SystemExit("Ejecuta primero src/evaluate.py")
    rep = json.loads(rep_path.read_text())

    rec = rep["metrics"]["recall"]["value"]
    pre = rep["metrics"]["precision"]["value"]
    n_test = rep["n_test"]
    prevalence = rep["prevalence"]
    n_pos = int(prevalence * n_test)

    m = CFG["impact"]["m_effectiveness"]
    review_cost = CFG["impact"]["review_cost_per_alert"]
    rc_pre = CFG["impact"]["rc_pre"]

    # Estimar pérdida media real
    loss_mean = estimate_loss_mean_from_stage1()
    if loss_mean is None:
        loss_mean = 10000.0  # fallback
    sim = simulate_impact(rec, pre, m, review_cost, loss_mean, n_pos, n_test)

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
        "rc": {"rc_pre": rc_pre, "reduction_rho": rho, "rc_post": rc_post},
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/impact_sim.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
