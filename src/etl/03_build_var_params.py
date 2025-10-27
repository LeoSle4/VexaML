import numpy as np
import pandas as pd
import yaml
from pathlib import Path

CFG = yaml.safe_load(Path("configs/config.yml").read_text())

STAGE1 = Path(CFG["data"]["stage1_path"])
PROC = Path(CFG["data"]["processed_path"])
OUT = Path(CFG["data"]["var_params_path"])

SEG_KEYS = CFG["var_build"][
    "segment_keys"
]  # p.ej. ["event_type","unit_impacted","product_service"]
RULE = CFG["var_build"].get("resample_rule", "MS")  # mensual: inicio de mes
P_VA = float(CFG["var_build"]["va_percentile"])
MIN_VA = float(CFG["var_build"]["min_va"])
W_LOW = float(CFG["var_build"]["winsor_low"])
W_HIGH = float(CFG["var_build"]["winsor_high"])
MIN_M = int(CFG["var_build"]["min_months"])

Z_DEFAULT = float(CFG["var"]["z_value"])
T_DEFAULT = float(CFG["var"]["horizon_T"])


def winsorize(s: pd.Series, low=1, high=99) -> pd.Series:
    if len(s) == 0:
        return s
    lo, hi = np.nanpercentile(s, [low, high])
    return s.clip(lo, hi)


def join_keys(df: pd.DataFrame, keys) -> pd.Series:
    arr = df[keys[0]].astype(str)
    for k in keys[1:]:
        arr = arr.str.cat(df[k].astype(str), sep="|")
    return arr


def monthly_sum(obj, col: str) -> pd.Series:
    """Acepta DataFrame (obj[col]) o Serie (obj); devuelve serie mensual resampleada y rellenada a índice continuo."""
    s = obj[col] if isinstance(obj, pd.DataFrame) else obj
    s = s.resample(RULE).sum()
    if not s.empty:
        idx = pd.period_range(
            s.index.min().to_period("M").start_time,
            s.index.max().to_period("M").end_time,
            freq="M",
        ).to_timestamp()
        s = s.reindex(idx, fill_value=0.0)  # <- acá sí se permite fill_value
    return s


def calc_params_from_series(monthly: pd.Series):
    """Devuelve (VA_seg, O_seg) desde una serie mensual de pérdidas USD."""
    if monthly.empty:
        return (MIN_VA, 0.0)
    va = (
        float(np.nanpercentile(monthly[monthly > 0], P_VA))
        if (monthly > 0).any()
        else 0.0
    )
    va = float(max(va, MIN_VA))
    rate = winsorize(monthly / va, W_LOW, W_HIGH) if va > 0 else monthly * 0.0
    oseg = float(np.nanstd(rate.values, ddof=1))
    return (va, oseg)


def main():
    if not STAGE1.exists():
        raise SystemExit(f"Falta {STAGE1}. Corre 01_read_normalize.py")

    s1 = pd.read_parquet(STAGE1)
    s1["date_ident"] = pd.to_datetime(s1["date_ident"], errors="coerce")
    # elige la mejor columna de pérdida en USD
    loss_col = "loss_value_usd" if "loss_value_usd" in s1.columns else "loss_net_usd"
    s1[loss_col] = pd.to_numeric(s1[loss_col], errors="coerce").fillna(0.0)

    # normaliza llaves del segmento
    for k in SEG_KEYS:
        if k not in s1.columns:
            s1[k] = "MISSING"
        s1[k] = s1[k].astype(str).fillna("MISSING")

    # lista de segmentos que nos interesan (los presentes en processed si existe, si no del stage1)
    if PROC.exists():
        proc = pd.read_parquet(PROC)
        for k in SEG_KEYS:
            if k not in proc.columns:
                proc[k] = "MISSING"
        segments = (
            join_keys(proc[SEG_KEYS].fillna("MISSING"), SEG_KEYS).unique().tolist()
        )
    else:
        segments = join_keys(s1[SEG_KEYS], SEG_KEYS).unique().tolist()

    s1 = s1.dropna(subset=["date_ident"]).set_index("date_ident").sort_index()

    rows = []
    for seg in segments:
        # desarma seg a dict {key: value}
        parts = seg.split("|")
        filt = pd.Series(True, index=s1.index)
        for k, v in zip(SEG_KEYS, parts):
            filt &= s1[k].astype(str) == v

        g = s1.loc[filt, [loss_col] + SEG_KEYS].copy()
        monthly = monthly_sum(g, loss_col)

        # 1) intento a nivel segmento exacto
        months = len(monthly)
        va, o = calc_params_from_series(monthly)
        src = "segment"
        # 2) fallback (evento, producto)
        if (months < MIN_M) or (o == 0.0 and monthly.sum() == 0.0):
            if len(SEG_KEYS) >= 2:
                k2 = [SEG_KEYS[0], SEG_KEYS[-1]]
                filt2 = (s1[SEG_KEYS[0]] == parts[0]) & (s1[SEG_KEYS[-1]] == parts[-1])
                monthly2 = monthly_sum(s1.loc[filt2, [loss_col]], loss_col)
                va2, o2 = calc_params_from_series(monthly2)
                if len(monthly2) >= max(3, MIN_M // 2) and (
                    o2 > 0 or monthly2.sum() > 0
                ):
                    va, o, src = va2, o2, "event+product"
                    months = len(monthly2)
        # 3) fallback (evento)
        if o == 0.0 and monthly.sum() == 0.0:
            filt3 = s1[SEG_KEYS[0]] == parts[0]
            monthly3 = monthly_sum(s1.loc[filt3, [loss_col]], loss_col)
            va3, o3 = calc_params_from_series(monthly3)
            if len(monthly3) >= max(3, MIN_M // 2) and (o3 > 0 or monthly3.sum() > 0):
                va, o, src = va3, o3, "event"
                months = len(monthly3)
        # 4) última opción: volatilidad de frecuencia (proxy)
        if o == 0.0:
            # std de tasa de eventos/mes como proxy
            mcount = g.resample(RULE)[loss_col].count()
            if not monthly.empty:
                mcount = mcount.reindex(monthly.index, fill_value=0.0)

            if not mcount.empty and mcount.sum() > 0:
                rate = winsorize(mcount / max(mcount.max(), 1.0), W_LOW, W_HIGH)
                o = float(np.nanstd(rate.values, ddof=1))
                src = src + "+freq"

        rows.append(
            {
                "segment": seg,
                "VA_seg": float(max(va, MIN_VA)),
                "O_seg": float(o),
                "months": int(months),
                "loss_month_sum": float(monthly.sum()),
                "source": src,
            }
        )

    var_params = pd.DataFrame(rows)
    var_params["Z"] = Z_DEFAULT
    var_params["T"] = T_DEFAULT

    OUT.parent.mkdir(parents=True, exist_ok=True)
    var_params.to_parquet(OUT, index=False)
    print(f"[OK] Guardado {OUT} con {len(var_params)} segmentos.")
    print(var_params.sort_values("O_seg", ascending=False).head(10))


if __name__ == "__main__":
    main()
