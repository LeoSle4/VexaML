# src/etl/02_quality_and_features.py
import pandas as pd, numpy as np, json
from pathlib import Path

IN_PATH = Path("data/processed/events_stage1.parquet")
OUT_PATH = Path("data/processed/events.parquet")
REPORT_PATH = Path("reports/etl_quality.json")

# Columnas que NO deben entrar al modelo (fuga de información / post-evento)
ANTI_LEAK = {
    "owner_person",
    "date_end",
    "date_report_ero",
    "event_state",
    "ero_reported_by",
    "doc_owner",
    "date_account",
    "account_code",
    "followup_owner",
    "action_plan",
    "date_committee",
    "board_flag",
    "date_recorded",
    "loss_gross_usd",
    "reco_ins",
    "reco_self",
    "reco_net",
    "loss_net",
    "loss_net_usd",
    "loss_value_usd",
    "currency",
}
TEXT_COLS = ["event_detail", "event_other"]


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date_ident"].dt.year
    df["month"] = df["date_ident"].dt.month
    df["quarter"] = df["date_ident"].dt.quarter
    df["weekday"] = df["date_ident"].dt.weekday
    df["is_month_end"] = df["date_ident"].dt.is_month_end.astype(int)
    if "date_start" in df.columns:
        lag = (df["date_ident"] - df["date_start"]).dt.days
        df["lag_start_ident_days"] = lag.fillna(lag.median())
    else:
        df["lag_start_ident_days"] = 0
    return df


def add_windows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    df = df.sort_values("date_ident").copy()
    for w in (30, 60, 90):
        df[f"n_events_{w}d_{key}"] = 0
    df[f"days_since_last_{key}"] = 9999
    for _, g in df.groupby(key, dropna=False):
        idx = g.index
        dates = g["date_ident"].values
        for i, d in enumerate(dates):
            df.loc[idx[i], f"n_events_30d_{key}"] = (
                (dates < d) & (dates >= d - np.timedelta64(30, "D"))
            ).sum()
            df.loc[idx[i], f"n_events_60d_{key}"] = (
                (dates < d) & (dates >= d - np.timedelta64(60, "D"))
            ).sum()
            df.loc[idx[i], f"n_events_90d_{key}"] = (
                (dates < d) & (dates >= d - np.timedelta64(90, "D"))
            ).sum()
            prev = dates[dates < d]
            if prev.size:
                df.loc[idx[i], f"days_since_last_{key}"] = int(
                    (d - prev[-1]) / np.timedelta64(1, "D")
                )
    return df


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"No existe {IN_PATH}. Corre primero 01_read_normalize.py")

    df = pd.read_parquet(IN_PATH)

    quality = {
        "rows": int(len(df)),
        "nulls_by_col": df.isna().sum().to_dict(),
        "dupes_event_id": int(
            df.duplicated("event_id").sum() if "event_id" in df.columns else 0
        ),
    }

    # Filtra registros válidos
    if "date_ident" not in df.columns:
        raise SystemExit("Falta la columna 'date_ident' en stage1.")
    df = df[df["date_ident"].notna()].copy()

    # Etiqueta (VD binaria para ML)
    df["loss_event"] = (df.get("loss_net_usd", 0).fillna(0) > 0).astype(int)

    # Derivados temporales y de recencia/frecuencia
    df = add_time(df)
    for key in ["unit_impacted", "process_impacted", "cat_n1"]:
        if key in df.columns:
            df = add_windows(df, key)

    # Texto simple
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("")
            df[f"{c}_len"] = df[c].str.len()
            df[f"kw_fraude_in_{c}"] = (
                df[c]
                .str.contains(
                    "fraude|phishing|suplantaci|apoderamiento", case=False, regex=True
                )
                .astype(int)
            )

    # Cat path
    if {"cat_n1", "cat_n2", "cat_n3"}.issubset(df.columns):
        df["cat_path"] = (
            df["cat_n1"].fillna("")
            + ">"
            + df["cat_n2"].fillna("")
            + ">"
            + df["cat_n3"].fillna("")
        )

    # Imputa categóricas nulas a 'MISSING' (robusto para OneHotEncoder)
    cat_cols = df.select_dtypes(include="object").columns.difference(TEXT_COLS)
    df[cat_cols] = df[cat_cols].fillna("MISSING")

    # Eliminar fuga + texto crudo + quitar fechas base del dataset final
    drop_cols = list(ANTI_LEAK.union(TEXT_COLS).union({"date_ident", "date_start"}))
    keep = [c for c in df.columns if c not in drop_cols]
    df_out = df[keep].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT_PATH, index=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print(f"[OK] {OUT_PATH} ({len(df_out)} filas, {len(df_out.columns)} columnas)")
    print(f"[OK] Reporte -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
