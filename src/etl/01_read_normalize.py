# src/etl/01_read_normalize.py
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/events.xlsx")
OUT_PATH = Path("data/processed/events_stage1.parquet")

COLUMN_MAP = {
    "Nro.": "event_id",
    "Unidad Impactada": "unit_impacted",
    "Prod. / Serv. Afectado": "product_service",
    "Proceso Afectado": "process_impacted",
    "Unidad Originadora": "unit_origin",
    "Area Originadora": "area_origin",
    "Proceso Afectado Originador": "process_origin",
    "Responsable del Evento": "owner_person",
    "Inicio del Evento": "date_start",
    "Fin del Evento": "date_end",
    "Fec. de Identificación": "date_ident",
    "Fuente de Origen": "source_origin",
    "Tipo de Evento": "event_type",
    "Detalle del Evento": "event_detail",
    "Riesgo Asociado": "risk_assoc",
    "Otro (descripción)": "event_other",
    "Moneda": "currency",
    "Pérdida Bruta USD": "loss_gross_usd",
    "Fecha de Recuperación por Seguros": "date_reco_ins",
    "Recupero por Seguros": "reco_ins",
    "Fecha de Recuperación por Gestión Propia": "date_reco_self",
    "Recupero por Gestión Propia": "reco_self",
    "Recuperación Neta": "reco_net",
    "Pérdida Neta": "loss_net",
    "Pérdida Neta USD": "loss_net_usd",
    "Zona Geográfica": "geo_zone",
    "Categoría del Evento (Nivel 1)": "cat_n1",
    "Categoría del Evento (Nivel 2)": "cat_n2",
    "Categoría del Evento (Nivel 3)": "cat_n3",
    "Otros Tipos de Riesgos": "risk_other_types",
    "Fecha de Reporte ERO": "date_report_ero",
    "Estado del Evento": "event_state",
    "ERO reportado por": "ero_reported_by",
    "Área": "area_generic",
    "Responsable de documentar": "doc_owner",
    "Fecha de Contabilización": "date_account",
    "Cuenta Contable Afectada": "account_code",
    "Responsable del Seguimiento": "followup_owner",
    "Plan de Acción Asociado": "action_plan",
    "Fec. de Present. C.R.": "date_committee",
    "Tablero Presentación": "board_flag",
    "F. Reg. Base": "date_recorded",
}


def main():
    df = pd.read_excel(RAW_PATH, dtype=str, engine="openpyxl").rename(
        columns=COLUMN_MAP
    )

    # Limpia espacios
    obj = df.select_dtypes(include="object").columns
    df[obj] = df[obj].apply(lambda s: s.str.strip())

    # Fechas
    for c in [
        "date_start",
        "date_end",
        "date_ident",
        "date_reco_ins",
        "date_reco_self",
        "date_report_ero",
        "date_account",
        "date_committee",
        "date_recorded",
    ]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Parser de dinero robusto
    def parse_money(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (1,234.56) -> -1,234.56
        s = s.str.replace(r"[^\d,.\-]", "", regex=True)  # quita símbolos
        mask_both = s.str.contains(",") & s.str.contains(r"\.")
        s = s.where(
            ~mask_both,
            s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        )  # 1.234,56 -> 1234.56
        mask_comma = s.str.contains(",") & ~mask_both
        s = s.where(
            ~mask_comma, s.str.replace(",", ".", regex=False)
        )  # 123,45 -> 123.45
        return pd.to_numeric(s, errors="coerce")

    for c in [
        "loss_gross_usd",
        "reco_ins",
        "reco_self",
        "reco_net",
        "loss_net",
        "loss_net_usd",
    ]:
        if c in df.columns:
            df[c] = parse_money(df[c])

    # --- Moneda + USD canónico ---
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    else:
        df["currency"] = "USD"

    fx_map = {"USD": 1.0, "US$": 1.0, "PEN": 1 / 3.70, "S/": 1 / 3.70}

    def to_usd(amount, cur):
        return float(amount) * float(fx_map.get(cur, 1.0))

    usd = df.get("loss_net_usd", 0).fillna(0)
    pen = df.get("loss_net", 0).fillna(0)
    df["loss_value_usd"] = usd.where(
        usd > 0, [to_usd(a, c) for a, c in zip(pen, df["currency"])]
    )

    # Auditoría
    pos = (df["loss_value_usd"].fillna(0) > 0).sum()
    total = float(df["loss_value_usd"].fillna(0).sum())
    print(f"[CHK] loss_value_usd > 0: {pos} | suma USD: {total:,.2f}")

    # ID: rellena faltantes y garantiza unicidad
    if "event_id" not in df.columns:
        df["event_id"] = ""
    miss = df["event_id"].isna() | (df["event_id"].astype(str).str.strip() == "")
    fb_all = (
        df.get("unit_impacted", "").astype(str).str.strip()
        + "|"
        + df.get("event_type", "").astype(str).str.strip()
        + "|"
        + df.get("date_ident", "").astype(str)
    )
    codes_all = pd.Series(pd.factorize(fb_all)[0].astype(str), index=df.index)
    df.loc[miss, "event_id"] = codes_all.loc[miss]
    dup_rank = df.groupby("event_id").cumcount()
    df.loc[dup_rank > 0, "event_id"] = (
        df.loc[dup_rank > 0, "event_id"] + "_" + dup_rank[dup_rank > 0].astype(str)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[OK] {OUT_PATH} ({len(df)} filas)")


if __name__ == "__main__":
    main()
