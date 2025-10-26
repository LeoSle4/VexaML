import pandas as pd
from pathlib import Path

CFG_IN = Path("configs/config.yml")
RAW_PATH = Path("data/raw/events.xlsx")  # cambia si tu archivo tiene otro nombre
OUT_PATH = Path("data/processed/events_stage1.parquet")

# Mapear cabeceras originales -> nombres canónicos (snake_case)
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
    df = pd.read_excel(RAW_PATH, dtype=str)
    df = df.rename(columns=COLUMN_MAP)

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

    # Numéricos
    for c in [
        "loss_gross_usd",
        "reco_ins",
        "reco_self",
        "reco_net",
        "loss_net",
        "loss_net_usd",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].str.replace(",", "", regex=False), errors="coerce"
            )

    # ID si falta
    if "event_id" not in df.columns or df["event_id"].isna().any():
        df["event_id"] = (
            df.get("unit_impacted", "").astype(str)
            + "|"
            + df.get("event_type", "").astype(str)
            + "|"
            + df.get("date_ident", "").astype(str)
        ).fillna("")
        df["event_id"] = df["event_id"].astype("category").cat.codes.astype(str)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[OK] {OUT_PATH} ({len(df)} filas)")


if __name__ == "__main__":
    main()
