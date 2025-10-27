# src/etl/03a_rc_build_from_monthly.py
from __future__ import annotations
import re, sys
import pandas as pd
from pathlib import Path

# === INPUT ===
IN_FILE = Path("data/finance/ib_2020_2024_mes.xlsx")  # pon aquí tu archivo
SHEET = 0  # si es Excel

# === OUTPUT esperado por rc.py ===
OUT_FILE = Path("data/finance/gross_income.csv")

# Columna con el nombre de la unidad (ajústalo si difiere)
PREFERRED_UNIT_COL = "INGRESO ENTIDADES (USD)"

# Mapeo opcional para alinear nombres a tu UI/ETL
UNIT_MAP = {
    "Credicorp Capital Bolsa": "Mercado de Capitales",
    "Credicorp Capital Fondos": "Gestión de Activos",
    "Credicorp Capital Servicios Financieros": "Finanzas Corporativas",
    "Credicorp Capital Sociedad Titulizadora": "Negocios de Confianza",
    # agrega más si necesitas
}

# Meses en español (acepta "set" o "sep")
MONTHS = {
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "set": 9,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
}
# Ej: "Ene-24", "Feb_2021", "Mar 20"
RE_M_Y = re.compile(r"^\s*([A-Za-z]{3,})[-_/ ]?(\d{2,4})\s*$", re.IGNORECASE)


def clean_money(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    # quita todo excepto dígitos, punto, coma y signo
    s = re.sub(r"[^\d,.\-]", "", s)
    # "1.234,56" -> "1234.56"
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    # "1,234,567" -> "1234567"
    elif "," in s and "." not in s:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def parse_month_col(name: str) -> tuple[int, int] | None:
    """
    Devuelve (year, month) para encabezados como 'Ene-20', 'Sep-2023', etc.
    Si la celda es una fecha real (excel), toma year/month de esa fecha.
    """
    # si es una fecha real parseable
    dt = pd.to_datetime(name, errors="coerce", dayfirst=True)
    if pd.notna(dt):
        return int(dt.year), int(dt.month)

    m = RE_M_Y.match(str(name))
    if not m:
        return None
    mon_raw, year_raw = m.group(1).strip().lower(), m.group(2)
    mon_key = mon_raw[:3]  # toma 3 chars (ene/feb/mar/abr/...)
    if mon_key not in MONTHS:
        return None
    month = MONTHS[mon_key]
    year = int(year_raw)
    if year < 100:  # 20 -> 2020
        year += 2000
    return year, month


def guess_unit_column(cols: list[str]) -> str:
    # 1) preferida explícita si existe
    for c in cols:
        if c.strip().lower() == PREFERRED_UNIT_COL.strip().lower():
            return c
    # 2) heurística
    heur = [
        c
        for c in cols
        if re.search(r"(unidad|entidad|empresa|per[úu]|nombre|descrip)", c, re.I)
    ]
    if heur:
        return heur[0]
    # 3) fallback: la primera no mes
    return cols[0]


def main():
    if not IN_FILE.exists():
        sys.exit(f"No existe {IN_FILE}. Ajusta IN_FILE en el script.")

    # lee Excel/CSV
    if IN_FILE.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(IN_FILE, sheet_name=SHEET, dtype=str)
    else:
        df = pd.read_csv(IN_FILE, dtype=str)

    df.columns = [str(c).strip() for c in df.columns]

    # separa columnas de meses (multi-año) de las otras
    month_cols, month_keys = [], {}  # name -> (year, month)
    for c in df.columns:
        ym = parse_month_col(c)
        if ym is not None:
            month_cols.append(c)
            month_keys[c] = ym

    if not month_cols:
        sys.exit(
            "No se detectaron columnas de meses (Ene-20 .. Dic-24). Revisa encabezados."
        )

    other_cols = [c for c in df.columns if c not in month_cols]
    if not other_cols:
        sys.exit("No se encontró columna de 'unidad'.")
    unit_col = guess_unit_column(other_cols)

    # normaliza unidad y mapea
    df[unit_col] = df[unit_col].astype(str).str.strip()
    df["unit_impacted"] = df[unit_col].map(UNIT_MAP).fillna(df[unit_col])

    # construye formato largo: (unit_impacted, year, month, ib_mes)
    long_rows = []
    for c in month_cols:
        y, m = month_keys[c]
        ser = df[c].apply(clean_money)
        long_rows.append(
            pd.DataFrame(
                {
                    "unit_impacted": df["unit_impacted"],
                    "year": y,
                    "month": m,
                    "ib_month": ser,
                }
            )
        )
    long_df = pd.concat(long_rows, ignore_index=True)
    long_df = long_df[long_df["unit_impacted"].notna()]

    # suma anual por unidad
    annual = (
        long_df.groupby(["year", "unit_impacted"], as_index=False)["ib_month"]
        .sum()
        .rename(columns={"ib_month": "ib"})
    )

    # guarda CSV multi-año
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    annual.sort_values(["year", "unit_impacted"]).to_csv(OUT_FILE, index=False)

    # prints de control
    print(f"[OK] Generado {OUT_FILE} con {len(annual)} filas.")
    print(annual.head(12).to_string(index=False))
    tot = annual.groupby("year")["ib"].sum()
    print("\nSuma total anual por año:")
    print(tot.to_string())


if __name__ == "__main__":
    main()
