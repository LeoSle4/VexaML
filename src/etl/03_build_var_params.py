# src/etl/03_build_var_params.py
import pandas as pd, numpy as np, yaml
from pathlib import Path

CFG = yaml.safe_load(Path("configs/config.yml").read_text())
PROCESSED = Path(CFG["data"]["processed_path"])
OUT = Path(CFG["data"]["var_params_path"])


# Ajusta esta función a tu realidad: cómo calcular VA (exposición) por segmento
def estimate_VA_for_segment(g: pd.DataFrame) -> float:
    """
    Ejemplos de opciones:
      - Si tienes una columna de exposición (ej. ingresos/volumen): usar mediana.
      - Si NO tienes nada: usa una constante consensuada o una proxy por producto.
    Aquí usamos una constante por demo.
    """
    return 10000.0  # <-- reemplázalo por tu proxy real


def main():
    df = pd.read_parquet(PROCESSED)

    # Define el segmento tal como lo usa la API
    df["segment"] = (
        df.get("event_type", "").astype(str)
        + "|"
        + df.get("unit_impacted", "").astype(str)
        + "|"
        + df.get("product_service", "").astype(str)
    )

    # Necesitamos pérdidas históricas para O (std de pérdidas relativas)
    # Tomaremos 'loss_net_usd' si existe en el stage1. Si no está en processed,
    # puedes volver a leer el stage1, unir por event_id y traer la columna.
    # Aquí asumimos que no está (anti-fuga), así que usamos una proxy: frecuencia*1000
    # -> Te recomiendo de verdad unir con la pérdida histórica para cada segmento.
    # Para demo, calculamos O desde una "tasa de pérdida" ficticia por segmento/mes.

    # Si tienes histórico por evento con loss_net_usd, descomenta y úsalo:
    # stage1 = pd.read_parquet("data/processed/events_stage1.parquet")
    # df = df.merge(stage1[["event_id","loss_net_usd"]], on="event_id", how="left")

    # Agrupa por segmento
    rows = []
    for seg, g in df.groupby("segment"):
        VA = estimate_VA_for_segment(g)
        # Si tuvieras pérdidas reales por evento:
        # loss_rel = (g["loss_net_usd"].fillna(0) / max(VA,1e-6)).values
        # O_seg = np.std(loss_rel)

        # Fallback didáctico: aproxima O según variación de frecuencia (NO regulatorio)
        # => Cambia esto por la desviación real de pérdidas relativas
        freq = g.set_index("date_ident").resample("M")["event_id"].count()
        O_seg = float(np.std(freq.values / max(VA, 1e-6)))

        rows.append({"segment": seg, "VA_seg": float(VA), "O_seg": O_seg})

    var_params = pd.DataFrame(rows)
    var_params["Z"] = float(CFG["var"]["z_value"])
    var_params["T"] = float(CFG["var"]["horizon_T"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    var_params.to_parquet(OUT, index=False)
    print(f"[OK] Guardado {OUT} con {len(var_params)} segmentos.")


if __name__ == "__main__":
    main()
