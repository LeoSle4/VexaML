import streamlit as st, requests
from datetime import date

st.set_page_config(page_title="Risk-ML", layout="centered")
st.title("Clasificador de Riesgo Operativo")

col1, col2 = st.columns(2)
unit_impacted = col1.text_input("Unidad Impactada", "Agencia A")
product_service = col1.text_input("Producto/Servicio", "Cuenta de Ahorros")
process_impacted = col1.text_input("Proceso Afectado", "Apertura de cuenta")
event_type = col2.selectbox("Tipo de Evento", ["Fraude", "Proceso", "TI", "Legal"])
risk_assoc = col2.text_input("Riesgo Asociado", "Fraude interno")

today = date.today()
year = st.number_input("Año", min_value=2000, max_value=2100, value=today.year)
month = st.number_input("Mes", min_value=1, max_value=12, value=today.month)
quarter = (month - 1) // 3 + 1
weekday = today.weekday()
is_month_end = 1 if month in (3, 6, 9, 12) else 0

st.subheader("Recencia/Frecuencia (opcional)")
n30 = st.number_input("Eventos 30d (unidad)", min_value=0, value=0)
n60 = st.number_input("Eventos 60d (unidad)", min_value=0, value=0)
n90 = st.number_input("Eventos 90d (unidad)", min_value=0, value=0)
dsl = st.number_input("Días desde último evento (unidad)", min_value=0, value=9999)

st.subheader("Consecuencia (C) para R = C × P̂")
consequence = st.slider(
    "Consecuencia", min_value=1.0, max_value=5.0, value=3.0, step=0.5
)

if st.button("Predecir"):
    payload = {
        "unit_impacted": unit_impacted,
        "product_service": product_service,
        "process_impacted": process_impacted,
        "event_type": event_type,
        "risk_assoc": risk_assoc,
        "year": int(year),
        "month": int(month),
        "quarter": int(quarter),
        "weekday": int(weekday),
        "is_month_end": int(is_month_end),
        "n_events_30d_unit_impacted": float(n30),
        "n_events_60d_unit_impacted": float(n60),
        "n_events_90d_unit_impacted": float(n90),
        "days_since_last_unit_impacted": float(dsl),
        "consequence": float(consequence),
    }
    try:
        r = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        if r.status_code == 200:
            out = r.json()
            st.metric("Probabilidad de pérdida", f"{out['risk_proba']:.2%}")
            st.metric("Etiqueta", "CON PÉRDIDA" if out["label"] == 1 else "SIN PÉRDIDA")
            st.metric(
                "Índice de Riesgo (R̂)",
                f"{out['R_index']:.3f}" if out["R_index"] is not None else "—",
            )
            st.metric("VaR̂ (segmento)", f"{out['VaR_hat']:.2f}")
            st.caption(
                f"Umbral: {out['used_threshold']:.3f}  |  Segmento: {out['segment']}"
            )
        else:
            st.error(f"Error {r.status_code}: {r.text}")
    except Exception as ex:
        st.error(f"No se pudo conectar con la API: {ex}")
