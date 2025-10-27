# ui/dashboard.py
import json
import requests
import pandas as pd
import streamlit as st
from datetime import date
from calendar import monthrange
from pathlib import Path

st.set_page_config(page_title="VEXA-ML", layout="centered")
st.title("Clasificador de Riesgo Operativo")

# ---------------------- Config / util ----------------------
API_URL_DEFAULT = "http://localhost:8000"


@st.cache_data(show_spinner=False)
def load_catalogs(proc_path: str = "data/processed/events.parquet"):
    """Lee catálogos únicos desde el dataset procesado y retorna (catalogs, random_row)."""
    p = Path(proc_path)
    if not p.exists():
        return None

    df = pd.read_parquet(p)

    def uniq(col):
        if col not in df.columns:
            return []
        vals = (
            df[col].astype(str).fillna("").map(lambda x: x.strip()).replace({"nan": ""})
        )
        return sorted([x for x in vals.unique().tolist() if x])

    catalogs = {
        "unit_impacted": uniq("unit_impacted"),
        "product_service": uniq("product_service"),
        "process_impacted": uniq("process_impacted"),
        "event_type": uniq("event_type"),
        "risk_assoc": uniq("risk_assoc"),
    }
    row = df.sample(1, random_state=7).to_dict(orient="records")[0] if len(df) else {}
    return catalogs, row


def api_health(api_url: str):
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return None, str(e)


def predict(api_url: str, payload: dict):
    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=15)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text}"
    except Exception as ex:
        return None, str(ex)


# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("Conexión")
    api_url = st.text_input("API URL", API_URL_DEFAULT)
    if "srv_threshold" not in st.session_state:
        st.session_state["srv_threshold"] = None

    if st.button("Probar conexión", use_container_width=True):
        h, err = api_health(api_url)
        if h:
            st.session_state["srv_threshold"] = h.get("threshold")
            st.success(
                f"OK · threshold={h.get('threshold'):.3f} · var_params={'sí' if h.get('has_var_params') else 'no'}"
            )
        else:
            st.error(f"No conecta: {err}")

    st.divider()
    data_info = load_catalogs()
    if data_info:
        catalogs, random_row = data_info
        st.caption("✅ Catálogos cargados desde data/processed/events.parquet")
    else:
        catalogs, random_row = None, {}
        st.caption(
            "ℹ️ No se encontró data/processed/events.parquet — usa entradas libres."
        )

# ---------------------- Formulario principal ----------------------
st.subheader("Datos del evento")

# Fecha de identificación -> derivadas correctas
pred_date = st.date_input("Fecha de identificación", value=date.today())
year = pred_date.year
month = pred_date.month
quarter = (month - 1) // 3 + 1
weekday = pred_date.weekday()
is_month_end = 1 if pred_date.day == monthrange(year, month)[1] else 0

col1, col2 = st.columns(2)


def sel_or_text(container, label, key, placeholder=""):
    if catalogs and catalogs.get(key):
        return container.selectbox(label, catalogs[key], index=0, key=f"sb_{key}")
    return container.text_input(label, placeholder, key=f"ti_{key}")


unit_impacted = sel_or_text(
    col1, "Unidad Impactada", "unit_impacted", "Ej.: Mesa de Dinero"
)
product_service = sel_or_text(
    col1, "Producto/Servicio", "product_service", "Ej.: Fondos Mutuos"
)
process_impacted = sel_or_text(
    col1, "Proceso Afectado", "process_impacted", "Ej.: Cierre diario"
)
event_type = sel_or_text(col2, "Tipo de Evento", "event_type", "Ej.: Evento de Pérdida")
risk_assoc = sel_or_text(
    col2, "Riesgo Asociado", "risk_assoc", "Ej.: Errores de registro/operativa"
)

st.caption(
    f"Año: **{year}** · Mes: **{month}** · Trimestre: **{quarter}** · Weekday: **{weekday}** · Month-end: **{is_month_end}**"
)

st.divider()
st.subheader("Recencia / Frecuencia")
c3, c4, c5, c6 = st.columns(4)
n30 = c3.number_input("Eventos 30d (unidad)", min_value=0, value=0)
n60 = c4.number_input("Eventos 60d (unidad)", min_value=0, value=0)
n90 = c5.number_input("Eventos 90d (unidad)", min_value=0, value=0)
dsl = c6.number_input("Días desde último (unidad)", min_value=0, value=9999)

st.divider()
st.subheader("Consecuencia (C) para R = C × P̂")
consequence = st.slider(
    "Consecuencia", min_value=1.0, max_value=5.0, value=3.0, step=0.5
)

# ---------------------- Ajustes avanzados ----------------------
with st.expander("Ajustes avanzados"):
    th_default = st.session_state.get("srv_threshold", None)
    th_override = st.slider(
        "Umbral de decisión (opcional)",
        0.0,
        1.0,
        float(th_default) if isinstance(th_default, (int, float)) else 0.0,
        0.001,
        help="Si lo estableces, se usará este umbral en lugar del del servidor para decidir la etiqueta.",
    )
    use_th_override = st.checkbox("Usar este umbral en la predicción", value=False)

# ---------------------- Botones de acción ----------------------
c7, c8 = st.columns([1, 1])


def build_payload_from_ui():
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
    if use_th_override:
        payload["threshold_override"] = float(th_override)
    return payload


def render_result(res: dict):
    colA, colB, colC = st.columns(3)
    colA.metric("Probabilidad de pérdida", f"{res['risk_proba']:.2%}")
    colB.metric("Etiqueta", "CON PÉRDIDA" if res["label"] == 1 else "SIN PÉRDIDA")
    colC.metric(
        "Índice de Riesgo (R̂)",
        f"{res['R_index']:.3f}" if res.get("R_index") is not None else "—",
    )

    # VaR̂ y detalles
    st.metric("VaR̂ (segmento)", f"{res.get('VaR_hat', 0.0):.2f}")
    seg = res.get("segment", "—")
    th = res.get("used_threshold", None)
    var_src = res.get("var_source", None)
    va_seg = res.get("VA_seg", None)
    o_seg = res.get("O_seg", None)

    extra = f"Umbral: {th:.3f}  |  Segmento: {seg}"
    if var_src is not None:
        extra += f"  |  VaR fuente: {var_src}"
    if va_seg is not None and o_seg is not None:
        extra += f"  |  VA={va_seg:.0f}, O={o_seg:.3f}"
    st.caption(extra)

    if float(res.get("VaR_hat", 0.0)) == 0.0:
        st.warning(
            "VaR̂ es 0 para este segmento. Asegúrate de haber generado "
            "`data/processed/var_params.parquet` y de que el segmento exista (event_type|unit_impacted|product_service)."
        )

    # RC (BIA)
    rc = res.get("rc", {}) or {}
    rc_val = rc.get("rc", None)
    if rc_val is not None:
        rc_level = rc.get("level", "bank")
        rc_cat = rc.get("category", "—")
        rc_years = rc.get("years_used", [])
        st.metric("RC (BIA)", f"${rc_val:,.0f}")
        st.caption(
            f"Nivel: {rc_level} · Categoría: {rc_cat} · Años usados: {', '.join(map(str, rc_years)) or '—'}"
        )

    with st.expander("Ver JSON de respuesta"):
        st.code(json.dumps(res, indent=2, ensure_ascii=False), language="json")


predict_clicked = c8.button("Predecir", type="primary", use_container_width=True)

# ---------------------- Predicción desde UI ----------------------
st.divider()
if predict_clicked:
    payload = build_payload_from_ui()
    with st.spinner("Calculando…"):
        res, err = predict(api_url, payload)
    if err:
        st.error(f"No se pudo obtener predicción: {err}")
    elif res:
        render_result(res)
