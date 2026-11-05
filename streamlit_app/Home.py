"""Public, read-only overview of persisted Alpha Engine facts."""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Community Cloud runs this file directly without installing the local package.
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from alpha_engine.services.public_dashboard import DashboardUnavailableError
from streamlit_app.ui import (
    auto_refresh_live, candle_chart, dashboard_data, data_mode_banner, format_duration_ms, format_lima,
    latest_visible_prediction, page, refresh_button, status_badge, storage_error, symbol_units,
)


settings = page("Alpha Engine", "Estado LIVE del pipeline y de la última predicción persistida.")
pair, base, quote = symbol_units(settings.symbol)
quote_suffix = f" {quote}" if quote else ""
st.markdown(f'<p class="ae-eyebrow">{pair} · intervalo {settings.interval}</p>', unsafe_allow_html=True)
refresh_button()
auto_refresh_live()

try:
    data = dashboard_data(settings, "home")
except DashboardUnavailableError:
    storage_error()
    st.stop()

data_mode_banner(data, fast_prediction=True)
st.subheader("Estado del sistema")
a, b, c, d = st.columns(4)
with a:
    st.metric("Estado basado en datos", data.system_state)
    status_badge(data.system_state)
with b:
    st.metric("Última ejecución", data.latest_run.status if data.latest_run else "N/A")
    st.caption(format_lima(data.latest_run.finished_at) if data.latest_run else "Sin datos disponibles")
with c:
    st.metric("Último éxito", data.last_successful_run.status if data.last_successful_run else "N/A")
    st.caption(format_lima(data.last_successful_run.finished_at) if data.last_successful_run else "Sin datos disponibles")
with d:
    st.metric("Latencia del último ciclo", format_duration_ms(data.latest_run.duration_ms) if data.latest_run else "N/A")
    st.caption(f"Storage: {data.storage}")

@st.fragment(run_every="5s")
def _latest_prediction_panel() -> None:
    st.subheader("Última vela y predicción")
    left, right = st.columns(2)
    with left:
        candle = data.latest_candle
        if candle:
            st.metric("Cierre de última vela", f"{candle.close:,.2f}{quote_suffix}")
            st.caption(f"Vela cerrada: {format_lima(candle.close_time, include_utc=True)}")
        else:
            st.info("Sin velas almacenadas.")
    with right:
        try:
            prediction = latest_visible_prediction(settings, data)
        except DashboardUnavailableError:
            st.info("La última predicción LIVE no está disponible en este momento.")
            return
        if prediction:
            st.metric("Clase prevista", prediction.predicted_class)
            st.caption(f"P(SUBE): {prediction.prob_up:.1%} · P(BAJA): {prediction.prob_down:.1%}")
            st.caption(f"Fuente: {format_lima(prediction.source_candle_open_time, include_utc=True)}")
            st.caption(f"Generada: {format_lima(prediction.generated_at, include_utc=True)}")
            st.caption(f"Válida hasta: {format_lima(prediction.valid_until, include_utc=True)}")
            status_badge(prediction.timing_status)
            status_badge(prediction.validity_at(datetime.now(timezone.utc)))
            st.caption(f"Modelo: {prediction.model_version} · pred: {prediction.prediction_id_short}")
        else:
            st.info("Sin predicción persistida.")


_latest_prediction_panel()

st.subheader("Mercado reciente")
candle_chart(data.candles[-120:])

with st.expander("Estado técnico"):
    if data.latest_run:
        st.write({
            "status": data.latest_run.status,
            "inicio": format_lima(data.latest_run.started_at, include_utc=True),
            "fin": format_lima(data.latest_run.finished_at, include_utc=True),
            "duración": format_duration_ms(data.latest_run.duration_ms),
            "vela": format_lima(data.latest_run.candle_open_time, include_utc=True),
            "predicción": data.latest_run.prediction_id_short or "N/A",
        })
    else:
        st.caption("Sin ejecuciones persistidas.")
    if data.recent_errors:
        st.caption("Errores recientes sanitizados")
        st.dataframe([
            {"Hora Lima": format_lima(run.finished_at), "Categoría": run.error_category or "UNEXPECTED", "Run": run.run_id_short}
            for run in data.recent_errors[:5]
        ], width="stretch", hide_index=True)

st.divider()
st.page_link("pages/1_Descriptivo.py", label="Descriptivo")
st.page_link("pages/2_Predictivo.py", label="Predictivo")
st.page_link("pages/3_Prescriptivo.py", label="Prescriptivo")
st.page_link("pages/4_Evaluacion.py", label="Evaluación")
st.page_link("pages/5_Simulacion.py", label="Simulación")
st.caption("Alpha Engine es un PMV académico de apoyo a decisiones. No ejecuta órdenes ni garantiza resultados.")
