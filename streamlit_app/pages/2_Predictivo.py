"""Shows persisted LIVE predictions; inference belongs exclusively to LiveCycle."""

import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from alpha_engine.services.public_dashboard import DashboardUnavailableError
from streamlit_app.ui import (
    auto_refresh_live, dashboard_data, data_mode_banner, format_lima, format_percent,
    latest_visible_prediction, live_evaluation_metrics, page, refresh_button,
    section_timing, status_badge, storage_error,
)


with section_timing("predictive.page"):
    settings = page("Predictivo", "Predicciones XGBoost ya generadas por el worker para el siguiente cierre de 1 minuto.")
refresh_button()
auto_refresh_live()

try:
    with section_timing("predictive.dashboard"):
        data = dashboard_data(settings, "predictive")
except DashboardUnavailableError:
    storage_error()
    st.stop()

data_mode_banner(data, fast_prediction=True)
if settings.storage_provider == "postgresql" and settings.app_env == "local":
    st.caption("Modo local: el historial de métricas incluye LOCAL_BOOTSTRAP causal y ciclos posteriores; no es una cohorte prospectiva pura.")


@st.fragment(run_every="5s")
def _latest_prediction_panel() -> None:
    st.subheader("Última predicción")
    try:
        with section_timing("predictive.latest"):
            prediction = latest_visible_prediction(settings, data)
    except DashboardUnavailableError:
        st.info("La última predicción LIVE no está disponible en este momento.")
        return
    if prediction is None:
        st.info("Sin predicciones persistidas.")
        return

    current_validity = prediction.validity_at(datetime.now(timezone.utc))
    a, b, c, d = st.columns(4)
    a.metric("Clase prevista", prediction.predicted_class)
    b.metric("P(SUBE)", f"{prediction.prob_up:.1%}")
    c.metric("P(BAJA)", f"{prediction.prob_down:.1%}")
    d.metric("Modelo", prediction.model_version)
    st.caption(f"Horizonte: {prediction.horizon} · clase UP si P(UP) > 50%; DOWN en caso contrario.")
    st.caption(f"Vela fuente: {format_lima(prediction.source_candle_open_time, include_utc=True)} · pred: {prediction.prediction_id_short}")
    st.caption(f"Generada: {format_lima(prediction.generated_at, include_utc=True)} · válida hasta: {format_lima(prediction.valid_until, include_utc=True)}")
    left, right = st.columns(2)
    with left:
        st.caption("Oportunidad histórica")
        status_badge(prediction.timing_status)
    with right:
        st.caption("Vigencia actual")
        status_badge(current_validity)
    if current_validity == "EXPIRED":
        st.warning("La última predicción se conserva para auditoría, pero ya no está vigente.")
    st.caption("P(SUBE) y P(BAJA) son probabilidades de la inferencia XGBoost. No son accuracy ni rentabilidad esperada.")
    confidence = max(prediction.prob_up, prediction.prob_down)
    st.caption(f"Gate prescriptivo: max(P) = {confidence:.1%} · umbral ≥ {settings.model_confidence_threshold:.0%}.")
    if confidence < settings.model_confidence_threshold:
        st.info("Gate no alcanzado: esperar / sin entrada. La clase UP/DOWN se conserva como predicción técnica.")
    else:
        st.caption("Gate alcanzado; considerar una acción requiere además datos actuales, vigencia, posición y controles de riesgo.")


_latest_prediction_panel()
prediction = data.latest_prediction
if prediction is None:
    st.stop()

st.subheader("Desempeño LIVE — contexto")


@st.fragment
def _live_metrics_panel() -> None:
    if data.mode == "FALLBACK":
        st.info("El modo FALLBACK no ofrece métricas LIVE actuales.")
        return
    # This full accumulated cohort may require many storage reads. Navigation
    # must render the prediction and recent history without waiting for it.
    if not st.button("Consultar desempeño LIVE", key="predictive_live_metrics"):
        st.caption("Consulta opcional del desempeño acumulado. La predicción y su historial están disponibles sin esperar este cálculo.")
        return
    try:
        with section_timing("predictive.metrics"):
            metrics = live_evaluation_metrics(settings, prediction.model_version)
    except DashboardUnavailableError:
        st.info("Las métricas LIVE no están disponibles en este momento.")
    else:
        if not metrics.evaluated:
            st.info("Aún no existen evaluaciones LIVE para este modelo.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Evaluadas LIVE", metrics.evaluated)
            b.metric("Accuracy LIVE", format_percent(metrics.accuracy))
            c.metric("Balanced accuracy", format_percent(metrics.balanced_accuracy))
            d.metric("Baseline siempre DOWN", format_percent(metrics.always_down_accuracy))
            st.caption(
                f"Velas fuente evaluadas: {format_lima(metrics.source_start, include_utc=True)} – "
                f"{format_lima(metrics.source_end, include_utc=True)}. "
                f"Predicho UP/DOWN: {format_percent(metrics.predicted_up_fraction)} / "
                f"{format_percent(metrics.predicted_down_fraction)}. "
                f"N evaluaciones disponibles dentro del período; puede haber huecos."
            )
            if metrics.scored_predictions != metrics.evaluated or metrics.inconsistent_correct_rows:
                st.warning("La cohorte contiene registros sin predicción comparable o inconsistencias de auditoría.")
            st.caption(
                "El baseline siempre DOWN es sólo una referencia sobre las mismas evaluaciones, "
                "no una estrategia recomendada. El desempeño histórico no garantiza esta predicción."
            )


_live_metrics_panel()

st.subheader("Historial LIVE reciente")
if not data.predictions:
    st.info("Sin historial disponible.")
else:
    st.dataframe([
        {
            "Hora fuente (Lima)": format_lima(item.source_candle_open_time),
            "Clase": item.predicted_class,
            "P(SUBE)": f"{item.prob_up:.1%}",
            "Generada (Lima)": format_lima(item.generated_at),
            "Válida hasta": format_lima(item.valid_until),
            "Timing": item.timing_status or "N/A",
            "Vigencia": item.validity_at(datetime.now(timezone.utc)) or "N/A",
            "Modelo": item.model_version,
            "ID": item.prediction_id_short,
        }
        for item in data.predictions[:20]
    ], width="stretch", hide_index=True)

st.caption("La página es de lectura: no ejecuta Binance, entrenamiento ni inferencia manual.")
