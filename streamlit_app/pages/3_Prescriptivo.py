"""Explicit, audited LLM requests. Nothing runs until the user presses the button."""

import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from alpha_engine.services.errors import StorageError
from alpha_engine.services.public_dashboard import DashboardUnavailableError
from alpha_engine.data.recommendation_response import RecommendationResult
from alpha_engine.data.news_context import NewsContext
from alpha_engine.services.recommendation_api import RecommendationApiClient
from alpha_engine.services.paper_api import paper_request
from alpha_engine.services.paper_policy import paper_decision
from streamlit_app.ui import (
    auto_refresh_live, dashboard_data, data_mode_banner, format_duration_ms, format_lima,
    info_note, latest_prescriptive_prediction, page,
    refresh_button, status_badge, storage_error,
)


STATUS_MESSAGES = {
    "SUCCESS": "Recomendación generada y disponible dentro del horizonte temporal.",
    "CACHED": "Origen: resultado previamente generado para esta predicción.",
    "LATE_RESULT": "La recomendación fue generada, pero terminó después del horizonte. Se conserva solo para auditoría.",
    "LLM_TIMEOUT": "El proveedor de IA no respondió dentro del tiempo disponible.",
    "PROVIDER_ERROR": "El proveedor de IA no pudo completar la solicitud.",
    "INVALID_RESPONSE": "La respuesta del proveedor no cumplió el formato requerido.",
    "GUARDRAIL_REJECTED": "ESPERAR / SIN ACCIÓN. Las reglas prescriptivas impidieron generar una recomendación operativa.",
    "EXPIRED_BEFORE_REQUEST": "La predicción venció antes de solicitar la recomendación.",
    "DEADLINE_TOO_CLOSE": "No quedaba tiempo útil para solicitar una recomendación.",
    "NO_ACTIVE_PREDICTION": "No existe una predicción vigente para generar recomendación.",
    "PREDICTION_NOT_FOUND": "La predicción solicitada no está disponible.",
    "PREDICTION_INVALID": "La predicción no contiene los datos necesarios para una solicitud segura.",
    "PERSISTENCE_ERROR": "No se pudo completar el registro de auditoría de la solicitud.",
}


def _display_status(status: str) -> str:
    return "ESPERAR / SIN ACCIÓN" if status == "GUARDRAIL_REJECTED" else status


def _show_news_context(context: NewsContext | None, *, enabled: bool) -> None:
    if not enabled:
        st.info("Noticias desactivadas en el PMV. No se consultan proveedores ni se publica contenido sin derechos confirmados.")
        return
    if context is not None and context.provider == "MARKETAUX":
        from streamlit_app.news_context import render_marketaux_context
        render_marketaux_context(context)
        return
    st.subheader("Contexto de noticias")
    if context is None:
        st.caption("Las noticias BTC se consultan o reutilizan sólo al pulsar Generar recomendación; navegar o actualizar datos no consume la API.")
        return
    if context.status == "AVAILABLE":
        a, b, c = st.columns(3)
        a.metric("Noticias BTC · 60 min", context.news_count)
        b.metric("Sentimiento BTC", f"{context.btc_sentiment_score:+.3f}" if context.btc_sentiment_score is not None else "N/A")
        c.metric("Última noticia", f"hace {max(0, int((datetime.now(timezone.utc) - context.latest_news_at).total_seconds() / 60))} min" if context.latest_news_at else "N/A")
        st.caption(f"Contexto obtenido: {format_lima(context.fetched_at, include_utc=True)} · Información adicional, no confirmación predictiva.")
    elif context.status == "NO_NEWS":
        st.caption(f"No se encontraron noticias BTC recientes en 60 min. Contexto obtenido: {format_lima(context.fetched_at, include_utc=True)}. Ausencia de noticias no equivale a sentimiento neutral.")
    else:
        if context.reason == "NOT_CONFIGURED":
            st.caption("Contexto de noticias no configurado. La solicitud usa el contexto técnico.")
        else:
            st.caption(
                "Contexto de noticias no disponible o desactualizado. La solicitud usa el contexto técnico."
                + (f" Última actualización: {format_lima(context.fetched_at, include_utc=True)}."
                   if context.fetched_at else "")
            )


def _show_result(result: RecommendationResult, selected_prediction_id: str | None) -> None:
    st.subheader("Resultado de la solicitud")
    if result.prediction_id:
        st.caption(f"Predicción solicitada: {result.prediction_id[:8]}…")
    if result.prediction_id != selected_prediction_id:
        st.info("Este resultado corresponde a otra predicción; no es el resultado de la predicción seleccionada ahora.")
    message = STATUS_MESSAGES.get(result.status, "La solicitud terminó con un estado no disponible para presentación.")
    if result.status == "GUARDRAIL_REJECTED":
        if "LOW_MODEL_CONFIDENCE" in result.guardrails:
            message = (
                "ESPERAR / SIN ACCIÓN. La predicción técnica no superó el gate "
                "prescriptivo experimental del 60%. No se consultaron noticias "
                "ni se llamó al LLM."
            )
        elif result.llm_latency_ms is None:
            message += " No se llamó al LLM."
    if result.status in {"SUCCESS", "CACHED"} and result.actionable:
        st.success(message)
    elif result.status == "LATE_RESULT":
        st.warning(message)
    elif result.status == "GUARDRAIL_REJECTED":
        with st.container(key="guardrail_info"):
            st.info(message)
    else:
        info_note(message)
    a, b, c, d = st.columns(4)
    a.metric("Estado", _display_status(result.status))
    b.metric("Latencia LLM", format_duration_ms(result.llm_latency_ms))
    c.metric("Latencia total", format_duration_ms(result.total_latency_ms))
    d.metric("Caché", "Sí" if result.cached else "No")
    st.caption(f"Solicitada: {format_lima(result.requested_at, include_utc=True)}")
    st.caption(f"Generada: {format_lima(result.generated_at, include_utc=True)} · válida hasta: {format_lima(result.valid_until, include_utc=True)}")
    status_badge(result.timing_status)
    status_badge(result.current_validity(datetime.now(timezone.utc)))
    if result.recommendation:
        st.metric("Riesgo evaluado", result.recommendation.risk_level)
        st.subheader("Explicación LLM")
        st.caption(f"Origen de la explicación: {result.recommendation.llm_model} · corresponde a la solicitud indicada.")
        if result.recommendation.summary:
            st.write(result.recommendation.summary)
        for factor in result.recommendation.key_factors:
            st.caption(f"Razón técnica o contextual: {factor}")
        for risk in result.recommendation.risks:
            st.caption(f"Riesgo: {risk}")
        if result.recommendation.warning:
            st.caption(result.recommendation.warning)
        st.caption("El LLM solo explica. La acción BUY/SELL/HOLD/WAIT la fijan reglas deterministas.")
    if result.guardrails and "LOW_MODEL_CONFIDENCE" not in result.guardrails:
        st.caption(f"Guardrails aplicados: {', '.join(result.guardrails)}")


settings = page("Prescriptivo", "Orientación por reglas y explicación opcional bajo demanda.")
st.markdown('<div class="ae-flow">Predicción persistida → política determinista → controles de datos y riesgo → explicación opcional → auditoría</div>', unsafe_allow_html=True)
refresh_button()
auto_refresh_live()

try:
    data = dashboard_data(settings, "prescriptive")
except DashboardUnavailableError:
    storage_error()
    st.stop()

data_mode_banner(data, fast_prediction=True)

token = st.session_state.get("paper_portfolio_id")
paper_state = None
if token:
    try:
        paper_reply = paper_request(settings, {"op": "read", "portfolio_id": token})
        if paper_reply.get("status") == "OK":
            paper_state = paper_reply["portfolio"]
    except Exception:
        st.warning("No se pudo leer la posición virtual; revise la página Simulación.")


@st.fragment(run_every="5s")
def _request_panel() -> None:
    """Read only the latest prediction and recompute its deadline on each tick."""
    try:
        prediction = (
            data.latest_prediction if data.mode == "FALLBACK"
            else latest_prescriptive_prediction(settings)
        )
    except DashboardUnavailableError:
        st.error("No se pudo consultar la predicción LIVE más reciente.")
        prediction = None
    now = datetime.now(timezone.utc)
    validity = prediction.validity_at(now) if prediction else None
    remaining = (
        (prediction.valid_until - now).total_seconds()
        if prediction is not None and prediction.valid_until is not None else 0.0
    )
    st.subheader("Predicción técnica")
    if prediction:
        direction, up, down = st.columns(3)
        direction.metric("Dirección", prediction.predicted_class)
        up.metric("P(SUBE)", f"{prediction.prob_up:.1%}")
        down.metric("P(BAJA)", f"{prediction.prob_down:.1%}")
        st.caption(f"ID: {prediction.prediction_id_short} · modelo: {prediction.model_version}")
        st.caption(f"Generada: {format_lima(prediction.generated_at)} · válida hasta: {format_lima(prediction.valid_until)}")
        st.caption(f"Tiempo restante: {max(0, int(remaining))} s")
        status_badge(prediction.timing_status)
        status_badge(validity)
    else:
        st.info("No existe una predicción persistida.")

    has_long = bool(paper_state and paper_state["position"] == "LONG")
    live_context = data.mode == "LIVE" and data.system_state == "CURRENT"
    if data.latest_candle is not None:
        age = (now - data.latest_candle.close_time).total_seconds()
        live_context = live_context and 0 <= age <= settings.max_candle_age_seconds
    decision = paper_decision(prediction if live_context else None, has_long=has_long,
                              threshold=settings.model_confidence_threshold, now=now)
    st.subheader("Posición y acción determinista")
    left, right, third = st.columns(3)
    left.metric("Estado usuario", "LONG" if has_long else "SIN POSICIÓN" if not token or paper_state else "NO DISPONIBLE")
    right.metric("Acción sugerida", decision.action if not token or paper_state else "N/A")
    third.metric("Confianza descriptiva", f"{decision.confidence:.1%}" if decision.confidence is not None else "N/A")
    st.caption(decision.reason if not token or paper_state else "No se calcula una acción sin conocer la posición virtual.")
    st.caption("Origen: política determinista, sin LLM. Esta orientación sigue disponible si falla el proveedor de explicaciones.")
    st.caption("Riesgo detallado: no evaluado en esta vista previa. La orientación no certifica los controles de riesgo ni ejecuta operaciones.")
    if not live_context:
        st.info("Sin orientación de entrada/salida: se requieren datos LIVE actuales. El snapshot histórico se conserva solo como contexto.")
    st.caption("Confianza = mayor probabilidad asignada por el modelo, no accuracy real ni probabilidad de beneficio. "
               "La decisión usa reglas; noticias y LLM solo aportan contexto.")
    if not token:
        st.caption("Sin cartera conectada: se muestra el caso SIN POSICIÓN. Cree una en Simulación para usar su estado real virtual.")

    current_data = data.model_copy(update={"latest_prediction": prediction})
    can_request = (current_data.recommendation_available_at(now, settings.llm_deadline_margin_seconds)
                   and live_context
                   and prediction.generated_at is not None and prediction.generated_at <= now
                   and (not token or paper_state is not None))
    st.subheader("Estado prescriptivo")
    gate_met = bool(prediction and max(prediction.prob_up, prediction.prob_down)
                    >= settings.model_confidence_threshold)
    if prediction and not gate_met:
        with st.container(key="guardrail_gate_info"):
            st.info("MANTENER / REEVALUAR" if has_long else "ESPERAR / NO ENTRAR")
        st.caption(
            "No se genera una recomendación operativa porque la predicción actual "
            f"no supera el gate prescriptivo experimental del {settings.model_confidence_threshold:.0%}."
        )
    elif gate_met and can_request:
        st.success("ANÁLISIS PRESCRIPTIVO HABILITADO")
    elif gate_met:
        st.info("Umbral experimental cumplido; el análisis requiere una predicción LIVE vigente.")
    else:
        st.caption("Pendiente de una predicción técnica.")
    st.caption(
        f"Gate prescriptivo experimental: max(P) ≥ {settings.model_confidence_threshold:.0%}. "
        "No representa una precisión histórica validada. La clase UP/DOWN "
        "usa por separado P(UP) > 50%."
    )

    if data.system_state == "STALE":
        st.warning("El pipeline está STALE según la antigüedad de la última vela; no se solicitará una recomendación LIVE.")
    elif prediction and validity == "EXPIRED":
        st.warning("No existe una predicción vigente para generar recomendación.")
    elif data.mode == "FALLBACK":
        st.info("El modo FALLBACK es de lectura y no solicita LLM.")

    prior_result = st.session_state.get("recommendation_result")
    _show_news_context(prior_result.news_context if isinstance(prior_result, RecommendationResult) else None,
                       enabled=settings.prescriptive_news_enabled)
    if (isinstance(prior_result, RecommendationResult) and prior_result.news_context is not None
            and prior_result.prediction_id != (prediction.prediction_id if prediction else None)):
        st.caption("Este contexto corresponde a la última solicitud, para otra predicción.")

    st.subheader("Recomendación")
    if st.button("Generar recomendación", type="primary", disabled=not can_request):
        st.session_state.pop("recommendation_result", None)
        with st.spinner("Solicitando recomendación y registrando auditoría…"):
            try:
                if settings.prescriptive_backend == "http":
                    client = RecommendationApiClient(settings)
                    result = (client.request(prediction.prediction_id, token) if paper_state
                              else client.request(prediction.prediction_id))
                else:
                    from alpha_engine.services.recommendation_use_case import RecommendationUseCase
                    with RecommendationUseCase.from_settings(settings) as use_case:
                        result = use_case.request(prediction.prediction_id,
                                                  portfolio_id=token if paper_state else None)
                st.session_state["recommendation_result"] = result
            except StorageError:
                st.error("No se pudo registrar la solicitud prescriptiva.")
            except Exception:
                st.error("No se pudo completar la solicitud prescriptiva.")
            else:
                from streamlit_app.ui import _load_dashboard, _load_latest_prediction
                _load_dashboard.clear()
                _load_latest_prediction.clear()
                st.rerun()

    result = st.session_state.get("recommendation_result")
    if isinstance(result, RecommendationResult):
        _show_result(result, prediction.prediction_id if prediction else None)
    else:
        if prediction and not gate_met:
            info_note("Puede solicitar el análisis: se registrará ESPERAR / SIN ACCIÓN sin consultar noticias ni llamar al LLM.")
        else:
            info_note("Genere una recomendación para la predicción vigente. Los intentos anteriores se resumen abajo sin exponer contexto privado.")


    if (settings.app_env == "local" and settings.storage_provider == "postgresql"
            and settings.local_demo_mode and prediction and not gate_met):
        st.divider()
        st.subheader("Modo demostración local")
        st.caption("Gate 60% no superado. Esta prueba genera una explicación, nunca una recomendación operativa.")
        if st.button("Probar explicación LLM (modo demostración)"):
            with st.spinner("Preparando explicación demostrativa…"):
                try:
                    st.session_state["local_demo_result"] = RecommendationApiClient(settings).demo_explanation(
                        prediction.prediction_id
                    )
                except Exception:
                    st.session_state["local_demo_result"] = {"status": "DEMO_UNAVAILABLE"}
        demo = st.session_state.get("local_demo_result")
        if isinstance(demo, dict):
            if demo.get("prediction_id") not in {None, prediction.prediction_id}:
                st.caption("La explicación anterior pertenece a otra predicción.")
            elif demo.get("status") == "LLM_NOT_CONFIGURED":
                st.info("LLM no configurado. Configure la clave solo en el .env local.")
            elif demo.get("status") == "DEMO_ONLY":
                st.warning("MODO DEMOSTRACIÓN · NO ES RECOMENDACIÓN OPERATIVA")
                explanation = demo.get("explanation") or {}
                st.write(explanation.get("summary", ""))
                for factor in explanation.get("key_factors", []):
                    st.write(f"• {factor}")
                if explanation.get("warning"):
                    st.caption(explanation["warning"])
                news_status = (demo.get("news_context") or {}).get("status")
                st.caption(f"Contexto de noticias de esta explicación: {news_status or 'no disponible'}.")
            else:
                st.info("La explicación demostrativa no está disponible en este momento.")


_request_panel()

st.subheader("Auditoría prescriptiva")
audit = data.recommendation_observability
if audit is None:
    st.info("Sin intentos prescriptivos auditados.")
else:
    a, b, c, d = st.columns(4)
    a.metric("Último intento", _display_status(audit.last_attempt_status) if audit.last_attempt_status else "N/A")
    b.metric("Latencia media", format_duration_ms(audit.average_llm_latency_ms))
    c.metric("TIMELY / LATE", f"{audit.timely_count} / {audit.late_count}")
    d.metric("Timeouts / errores", f"{audit.timeout_count} / {audit.provider_error_count}")
    st.caption(f"Último intento: {format_lima(audit.last_attempt_at)} · último éxito: {format_lima(audit.last_success_at)}")

st.caption("Se muestran solo campos sanitizados. El prompt, contexto, respuesta raw y errores técnicos permanecen privados.")
st.caption("Alpha Engine es un PMV académico. No ejecuta órdenes ni constituye asesoría financiera.")
