"""Public ALPHA ENGINE pages: reviewed presentation, GET-only facts, pure calculations."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from alpha_engine.features.indicators import build_feature_frame
from deploy.northflank.public_config import PublicConfig
from deploy.northflank.public_data import PublicReader, ReadUnavailable, prescriptive_scenarios
from deploy.northflank.public_presentation import (
    apply_theme, candle_chart, format_lima, format_percent, status_badge, symbol_units,
)

ROOT = Path(__file__).resolve().parents[2]


def config() -> PublicConfig:
    return PublicConfig.from_environment(dict(os.environ))


def source_id() -> str:
    cfg = config()
    return hashlib.sha256((cfg.supabase_url + "\0" + cfg.supabase_key + "\0" + cfg.symbol).encode()).hexdigest()


def make_reader() -> PublicReader:
    return PublicReader(config())


@st.cache_data(ttl=55, show_spinner=False)
def snapshot(section: str, identity: str) -> dict:
    if section not in ROUTES:
        raise ValueError("Unknown public page")
    with make_reader() as reader:
        if section == "descriptivo":
            return {"candles": reader.candles()}
        if section == "predictivo":
            return {"predictions": reader.predictions()}
        if section == "evaluacion":
            return {"evaluations": reader.evaluations(50)}
        if section == "simulacion":
            return {"predictions": reader.predictions(360), "candles": reader.candles(1000)}
        return {"predictions": reader.predictions(1),
                "candles": reader.candles(120 if section == "inicio" else 1)}


@st.cache_data(ttl=4, show_spinner=False)
def latest_prediction(identity: str):
    with make_reader() as reader:
        rows = reader.predictions(1)
        return rows[0] if rows else None


@st.cache_data(ttl=4, show_spinner=False)
def latest_candle(identity: str):
    with make_reader() as reader:
        rows = reader.candles(1)
        return rows[-1] if rows else None


@st.cache_data(ttl=55, show_spinner=False)
def metrics(identity: str) -> dict:
    with make_reader() as reader:
        return reader.recent_metrics()


def time_label(value) -> str:
    return format_lima(value, include_utc=True)


def percent(value) -> str:
    return format_percent(value)


def begin(title: str, question: str) -> dict:
    cfg = config()
    apply_theme()
    st.title(title)
    st.caption("ALPHA ENGINE / PMV ACADÉMICO / CONSULTA PÚBLICA DE SOLO LECTURA")
    st.markdown(f'<p class="ae-subtitle">{question}</p>', unsafe_allow_html=True)
    pair, _, _ = symbol_units(cfg.symbol)
    st.markdown(f'<p class="ae-eyebrow">{pair} · intervalo 1m · fuente persistida</p>', unsafe_allow_html=True)
    with st.container(key="refresh_control"):
        refresh = st.button("Actualizar datos", icon="🔄", key="refresh_" + title,
                            help="Vuelve a consultar los datos públicos almacenados.")
    if refresh:
        for cache in (snapshot, latest_prediction, latest_candle, metrics):
            cache.clear()
    # Match the original UI cadence without invoking schedulers or persistent work.
    st.session_state["_public_last_full_run"] = time.monotonic()

    @st.fragment(run_every="60s")
    def refresh_tick():
        if time.monotonic() - st.session_state["_public_last_full_run"] >= 58:
            st.rerun()
    if title != "Simulación":
        refresh_tick()
    section = "inicio" if title == "Alpha Engine" else next(
        key for key, (label, _) in ROUTES.items() if label == title
    )
    st.caption("Modo: REPLAY · muestra persistida · actualización manual" if section == "simulacion" else
               "Modo: LIVE · Storage: Supabase · hora Lima / UTC · actualización de la página ~60 s")
    try:
        return snapshot(section, source_id())
    except (ReadUnavailable, ValueError):
        st.error("La información pública no está disponible en este momento.")
        st.stop()


def market_state(candles: list) -> bool:
    if not candles:
        status_badge("NO_DATA")
        st.info("Sin velas cerradas almacenadas.")
        return False
    last = candles[-1]
    current = 0 <= (datetime.now(timezone.utc) - last.close_time).total_seconds() <= 120
    status_badge("CURRENT" if current else "STALE")
    st.caption(f"Última vela cerrada: {time_label(last.close_time)}")
    if not current:
        st.warning("Datos atrasados (STALE). Los valores mostrados son históricos; no justifican una acción actual.")
    return current


def prediction_panel(prediction) -> None:
    if prediction is None:
        st.info("Sin predicciones persistidas.")
        return
    now = datetime.now(timezone.utc)
    a, b, c, d = st.columns(4)
    a.metric("Clase prevista", prediction.predicted_class)
    b.metric("P(SUBE)", percent(prediction.prob_up))
    c.metric("P(BAJA)", percent(prediction.prob_down))
    d.metric("Modelo", prediction.model_version)
    st.caption(f"Horizonte: {prediction.horizon} · clase UP si P(UP) > 50%; DOWN en caso contrario.")
    st.caption(f"Vela fuente: {time_label(prediction.timestamp)} · pred: {prediction.prediction_id[:8]}…")
    st.caption(f"Generada: {time_label(prediction.generated_at)} · válida hasta: {time_label(prediction.valid_until)}")
    left, right = st.columns(2)
    with left:
        st.caption("Oportunidad histórica")
        status_badge(prediction.timing_status)
    with right:
        st.caption("Vigencia actual")
        status_badge(prediction.current_validity(now))
    if prediction.current_validity(now) != "ACTIVE":
        st.warning("La última predicción se conserva para auditoría, pero ya no está vigente.")
    confidence = max(prediction.prob_up, prediction.prob_down)
    st.caption("P(SUBE) y P(BAJA) son probabilidades XGBoost. No son accuracy ni rentabilidad esperada.")
    st.caption(f"Gate prescriptivo: max(P) = {confidence:.1%} · umbral ≥ 60% (0.60).")
    if confidence < 0.60:
        st.info("Gate no alcanzado: esperar / sin entrada. La clase UP/DOWN se conserva como predicción técnica.")
    else:
        st.caption("Gate alcanzado; una orientación requiere además datos actuales, vigencia y contexto de posición.")


def render_home() -> None:
    data = begin("Alpha Engine", "Estado de los datos LIVE y de la última predicción persistida.")
    st.subheader("Estado del sistema")
    current = market_state(data["candles"])
    prediction = next(iter(data["predictions"]), None)
    pair, _, quote = symbol_units(config().symbol)
    candle = data["candles"][-1] if data["candles"] else None
    a, b, c, d = st.columns(4)
    a.metric("Estado basado en datos", "CURRENT" if current else "STALE" if candle else "NO_DATA")
    b.metric("Cierre de última vela", f"{candle.close:,.2f} {quote}" if candle else "N/A")
    c.metric("Última predicción", prediction.predicted_class if prediction else "N/A")
    d.metric("Modelo", prediction.model_version if prediction else "N/A")
    st.caption("El estado resume la frescura de las velas; no confirma la disponibilidad del sistema completo.")

    @st.fragment(run_every="5s")
    def latest_panel():
        st.subheader("Última vela y predicción")
        try:
            prediction_panel(latest_prediction(source_id()))
        except (ReadUnavailable, ValueError):
            st.info("La última predicción LIVE no está disponible en este momento.")
    latest_panel()
    st.subheader("Mercado reciente")
    candle_chart(data["candles"][-120:])
    st.caption(f"{pair} · velas cerradas persistidas. La auditoría interna del pipeline no se publica en esta vista.")
    st.divider()
    for route, (label, page) in ROUTES.items():
        if route != "inicio":
            st.page_link(st.Page(page, title=label, url_path=route), label=label)
    st.caption("Alpha Engine es un PMV académico de apoyo a decisiones. No ejecuta órdenes ni garantiza resultados.")


def render_descriptive() -> None:
    data = begin("Descriptivo", "Velas cerradas e indicadores causales calculados desde el almacenamiento.")
    candles = data["candles"]
    market_state(candles)
    if not candles:
        return
    latest = candles[-1]
    pair, base, quote = symbol_units(config().symbol)
    consecutive = len(candles) >= 2 and (latest.open_time - candles[-2].open_time).total_seconds() == 60
    change = latest.close / candles[-2].close - 1 if consecutive else None
    a, b, c, d = st.columns(4)
    a.metric("Precio de cierre", f"{latest.close:,.2f} {quote}")
    b.metric("Variación 1 minuto", f"{change * 100:+.3f}%" if change is not None else "N/A")
    c.metric("Volumen", f"{latest.volume:,.4f} {base}")
    d.metric("Última vela cerrada", latest.close_time.strftime("%H:%M:%S UTC"))
    if change is None:
        st.caption("Variación de un minuto no disponible: faltan dos velas consecutivas.")
    st.subheader("OHLC y volumen")
    candle_chart(candles, show_volume=True)
    st.caption(f"Símbolo: {pair} · intervalo: 1m · última vela cerrada: {time_label(latest.close_time)}")
    st.subheader("Indicadores técnicos")
    try:
        features = build_feature_frame(candles).iloc[-1]
    except ValueError:
        st.info("Indicadores no disponibles: se requiere historia continua suficiente.")
        return
    required = ["rsi_14", "ema_20", "macd", "rolling_volatility"]
    if any(pd.isna(features[name]) for name in required):
        st.info("Indicadores no disponibles: se requiere historia continua suficiente.")
        return
    a, b, c, d = st.columns(4)
    a.metric("RSI 14", f"{features.rsi_14:.2f}")
    b.metric("EMA 20", f"{features.ema_20:,.2f}")
    c.metric("MACD", f"{features.macd:.4f}")
    d.metric("Volatilidad móvil", f"{features.rolling_volatility:.4%}")
    st.caption(f"MACD señal: {features.macd_signal:.4f} · histograma: {features.macd_histogram:.4f} · cambio de volumen: {features.volume_change:+.2%}")
    relation = "sobre" if latest.close > features.ema_20 else "bajo" if latest.close < features.ema_20 else "en"
    st.caption(f"Retorno 1 minuto: {features.return_1m:+.3%} · Cierre {relation} EMA 20. Esta relación describe el precio observado; no predice su dirección siguiente.")
    st.caption("Fuente: velas cerradas ya almacenadas por el worker. Esta página no consulta Binance ni genera predicciones.")


@st.fragment
def metrics_panel(detailed: bool = False) -> None:
    key = "evaluation_live_metrics" if detailed else "predictive_live_metrics"
    if not st.button("Consultar desempeño LIVE", key=key):
        st.caption("Consulta opcional de una muestra reciente; la predicción y el historial no esperan este cálculo.")
        return
    try:
        result = metrics(source_id())
    except (ReadUnavailable, ValueError):
        st.info("Las métricas LIVE no están disponibles en este momento.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Evaluadas LIVE", result["evaluated"])
    b.metric("Accuracy LIVE", percent(result["accuracy"]))
    c.metric("Balanced accuracy", percent(result["balanced_accuracy"]))
    d.metric("Baseline siempre DOWN", percent(result["always_down_accuracy"]))
    st.caption("Muestra reciente: hasta 500 evaluaciones y 1000 predicciones almacenadas, solo V1. No es la cohorte histórica completa.")
    st.caption(f"Velas fuente evaluadas: {time_label(result['source_start'])} – {time_label(result['source_end'])}; comparables: {result['scored_predictions']}.")
    st.caption(f"Predicho UP/DOWN: {percent(result['predicted_up_fraction'])} / {percent(result['predicted_down_fraction'])}. El baseline es una referencia sobre la misma muestra, no una estrategia recomendada.")
    if result["inconsistent_correct_rows"] or result["missing_predictions"]:
        st.warning("La muestra contiene registros no comparables o inconsistentes.")
    if detailed:
        a, b, c, d = st.columns(4)
        a.metric("Correctas", result["correct"])
        b.metric("Incorrectas", result["incorrect"])
        c.metric("TIMELY / LATE", f"{result['timely']} / {result['late']}")
        d.metric("Macro F1", percent(result["macro_f1"]))
        st.subheader("Calidad por clase")
        st.dataframe([{"Clase": label, "Precision": percent(result['precision_' + suffix]),
                       "Recall": percent(result['recall_' + suffix]), "F1": percent(result['f1_' + suffix])}
                      for label, suffix in (("UP", "up"), ("DOWN", "down"))], hide_index=True, width="stretch")
        st.caption(f"ROC AUC UP: {result['roc_auc_up']:.3f}" if result["roc_auc_up"] is not None else "ROC AUC UP: N/A")
        st.subheader("Matriz de confusión LIVE")
        st.caption("Filas: clase predicha. Columnas: clase real.")
        st.dataframe([{"Predicción": predicted, "Real DOWN": result[f'pred_{suffix}_actual_down'],
                       "Real UP": result[f'pred_{suffix}_actual_up']}
                      for predicted, suffix in (("DOWN", "down"), ("UP", "up"))], hide_index=True, width="stretch")
        st.subheader("Distribución y baseline")
        st.dataframe([{"Clase": label, "Predicha": percent(result['predicted_' + suffix + '_fraction']),
                       "Real": percent(result['actual_' + suffix + '_fraction'])}
                      for label, suffix in (("UP", "up"), ("DOWN", "down"))], hide_index=True)


def render_predictive() -> None:
    data = begin("Predictivo", "Predicciones XGBoost ya generadas por el worker para el siguiente cierre de 1 minuto.")

    @st.fragment(run_every="5s")
    def latest_panel():
        st.subheader("Última predicción")
        try:
            prediction_panel(latest_prediction(source_id()))
        except (ReadUnavailable, ValueError):
            st.info("La última predicción no está disponible.")
    latest_panel()
    st.subheader("Desempeño LIVE — contexto")
    metrics_panel()
    st.subheader("Historial LIVE reciente")
    now = datetime.now(timezone.utc)
    st.dataframe([{"Hora fuente (Lima)": format_lima(p.timestamp), "Clase": p.predicted_class,
                   "P(SUBE)": percent(p.prob_up), "P(BAJA)": percent(p.prob_down),
                   "Generada (Lima)": format_lima(p.generated_at), "Válida hasta": format_lima(p.valid_until),
                   "Timing": p.timing_status or "N/A", "Vigencia": p.current_validity(now) or "N/A",
                   "Modelo": p.model_version, "ID": p.prediction_id[:8]}
                  for p in data["predictions"]], hide_index=True, width="stretch")
    st.caption("La página es de lectura: no ejecuta Binance, entrenamiento ni inferencia manual.")


def render_prescriptive() -> None:
    begin("Prescriptivo", "Orientación por reglas sobre predicciones persistidas, sin acciones externas.")
    st.markdown('<div class="ae-flow">Predicción persistida → política determinista → escenarios informativos</div>', unsafe_allow_html=True)

    @st.fragment(run_every="5s")
    def policy_panel():
        try:
            prediction = latest_prediction(source_id())
            candle = latest_candle(source_id())
        except (ReadUnavailable, ValueError):
            st.info("La orientación no está disponible en este momento.")
            return
        market_state([candle] if candle else [])
        prediction_panel(prediction)
        st.subheader("Posición y acción determinista")
        choices = prescriptive_scenarios(prediction, candle)
        selected = st.selectbox("Escenario de posición", [s["escenario"] for s in choices], key="public_position_scenario")
        scenario = next(s for s in choices if s["escenario"] == selected)
        a, b, c = st.columns(3)
        a.metric("Estado hipotético", selected)
        b.metric("Acción de la regla", scenario["accion"])
        c.metric("Confianza descriptiva", percent(max(prediction.prob_up, prediction.prob_down)) if prediction else "N/A")
        st.caption(scenario["motivo"])
        st.caption("Escenario informativo con gate 0.60. No consulta su posición real o virtual ni registra operaciones; el riesgo detallado no está evaluado.")
        st.subheader("Estado prescriptivo")
        if prediction and max(prediction.prob_up, prediction.prob_down) < 0.60:
            st.info("MANTENER / REEVALUAR" if selected != choices[0]["escenario"] else "ESPERAR / NO ENTRAR")
        st.dataframe(choices, hide_index=True, width="stretch")
        st.info("Las explicaciones LLM, las noticias y las carteras pertenecen al entorno privado. Esta vista no llama proveedores ni genera recomendaciones nuevas.")
    policy_panel()
    from deploy.northflank.academic_panel import render_panel
    try:
        current = latest_prediction(source_id())
    except (ReadUnavailable, ValueError):
        current = None
    render_panel(current.prediction_id if current else None)


@st.cache_data(show_spinner=False)
def historical_reports() -> dict:
    reports = {}
    for key, filename in (("model", "xgboost_v1_metadata.json"), ("baseline", "baseline_v1.json"), ("economic", "economic_summary_v1.json")):
        try:
            reports[key] = json.loads((ROOT / "models" / filename).read_text())
        except (OSError, ValueError):
            reports[key] = {}
    return reports


def render_evaluation() -> None:
    data = begin("Evaluación", "Resultados LIVE persistidos, TEST histórico y simulación económica en secciones separadas.")
    live_tab, test_tab, simulation_tab = st.tabs(["LIVE", "TEST HISTÓRICO", "SIMULACIÓN ECONÓMICA"])
    with live_tab:
        st.caption("Evaluaciones persistidas. Las métricas de muestra se consultan bajo demanda; no se generan evaluaciones nuevas.")
        metrics_panel(detailed=True)
        st.subheader("Evaluaciones LIVE recientes")
        st.dataframe([{"Evaluada (Lima)": format_lima(e.evaluated_at), "Movimiento observado": e.observed_move,
                       "Clase binaria": e.observed_class, "Correcta": "Sí" if e.correct else "No",
                       "Cierre origen": e.source_close, "Cierre objetivo": e.target_close,
                       "Predicción": e.prediction_id[:8]} for e in data["evaluations"]], hide_index=True, width="stretch")
        st.caption("FLAT significa cierres iguales; por la definición binaria se etiqueta como DOWN.")
    reports = historical_reports()
    with test_tab:
        st.caption("Métricas históricas del artefacto XGBoost; no son rendimiento LIVE ni prometen rentabilidad.")
        report = reports["model"]
        test = report.get("evaluation", {}).get("test", {})
        if not test:
            st.info("Informe TEST no disponible.")
        else:
            for column, label, name in zip(st.columns(4), ("Accuracy", "Precision", "Recall", "F1"), ("accuracy", "precision", "recall", "f1")):
                value = test.get(name)
                column.metric(label + " TEST", f"{value:.3f}" if value is not None else "N/A")
            baseline = reports["baseline"].get("test", {}).get("accuracy")
            st.caption(f"Modelo: {report.get('model_version', 'N/A')} · muestras TEST: {test.get('sample_count', 'N/A')} · baseline accuracy: {baseline:.3f}" if baseline is not None else "Baseline no disponible.")
            st.caption("Target: " + report.get("target_definition", "N/A"))
            trained = report.get("trained_at")
            st.caption("Entrenamiento: " + time_label(datetime.fromisoformat(trained) if trained else None))
    with simulation_tab:
        st.caption("SIMULACIÓN histórica con supuestos. No representa fills ni ejecución real.")
        report = reports["economic"]
        rows = [{"Escenario de costo": run["cost_scenario"], "Latencia asumida (s)": run["latency_seconds_assumed"],
                 "Filas TEST": run["total"]["tested_rows"], "BUY ejecutadas": run["total"]["executed_buy_trades"],
                 "Retorno neto estimado": percent(run["total"].get("net_compounded_return_estimated")),
                 "Drawdown neto estimado": percent(run["total"].get("max_drawdown_net_estimated"))}
                for run in report.get("runs", [])]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("Son estimaciones con costos y latencias asumidos, no rendimientos de una cartera personal.")
            st.caption("Límites declarados por el reporte: " + str(report.get("limits", "N/A")))
        else:
            st.info("Reporte económico no disponible.")
    st.caption("Los conjuntos TEST y LIVE son distintos. El desempeño histórico no garantiza resultados futuros.")


def render_simulation() -> None:
    from deploy.northflank.public_replay import render_replay
    data = begin("Simulación", "Replay histórico V1 en memoria, sin cartera LIVE ni operaciones persistentes.")
    render_replay(data)


ROUTES = {
    "inicio": ("Home", render_home),
    "descriptivo": ("Descriptivo", render_descriptive),
    "predictivo": ("Predictivo", render_predictive),
    "prescriptivo": ("Prescriptivo", render_prescriptive),
    "evaluacion": ("Evaluación", render_evaluation),
    "simulacion": ("Simulación", render_simulation),
}


def resolve_page(route: str):
    if route not in ROUTES:
        raise ValueError("Unknown public route")
    return ROUTES[route][1]
