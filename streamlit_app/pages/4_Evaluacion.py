"""Separates persisted LIVE evaluation from historical TEST and simulation reports."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from alpha_engine.services.public_dashboard import DashboardUnavailableError
from streamlit_app.ui import (
    auto_refresh_live, dashboard_data, data_mode_banner, format_lima, format_percent,
    live_evaluation_metrics, page, refresh_button, storage_error,
)


settings = page("Evaluación", "Resultados LIVE persistidos, TEST histórico y simulación económica en secciones separadas.")
refresh_button()
auto_refresh_live()

try:
    data = dashboard_data(settings, "evaluation")
except DashboardUnavailableError:
    storage_error()
    st.stop()

data_mode_banner(data)
if settings.storage_provider == "postgresql" and settings.app_env == "local":
    st.caption("Modo local: las evaluaciones incluyen el replay causal LOCAL_BOOTSTRAP y los ciclos posteriores; no equivalen a resultados LIVE prospectivos puros.")
live_tab, test_tab, simulation_tab = st.tabs(["LIVE", "TEST HISTÓRICO", "SIMULACIÓN ECONÓMICA"])

with live_tab:
    st.caption("Evaluaciones posteriores de predicciones LIVE. No modifica artefactos TRAIN, VALIDATION ni TEST.")
    metrics = None
    if data.mode == "LIVE":
        model_version = data.test_metrics.model_version if data.test_metrics and data.test_metrics.model_version else settings.model_path.stem
        try:
            metrics = live_evaluation_metrics(settings, model_version)
        except DashboardUnavailableError:
            st.info("No se pudieron consultar las métricas LIVE en este momento.")
    if metrics is None or metrics.evaluated == 0:
        st.info(f"Evaluaciones LIVE: {metrics.evaluated if metrics else 'N/A'} · Accuracy LIVE: N/A")
    else:
        a, b, c, d, e = st.columns(5)
        a.metric("Evaluadas", metrics.evaluated)
        b.metric("Correctas", metrics.correct)
        c.metric("Incorrectas", metrics.incorrect)
        d.metric("Accuracy LIVE", format_percent(metrics.accuracy))
        e.metric("TIMELY / LATE", f"{metrics.timely} / {metrics.late}")
        st.caption(
            f"Modelo: {model_version} · velas fuente evaluadas: "
            f"{format_lima(metrics.source_start, include_utc=True)} – "
            f"{format_lima(metrics.source_end, include_utc=True)}. "
            "N evaluaciones disponibles dentro del período; puede haber huecos."
        )
        if metrics.scored_predictions != metrics.evaluated or metrics.inconsistent_correct_rows:
            st.warning("Hay evaluaciones sin predicción comparable o con auditoría inconsistente; revise la cohorte.")

        st.subheader("Calidad por clase")
        a, b, c = st.columns(3)
        a.metric("Balanced accuracy", format_percent(metrics.balanced_accuracy))
        b.metric("Macro F1", format_percent(metrics.macro_f1))
        c.metric("ROC AUC UP", f"{metrics.roc_auc_up:.3f}" if metrics.roc_auc_up is not None else "N/A")
        st.dataframe([
            {"Clase": "UP", "Precision": format_percent(metrics.precision_up),
             "Recall": format_percent(metrics.recall_up), "F1": format_percent(metrics.f1_up)},
            {"Clase": "DOWN", "Precision": format_percent(metrics.precision_down),
             "Recall": format_percent(metrics.recall_down), "F1": format_percent(metrics.f1_down)},
        ], width="stretch", hide_index=True)

        st.subheader("Matriz de confusión LIVE")
        st.caption("Filas: clase predicha. Columnas: clase real.")
        st.dataframe([
            {"Predicción": "DOWN", "Real DOWN": metrics.pred_down_actual_down,
             "Real UP": metrics.pred_down_actual_up},
            {"Predicción": "UP", "Real DOWN": metrics.pred_up_actual_down,
             "Real UP": metrics.pred_up_actual_up},
        ], width="stretch", hide_index=True)

        st.subheader("Distribución y baseline")
        a, b, c, d = st.columns(4)
        a.metric("Predicho UP", format_percent(metrics.predicted_up_fraction))
        b.metric("Predicho DOWN", format_percent(metrics.predicted_down_fraction))
        c.metric("Real UP", format_percent(metrics.actual_up_fraction))
        d.metric("Real DOWN", format_percent(metrics.actual_down_fraction))
        st.caption(
            f"Accuracy {model_version}: {format_percent(metrics.accuracy)} · "
            f"baseline siempre DOWN: {format_percent(metrics.always_down_accuracy)} "
            "sobre la misma cohorte. El baseline es una referencia, no una estrategia recomendada."
        )
    if data.evaluations:
        st.dataframe([
            {
                "Evaluada (Lima)": format_lima(item.evaluated_at),
                "Movimiento observado": item.observed_move,
                "Clase binaria": item.observed_class,
                "Correcta": "Sí" if item.correct else "No",
                "Cierre origen": item.source_close,
                "Cierre objetivo": item.target_close,
                "Predicción": item.prediction_id_short,
            }
            for item in data.evaluations[:50]
        ], width="stretch", hide_index=True)
        st.caption("FLAT significa que los cierres fueron iguales; por la definición binaria del modelo se etiqueta como DOWN.")
    elif metrics is not None:
        st.caption("Aún no existen evaluaciones LIVE persistidas.")

with test_tab:
    st.caption("Métricas históricas del artefacto XGBoost. No son rendimiento LIVE ni una promesa de rentabilidad.")
    report = data.test_metrics
    if report is None:
        st.info("No está disponible el artefacto de métricas TEST en este entorno.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Accuracy TEST", f"{report.accuracy:.3f}" if report.accuracy is not None else "N/A")
        b.metric("Precision TEST", f"{report.precision:.3f}" if report.precision is not None else "N/A")
        c.metric("Recall TEST", f"{report.recall:.3f}" if report.recall is not None else "N/A")
        d.metric("F1 TEST", f"{report.f1:.3f}" if report.f1 is not None else "N/A")
        st.caption(
            f"Modelo: {report.model_version or 'N/A'} · entrenamiento: {format_lima(report.trained_at)} · "
            f"muestras TEST: {report.sample_count or 'N/A'} · baseline accuracy: "
            f"{report.baseline_accuracy:.3f}" if report.baseline_accuracy is not None else
            f"Modelo: {report.model_version or 'N/A'} · muestras TEST: {report.sample_count or 'N/A'}"
        )
        if report.target_definition:
            st.caption(f"Target: {report.target_definition}")

with simulation_tab:
    st.caption("SIMULACIÓN histórica con supuestos. No representa fills, ejecución real ni rentabilidad futura.")
    if not data.economic_scenarios:
        st.info("No está disponible el reporte de simulación económica en este entorno.")
    else:
        st.dataframe([
            {
                "Escenario de costo": item.cost_scenario,
                "Latencia asumida (s)": item.latency_seconds_assumed,
                "Filas TEST": item.tested_rows,
                "BUY ejecutadas": item.executed_buy_trades,
                "Retorno neto estimado": (
                    f"{item.net_compounded_return_estimated:.2%}"
                    if item.net_compounded_return_estimated is not None else "N/A"
                ),
                "Drawdown neto estimado": (
                    f"{item.max_drawdown_net_estimated:.2%}"
                    if item.max_drawdown_net_estimated is not None else "N/A"
                ),
            }
            for item in data.economic_scenarios
        ], width="stretch", hide_index=True)
        limits = next((item.limits for item in data.economic_scenarios if item.limits), None)
        if limits:
            st.caption(f"Límites declarados por el reporte: {limits}")
