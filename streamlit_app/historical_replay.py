"""Historical read-only replay panel; no LIVE wallet or paper API access."""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpha_engine.config.settings import Settings, get_settings
from alpha_engine.repositories.factory import create_market_repository, create_prediction_repository
from alpha_engine.services.historical_replay import (
    GATE, MODEL_VERSION, ReplayConfig, build_replay, find_event_window, load_window,
)
from streamlit_app.ui import format_lima


@st.cache_data(ttl=300, show_spinner=False)
def _dataset(storage_provider: str, symbol: str, interval: str, count: int, before):
    settings = get_settings()
    predictions_repo = create_prediction_repository(settings)
    market_repo = None
    try:
        market_repo = create_market_repository(settings)
        return load_window(predictions_repo, market_repo, symbol, interval, count, before)
    finally:
        predictions_repo.close()
        if market_repo is not None:
            market_repo.close()


def _find_events(settings: Settings, count: int):
    repo = create_prediction_repository(settings)
    try:
        return find_event_window(repo, settings.symbol, count)
    finally:
        repo.close()


def _money(value: float) -> str:
    return f"USD {value:,.2f}"


def _charts(snapshot: dict) -> None:
    steps = snapshot["steps"]
    if not steps:
        st.caption("Avance el replay para ver la trayectoria.")
        return
    times = [row["timestamp"] for row in steps]
    equity = go.Figure(go.Scatter(x=times, y=[row["equity"] for row in steps],
                                  mode="lines+markers", name="Cartera virtual"))
    equity.update_layout(title="Equity curve", xaxis_title="UTC", yaxis_title="USD",
                         height=300, margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(equity, width="stretch", config={"displaylogo": False})

    price = go.Figure(go.Scatter(x=times, y=[row["price"] for row in steps],
                                 mode="lines", name="BTC open"))
    for action, color, symbol in (("BUY", "#22C55E", "triangle-up"),
                                  ("SELL", "#FB7185", "triangle-down")):
        marked = [row for row in steps if row["action"] == action]
        if marked:
            price.add_trace(go.Scatter(x=[row["timestamp"] for row in marked],
                                       y=[row["price"] for row in marked],
                                       mode="markers", name=action,
                                       marker=dict(color=color, symbol=symbol, size=12)))
    price.update_layout(title="BTC y operaciones simuladas", xaxis_title="UTC",
                        yaxis_title="USD/ BTC", height=300,
                        margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(price, width="stretch", config={"displaylogo": False})


def render_historical_replay(settings: Settings) -> None:
    st.subheader("Replay histórico / simulación retrospectiva")
    st.info("Capital virtual. No ejecuta operaciones reales. Los resultados históricos no garantizan resultados futuros.")
    st.caption("Predicciones reales persistidas de xgboost_v1 · Gate prescriptivo 0.60 · Política determinista del módulo Prescriptivo. El replay no modifica la cartera LIVE.")

    count = st.selectbox("Período histórico", (60, 180, 360), format_func=lambda n: f"Últimas {n} predicciones V1", key="replay_window")
    a, b = st.columns(2)
    if a.button("Buscar período con eventos", key="replay_find"):
        with st.spinner("Buscando únicamente señales históricas reales…"):
            try:
                before, message, scanned = _find_events(settings, count)
            except Exception:
                st.error("No se pudo buscar el período histórico.")
            else:
                st.session_state["replay_search_message"] = f"{message} Señales de entrada elegibles revisadas: {scanned}."
                if before is not None:
                    st.session_state["replay_before"] = before
                else:
                    st.session_state.pop("replay_before", None)
                st.rerun()
    if b.button("Volver al período más reciente", key="replay_recent"):
        st.session_state.pop("replay_before", None)
        st.session_state.pop("replay_search_message", None)
        st.rerun()
    if message := st.session_state.get("replay_search_message"):
        st.caption(message)

    st.markdown("**Parámetros de simulación — no son una recomendación financiera ni tarifas reales verificadas**")
    a, b, c, d = st.columns(4)
    capital = a.number_input("Capital inicial USD", min_value=1.0, max_value=1_000_000.0,
                             value=1000.0, step=100.0, key="replay_capital")
    allocation = b.number_input("Tamaño por BUY % del cash", min_value=1.0, max_value=100.0,
                                value=20.0, step=1.0, key="replay_allocation") / 100
    fee = c.number_input("Fee por lado %", min_value=0.0, max_value=99.0,
                         value=0.10, step=0.01, format="%.2f", key="replay_fee") / 100
    slippage = d.number_input("Slippage por lado %", min_value=0.0, max_value=99.0,
                              value=0.02, step=0.01, format="%.2f", key="replay_slippage") / 100
    config = ReplayConfig(capital, allocation, fee, slippage)
    before = st.session_state.get("replay_before")
    started = time.perf_counter()
    try:
        predictions, candles = _dataset(settings.storage_provider, settings.symbol,
                                        settings.interval, count, before)
        replay = build_replay(predictions, candles, config)
    except Exception:
        st.error("No se pudo cargar el replay histórico. Compruebe las predicciones y velas persistidas.")
        return
    load_ms = (time.perf_counter() - started) * 1000
    st.caption(f"Fuente: {len(predictions)} predicciones V1 y {len(candles)} velas reales; "
               f"{replay.omitted} señales sin fill causal omitidas. "
               f"Carga/cálculo observado: {load_ms:.0f} ms. "
               "Carga BD: 1 lectura de predicciones + 1 lectura de velas (paginada si fuera necesario); 0 consultas por paso.")
    st.caption("Decisión al generated_at vigente de la predicción de la vela T. "
               "Una orden virtual presentada entonces se valora al open real de la vela T+2 minutos, "
               "el primer open completo posterior. Fee sobre valor USD y slippage por lado. "
               "No se utiliza el close posterior para decidir; no se fuerza venta al terminar.")
    if not replay.steps:
        st.warning("No hay señales V1 con precio de ejecución causal disponible en este período.")
        return
    full = replay.at(len(replay.steps))
    if full["BUY"] == 0:
        st.warning("En este período no hubo señal de entrada que superara el gate experimental de 0.60. Sin entradas elegibles en este período.")

    fingerprint = (tuple(row["timestamp"] for row in replay.steps),
                   config, before)
    if st.session_state.get("replay_fingerprint") != fingerprint:
        st.session_state["replay_fingerprint"] = fingerprint
        st.session_state["replay_index"] = 0
        st.session_state["replay_playing"] = False
        st.session_state["replay_last_tick"] = time.monotonic()

    @st.fragment(run_every="1s")
    def _player():
        left, next_col, play_col, pause_col, speed_col = st.columns([1.2, 1.2, 1, 1, 1])
        if left.button("|◀ Reiniciar", key="replay_reset"):
            st.session_state["replay_index"] = 0
            st.session_state["replay_playing"] = False
        if next_col.button("Siguiente minuto >", key="replay_next"):
            st.session_state["replay_index"] = min(len(replay.steps),
                                                    st.session_state["replay_index"] + 1)
            st.session_state["replay_last_tick"] = time.monotonic()
        if play_col.button("▶ Reproducir", key="replay_play"):
            st.session_state["replay_playing"] = True
            st.session_state["replay_last_tick"] = time.monotonic()
        if pause_col.button("⏸ Pausar", key="replay_pause"):
            st.session_state["replay_playing"] = False
        speed = speed_col.selectbox("Velocidad", (1, 5, 10),
                                     format_func=lambda x: f"{x}x", key="replay_speed")
        if st.session_state.get("replay_playing"):
            now = time.monotonic()
            if now - st.session_state.get("replay_last_tick", now) >= 0.9:
                st.session_state["replay_index"] = min(len(replay.steps),
                                                        st.session_state["replay_index"] + speed)
                st.session_state["replay_last_tick"] = now
            if st.session_state["replay_index"] == len(replay.steps):
                st.session_state["replay_playing"] = False
        snapshot = replay.at(st.session_state["replay_index"])
        current = snapshot["current"]
        st.progress(snapshot["index"] / len(replay.steps),
                    text=f"Paso {snapshot['index']} de {len(replay.steps)}")
        if current:
            a, b, c, d, e = st.columns(5)
            a.metric("Timestamp histórico", format_lima(current["timestamp"]))
            b.metric("Precio BTC", _money(current["price"]))
            c.metric("P(UP)", f"{current['prob_up']:.1%}")
            d.metric("P(DOWN)", f"{current['prob_down']:.1%}")
            e.metric("Predicción técnica", current["predicted_class"])
            st.metric("Acción determinista", current["action"])
            st.caption(current["reason"])
        else:
            st.caption("Cartera inicial. Pulse Siguiente minuto o Reproducir.")
        a, b, c, d = st.columns(4)
        a.metric("Estado", snapshot["position"])
        b.metric("Cash", _money(snapshot["cash"]))
        c.metric("BTC virtual", f"{snapshot['btc']:.8f}")
        d.metric("Valor de cartera", _money(snapshot["capital_final"]))
        a, b, c, d = st.columns(4)
        a.metric("PnL realizado", _money(snapshot["realized_pnl"]))
        b.metric("PnL no realizado", _money(snapshot["unrealized_pnl"]))
        c.metric("PnL total neto", _money(snapshot["total_pnl"]))
        d.metric("Fees acumuladas", _money(snapshot["fees"]))

        st.subheader("Resumen del replay hasta el paso actual")
        st.caption(f"Capital inicial {_money(snapshot['capital_initial'])} · final {_money(snapshot['capital_final'])} · "
                   f"return {snapshot['return_pct']:+.2f}% · BUY {snapshot['BUY']} · SELL {snapshot['SELL']} · "
                   f"HOLD {snapshot['HOLD']} · WAIT {snapshot['WAIT']}")
        st.caption(f"Trades cerrados {snapshot['closed_trades']} · ganadores {snapshot['wins']} · "
                   f"perdedores {snapshot['losses']} · win rate "
                   f"{snapshot['win_rate_pct']:.1f}%" if snapshot["win_rate_pct"] is not None else
                   f"Trades cerrados {snapshot['closed_trades']} · ganadores {snapshot['wins']} · "
                   f"perdedores {snapshot['losses']} · win rate N/A")
        st.caption(f"Max drawdown {snapshot['max_drawdown_pct']:.2f}% · exposición aproximada "
                   f"{snapshot['exposure_pct']:.1f}% · fees {_money(snapshot['fees'])}")
        baseline = pd.DataFrame([
            {"Política": "Alpha Engine", "Capital USD": snapshot["capital_final"]},
            {"Política": "No operar", "Capital USD": snapshot["no_trade_capital"]},
            {"Política": "Buy & hold (marcado a mercado)", "Capital USD": snapshot["buy_hold_capital"]},
        ])
        st.dataframe(baseline, width="stretch", hide_index=True)
        _charts(snapshot)
        st.subheader("Ledger del replay (solo BUY/SELL)")
        if snapshot["ledger"]:
            frame = pd.DataFrame(snapshot["ledger"]).drop(columns="step")
            st.dataframe(frame, width="stretch", hide_index=True)
        else:
            st.caption("Aún no hay operaciones virtuales en el replay.")

    _player()
