"""Historical replay UI. All inputs are already-read DTOs; no repositories or APIs.

The unchanged historical_replay module supplies the original causal V1 policy.
Only browser-session state changes; no LIVE portfolio, account IDs or persistence.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpha_engine.services.historical_replay import ReplayConfig, build_replay, find_event_window
from deploy.northflank.public_presentation import COLORS, format_lima, status_badge


def money(value):
    return "N/A" if value is None else f"USD {value:,.2f}"


def replay_from_sample(predictions, candles, count, config, before=None):
    if count not in (60, 180, 360):
        raise ValueError("Invalid bounded replay window")
    selected = sorted((p for p in predictions if before is None or p.timestamp <= before),
                      key=lambda p: (p.timestamp, p.prediction_id), reverse=True)[:count]
    return build_replay(selected, candles, config)


def find_sample_events(predictions, count):
    # The existing search helper receives an in-memory reader, never a DB adapter.
    def entries(symbol, model_version, minimum_up, limit=100):
        return [p for p in predictions if p.model_version == model_version and p.prob_up >= minimum_up][:limit]
    return find_event_window(SimpleNamespace(list_entry_signals=entries), "BTCUSDT", count)


def charts(snapshot):
    steps = snapshot["steps"]
    if not steps:
        st.caption("Avance el replay para ver la trayectoria.")
        return
    times = [row["timestamp"] for row in steps]
    equity = go.Figure(go.Scatter(x=times, y=[row["equity"] for row in steps],
                                  mode="lines+markers", name="Cartera virtual"))
    price = go.Figure(go.Scatter(x=times, y=[row["price"] for row in steps],
                                 mode="lines", name="Precio de apertura"))
    for action, color, symbol in (("BUY", COLORS["green"], "triangle-up"),
                                  ("SELL", COLORS["red"], "triangle-down")):
        rows = [row for row in steps if row["action"] == action]
        if rows:
            price.add_trace(go.Scatter(x=[r["timestamp"] for r in rows], y=[r["price"] for r in rows],
                                       mode="markers", name=action, marker=dict(color=color, symbol=symbol, size=12)))
    for figure, title in ((equity, "Equity curve"), (price, "Precio y operaciones simuladas")):
        figure.update_layout(title=title, xaxis_title="UTC", yaxis_title="USD", height=300,
                             paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
                             font_color=COLORS["text"], margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_replay(data):
    st.subheader("Replay histórico / simulación retrospectiva")
    st.info("Capital virtual en memoria. No crea ni modifica carteras LIVE, no registra operaciones y no ejecuta órdenes reales.")
    st.caption("Predicciones persistidas xgboost_v1 · gate prescriptivo 0.60 · política determinista original.")
    count = st.selectbox("Período histórico", (60, 180, 360),
                         format_func=lambda n: f"Últimas {n} predicciones V1", key="public_replay_window")
    a, b = st.columns(2)
    if a.button("Buscar período con eventos", key="public_replay_find"):
        before, message, scanned = find_sample_events(data["predictions"], count)
        st.session_state["public_replay_before"] = before
        st.session_state["public_replay_message"] = f"{message} Señales revisadas en la muestra cargada: {scanned}."
    if b.button("Volver al período más reciente", key="public_replay_recent"):
        st.session_state.pop("public_replay_before", None)
        st.session_state.pop("public_replay_message", None)
    if message := st.session_state.get("public_replay_message"):
        st.caption(message)
    st.markdown("**Parámetros de simulación — no son tarifas verificadas ni una recomendación financiera**")
    a, b, c, d = st.columns(4)
    capital = a.number_input("Capital inicial USD", min_value=1.0, max_value=1_000_000.0,
                             value=1000.0, step=100.0, key="public_replay_capital")
    allocation = b.number_input("Tamaño por BUY % del cash", min_value=1.0, max_value=100.0,
                                value=20.0, step=1.0, key="public_replay_allocation") / 100
    fee = c.number_input("Fee por lado %", min_value=0.0, max_value=99.0,
                         value=0.10, step=0.01, format="%.2f", key="public_replay_fee") / 100
    slippage = d.number_input("Slippage por lado %", min_value=0.0, max_value=99.0,
                              value=0.02, step=0.01, format="%.2f", key="public_replay_slippage") / 100
    config = ReplayConfig(capital, allocation, fee, slippage)
    before = st.session_state.get("public_replay_before")
    replay = replay_from_sample(data["predictions"], data["candles"], count, config, before)
    st.caption(f"Muestra pública acotada: hasta 360 predicciones V1 y 1000 velas cerradas; "
               f"{len(replay.steps)} pasos causales y {replay.omitted} señales sin fill causal omitidas. "
               "La búsqueda usa solo esta muestra, no todo el histórico. Los pasos y gráficos no realizan consultas adicionales.")
    st.caption("La decisión usa generated_at vigente. El fill hipotético usa el open real T+2, posterior a la decisión; "
               "no el close futuro. Fee y slippage por lado. No se fuerza venta al terminar.")
    if not replay.steps:
        st.warning("No hay señales V1 con precio de ejecución causal disponible en este período.")
        return
    if not replay.at(len(replay.steps))["BUY"]:
        st.warning("En este período no hubo una entrada elegible que superara el gate 0.60.")
    fingerprint = (tuple(row["timestamp"] for row in replay.steps),
                   tuple(p.prediction_id for p in data["predictions"]), config, before)
    if st.session_state.get("public_replay_fingerprint") != fingerprint:
        st.session_state["public_replay_fingerprint"] = fingerprint
        st.session_state["public_replay_index"] = 0
        st.session_state["public_replay_playing"] = False
        st.session_state["public_replay_last_tick"] = time.monotonic()

    @st.fragment(run_every="1s")
    def player():
        reset, next_col, play, pause, speed_col = st.columns([1.2, 1.2, 1, 1, 1])
        if reset.button("|◀ Reiniciar", key="public_replay_reset"):
            st.session_state["public_replay_index"] = 0
            st.session_state["public_replay_playing"] = False
        if next_col.button("Siguiente minuto >", key="public_replay_next"):
            st.session_state["public_replay_index"] = min(len(replay.steps), st.session_state["public_replay_index"] + 1)
        if play.button("▶ Reproducir", key="public_replay_play"):
            st.session_state["public_replay_playing"] = True
            st.session_state["public_replay_last_tick"] = time.monotonic()
        if pause.button("⏸ Pausar", key="public_replay_pause"):
            st.session_state["public_replay_playing"] = False
        speed = speed_col.selectbox("Velocidad", (1, 5, 10), format_func=lambda x: f"{x}x", key="public_replay_speed")
        if st.session_state.get("public_replay_playing"):
            now = time.monotonic()
            if now - st.session_state.get("public_replay_last_tick", now) >= .9:
                st.session_state["public_replay_index"] = min(len(replay.steps), st.session_state["public_replay_index"] + speed)
                st.session_state["public_replay_last_tick"] = now
            if st.session_state["public_replay_index"] == len(replay.steps):
                st.session_state["public_replay_playing"] = False
        snapshot = replay.at(st.session_state["public_replay_index"])
        st.progress(snapshot["index"] / len(replay.steps), text=f"Paso {snapshot['index']} de {len(replay.steps)}")
        if current := snapshot["current"]:
            st.caption(f"Instante: {format_lima(current['timestamp'], include_utc=True)} · acción: {current['action']}")
            status_badge(current["action"])
            st.caption(current["reason"])
        else:
            st.caption("Cartera inicial. Pulse Siguiente minuto o Reproducir.")
        a, b, c, d = st.columns(4)
        a.metric("Estado", snapshot["position"])
        b.metric("Cash", money(snapshot["cash"]))
        c.metric("BTC virtual", f"{snapshot['btc']:.8f}")
        d.metric("Valor de cartera", money(snapshot["capital_final"]))
        a, b, c, d = st.columns(4)
        a.metric("PnL realizado", money(snapshot["realized_pnl"]))
        b.metric("PnL no realizado", money(snapshot["unrealized_pnl"]))
        c.metric("PnL total neto", money(snapshot["total_pnl"]))
        d.metric("Fees acumuladas", money(snapshot["fees"]))
        st.subheader("Resumen del replay hasta el paso actual")
        st.caption(f"Capital inicial {money(snapshot['capital_initial'])} · final {money(snapshot['capital_final'])} · "
                   f"return {snapshot['return_pct']:+.2f}% · BUY {snapshot['BUY']} · SELL {snapshot['SELL']} · "
                   f"HOLD {snapshot['HOLD']} · WAIT {snapshot['WAIT']}")
        st.caption(f"Trades cerrados {snapshot['closed_trades']} · ganadores {snapshot['wins']} · perdedores {snapshot['losses']} · "
                   + (f"win rate {snapshot['win_rate_pct']:.1f}%" if snapshot['win_rate_pct'] is not None else "win rate N/A"))
        st.caption(f"Max drawdown {snapshot['max_drawdown_pct']:.2f}% · exposición aproximada {snapshot['exposure_pct']:.1f}%")
        st.dataframe([{"Política": label, "Capital USD": snapshot[key]}
                      for label, key in (("Alpha Engine", "capital_final"), ("No operar", "no_trade_capital"),
                                         ("Buy & hold (marcado a mercado)", "buy_hold_capital"))], hide_index=True, width="stretch")
        charts(snapshot)
        st.subheader("Ledger del replay (solo BUY/SELL)")
        if snapshot["ledger"]:
            st.dataframe(pd.DataFrame(snapshot["ledger"]).drop(columns="step"), hide_index=True, width="stretch")
        else:
            st.caption("Aún no hay operaciones virtuales en el replay.")
    player()
