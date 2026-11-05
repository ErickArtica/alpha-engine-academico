"""Persisted market data only; this page never queries Binance."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from alpha_engine.services.public_dashboard import DashboardUnavailableError
from alpha_engine.tools.market import AnalyticsDataError, get_market_summary
from streamlit_app.ui import (
    auto_refresh_live, candle_chart, dashboard_data, data_mode_banner, format_lima, page,
    refresh_button, storage_error, symbol_units,
)


settings = page("Descriptivo", "Velas cerradas e indicadores causales calculados desde el almacenamiento.")
pair, base, quote = symbol_units(settings.symbol)
base_suffix = f" {base}" if base else ""
quote_suffix = f" {quote}" if quote else ""
st.markdown(f'<p class="ae-eyebrow">{pair} · fuente persistida · {settings.interval}</p>', unsafe_allow_html=True)
refresh_button()
auto_refresh_live()

try:
    data = dashboard_data(settings, "descriptive")
except DashboardUnavailableError:
    storage_error()
    st.stop()

data_mode_banner(data)
if not data.candles:
    st.info("Sin velas almacenadas.")
    st.stop()

latest = data.candles[-1]
try:
    change = get_market_summary(data.candles).change_1m_pct
except AnalyticsDataError:
    change = None
a, b, c, d = st.columns(4)
a.metric("Precio de cierre", f"{latest.close:,.2f}{quote_suffix}")
b.metric("Variación 1 minuto", f"{change:+.3f}%" if change is not None else "N/A")
c.metric("Volumen", f"{latest.volume:,.4f}{base_suffix}")
d.metric("Última vela cerrada", latest.close_time.strftime("%H:%M:%S UTC"))
if change is None:
    st.caption("Variación de un minuto no disponible: faltan dos velas consecutivas.")

st.subheader("OHLC y volumen")
candle_chart(data.candles, show_volume=True)
st.caption(f"Símbolo: {pair} · intervalo: {latest.interval} · última vela cerrada: {format_lima(latest.close_time, include_utc=True)}")

st.subheader("Indicadores técnicos")
if data.indicators is None:
    st.info("Indicadores no disponibles: se requiere historia continua suficiente.")
else:
    indicators = data.indicators
    a, b, c, d = st.columns(4)
    a.metric("RSI 14", f"{indicators.rsi_14:.2f}")
    b.metric("EMA 20", f"{indicators.ema_20:,.2f}")
    c.metric("MACD", f"{indicators.macd:.4f}")
    d.metric("Volatilidad móvil", f"{indicators.rolling_volatility:.4%}")
    st.caption(
        f"MACD señal: {indicators.macd_signal:.4f} · histograma: {indicators.macd_histogram:.4f} · "
        f"cambio de volumen: {indicators.volume_change:+.2%}"
    )
    st.caption(f"Retorno 1 minuto: {indicators.return_1m:+.3%} · "
               f"Cierre {'sobre' if latest.close > indicators.ema_20 else 'bajo' if latest.close < indicators.ema_20 else 'en'} EMA 20. "
               "Esta relación describe el precio observado; no predice su dirección siguiente.")

st.caption("Fuente: velas cerradas ya almacenadas por el worker. Esta página no consulta Binance ni genera predicciones.")
