"""Presentation helpers for the read-only LIVE Streamlit frontend."""

import base64
from contextlib import contextmanager
import html
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpha_engine.config.settings import PROJECT_ROOT, Settings, get_settings
from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.services.public_dashboard import (
    DashboardUnavailableError,
    PublicCandleView,
    PublicDashboardData,
    PublicDashboardService,
    PublicFallbackDashboardService,
    PublicLiveMetrics,
    PublicPredictionView,
    read_latest_public_prediction,
    read_public_live_metrics,
)


COLORS = {
    "background": "#020617", "panel": "#0F172A", "text": "#F8FAFC",
    "muted": "#A8B6C9", "green": "#22C55E", "red": "#FB7185", "grid": "#334155",
}
QUOTE_ASSETS = ("FDUSD", "USDT", "USDC", "TUSD", "BUSD", "DAI", "EUR", "TRY", "BRL", "BTC", "ETH", "BNB")
LIMA = ZoneInfo("America/Lima")


@contextmanager
def section_timing(section: str):
    """Opt-in local timing; logs no settings, credentials or row contents."""
    if os.environ.get("ALPHA_UI_TIMINGS") != "1":
        yield
        return
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        logger = logging.getLogger("alpha_engine.ui_timing")
        logger.setLevel(logging.INFO)
        logger.info("section=%s elapsed_ms=%.1f outcome=%s", section,
                    (time.perf_counter() - started) * 1000, outcome)


def symbol_units(symbol: str) -> tuple[str, str, str]:
    for quote in QUOTE_ASSETS:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[:-len(quote)]
            return f"{base}/{quote}", base, quote
    return symbol, "", ""


def page(title: str, question: str | None = None) -> Settings:
    # Root-level Streamlit Secrets become environment variables after loading.
    try:
        _ = list(st.secrets)
    except FileNotFoundError:
        pass
    st.set_page_config(page_title=f"{title} · Alpha Engine", page_icon="📈", layout="wide")
    st.markdown(
        """<style>
        .stApp {background:#020617;color:#F8FAFC}
        h1,h2,h3 {letter-spacing:-.025em}
        [data-testid="stMetric"] {background:#0F172A;border:1px solid #334155;border-radius:12px;padding:1rem}
        [data-testid="stMetricLabel"] {color:#CBD5E1}
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] p {color:#F8FAFC !important;font-size:1.55rem !important;line-height:1.25;overflow-wrap:anywhere}
        button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-secondary"] p {color:#0F172A !important}
        .st-key-refresh_control {margin:.45rem 0 1.1rem}
        .st-key-refresh_control button[data-testid="stBaseButton-secondary"] {
          background:#E0F2FE !important;color:#082F49 !important;
          border:2px solid #38BDF8 !important;border-radius:10px !important;
          padding:.45rem .9rem !important;min-height:2.65rem;
          font-weight:700;transition:background-color .15s,border-color .15s,box-shadow .15s
        }
        .st-key-refresh_control button[data-testid="stBaseButton-secondary"] p,
        .st-key-refresh_control button[data-testid="stBaseButton-secondary"] span {color:inherit !important}
        .st-key-refresh_control button[data-testid="stBaseButton-secondary"]:hover {
          background:#BAE6FD !important;border-color:#0EA5E9 !important;
          box-shadow:0 2px 10px #0284C744
        }
        .st-key-refresh_control button[data-testid="stBaseButton-secondary"]:active {
          background:#7DD3FC !important;box-shadow:none
        }
        .st-key-refresh_control button[data-testid="stBaseButton-secondary"]:focus-visible {
          outline:3px solid #FACC15 !important;outline-offset:3px
        }
        .st-key-refresh_control button[data-testid="stBaseButton-secondary"]:disabled {
          background:#334155 !important;color:#CBD5E1 !important;
          border-color:#475569 !important;box-shadow:none !important;
          cursor:not-allowed;opacity:1
        }
        [data-baseweb="tab"], [data-baseweb="tab"] p {color:#CBD5E1 !important}
        .ae-eyebrow {color:#A8B6C9;text-transform:uppercase;letter-spacing:.13em;font-size:.77rem;font-weight:700}
        .ae-subtitle {color:#CBD5E1;font-size:1.05rem;line-height:1.55}
        .ae-flow {padding:1rem;border:1px solid #334155;border-radius:12px;background:#0F172A;color:#DCE7F3}
        .ae-badge {display:inline-block;border:1px solid #475569;border-radius:999px;padding:.15rem .5rem;margin-right:.35rem;font-size:.78rem;font-weight:700}
        .ae-info {background:#13243D;border:1px solid #60A5FA;border-radius:10px;color:#E2E8F0;padding:.8rem 1rem;line-height:1.5}
        .st-key-guardrail_info [data-testid="stAlert"], .st-key-guardrail_gate_info [data-testid="stAlert"] {background:#13243D !important;border:1px solid #60A5FA !important}
        .st-key-guardrail_info [data-testid="stAlert"] *, .st-key-guardrail_gate_info [data-testid="stAlert"] * {color:#E2E8F0 !important}
        </style>""",
        unsafe_allow_html=True,
    )
    st.caption("ALPHA ENGINE  /  PMV ACADÉMICO  /  SIN OPERACIONES REALES")
    st.title(title)
    if question:
        st.markdown(f'<p class="ae-subtitle">{question}</p>', unsafe_allow_html=True)
    settings = get_settings()
    if settings.prescriptive_backend == "http":
        local_compose = settings.app_env == "local" and settings.storage_provider == "postgresql"
        if (settings.storage_provider != "supabase" and not local_compose) or settings.dashboard_fallback_enabled:
            st.error("ConfiguraciÃ³n CLOUD insegura o incompleta.")
            st.stop()
        if settings.storage_provider == "supabase" and _is_supabase_write_key(settings.supabase_key.get_secret_value()):
            st.error("La clave de escritura de Supabase no puede usarse en Streamlit.")
            st.stop()
    return settings


def _is_supabase_write_key(key: str) -> bool:
    if key.startswith("sb_secret_"):
        return True
    try:
        encoded = key.split(".")[1]
        encoded += "=" * (-len(encoded) % 4)
        claims = json.loads(base64.urlsafe_b64decode(encoded))
        return claims.get("role") == "service_role"
    except (IndexError, ValueError, TypeError):
        return False


@st.cache_data(ttl=55, show_spinner=False)
def _load_dashboard(
    storage_provider: str, symbol: str, interval: str,
    fallback_enabled: bool, fallback_snapshot: str, section: str = "all",
) -> PublicDashboardData:
    """The cache contains safe DTOs only; no repository or credential is cached."""
    settings = get_settings()
    try:
        with PublicDashboardService.from_settings(settings) as service:
            return service.dashboard(section=section)
    except DashboardUnavailableError:
        if not fallback_enabled:
            raise
        path = Path(fallback_snapshot)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return PublicFallbackDashboardService(path).dashboard()


def dashboard_data(settings: Settings, section: str = "all") -> PublicDashboardData:
    data = _load_dashboard(
        settings.storage_provider, settings.symbol, settings.interval,
        settings.dashboard_fallback_enabled, str(settings.dashboard_fallback_snapshot), section,
    )
    if data.mode != "LIVE":
        return data
    # Cache persisted facts across page navigation, while freshness and
    # prediction validity always use the current UTC clock.
    now = datetime.now(timezone.utc)
    candle = data.latest_candle
    state = (
        "NO_DATA" if candle is None else
        "STALE" if (now - candle.close_time).total_seconds() > settings.max_candle_age_seconds
        else "CURRENT"
    )
    prediction = data.latest_prediction
    if prediction is not None:
        prediction = prediction.model_copy(update={"current_validity": prediction.validity_at(now)})
    return data.model_copy(update={"system_state": state, "latest_prediction": prediction})


@st.cache_data(ttl=4, show_spinner=False)
def _load_latest_prediction(storage_provider: str, symbol: str, interval: str) -> PublicPredictionView | None:
    settings = get_settings()
    return read_latest_public_prediction(settings)


def latest_prescriptive_prediction(settings: Settings) -> PublicPredictionView | None:
    return _load_latest_prediction(settings.storage_provider, settings.symbol, settings.interval)


def latest_visible_prediction(settings: Settings, data: PublicDashboardData) -> PublicPredictionView | None:
    """Read one LIVE prediction; preserve the offline snapshot in fallback mode."""
    if data.mode == "FALLBACK":
        return data.latest_prediction
    return latest_prescriptive_prediction(settings)


@st.cache_data(ttl=55, show_spinner=False)
def _load_live_metrics(storage_provider: str, symbol: str, interval: str, model_version: str) -> PublicLiveMetrics:
    """One bounded-age, public cohort shared by Predictivo and Evaluación."""
    return read_public_live_metrics(get_settings(), model_version)


def live_evaluation_metrics(settings: Settings, model_version: str) -> PublicLiveMetrics:
    return _load_live_metrics(settings.storage_provider, settings.symbol, settings.interval, model_version)


def info_note(message: str) -> None:
    """Readable informational copy against the fixed dark application surface."""
    st.markdown(f'<div class="ae-info">{html.escape(message)}</div>', unsafe_allow_html=True)


def refresh_button() -> None:
    with st.container(key="refresh_control"):
        pressed = st.button("Actualizar datos", icon="🔄", help="Vuelve a consultar el almacenamiento configurado.")
    if pressed:
        _load_dashboard.clear()
        _load_latest_prediction.clear()
        _load_live_metrics.clear()
        st.rerun()


def auto_refresh_live() -> None:
    """Refresh the visible page once per minute while its browser session is open."""
    st.session_state["_ae_last_full_run"] = time.monotonic()

    @st.fragment(run_every=60)
    def _refresh_tick() -> None:
        if time.monotonic() - st.session_state["_ae_last_full_run"] >= 58:
            st.rerun()

    _refresh_tick()


def format_lima(value: datetime | None, *, include_utc: bool = False) -> str:
    if value is None:
        return "N/A"
    local = value.astimezone(LIMA)
    shown = local.strftime("%Y-%m-%d %H:%M:%S PET")
    return f"{shown} · {value.astimezone(timezone.utc):%H:%M:%S UTC}" if include_utc else shown


def format_duration_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.0f} ms"


def format_percent(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}%}"


def status_badge(value: str | None) -> None:
    if value is None:
        st.caption("N/A")
        return
    colors = {
        "CURRENT": "#22C55E", "ACTIVE": "#22C55E", "TIMELY": "#22C55E", "SUCCESS": "#22C55E",
        "STALE": "#F59E0B", "EXPIRED": "#F59E0B", "LATE": "#FB7185", "FAILED": "#FB7185",
        "NO_DATA": "#94A3B8",
    }
    color = colors.get(value, "#94A3B8")
    st.markdown(
        f'<span class="ae-badge" style="border-color:{color};color:{color}">{value}</span>',
        unsafe_allow_html=True,
    )


def data_mode_banner(data: PublicDashboardData, *, fast_prediction: bool = False) -> None:
    if data.mode == "FALLBACK":
        st.warning(
            f"MODO FALLBACK · corte del snapshot: {format_lima(data.snapshot_at, include_utc=True)}. "
            "No son datos LIVE."
        )
    else:
        cadence = (
            "última predicción ~5 s · resto de la página ~60 s"
            if fast_prediction else "actualización automática ~60 s"
        )
        st.caption(
            f"Modo: LIVE · Storage: {data.storage} · Consultado: "
            f"{format_lima(data.refreshed_at, include_utc=True)} · {cadence}"
        )
        if data.latest_candle is not None:
            st.caption(f"Última vela cerrada: {format_lima(data.latest_candle.close_time, include_utc=True)}")
        if data.system_state == "STALE":
            st.warning("Datos atrasados (STALE). Los valores mostrados son históricos; no justifican una acción actual.")
        elif data.system_state == "NO_DATA" and data.latest_prediction is None:
            st.info("No hay datos de mercado disponibles en esta vista.")


def storage_error() -> None:
    st.error("No se pudo consultar el estado LIVE. No se muestran datos históricos como si fueran actuales.")


def candles_frame(candles: list[PublicCandleView]) -> pd.DataFrame:
    return pd.DataFrame([
        {"time": c.open_time, "open": c.open, "high": c.high, "low": c.low,
         "close": c.close, "volume": c.volume}
        for c in candles
    ])


def candle_chart(candles: list[PublicCandleView], *, show_volume: bool = False) -> None:
    if not candles:
        st.info("Sin velas almacenadas para mostrar.")
        return
    frame = candles_frame(candles)
    pair, base, quote = symbol_units(candles[0].symbol)
    figure = go.Figure(go.Candlestick(
        x=frame["time"], open=frame["open"], high=frame["high"], low=frame["low"], close=frame["close"],
        increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"], name=pair,
    ))
    figure.update_layout(
        height=410, margin=dict(l=8, r=8, t=24, b=8), paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"], font_color=COLORS["text"], xaxis_rangeslider_visible=False,
        xaxis_title="Hora UTC", yaxis_title=f"Precio ({quote})" if quote else "Precio",
        xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(gridcolor=COLORS["grid"]), showlegend=False,
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    if show_volume:
        volume = go.Figure(go.Bar(x=frame["time"], y=frame["volume"], marker_color=COLORS["muted"]))
        volume.update_layout(
            height=180, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor=COLORS["panel"],
            plot_bgcolor=COLORS["panel"], font_color=COLORS["text"], xaxis_title="Hora UTC",
            yaxis_title=f"Volumen ({base})" if base else "Volumen",
            xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(gridcolor=COLORS["grid"]),
        )
        st.plotly_chart(volume, width="stretch", config={"displaylogo": False})


def prediction_is_actionable(
    candles: list[Candle], prediction: PredictionRecord | None, now: datetime | None = None,
) -> bool:
    """Legacy helper retained for temporal compatibility tests.

    LIVE pages use ``PredictionRecord.current_validity`` through public DTOs.
    """
    if not candles or prediction is None or prediction.timestamp != candles[-1].open_time:
        return False
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now debe incluir zona horaria")
    if prediction.valid_until is not None:
        return prediction.current_validity(current_time) == "ACTIVE"
    return candles[-1].close_time <= current_time < candles[-1].close_time + timedelta(minutes=1)
