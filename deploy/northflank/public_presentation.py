"""Reviewed presentation-only helpers from the current ALPHA ENGINE UI.

No Settings, dotenv, repositories, provider clients or original page imports.
"""
from __future__ import annotations
import html
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alpha_engine.data.schemas import Candle

LIMA = ZoneInfo("America/Lima")



COLORS = {
    "background": "#020617", "panel": "#0F172A", "text": "#F8FAFC",
    "muted": "#A8B6C9", "green": "#22C55E", "red": "#FB7185", "grid": "#334155",
}


QUOTE_ASSETS = ("FDUSD", "USDT", "USDC", "TUSD", "BUSD", "DAI", "EUR", "TRY", "BRL", "BTC", "ETH", "BNB")


STYLE = """<style>
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
        </style>"""


def apply_theme() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    # The original simulation uses light secondary buttons on the dark surface.
    # Keep that contrast when the shared theme is explicitly dark in Docker.
    st.markdown('''<style>
    button[data-testid="stBaseButton-secondary"] {
      background:#F8FAFC !important;color:#0F172A !important;
    }
    button[data-testid="stBaseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] span {color:inherit !important}
    button[data-testid="stBaseButton-secondary"]:hover {background:#E2E8F0 !important}
    button[data-testid="stBaseButton-secondary"]:focus-visible {
      outline:3px solid #FACC15 !important;outline-offset:2px;
    }
    button[data-testid="stBaseButton-secondary"]:disabled {
      background:#E2E8F0 !important;color:#475569 !important;opacity:1;
    }
    </style>''', unsafe_allow_html=True)


def status_badge(value: str | None) -> None:
    if value is None:
        st.caption("N/A")
        return
    colors = {"CURRENT": "#22C55E", "ACTIVE": "#22C55E", "TIMELY": "#22C55E",
              "SUCCESS": "#22C55E", "STALE": "#F59E0B", "EXPIRED": "#F59E0B",
              "LATE": "#FB7185", "FAILED": "#FB7185", "NO_DATA": "#94A3B8"}
    color = colors.get(value, "#94A3B8")
    st.markdown(f'<span class="ae-badge" style="border-color:{color};color:{color}">'
                + html.escape(str(value)) + '</span>', unsafe_allow_html=True)


def symbol_units(symbol: str) -> tuple[str, str, str]:
    for quote in QUOTE_ASSETS:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[:-len(quote)]
            return f"{base}/{quote}", base, quote
    return symbol, "", ""


def format_lima(value: datetime | None, *, include_utc: bool = False) -> str:
    if value is None:
        return "N/A"
    local = value.astimezone(LIMA)
    shown = local.strftime("%Y-%m-%d %H:%M:%S PET")
    return f"{shown} · {value.astimezone(timezone.utc):%H:%M:%S UTC}" if include_utc else shown


def format_percent(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}%}"


def candles_frame(candles: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame([
        {"time": c.open_time, "open": c.open, "high": c.high, "low": c.low,
         "close": c.close, "volume": c.volume}
        for c in candles
    ])


def candle_chart(candles: list[Candle], *, show_volume: bool = False) -> None:
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
