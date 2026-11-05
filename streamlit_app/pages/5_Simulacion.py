"""Manual assisted virtual BTC portfolio; every write is an explicit click."""

import sys
from pathlib import Path
from uuid import uuid4

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd

from alpha_engine.services.paper_api import PaperApiError, paper_request
from streamlit_app.historical_replay import render_historical_replay
from streamlit_app.ui import auto_refresh_live, page, refresh_button


settings = page("Simulación", "Cartera BTC virtual LIVE y replay histórico de predicciones V1.")
st.markdown("""<style>
  button[data-testid="stBaseButton-secondary"] {
    background:#F8FAFC !important;color:#0F172A !important;
  }
  button[data-testid="stBaseButton-secondary"] p,
  button[data-testid="stBaseButton-secondary"] span {
    color:inherit !important;
  }
  button[data-testid="stBaseButton-secondary"]:hover {
    background:#E2E8F0 !important;
  }
  button[data-testid="stBaseButton-secondary"]:active {
    background:#CBD5E1 !important;
  }
  button[data-testid="stBaseButton-secondary"]:focus-visible {
    outline:3px solid #FACC15 !important;outline-offset:2px;
  }
  button[data-testid="stBaseButton-secondary"]:disabled {
    background:#E2E8F0 !important;color:#475569 !important;opacity:1;
  }
  button[data-testid="stBaseButton-primary"] {
    background:#B91C1C !important;border-color:#B91C1C !important;
    color:#FFFFFF !important;
  }
  button[data-testid="stBaseButton-primary"] p,
  button[data-testid="stBaseButton-primary"] span {
    color:inherit !important;
  }
  button[data-testid="stBaseButton-primary"]:hover {
    background:#991B1B !important;border-color:#991B1B !important;
  }
  button[data-testid="stBaseButton-primary"]:active {
    background:#7F1D1D !important;border-color:#7F1D1D !important;
  }
  button[data-testid="stBaseButton-primary"]:focus-visible {
    outline:3px solid #FACC15 !important;outline-offset:2px;
  }
  button[data-testid="stBaseButton-primary"]:disabled {
    background:#334155 !important;border-color:#334155 !important;
    color:#CBD5E1 !important;opacity:1;
  }
  [data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"] {
    background:#F8FAFC !important;color:#0F172A !important;
  }
  [data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"] p,
  [data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"] span {
    color:inherit !important;
  }
  [data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    background:#E2E8F0 !important;
  }
  [data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"]:active {
    background:#CBD5E1 !important;
  }
  [data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"]:focus-visible {
    outline:3px solid #FACC15 !important;outline-offset:2px;
  }
  [data-testid="stForm"] button[data-testid="stBaseButton-secondaryFormSubmit"]:disabled {
    background:#E2E8F0 !important;color:#475569 !important;opacity:1;
  }
</style>""", unsafe_allow_html=True)
st.info("Simulación con capital virtual. No ejecuta operaciones reales.")
mode = st.radio("Modo de simulación", ("Simulación LIVE", "Replay Histórico"),
                horizontal=True, key="simulation_mode")
if mode == "Replay Histórico":
    render_historical_replay(settings)
    st.stop()
st.caption("Modo MANUAL ASSISTED · Spot long only · sin short, margen ni órdenes automáticas")
refresh_button()
auto_refresh_live()


def call(payload):
    try:
        return paper_request(settings, payload)
    except PaperApiError:
        st.error("No se pudo consultar la cartera virtual. Intente de nuevo.")
        return None
    except Exception:
        st.error("El servicio de simulación no está disponible.")
        return None


token = st.session_state.get("paper_portfolio_id")
if not token:
    st.subheader("Crear o recuperar cartera")
    st.caption(f"Capital inicial USD {settings.paper_initial_capital:,.2f}; asignación por BUY "
               f"{settings.paper_allocation:.0%} del cash; fee {settings.paper_fee_rate:.2%} "
               f"y slippage {settings.paper_slippage_rate:.2%} por lado. "
               "Son parámetros de simulación, no tarifas verificadas ni recomendación financiera.")
    if st.button("Crear cartera virtual", type="primary"):
        result = call({"op": "create"})
        if result and result["status"] == "CREATED":
            st.session_state["paper_portfolio_id"] = result["portfolio"]["portfolio_id"]
            st.rerun()
    with st.form("recover_paper"):
        restored = st.text_input("Código de recuperación de cartera virtual")
        submitted = st.form_submit_button("Recuperar cartera")
    if submitted:
        result = call({"op": "read", "portfolio_id": restored.strip()})
        if result and result["status"] == "OK":
            st.session_state["paper_portfolio_id"] = result["portfolio"]["portfolio_id"]
            st.rerun()
        elif result:
            st.error("Código no válido o cartera virtual no encontrada.")
    st.caption("Guarde el código que aparecerá después de crear la cartera. "
               "Se necesita para recuperarla en otra sesión; quien lo posea puede modificar solo esta cartera virtual.")
    st.stop()

result = call({"op": "read", "portfolio_id": token})
if not result or result["status"] != "OK":
    if result:
        st.error("No se encontró esta cartera virtual.")
        if st.button("Olvidar código local"):
            st.session_state.pop("paper_portfolio_id", None)
            st.rerun()
    st.stop()

portfolio = result["portfolio"]
st.subheader("Cartera virtual")
st.caption("Código de recuperación: guárdelo en privado para volver a abrir esta cartera virtual.")
st.code(token, language=None)


def money(value):
    return "N/A" if value is None else f"USD {value:,.2f}"


a, b, c, d = st.columns(4)
a.metric("Capital inicial", money(portfolio["initial_capital"]))
b.metric("Capital actual estimado", money(portfolio["current_capital"]))
c.metric("Cash virtual", money(portfolio["cash"]))
d.metric("BTC virtual", f"{portfolio['btc']:.8f}")
a, b, c, d = st.columns(4)
a.metric("Posición", "LONG" if portfolio["position"] == "LONG" else "SIN POSICIÓN")
b.metric("Precio de entrada", money(portfolio["entry_price"]))
c.metric("Precio actual estimado", money(portfolio["current_price"]))
d.metric("Rentabilidad", "N/A" if portfolio["return_pct"] is None else f"{portfolio['return_pct']:+.2f}%")
a, b, c, d = st.columns(4)
a.metric("PnL realizado", money(portfolio["realized_pnl"]))
b.metric("PnL no realizado", money(portfolio["unrealized_pnl"]))
c.metric("PnL total", money(portfolio["total_pnl"]))
d.metric("Fees pagadas", money(portfolio["fees_paid"]))
st.caption(f"Operaciones aplicadas: {portfolio['operation_count']} · Operaciones cerradas: {portfolio['closed_trades']} · Win rate: "
           f"{portfolio['win_rate_pct']:.1f}%" if portfolio["win_rate_pct"] is not None
           else f"Operaciones aplicadas: {portfolio['operation_count']} · Operaciones cerradas: {portfolio['closed_trades']} · Win rate: N/A")
st.caption("Precio actual = punto medio del bid/ask público recibido por el servidor. "
           "El fill virtual usa ask para BUY o bid para SELL, más slippage configurado. "
           "La valoración no garantiza un precio ejecutable.")

st.divider()
st.subheader("Acción sugerida")
prediction = portfolio["prediction"]
if prediction:
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicción técnica", prediction["predicted_class"])
    p2.metric("P(UP)", f"{prediction['prob_up']:.1%}")
    p3.metric("P(DOWN)", f"{prediction['prob_down']:.1%}")
    st.caption(f"Modelo: {prediction['model_version']} · vigencia UTC: {prediction['valid_until']}")
decision = portfolio["suggestion"]
st.metric("Regla determinista", decision["action"])
st.caption(decision["reason"])
st.caption(f"Gate conservado: {settings.model_confidence_threshold:.0%}. "
           "Confianza = probabilidad asignada por el modelo; no es accuracy observada ni probabilidad de beneficio.")
if decision["action"] == "BUY":
    st.info("CONSIDERAR COMPRA virtual. Revise la señal y aplique solo si desea registrar la operación simulada.")
elif decision["action"] == "SELL":
    st.info("CONSIDERAR VENTA / SALIDA virtual. La posición seguirá abierta hasta aplicar la operación simulada.")
elif decision["action"] == "HOLD":
    st.info("MANTENER / REEVALUAR. No hay operación para aplicar.")
else:
    st.info("ESPERAR / NO ENTRAR. No hay operación para aplicar.")

can_apply = bool(decision["executable"] and portfolio["quote_received_at"] and prediction)
if st.button("Aplicar operación simulada", type="primary", disabled=not can_apply):
    request_id = st.session_state.setdefault("paper_pending_apply", str(uuid4()))
    with st.spinner("Registrando operación virtual…"):
        applied = call({"op": "apply", "portfolio_id": token,
                        "request_id": request_id, "expected_version": portfolio["version"],
                        "prediction_id": prediction["prediction_id"]})
    if applied and applied["status"] in {"APPLIED", "DUPLICATE"}:
        st.session_state.pop("paper_pending_apply", None)
        st.success("Operación virtual registrada." if applied["status"] == "APPLIED"
                   else "La operación virtual ya estaba registrada.")
        st.rerun()
    elif applied:
        st.session_state.pop("paper_pending_apply", None)
        st.warning("No se aplicó la operación: la señal, la cotización o el estado de la cartera cambió. Actualice la página.")

st.divider()
st.subheader("Ledger de la simulación actual")
if portfolio["ledger"]:
    columns = ("timestamp", "action", "price", "quantity_btc", "value_usd", "fee",
               "cash_after", "btc_after", "realized_pnl", "model_version", "signal_probability")
    frame = pd.DataFrame(portfolio["ledger"])[list(columns)]
    st.dataframe(frame, width="stretch", hide_index=True)
else:
    st.caption("Aún no hay operaciones virtuales.")

with st.expander("Reiniciar simulación"):
    st.warning("Reiniciar dejará esta cartera con su capital inicial y ocultará el ledger anterior. "
               "No altera predicciones, evaluaciones, pipeline ni recomendaciones.")
    confirmed = st.checkbox("Confirmo que deseo reiniciar esta cartera virtual")
    if st.button("Reiniciar simulación", disabled=not confirmed):
        request_id = st.session_state.setdefault("paper_pending_reset", str(uuid4()))
        reset = call({"op": "reset", "portfolio_id": token,
                      "request_id": request_id, "expected_version": portfolio["version"]})
        if reset and reset["status"] in {"RESET", "DUPLICATE"}:
            st.session_state.pop("paper_pending_reset", None)
            st.rerun()
        elif reset:
            st.session_state.pop("paper_pending_reset", None)
            st.warning("La cartera cambió. Revise su estado antes de volver a confirmar el reinicio.")
