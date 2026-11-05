"""Explicit authenticated classroom panel; no cached tokens or anonymous actions."""
import os
import time

import streamlit as st

from deploy.northflank.academic_client import DemoClient, DemoUnavailable, endpoint

SESSION = "_academic_session"
RESULT = "_academic_result"


def clear_session():
    st.session_state.pop(SESSION, None)
    st.session_state.pop(RESULT, None)
    st.session_state.pop("_academic_news", None)


def show_news(news):
    news = dict(news)
    fetched = news.get("fetched_at")
    if fetched and news.get("status") == "LIVE":
        from datetime import datetime
        if time.time() - datetime.fromisoformat(fetched).timestamp() >= 1800:
            news["status"] = "STALE"
    st.subheader("Sentimiento contextual · Marketaux")
    a, b, c = st.columns(3)
    a.metric("Estado del sentimiento", news.get("status", "N/A"))
    b.metric("Clasificación", news.get("classification", "N/A"))
    score = news.get("score")
    c.metric("Score agregado", f"{score:+.4f}" if isinstance(score, (int, float)) else "N/A")
    st.caption(f"Noticias realmente consideradas: {news.get('considered', 0)} · media simple de scores BTC válidos")
    st.caption("Fuente: Marketaux · última consulta: " + str(news.get("fetched_at") or "N/A"))
    st.caption("Última noticia considerada: " + str(news.get("latest_news_at") or "N/A"))
    st.info("Información contextual; N/A no equivale a neutral. No modifica XGBoost ni autoriza una operación.")


@st.fragment(run_every="15s")
def render_panel(prediction_id=None):
    st.subheader("DEMO académico autorizado")
    try:
        origin = endpoint(dict(os.environ))
    except (DemoUnavailable, ValueError):
        st.info("Acceso DEMO no configurado. La consulta pública sigue disponible.")
        return
    if origin is None:
        st.caption("Noticias y explicaciones: N/A en modo público; los proveedores no se consultan automáticamente.")
        return
    client = DemoClient(origin)
    session = st.session_state.get(SESSION)
    if session and session["expires_at"] <= time.time():
        clear_session()
        session = None
        st.info("La sesión DEMO venció. Vuelva a ingresar.")
    if session is None:
        st.caption("Solo usuarios autorizados. La sesión dura como máximo quince minutos; no habilita trading ni carteras persistentes.")
        with st.form("academic_login", clear_on_submit=True):
            user = st.text_input("Usuario DEMO", max_chars=40)
            password = st.text_input("Contraseña DEMO", type="password", max_chars=256)
            submitted = st.form_submit_button("Ingresar al DEMO")
        if submitted:
            try:
                code, reply = client.post("login", {"user": user.strip(), "password": password})
                if (code == 200 and reply.get("status") == "AUTHORIZED"
                        and isinstance(reply.get("access_token"), str)
                        and isinstance(reply.get("expires_at"), (int, float))):
                    st.session_state[SESSION] = {"token": reply["access_token"], "expires_at": reply["expires_at"]}
                    st.rerun()
                st.warning("Acceso no autorizado o límite de intentos alcanzado.")
            except DemoUnavailable:
                st.warning("El servicio DEMO no está disponible.")
        return
    st.success("Sesión DEMO autorizada · sin operaciones persistentes")
    if st.button("Cerrar sesión DEMO"):
        try:
            status, _ = client.post("logout", {}, session["token"])
            if status != 200:
                st.warning("No se confirmó la revocación remota; se cerrará la sesión local.")
        except DemoUnavailable:
            st.warning("No se confirmó la revocación remota; la sesión local se cerró y el token vence en quince minutos como máximo.")
        clear_session()
        st.rerun()
    st.caption("La decisión proviene de reglas. NVIDIA solo puede explicarla; la posición es hipotética.")
    selected = st.text_input("ID de predicción persistida", value=prediction_id or "", max_chars=64,
                             key="academic_prediction_id")
    has_long = st.selectbox("Posición académica hipotética", [False, True],
                           format_func=lambda value: "LONG hipotético" if value else "Sin posición")
    historical = st.checkbox("Evaluar reglas en el momento histórico (no LIVE)", value=False)
    if historical:
        st.caption("Se usa el instante real de generación guardado. No incorpora noticias actuales ni representa una acción vigente.")
    payload = {"prediction_id": selected.strip(), "has_long": has_long, "historical": historical}
    action = None
    a, b = st.columns(2)
    if a.button("Consultar contexto DEMO"):
        action = "context"
    if b.button("Explicar con NVIDIA"):
        action = "explain"
    c, d = st.columns(2)
    if c.button("Leer sentimiento guardado"):
        action = "news"
    if d.button("Actualizar sentimiento (controlado)"):
        action = "refresh-news"
    if action:
        try:
            with st.spinner("Consultando el servicio académico..."):
                code, reply = client.post(action, {} if "news" in action else payload, session["token"])
            if code == 401:
                clear_session()
                st.warning("La sesión no está autorizada o venció.")
                return
            if code == 200:
                if action in {"news", "refresh-news"}:
                    st.session_state["_academic_news"] = reply.get("news", {})
                else:
                    st.session_state[RESULT] = {"payload": payload, "reply": reply}
            else:
                st.warning("Operación no completada: " + str(reply.get("status", "UNAVAILABLE")))
        except DemoUnavailable:
            st.warning("El servicio no respondió correctamente. No se reintentó la solicitud.")
    result = st.session_state.get(RESULT)
    if result and result["payload"] == payload:
        reply = result["reply"]
        prediction = reply.get("prediction", {})
        a, b = st.columns(2)
        for column, label, key in ((a, "P(UP) de esta predicción", "prob_up"),
                                   (b, "P(DOWN) de esta predicción", "prob_down")):
            value = prediction.get(key)
            column.metric(label, f"{value:.1%}" if isinstance(value, (int, float)) else "N/A")
        st.caption("Predicción consultada: " + str(reply.get("prediction_id", "N/A"))
                   + " · modelo: " + str(prediction.get("model_version", "N/A")))
        st.metric("Decisión determinista", reply.get("decision", {}).get("action", "N/A"))
        source = reply.get("source", {})
        # A cached LIVE receipt is not evidence that its prediction remains current.
        until = source.get("valid_until")
        if until:
            from datetime import datetime
            expired = datetime.fromisoformat(until).timestamp() <= time.time()
        else:
            expired = True
        label = "HISTÓRICO · predicción vencida" if expired else source.get("source_status", "N/A")
        st.caption("Fuente de la explicación: " + label + " · vencimiento: " + str(until or "N/A"))
        if source.get("historical_evaluated_at"):
            st.caption("Reglas evaluadas en el instante histórico: " + source["historical_evaluated_at"])
        st.caption(reply.get("decision", {}).get("reason", ""))
        st.caption("Trazabilidad: " + str(reply.get("decision_hash", "N/A")))
        explanation = reply.get("explanation")
        if explanation:
            st.subheader("Explicación NVIDIA · " + str(reply.get("model", "N/A")))
            st.text(explanation.get("summary", ""))
            for item in explanation.get("key_factors", []):
                st.text("• " + item)
            for item in explanation.get("risks", []):
                st.text("Riesgo: " + item)
            st.text(explanation.get("warning", ""))
            st.caption("Explicación recuperada de caché" if reply.get("cached") else "Explicación generada para la solicitud guardada en esta sesión")
        show_news(reply.get("news", {}))
    if "_academic_news" in st.session_state:
        show_news(st.session_state["_academic_news"])
