"""Read-only Marketaux presentation. No repository, provider or refresh calls."""
from datetime import datetime, timezone

import streamlit as st

from alpha_engine.data.news_context import NewsContext
from alpha_engine.news.marketaux import public_article_url

REASONS = {
    "NOT_CONFIGURED":"Proveedor de noticias no configurado.",
    "NO_RECENT_NEWS":"No hay noticias de Bitcoin publicadas en los últimos 60 minutos.",
    "NO_NEWS_IN_WINDOW":"No hay noticias disponibles en la muestra de las últimas 24 horas.",
    "NO_VALID_SENTIMENT":"Hay noticias recientes, pero carecen de un score BTC válido.",
    "CACHE_EXPIRED":"El snapshot necesita actualizarse mediante una solicitud explícita.",
    "DAILY_BUDGET_EXHAUSTED":"Se alcanzó el presupuesto diario de consultas.",
    "REFRESH_IN_PROGRESS":"Otra solicitud ya está consultando las noticias.",
    "RATE_LIMIT":"El proveedor o el límite de consultas impide actualizar ahora.",
    "PLAN_RESTRICTION":"El plan del proveedor no permite la consulta.",
    "AUTHENTICATION_FAILED":"No se pudo autenticar al proveedor de noticias.",
    "STORAGE_UNAVAILABLE":"El caché de noticias no está disponible.",
}


def render_marketaux_context(context: NewsContext, *, now: datetime | None = None,
                             publication_authorized: bool = False) -> None:
    # No product call site grants publication rights. Hide even cached API/DTO
    # content until a separate rights review explicitly approves publication.
    if not publication_authorized:
        st.info("Contexto Marketaux no disponible: publicación pendiente de autorización de derechos.")
        return
    context = context.at_time(now or datetime.now(timezone.utc))
    st.subheader("Contexto de noticias · Marketaux")
    if context.status == "AVAILABLE" and context.btc_sentiment_score is not None:
        st.success("Sentimiento BTC actual disponible · vigencia de noticias: 60 minutos")
        st.metric("Sentimiento BTC", f"{context.btc_sentiment_score:+.4f}")
        st.caption(f"Media simple de {context.scored_news_count} artículos con score válido; sin pesos de relevancia.")
    else:
        st.info("Sentimiento BTC actual no disponible. " + REASONS.get(context.reason, "No hay datos actuales utilizables."))
        st.caption("No disponible no equivale a neutral: no se asigna cero a datos ausentes.")
    if context.fetched_at:
        st.caption("Snapshot obtenido: " + context.fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    st.caption(f"{len(context.recent_articles)} noticias recientes; {len(context.historical_articles)} históricas. "
               "Muestra de hasta tres artículos; información contextual, no confirmación predictiva.")
    for label, articles, historical in (
        ("Noticias recientes · hasta 60 minutos", context.recent_articles, False),
        ("Contexto histórico · de 60 minutos a 24 horas", context.historical_articles, True),
    ):
        if not articles:
            continue
        st.subheader(label)
        if historical:
            st.caption("Estas noticias no se incluyen en el sentimiento actual.")
        for article in articles:
            # Provider text is plain text, never unsafe HTML/Markdown.
            st.text(article.title)
            st.caption(f"Fuente: {article.source or 'no informada'} · "
                       + article.published_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
            url = public_article_url(article.public_url)
            if url:
                st.link_button("Leer en la fuente original", url)
    st.caption("Metadatos y sentimiento: Marketaux. Los artículos pertenecen a sus fuentes originales.")

