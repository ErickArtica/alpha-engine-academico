"""Northflank-only router. The original pages are neither registered nor loaded."""
import streamlit as st

from deploy.northflank.public_config import ConfigurationError
from deploy.northflank import public_views as views


def main() -> None:
    try:
        views.config()
    except (ConfigurationError, ValueError):
        st.error("El perfil público no está configurado correctamente.")
        st.stop()
    st.set_page_config(page_title="ALPHA ENGINE", page_icon="📈", layout="wide")
    pages = [st.Page(views.resolve_page(route), title=title, url_path=route, default=route == "inicio")
             for route, (title, _) in views.ROUTES.items()]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
