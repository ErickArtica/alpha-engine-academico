"""Renderiza las páginas sin navegador, usando sólo datos locales."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "streamlit_app" / "Home.py", *sorted((ROOT / "streamlit_app" / "pages").glob("[0-9]_*.py"))]


def main() -> None:
    for path in PAGES:
        result = AppTest.from_file(path, default_timeout=30).run()
        if result.exception:
            raise RuntimeError(f"Falló {path.name}: {result.exception[0].message}")
        if result.error:
            raise RuntimeError(f"La página {path.name} mostró un error: {result.error[0].value}")
        print(f"OK {path.name}")


if __name__ == "__main__":
    main()
