"""Prueba los botones de predicción y recomendación con servicios locales."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


PAGES = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages"


def check_button(filename: str) -> None:
    app = AppTest.from_file(PAGES / filename, default_timeout=60).run()
    if app.exception or app.error:
        raise RuntimeError(f"La página {filename} falló antes del botón")
    app.button[0].click().run(timeout=60)
    if app.exception or app.error:
        detail = app.exception[0].message if app.exception else app.error[0].value
        raise RuntimeError(f"El botón de {filename} falló: {detail}")
    print(f"OK acción {filename}")


def main() -> None:
    check_button("2_Predictivo.py")
    check_button("3_Prescriptivo.py")


if __name__ == "__main__":
    main()
