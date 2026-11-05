"""La demo pública debe abrir el corte sin servicios locales."""

from pathlib import Path
import pytest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "public_demo" / "app.py"
SNAPSHOT = APP.parent / "data" / "snapshot.json"


def test_public_demo_renders_from_bundled_snapshot():
    if not SNAPSHOT.is_file():
        pytest.skip("Snapshot histórico privado no incluido en Git")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert len(app.tabs) == 4
    assert any("Datos históricos de sólo lectura" in warning.value for warning in app.warning)
