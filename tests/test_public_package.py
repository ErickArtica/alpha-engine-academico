"""El paquete público se ejecuta aislado del repositorio principal."""

from zipfile import ZipFile
import pytest

from streamlit.testing.v1 import AppTest

from scripts.package_public_demo import FILES, package


def test_public_package_contains_only_standalone_demo(tmp_path):
    from scripts.package_public_demo import SOURCE
    if not (SOURCE / "data" / "snapshot.json").is_file():
        pytest.skip("Snapshot histórico privado no incluido en Git")
    archive_path = tmp_path / "release.zip"
    digest = package(archive_path)
    assert len(digest) == 64
    extracted = tmp_path / "extracted"
    with ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(FILES)
        archive.extractall(extracted)
    app = AppTest.from_file(extracted / "app.py", default_timeout=30).run()
    assert not app.exception
    assert not app.error
