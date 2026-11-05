"""Empaqueta únicamente la demo pública revisada, sin archivos locales."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from alpha_engine.config.settings import PROJECT_ROOT


SOURCE = PROJECT_ROOT / "public_demo"
DESTINATION = PROJECT_ROOT / "runtime" / "public_demo_release.zip"
FILES = ("app.py", "requirements.txt", "README.md", "data/snapshot.json")
PRIVATE_KEYS = {"context_json", "response_json", "api_key", "token", "secret"}


def _check_public_json(value: object) -> None:
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(key.lower() for key in value):
            raise ValueError("El corte contiene una clave privada o contexto LLM")
        for child in value.values():
            _check_public_json(child)
    elif isinstance(value, list):
        for child in value:
            _check_public_json(child)


def package(destination: Path = DESTINATION) -> str:
    snapshot = json.loads((SOURCE / "data" / "snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("snapshot_version") != 1 or not snapshot.get("candles"):
        raise ValueError("Corte público ausente o incompatible")
    _check_public_json(snapshot)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(SOURCE / name, arcname=name)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


if __name__ == "__main__":
    digest = package()
    print(f"Paquete: {DESTINATION} · SHA-256: {digest}")
