"""Configura la instancia local sin imprimir ni incluir tokens en el código."""

import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "runtime" / "influxdb3" / "bin" / "influxdb3.exe"
SECRET_FILE = ROOT / "runtime" / "influxdb3" / "operator-token.json"
ENV_FILE = ROOT / ".env"


def set_env(text: str, name: str, value: str) -> str:
    line = f"{name}={value}"
    pattern = re.compile(rf"^{re.escape(name)}=.*$", re.MULTILINE)
    return pattern.sub(lambda _: line, text) if pattern.search(text) else text.rstrip() + "\n" + line + "\n"


def main() -> None:
    if not EXE.exists():
        raise RuntimeError("Instala primero influxdb3.exe")
    if SECRET_FILE.exists():
        token = json.loads(SECRET_FILE.read_text(encoding="utf-8"))["token"]
    else:
        result = subprocess.run(
            [str(EXE), "create", "token", "--admin", "--format", "json"],
            capture_output=True, text=True, check=True,
        )
        # Guardar inmediatamente: el token de operador sólo aparece una vez.
        SECRET_FILE.write_text(result.stdout, encoding="utf-8")
        token = json.loads(result.stdout)["token"]
    if not token.startswith("apiv3_"):
        raise RuntimeError("Formato inesperado de token InfluxDB 3")
    environment = os.environ.copy()
    environment["INFLUXDB3_AUTH_TOKEN"] = token
    result = subprocess.run(
        [str(EXE), "create", "database", "alpha_engine"],
        capture_output=True, text=True, env=environment,
    )
    if result.returncode and "already exists" not in result.stderr.lower():
        raise RuntimeError(f"No se pudo crear alpha_engine: {result.stderr}")
    contents = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else (ROOT / ".env.example").read_text(encoding="utf-8")
    for key, value in {
        "STORAGE_PROVIDER": "influxdb3",
        "INFLUX3_URL": "http://127.0.0.1:8181",
        "INFLUX3_DATABASE": "alpha_engine",
        "INFLUX3_TOKEN": token,
        "OLLAMA_MODEL": "qwen3:4b",
    }.items():
        contents = set_env(contents, key, value)
    ENV_FILE.write_text(contents, encoding="utf-8")
    print("Base de datos alpha_engine creada; token guardado en archivos ignorados por Git.")


if __name__ == "__main__":
    main()
