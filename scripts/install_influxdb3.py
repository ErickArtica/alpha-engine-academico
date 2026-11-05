"""Verifica y extrae el artefacto oficial descargado de InfluxDB 3 Core."""

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile


VERSION = "3.11.4"
URL = f"https://dl.influxdata.com/influxdb/releases/influxdb3-core-{VERSION}-windows_amd64.zip"
ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "runtime" / "influxdb3" / "bin"


def main() -> None:
    BIN.mkdir(parents=True, exist_ok=True)
    destination = BIN / Path(URL).name
    checksum = BIN / (destination.name + ".sha256")
    expected = checksum.read_text(encoding="utf-8").split()[0].lower()
    actual = sha256(destination.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("SHA-256 del binario oficial no coincide")
    with ZipFile(destination) as archive:
        for member in archive.infolist():
            target = (BIN / member.filename).resolve()
            if not target.is_relative_to(BIN.resolve()):
                raise RuntimeError("Ruta insegura dentro del ZIP")
        archive.extractall(BIN)
    binaries = list(BIN.rglob("influxdb3.exe"))
    if not binaries:
        raise RuntimeError("El ZIP no contiene influxdb3.exe")
    print(f"SHA-256 verificado: {actual}")
    print(f"Ejecutable: {binaries[0]}")


if __name__ == "__main__":
    main()
