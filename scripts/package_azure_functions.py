"""Build an ignored publish directory with the exact worker files."""

from pathlib import Path
import shutil
import hashlib


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "runtime" / "azure_functions_package"
MODEL = ROOT / "models" / "xgboost_v1.joblib"
METADATA = ROOT / "models" / "xgboost_v1_metadata.json"
CHECKSUM = ROOT / "models" / "xgboost_v1.sha256"


def package() -> Path:
    runtime_root = (ROOT / "runtime").resolve()
    if not TARGET.resolve().is_relative_to(runtime_root):
        raise ValueError("Package target escapes runtime directory")
    if not MODEL.is_file() or not METADATA.is_file() or not CHECKSUM.is_file():
        raise FileNotFoundError("Model artifact or metadata is missing")
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != CHECKSUM.read_text().strip():
        raise ValueError("Model artifact checksum mismatch")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    shutil.copytree(ROOT / "alpha_engine", TARGET / "alpha_engine",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (TARGET / "models").mkdir()
    shutil.copy2(MODEL, TARGET / "models" / MODEL.name)
    shutil.copy2(METADATA, TARGET / "models" / METADATA.name)
    shutil.copy2(CHECKSUM, TARGET / "models" / CHECKSUM.name)
    for filename in ("function_app.py", "host.json"):
        shutil.copy2(ROOT / "deploy" / "azure_functions" / filename, TARGET / filename)
    shutil.copy2(ROOT / "requirements-worker.txt", TARGET / "requirements.txt")
    return TARGET


if __name__ == "__main__":
    print(package())
