"""Ejecutar: python -m alpha_engine.ml.validate_prescription."""

import json
import os
import tempfile
from datetime import datetime, timedelta

from alpha_engine.config.settings import PROJECT_ROOT, get_settings
from alpha_engine.ml.prescriptive_check import evaluate_prescriptive_rules
from alpha_engine.repositories.factory import create_market_repository
from alpha_engine.services.prediction import load_model_bundle


def main() -> int:
    settings = get_settings()
    model_path = settings.model_path if settings.model_path.is_absolute() else PROJECT_ROOT / settings.model_path
    bundle = load_model_bundle(model_path)
    metadata_path = model_path.with_name(model_path.stem + "_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(metadata["data_range"]["start"])
    stop = datetime.fromisoformat(metadata["data_range"]["end"]) + timedelta(minutes=1)
    repository = create_market_repository(settings)
    try:
        candles = repository.get_candles(settings.symbol, settings.interval, start, stop)
    finally:
        repository.close()
    result = evaluate_prescriptive_rules(candles, bundle, metadata, settings)
    destination = model_path.with_name("prescriptive_check_v1.json")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", dir=destination.parent,
        prefix=destination.stem, delete=False,
    ) as handle:
        temporary = handle.name
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
