"""Run one provider-neutral LIVE cycle and exit."""

import json
import sys

from alpha_engine.services.errors import MarketDataError, StorageError
from alpha_engine.services.live_cycle import run_live_cycle
from alpha_engine.services.prediction import ModelNotTrainedError, PredictionUnavailableError
from alpha_engine.utils.logging import configure_logging


def main() -> int:
    configure_logging()
    try:
        result = run_live_cycle()
    except (MarketDataError, StorageError, ModelNotTrainedError,
            PredictionUnavailableError, ValueError) as exc:
        print(json.dumps({"completed": False, "error_type": type(exc).__name__}), file=sys.stderr)
        return 2
    except Exception:
        print(json.dumps({"completed": False, "error_type": "UnexpectedError"}), file=sys.stderr)
        return 3
    print(json.dumps({
        "completed": result.completed,
        "status": result.status,
        "run_id": result.run_id,
        "duration_ms": result.duration_ms,
        "as_of": result.as_of.isoformat(),
        "candles_ingested": result.candles_ingested,
        "new_candle": result.new_candle,
        "candle_open_time": result.candle.open_time.isoformat() if result.candle else None,
        "prediction_id": result.prediction_id,
        "prediction_created": result.prediction_created,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
