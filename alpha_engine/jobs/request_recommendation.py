"""Request exactly one audited, on-demand recommendation and exit."""

import argparse
import json
import sys

from alpha_engine.services.errors import StorageError
from alpha_engine.services.recommendation_use_case import RecommendationUseCase
from alpha_engine.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Solicita una recomendacion academica bajo demanda")
    parser.add_argument("--prediction-id", help="Prediccion concreta; si se omite resuelve la ultima activa")
    args = parser.parse_args()
    configure_logging()
    try:
        with RecommendationUseCase.from_settings() as use_case:
            result = use_case.request(args.prediction_id)
    except StorageError:
        print(json.dumps({"completed": False, "error_type": "StorageError"}), file=sys.stderr)
        return 2
    except Exception:
        print(json.dumps({"completed": False, "error_type": "RecommendationRequestError"}), file=sys.stderr)
        return 3
    print(json.dumps({
        "completed": True, "request_id": str(result.request_id),
        "prediction_id": result.prediction_id, "status": result.status,
        "requested_at": result.requested_at.isoformat(),
        "generated_at": result.generated_at.isoformat() if result.generated_at else None,
        "valid_until": result.valid_until.isoformat() if result.valid_until else None,
        "timing_status": result.timing_status, "current_validity": result.current_validity(result.finished_at),
        "llm_latency_ms": result.llm_latency_ms, "total_latency_ms": result.total_latency_ms,
        "cached": result.cached, "actionable": result.actionable,
        "final_signal": result.recommendation.final_signal if result.recommendation else None,
        "guardrail_count": len(result.guardrails),
    }, separators=(",", ":")))
    return 0 if result.status in ("SUCCESS", "CACHED", "GUARDRAIL_REJECTED", "LATE_RESULT",
                                  "EXPIRED_BEFORE_REQUEST", "DEADLINE_TOO_CLOSE", "NO_ACTIVE_PREDICTION",
                                  "PREDICTION_NOT_FOUND", "PREDICTION_INVALID") else 2


if __name__ == "__main__":
    raise SystemExit(main())
