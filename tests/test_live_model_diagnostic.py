"""Checks for the read-only LIVE diagnostic's label and probability arithmetic."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
import pytest

from scripts import live_model_diagnostic as diagnostic


SOURCE = datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc)
CUTOFF = SOURCE + timedelta(hours=1)


def sample(index: int, probability_up: float, observed: str, *, correct: bool | None = None):
    source = SOURCE + timedelta(minutes=index)
    target_open = source + timedelta(minutes=1)
    source_close = 100.0 + index
    target_close = source_close + (1 if observed == "UP" else -1 if observed == "DOWN" else 0)
    predicted = "UP" if probability_up > 0.5 else "DOWN"
    observed_class = "UP" if observed == "UP" else "DOWN"
    return {
        "evaluation_id": str(UUID(int=index + 1)),
        "prediction_id": f"prediction-{index}",
        "symbol": "BTCUSDT",
        "evaluated_at": (target_open + timedelta(minutes=1, seconds=3)).isoformat(),
        "source_close": source_close,
        "target_open_time": target_open.isoformat(),
        "target_close_time": (target_open + timedelta(minutes=1, milliseconds=-1)).isoformat(),
        "target_close": target_close,
        "observed_move": observed,
        "observed_class": observed_class,
        "correct": predicted == observed_class if correct is None else correct,
        "predictions": {
            "prediction_id": f"prediction-{index}",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "candle_open_time": source.isoformat(),
            "horizon": "1m",
            "prob_up": probability_up,
            "prob_down": 1 - probability_up,
            "predicted_class": predicted,
            "model_version": "xgboost_v1",
            "trained_at": None,
            "generated_at": (source + timedelta(minutes=1, seconds=2)).isoformat(),
            "valid_until": (source + timedelta(minutes=2, milliseconds=-1)).isoformat(),
        },
    }


def test_confusion_per_class_thresholds_and_scores():
    rows = [
        sample(0, 0.4, "DOWN"),
        sample(1, 0.7, "UP"),
        sample(2, 0.45, "UP"),
        sample(3, 0.8, "DOWN"),
    ]
    report = diagnostic.analyze_rows(rows, [], cutoff=CUTOFF)
    assert report["evaluated"] == 4
    assert report["correct"] == 2
    assert report["confusion_matrix_predicted_rows_actual_columns"] == {
        "DOWN": {"DOWN": 1, "UP": 1}, "UP": {"DOWN": 1, "UP": 1},
    }
    assert report["per_class"]["UP"]["precision"] == 0.5
    assert report["per_class"]["DOWN"]["recall"] == 0.5
    assert report["macro_f1"] == 0.5
    assert report["balanced_accuracy"] == 0.5
    assert report["brier_up"] == pytest.approx((0.4**2 + 0.3**2 + 0.55**2 + 0.8**2) / 4)
    assert report["log_loss_up"] > 0
    assert next(item for item in report["thresholds"] if item["threshold"] == 0.6)["count"] == 3
    assert report["confidence_buckets"][5]["count"] == 1  # 60% enters [60,65).
    assert sum(item["count"] for item in report["confidence_buckets"]) == 4


def test_flat_is_down_and_previous_movement_uses_only_prior_closed_candle():
    row = sample(0, 0.4, "FLAT")
    previous = SOURCE - timedelta(minutes=1)
    candles = [
        {"open_time": previous.isoformat(), "close": 99.0},
        {"open_time": SOURCE.isoformat(), "close": 100.0},
    ]
    report = diagnostic.analyze_rows([row], candles, cutoff=CUTOFF)
    assert report["observed_move"] == {"FLAT": 1}
    assert report["observed_classes"]["DOWN"]["count"] == 1
    assert report["baselines"]["previous_movement_same"] == {
        "count": 1, "correct": 0, "accuracy": 0.0,
        "model_accuracy_same_rows": 1.0, "missing_previous_candle": 0,
    }


def test_inconsistent_persisted_correctness_is_not_silently_recomputed():
    with pytest.raises(ValueError, match="Stored correctness"):
        diagnostic.analyze_rows([sample(0, 0.7, "DOWN", correct=True)], [], cutoff=CUTOFF)


def test_http_pagination_checks_exact_count(monkeypatch):
    monkeypatch.setattr(diagnostic, "PAGE_SIZE", 2)

    def reply(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        values = [{"id": n} for n in range(3)][offset:offset + 2]
        return httpx.Response(206, json=values, headers={"Content-Range": f"{offset}-{offset + len(values) - 1}/3"})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        rows, count = diagnostic._fetch_all(client, "https://example.supabase.co", "public-key", "dummy", [])
    assert count == 3
    assert rows == [{"id": 0}, {"id": 1}, {"id": 2}]


def test_diagnostic_refuses_privileged_key():
    diagnostic._require_public_read_key("sb_publishable_example")
    with pytest.raises(ValueError, match="server/secret"):
        diagnostic._require_public_read_key("sb_secret_example")
