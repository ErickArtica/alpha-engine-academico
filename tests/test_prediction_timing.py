from datetime import datetime, timedelta, timezone

import pytest

from alpha_engine.data.schemas import PredictionRecord
from alpha_engine.data.timing import prediction_expires_at

OPEN = datetime(2026, 1, 1, 10, 4, tzinfo=timezone.utc)
CLOSE = OPEN + timedelta(minutes=1) - timedelta(milliseconds=1)
VALID_UNTIL = CLOSE + timedelta(minutes=1)


def record(generated_at=None, valid_until=None):
    return PredictionRecord(prediction_id="p1", symbol="BTCUSDT", timestamp=OPEN,
                            horizon="1m", prob_up=0.7, prob_down=0.3,
                            predicted_class="UP", model_version="xgboost_v1",
                            generated_at=generated_at, valid_until=valid_until)


def test_timestamp_remains_source_open_and_old_prediction_is_readable():
    old = record()
    assert old.timestamp == old.source_candle_open_time == OPEN
    assert old.timing_status is None and old.current_validity(OPEN) is None


def test_timely_and_late_boundaries_are_historical():
    timely = record(VALID_UNTIL - timedelta(microseconds=1), VALID_UNTIL)
    equality = record(VALID_UNTIL, VALID_UNTIL)
    late = record(VALID_UNTIL + timedelta(microseconds=1), VALID_UNTIL)
    assert timely.timing_status == "TIMELY"
    assert equality.timing_status == late.timing_status == "LATE"
    assert timely.timing_status == "TIMELY"  # Remains timely long after expiry.
    assert timely.current_validity(VALID_UNTIL - timedelta(microseconds=1)) == "ACTIVE"
    assert timely.current_validity(VALID_UNTIL) == "EXPIRED"
    assert timely.current_validity(VALID_UNTIL + timedelta(hours=1)) == "EXPIRED"


def test_valid_until_is_target_candle_close_from_existing_horizon_rule():
    from alpha_engine.ai.guardrails import prediction_expires_at as guardrail_expiry
    assert prediction_expires_at(CLOSE, "1m") == VALID_UNTIL
    assert guardrail_expiry(CLOSE, "1m") == VALID_UNTIL
    with pytest.raises(ValueError, match="zona horaria"):
        record(VALID_UNTIL.replace(tzinfo=None), VALID_UNTIL)
    with pytest.raises(ValueError, match="juntos"):
        record(VALID_UNTIL, None)
