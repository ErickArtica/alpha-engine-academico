from datetime import datetime, timedelta, timezone

from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.services.history_audit import audit_history, history_summary


OPEN = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
CLOSE = OPEN + timedelta(seconds=59, milliseconds=999)
EXPIRY = CLOSE + timedelta(minutes=1)


def candle(at, close):
    return Candle(
        symbol="BTCUSDT", interval="1m", open_time=at,
        open=100, high=max(100, close), low=min(100, close), close=close,
        volume=1, close_time=at + timedelta(seconds=59, milliseconds=999),
    )


def prediction(prediction_id="p1", prob_up=0.7):
    return PredictionRecord(
        prediction_id=prediction_id, symbol="BTCUSDT", timestamp=OPEN, horizon="1m",
        prob_up=prob_up, prob_down=1 - prob_up,
        predicted_class="UP" if prob_up > 0.5 else "DOWN", model_version="m1",
    )


def recommendation(prediction_id="p1", seconds=5, signal="BUY", prompt="v1"):
    return RecommendationRecord(
        prediction_id=prediction_id, prompt_version=prompt, llm_model="llm",
        timestamp=OPEN, generated_at=CLOSE + timedelta(seconds=seconds),
        candidate_signal=signal, llm_signal=signal, final_signal=signal,
        risk_level="LOW", context_json="{}", guardrails_json="[]",
    )


def test_observed_candle_and_directional_hit():
    rows = audit_history(
        [recommendation()], {"p1": prediction()},
        [candle(OPEN, 100), candle(OPEN + timedelta(minutes=1), 101)],
        EXPIRY + timedelta(seconds=2),
    )
    row = rows[0]
    assert row["validity"] == "vencida ahora"
    assert row["expires_at"] == EXPIRY.isoformat()
    assert row["latency_seconds"] == 5
    assert row["outcome_status"] == "observado"
    assert row["observed_move"] == "UP"
    assert row["model_target_correct"] is True
    assert row["signal_direction_correct"] is True
    assert history_summary(rows)["directional_hit_rate"] == 1


def test_future_target_is_pending_even_if_present_in_repository():
    rows = audit_history(
        [recommendation()], {"p1": prediction()},
        [candle(OPEN, 100), candle(OPEN + timedelta(minutes=1), 101)],
        CLOSE + timedelta(seconds=20),
    )
    assert rows[0]["outcome_status"] == "pendiente"
    assert rows[0]["target_close"] is None
    assert rows[0]["validity"] == "vigente"


def test_missing_target_and_late_signal_are_not_counted():
    rows = audit_history(
        [recommendation(seconds=70)], {"p1": prediction()},
        [candle(OPEN, 100)], EXPIRY + timedelta(seconds=20),
    )
    assert rows[0]["outcome_status"] == "sin dato"
    assert rows[0]["validity"] == "emitida tarde"
    assert rows[0]["counted_for_metric"] is False
    assert history_summary(rows)["directional_hit_rate"] is None


def test_only_first_timely_directional_version_counts_per_candle():
    records = [
        recommendation(seconds=2, signal="HOLD"),
        recommendation(seconds=5, signal="BUY", prompt="v2"),
        recommendation(prediction_id="p2", seconds=8, signal="SELL"),
    ]
    rows = audit_history(
        records, {"p1": prediction(), "p2": prediction("p2", prob_up=0.3)},
        [candle(OPEN, 100), candle(OPEN + timedelta(minutes=1), 99)],
        EXPIRY + timedelta(seconds=2),
    )
    selected = [row for row in rows if row["counted_for_metric"]]
    assert len(selected) == 1
    assert selected[0]["final_signal"] == "BUY"
    assert selected[0]["signal_direction_correct"] is False
    assert all(row["versions_in_candle"] == 3 for row in rows)
    assert history_summary(rows)["unique_timely_signals"] == 1


def test_flat_target_is_not_a_correct_sell():
    rows = audit_history(
        [recommendation(signal="SELL")], {"p1": prediction(prob_up=0.3)},
        [candle(OPEN, 100), candle(OPEN + timedelta(minutes=1), 100)],
        EXPIRY + timedelta(seconds=2),
    )
    assert rows[0]["observed_move"] == "FLAT"
    assert rows[0]["target_class"] == "DOWN"
    assert rows[0]["model_target_correct"] is True
    assert rows[0]["signal_direction_correct"] is False
