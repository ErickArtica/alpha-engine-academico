"""Una señal de un minuto deja de ser vigente al cerrar la vela objetivo."""

from datetime import datetime, timedelta, timezone

from alpha_engine.data.schemas import Candle, PredictionRecord
from streamlit_app.ui import prediction_is_actionable


def test_prediction_actionability_ends_at_horizon():
    opened = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    candle = Candle(
        symbol="BTCUSDT", interval="1m", open_time=opened,
        close_time=opened + timedelta(milliseconds=59_999),
        open=100, high=101, low=99, close=100, volume=10,
    )
    prediction = PredictionRecord(
        prediction_id="p", symbol="BTCUSDT", timestamp=opened, horizon="1m",
        prob_up=0.7, prob_down=0.3, predicted_class="UP", model_version="v1",
    )
    assert prediction_is_actionable([candle], prediction, candle.close_time + timedelta(seconds=5))
    assert not prediction_is_actionable([candle], prediction, candle.close_time + timedelta(minutes=1))
    assert not prediction_is_actionable([candle], prediction, candle.close_time - timedelta(seconds=1))
    assert not prediction_is_actionable([], prediction, candle.close_time + timedelta(seconds=5))
