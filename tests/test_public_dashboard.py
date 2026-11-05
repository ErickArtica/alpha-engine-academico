"""Public frontend DTOs must be read-only and omit private prescriptive fields."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from alpha_engine.config.settings import Settings
from alpha_engine.data.live_schemas import LiveEvaluationRecord, PipelineRunRecord
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord
from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.services.public_dashboard import (
    DashboardUnavailableError, PublicDashboardService, PublicFallbackDashboardService,
    read_latest_public_prediction,
)
from streamlit_app.ui import format_lima


OPEN = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
CLOSE = OPEN + timedelta(minutes=1) - timedelta(milliseconds=1)


def candles(count=40):
    return [Candle(
        symbol="BTCUSDT", interval="1m", open_time=OPEN + timedelta(minutes=index),
        close_time=OPEN + timedelta(minutes=index + 1) - timedelta(milliseconds=1),
        open=100 + index, high=101 + index, low=99 + index, close=100.5 + index, volume=10 + index,
    ) for index in range(count)]


def prediction():
    source = OPEN + timedelta(minutes=39)
    return PredictionRecord(
        prediction_id="prediction-abcdefgh", symbol="BTCUSDT", timestamp=source, horizon="1m",
        prob_up=0.7, prob_down=0.3, predicted_class="UP", model_version="xgboost_v1",
        generated_at=source + timedelta(minutes=1, seconds=5),
        valid_until=source + timedelta(minutes=2) - timedelta(milliseconds=1),
    )


class Market:
    def __init__(self, values):
        self.values = values
        self.write_calls = 0

    def get_latest_candle(self, symbol, interval):
        return self.values[-1] if self.values else None

    def get_candles(self, symbol, interval, start, stop):
        return [item for item in self.values if start <= item.open_time < stop]


class Predictions:
    def __init__(self, value):
        self.value = value
        self.write_calls = 0
        self.closed = False

    def close(self):
        self.closed = True

    def get_latest(self, symbol):
        return self.value

    def get_by_id(self, prediction_id):
        return self.value if self.value and prediction_id == self.value.prediction_id else None

    def list_recent(self, symbol, limit=100, offset=0):
        return [self.value][offset:offset + limit] if self.value else []


class Runs:
    def __init__(self, value):
        self.value = value
        self.write_calls = 0

    def get_latest(self, symbol, status=None):
        if self.value is None:
            return None
        return self.value if status is None or status == self.value.status else None

    def list_recent(self, symbol, limit=20, status=None):
        if self.value and (status is None or status == self.value.status):
            return [self.value]
        return []


class Evaluations:
    def __init__(self, value):
        self.value = value
        self.write_calls = 0

    def get_latest(self, symbol):
        return self.value

    def get_by_prediction(self, prediction_id):
        return self.value if self.value and self.value.prediction_id == prediction_id else None

    def list_recent(self, symbol, limit=100, offset=0):
        return [self.value][offset:offset + limit] if self.value else []


class Requests:
    def __init__(self, values):
        self.values = values
        self.write_calls = 0

    def list_recent(self, symbol, limit=20, status=None):
        return [item for item in self.values if status is None or item.status == status][:limit]


class Recommendations:
    def __init__(self, values):
        self.values = values
        self.write_calls = 0

    def list_recent(self, limit=20):
        return self.values[:limit]


def service(values=None):
    market_values = candles() if values is None else values
    pred = prediction() if market_values else None
    run = PipelineRunRecord(
        run_id=UUID("00000000-0000-0000-0000-000000000001"), started_at=OPEN,
        finished_at=OPEN + timedelta(seconds=1), duration_ms=1000, status="SUCCESS",
        symbol="BTCUSDT", interval="1m", candle_open_time=OPEN, prediction_id=pred.prediction_id if pred else None,
        new_candle=True, new_prediction=True, candles_ingested=1,
    )
    evaluation = LiveEvaluationRecord(
        evaluation_id=UUID("00000000-0000-0000-0000-000000000002"), prediction_id=pred.prediction_id,
        symbol="BTCUSDT", evaluated_at=OPEN + timedelta(minutes=41), source_close=139.5,
        target_open_time=OPEN + timedelta(minutes=40),
        target_close_time=OPEN + timedelta(minutes=41) - timedelta(milliseconds=1),
        target_close=141, observed_move="UP", observed_class="UP", correct=True,
    ) if pred else None
    request = RecommendationRequestRecord(
        request_id=UUID("00000000-0000-0000-0000-000000000003"), prediction_id=pred.prediction_id,
        symbol="BTCUSDT", interval="1m", provider="ollama", llm_model="qwen",
        prompt_version="recommendation_v3", requested_at=OPEN + timedelta(minutes=40, seconds=5),
        generated_at=OPEN + timedelta(minutes=40, seconds=6),
        finished_at=OPEN + timedelta(minutes=40, seconds=6),
        valid_until=OPEN + timedelta(minutes=41) - timedelta(milliseconds=1),
        llm_latency_ms=500, total_latency_ms=600, status="SUCCESS", timing_status="TIMELY",
    ) if pred else None
    recommendation = RecommendationRecord(
        prediction_id=pred.prediction_id, prompt_version="recommendation_v3", llm_model="qwen",
        timestamp=pred.timestamp, requested_at=request.requested_at, generated_at=request.generated_at,
        valid_until=request.valid_until, llm_latency_ms=500, total_latency_ms=600,
        candidate_signal="BUY", llm_signal="BUY", final_signal="BUY", risk_level="LOW",
        context_json='{"private":"never public"}', response_json='{"raw":"never public"}', guardrails_json="[]",
    ) if pred else None
    dependencies = (Market(market_values), Predictions(pred), Runs(run), Evaluations(evaluation),
                    Requests([request] if request else []))
    return PublicDashboardService(Settings(_env_file=None, storage_provider="influxdb3"), *dependencies), dependencies


def test_dashboard_reads_persisted_services_without_writing_or_private_fields():
    dashboard, dependencies = service()
    data = dashboard.dashboard(now=OPEN + timedelta(minutes=40, seconds=10))
    assert data.mode == "LIVE" and data.system_state == "CURRENT"
    assert data.latest_prediction.current_validity == "ACTIVE"
    assert data.live_metrics.accuracy == 1
    def keys(value):
        if isinstance(value, dict):
            yield from value
            for nested in value.values():
                yield from keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from keys(nested)

    public_keys = set(keys(data.model_dump(mode="json")))
    for forbidden in ("context_json", "response_json", "raw_response", "prompt", "error_message", "service_role"):
        assert forbidden not in public_keys
    assert all(item.write_calls == 0 for item in dependencies)


def test_descriptive_dashboard_reads_only_market_data(monkeypatch):
    dashboard, dependencies = service()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("This page must not query unrelated repositories")

    for repository in dependencies[1:]:
        for method in ("get_latest", "list_recent"):
            if hasattr(repository, method):
                monkeypatch.setattr(repository, method, forbidden)

    data = dashboard.dashboard(now=OPEN + timedelta(minutes=40), section="descriptive")

    assert data.candles and data.indicators is not None
    assert data.predictions == [] and data.live_metrics is None


def test_home_and_prescriptive_skip_live_metrics_and_unneeded_history(monkeypatch):
    dashboard, dependencies = service()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("LIVE metric history is not needed on this page")

    monkeypatch.setattr(dependencies[3], "list_recent", forbidden)
    home = dashboard.dashboard(now=OPEN + timedelta(minutes=40), section="home")
    assert home.latest_run is not None and home.latest_prediction is not None
    assert home.live_metrics is None and home.recommendation_observability is None

    monkeypatch.setattr(dependencies[2], "list_recent", forbidden)
    monkeypatch.setattr(dependencies[1], "get_latest", forbidden)
    prescriptive = dashboard.dashboard(now=OPEN + timedelta(minutes=40), section="prescriptive")
    assert prescriptive.system_state == "CURRENT"
    assert prescriptive.recommendation_observability is not None
    assert prescriptive.latest_prediction is None  # Selected separately at request time.


def test_prescriptive_availability_uses_current_time_not_cached_validity():
    dashboard, _ = service()
    data = dashboard.dashboard(now=OPEN + timedelta(minutes=40, seconds=10))
    assert data.latest_prediction.current_validity == "ACTIVE"
    deadline = data.latest_prediction.valid_until

    assert data.latest_prediction.validity_at(deadline) == "EXPIRED"
    assert data.recommendation_available_at(deadline - timedelta(seconds=30), 1.0)
    assert not data.recommendation_available_at(deadline - timedelta(milliseconds=500), 1.0)
    assert not data.recommendation_available_at(deadline, 1.0)
    assert not data.recommendation_available_at(deadline + timedelta(seconds=28), 1.0)
    assert not data.model_copy(update={"system_state": "STALE"}).recommendation_available_at(
        deadline - timedelta(seconds=30), 1.0,
    )


def test_public_prediction_reader_fetches_new_value_without_reloading_dashboard(monkeypatch):
    dashboard, dependencies = service()
    old = dashboard.dashboard(now=OPEN + timedelta(minutes=40, seconds=10))
    newer = prediction().model_copy(update={"prediction_id": "new-prediction"})
    dependencies[1].value = newer
    monkeypatch.setattr(
        "alpha_engine.services.public_dashboard.create_prediction_repository",
        lambda _settings: dependencies[1],
    )

    selected = read_latest_public_prediction(
        Settings(_env_file=None, storage_provider="supabase"),
        now=OPEN + timedelta(minutes=40, seconds=20),
    )

    assert old.latest_prediction.prediction_id != selected.prediction_id
    assert selected.prediction_id == newer.prediction_id
    assert selected.current_validity == "ACTIVE"
    assert dependencies[1].closed
    assert all(item.write_calls == 0 for item in dependencies)


def test_dashboard_marks_stale_and_empty_without_inventing_live_data():
    dashboard, _ = service()
    stale = dashboard.dashboard(now=OPEN + timedelta(minutes=45))
    assert stale.system_state == "STALE"
    empty, _ = service(values=[])
    no_data = empty.dashboard(now=OPEN)
    assert no_data.system_state == "NO_DATA"
    assert no_data.latest_candle is no_data.latest_prediction is None


def test_dashboard_converts_storage_failure_to_a_controlled_public_error():
    dashboard, dependencies = service()

    def broken_latest(symbol, interval):
        raise RuntimeError("connection detail that must not reach the frontend")

    dependencies[0].get_latest_candle = broken_latest
    try:
        dashboard.dashboard(now=OPEN)
    except DashboardUnavailableError as exc:
        assert str(exc) == "No se pudo consultar el estado persistido"
    else:
        raise AssertionError("La falla de storage debía propagarse como error controlado")


def test_fallback_is_explicit_and_snapshot_timestamp_is_preserved(tmp_path):
    snapshot = {
        "exported_at": "2026-01-01T12:05:00Z", "symbol": "BTCUSDT", "interval": "1m",
        "candles": [{"open_time": "2026-01-01T12:00:00Z", "open": 100, "high": 101,
                     "low": 99, "close": 100, "volume": 1}],
        "latest_prediction": None, "model": {},
    }
    path = Path(tmp_path) / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    data = PublicFallbackDashboardService(path).dashboard(now=OPEN + timedelta(minutes=6))
    assert data.mode == "FALLBACK" and data.snapshot_at == OPEN + timedelta(minutes=5)
    assert data.system_state == "STALE" and data.latest_candle.close_time == CLOSE


def test_lima_format_changes_only_the_presentation_timezone():
    assert format_lima(OPEN) == "2026-01-01 07:00:00 PET"
    assert OPEN.tzinfo == timezone.utc


def test_read_pages_do_not_import_pipeline_or_market_side_effects():
    root = Path(__file__).resolve().parents[1] / "streamlit_app"
    for relative in ("Home.py", "pages/1_Descriptivo.py", "pages/2_Predictivo.py", "pages/4_Evaluacion.py"):
        source = (root / relative).read_text(encoding="utf-8")
        for forbidden in ("create_market_repository", "PredictionService", "BinanceBookTickerClient", "RecommendationUseCase"):
            assert forbidden not in source
    prescriptive = (root / "pages/3_Prescriptivo.py").read_text(encoding="utf-8")
    assert "RecommendationUseCase" in prescriptive
    assert 'if st.button("Generar recomendación"' in prescriptive
