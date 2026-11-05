"""Contratos SQL del adaptador InfluxDB 3 sin requerir servicio externo."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.data.live_schemas import LiveEvaluationRecord, PipelineRunRecord
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord
from alpha_engine.repositories.influx3 import (
    Influx3MarketRepository, Influx3PredictionRepository,
    Influx3RecommendationRepository, Influx3PipelineRunRepository,
    Influx3LiveEvaluationRepository,
    Influx3RecommendationRequestRepository,
    _ms,
)


class FakeConnection:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.written = ""
        self.sql = ""
        self.params: dict = {}

    def write(self, line_protocol: str) -> None:
        self.written = line_protocol

    def query(self, sql: str, params: dict) -> list[dict]:
        self.sql = sql
        self.params = params
        return self.rows

    def close(self) -> None:
        pass


def test_market_roundtrip_uses_sql_parameters_and_utc() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candle = Candle(
        symbol="BTCUSDT", interval="1m", open_time=start,
        open=100.0, high=102.0, low=99.0, close=101.0, volume=5.0,
        close_time=start + timedelta(milliseconds=59_999),
    )
    row = {"symbol": "BTCUSDT", "interval": "1m", "time": "2026-01-01T00:00:00",
           "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
           "volume": 5.0, "close_time_ms": int(candle.close_time.timestamp() * 1000)}
    connection = FakeConnection([row])
    repository = Influx3MarketRepository(connection)
    assert repository.upsert_candles([candle]) == 1
    assert "candles,symbol=BTCUSDT,interval=1m" in connection.written
    assert repository.get_candles("BTCUSDT", "1m", start, start + timedelta(minutes=1)) == [candle]
    assert '"interval" = $interval' in connection.sql
    assert connection.params["symbol"] == "BTCUSDT"
    assert connection.params["start"] == start.isoformat()


def test_recommendation_history_parses_persisted_fields() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    record = RecommendationRecord(
        prediction_id="id", prompt_version="recommendation_v1", llm_model="qwen3:4b",
        timestamp=start, generated_at=start + timedelta(seconds=1),
        candidate_signal="HOLD", llm_signal=None, final_signal="HOLD", risk_level="HIGH",
        context_json="{}", response_json=None, guardrails_json='["LOW_MODEL_CONFIDENCE"]',
    )
    row = {**record.model_dump(mode="json"), "time": "2026-01-01T00:00:00Z",
           "llm_signal": "", "response_json": ""}
    connection = FakeConnection([row])
    repository = Influx3RecommendationRepository(connection)
    assert repository.list_recent(1) == [record]
    assert "ORDER BY time DESC LIMIT 1" in connection.sql


def test_recommendation_versions_are_appended_and_merged_with_cache() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = RecommendationRecord(
        prediction_id="id", prompt_version="v1", llm_model="llm",
        timestamp=start, generated_at=start + timedelta(seconds=1),
        candidate_signal="BUY", llm_signal="BUY", final_signal="BUY",
        risk_level="LOW", context_json="{}", guardrails_json="[]",
    )
    newer = older.model_copy(update={
        "generated_at": start + timedelta(seconds=2), "final_signal": "HOLD",
        "guardrails_json": '["PREDICTION_EXPIRED"]',
    })

    class VersionConnection(FakeConnection):
        def query(self, sql: str, params: dict) -> list[dict]:
            if "FROM recommendation_history" in sql:
                return [
                    {**record.model_dump(mode="json"),
                     "candle_open_at": record.timestamp.isoformat(),
                     "time": record.generated_at.isoformat(), "revision_id": f"v{i}",
                     "response_json": ""}
                    for i, record in enumerate((newer, older))
                ]
            return [{**newer.model_dump(mode="json"), "time": newer.timestamp.isoformat(), "response_json": ""}]

    connection = VersionConnection([])
    repository = Influx3RecommendationRepository(connection)
    repository.save(older)
    assert "recommendations," in connection.written
    assert "\nrecommendation_history," in connection.written
    assert "candle_open_at=" in connection.written
    assert repository.list_recent(10) == [newer, older]


def test_influx3_prediction_timing_fields_and_legacy_record():
    opened = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh = PredictionRecord(
        prediction_id="p1", symbol="BTCUSDT", timestamp=opened,
        horizon="1m", prob_up=0.7, prob_down=0.3, predicted_class="UP",
        model_version="xgboost_v1", generated_at=opened + timedelta(minutes=1),
        valid_until=opened + timedelta(minutes=2),
    )
    connection = FakeConnection([])
    repo = Influx3PredictionRepository(connection)
    repo.save(fresh)
    assert "generated_at=" in connection.written and "valid_until=" in connection.written
    row = {**fresh.model_dump(mode="json"), "time": opened.isoformat()}
    assert repo._record(row) == fresh
    row.pop("generated_at")
    row.pop("valid_until")
    legacy = repo._record(row)
    assert legacy.timestamp == opened and legacy.timing_status is None


def test_influx3_pipeline_run_and_evaluation_use_distinct_measurements():
    opened = datetime(2026, 1, 1, tzinfo=timezone.utc)
    connection = FakeConnection([])
    runs = Influx3PipelineRunRepository(connection)
    run = PipelineRunRecord(
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        started_at=opened, finished_at=opened + timedelta(seconds=1),
        duration_ms=1000, status="FAILED", symbol="BTCUSDT", interval="1m",
        error_type="MarketDataError", error_message="Market data request failed",
    )
    runs.save(run)
    assert connection.written.startswith("pipeline_runs,")
    assert "error_message=" in connection.written
    row = {**run.model_dump(mode="json"), "time": opened.isoformat()}
    assert runs._record(row) == run
    assert runs.get_latest("BTCUSDT") is None
    observed = LiveEvaluationRecord(
        evaluation_id=UUID("00000000-0000-0000-0000-000000000002"),
        prediction_id="p1", symbol="BTCUSDT",
        evaluated_at=opened + timedelta(minutes=2), source_close=100,
        target_open_time=opened + timedelta(minutes=1),
        target_close_time=opened + timedelta(minutes=2) - timedelta(milliseconds=1),
        target_close=101, observed_move="UP", observed_class="UP", correct=True,
    )
    evaluations = Influx3LiveEvaluationRepository(connection)
    assert evaluations.save(observed) is True
    assert connection.written.startswith("live_evaluations,")
    assert f" {_ms(observed.target_close_time)}" in connection.written
    row = {**observed.model_dump(mode="json"), "time": observed.target_close_time.isoformat()}
    assert evaluations._record(row) == observed


def test_influx3_recommendation_request_round_trip_uses_utc_and_audit_measurement():
    opened = datetime(2026, 1, 1, tzinfo=timezone.utc)
    request = RecommendationRequestRecord(
        request_id=UUID("00000000-0000-0000-0000-000000000010"), prediction_id="p1",
        symbol="BTCUSDT", interval="1m", provider="ollama", llm_model="qwen",
        prompt_version="recommendation_v3", requested_at=opened,
        generated_at=opened + timedelta(seconds=1), finished_at=opened + timedelta(seconds=1),
        valid_until=opened + timedelta(minutes=1), llm_latency_ms=900,
        total_latency_ms=1000, status="SUCCESS", timing_status="TIMELY",
    )
    connection = FakeConnection([])
    repository = Influx3RecommendationRequestRepository(connection)
    repository.save(request)
    assert connection.written.startswith("recommendation_requests,")
    assert f" {_ms(opened)}" in connection.written
    row = {**request.model_dump(mode="json"), "time": opened.isoformat()}
    assert repository._record(row) == request
