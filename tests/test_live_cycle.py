from datetime import datetime, timedelta, timezone
from math import sin
from uuid import uuid4

import httpx
import numpy as np
import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle
from alpha_engine.data.live_schemas import PipelineRunRecord
from alpha_engine.features import FEATURE_COLUMNS
from alpha_engine.observability.telemetry import TelemetryStore
from alpha_engine.services.errors import MarketDataError, StorageError
from alpha_engine.services.live_cycle import LiveCycle, run_live_cycle
from alpha_engine.services.prediction import PredictionUnavailableError
from alpha_engine.services.system_status import SystemStatusService

START = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def candle(index):
    opened = START + timedelta(minutes=index)
    price = 100 + index * 0.1 + sin(index / 4)
    return Candle(symbol="BTCUSDT", interval="1m", open_time=opened,
                  close_time=opened + timedelta(minutes=1) - timedelta(milliseconds=1),
                  open=price, high=price + 1, low=price - 1, close=price,
                  volume=10 + index % 3)


class Market:
    def __init__(self, rows=()):
        self.rows = {(r.symbol, r.interval, r.open_time): r for r in rows}
        self.closed = False
        self.fail_upsert = False

    def get_latest_candle(self, symbol, interval):
        return max((r for r in self.rows.values() if r.symbol == symbol and r.interval == interval),
                   key=lambda r: r.open_time, default=None)

    def get_candles(self, symbol, interval, start, stop):
        return sorted((r for r in self.rows.values() if r.symbol == symbol and r.interval == interval
                       and start <= r.open_time < stop), key=lambda r: r.open_time)

    def upsert_candles(self, rows):
        if self.fail_upsert:
            raise RuntimeError("storage unavailable")
        for row in rows:
            self.rows[(row.symbol, row.interval, row.open_time)] = row
        return len(rows)

    def close(self):
        self.closed = True


class Predictions:
    def __init__(self):
        self.rows = {}
        self.save_calls = 0
        self.closed = False
        self.race = False

    def get_by_id(self, prediction_id):
        return self.rows.get(prediction_id)

    def save(self, prediction):
        self.save_calls += 1
        self.rows[prediction.prediction_id] = prediction
        return not self.race

    def list_recent(self, symbol, limit=100, offset=0):
        return sorted((row for row in self.rows.values() if row.symbol == symbol),
                      key=lambda row: row.timestamp, reverse=True)[offset:offset + limit]

    def get_latest(self, symbol):
        rows = self.list_recent(symbol, 1)
        return rows[0] if rows else None

    def close(self):
        self.closed = True


class Source:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.closed = False
        self.fail = False
        self.calls = []

    def fetch_page(self, symbol, interval, start, stop, limit=1000):
        self.calls.append((start, stop))
        if self.fail:
            raise httpx.ConnectError("market unavailable")
        # BinanceKlineClient applies the same closed-candle filter.
        return [row for row in self.rows if row.symbol == symbol and row.interval == interval
                and start <= row.open_time < stop and row.close_time < stop][:limit]

    def close(self):
        self.closed = True


class Runs:
    def __init__(self):
        self.rows = []
        self.closed = False

    def save(self, row):
        self.rows.append(row)

    def get_latest(self, symbol, status=None):
        rows = self.list_recent(symbol, status=status)
        return rows[0] if rows else None

    def list_recent(self, symbol, limit=20, status=None):
        return sorted((row for row in self.rows if row.symbol == symbol
                       and (status is None or row.status == status)),
                      key=lambda row: row.started_at, reverse=True)[:limit]

    def close(self):
        self.closed = True


class Evaluations:
    def __init__(self):
        self.rows = {}
        self.closed = False

    def save(self, row):
        if row.prediction_id in self.rows:
            return False
        self.rows[row.prediction_id] = row
        return True

    def get_by_prediction(self, prediction_id):
        return self.rows.get(prediction_id)

    def get_latest(self, symbol):
        rows = self.list_recent(symbol)
        return rows[0] if rows else None

    def list_recent(self, symbol, limit=100, offset=0):
        return sorted((row for row in self.rows.values() if row.symbol == symbol),
                      key=lambda row: row.evaluated_at, reverse=True)[offset:offset + limit]

    def close(self):
        self.closed = True


class Estimator:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities=(0.3, 0.7)):
        self.probabilities = probabilities
        self.calls = 0

    def predict_proba(self, frame):
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        self.calls += 1
        return np.array([self.probabilities])


def bundle(estimator):
    return {"estimator": estimator, "features": list(FEATURE_COLUMNS),
            "model_version": "xgboost_v1", "trained_at": "2026-01-01T00:00:00+00:00",
            "symbol": "BTCUSDT", "interval": "1m", "horizon": "1m"}


def make_cycle(market=None, predictions=None, source=None, estimator=None, clock=None, loader=None,
               runs=None, evaluations=None, monotonic=None):
    market = market or Market(candle(i) for i in range(119))
    predictions = predictions or Predictions()
    source = source or Source([candle(119)])
    estimator = estimator or Estimator()
    calls = []
    def load(path):
        calls.append(path)
        return bundle(estimator) if loader is None else loader(path)
    kwargs = {"runs": runs, "evaluations": evaluations}
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    cycle = LiveCycle(Settings(_env_file=None, storage_provider="influxdb3"), source,
                      market, predictions, clock=clock or (lambda: START + timedelta(minutes=120, seconds=5)),
                      model_loader=load, **kwargs)
    return cycle, market, predictions, source, estimator, calls


def test_new_closed_candle_generates_one_prediction_and_reuses_model():
    cycle, market, predictions, source, estimator, loads = make_cycle()
    with cycle:
        first = cycle.run()
        second = cycle.run()
    assert first.status == "SUCCESS" and first.completed
    assert first.new_candle and first.prediction_created and first.candles_ingested == 1
    assert first.candle.open_time == START + timedelta(minutes=119)
    assert first.prediction_id == second.prediction_id
    assert second.status == "NO_NEW_CANDLE" and not second.prediction_created
    assert len(market.rows) == 120 and len(predictions.rows) == 1 and predictions.save_calls == 1
    assert estimator.calls == 1 and len(loads) == 1
    assert market.closed and predictions.closed and source.closed


@pytest.mark.parametrize("shadow_fails", [False, True])
def test_v3_shadow_does_not_change_v1_prediction_or_persistence(monkeypatch, shadow_fails):
    from alpha_engine.services import live_cycle as module

    baseline, _, baseline_predictions, _, _, _ = make_cycle()
    with baseline:
        expected = baseline.run(now=START + timedelta(minutes=120, seconds=5))

    calls = []
    class Shadow:
        def __init__(self, source):
            pass

        def predict(self, symbol, interval, opened):
            calls.append((symbol, interval, opened))
            if shadow_fails:
                raise ValueError("shadow unavailable")
            from alpha_engine.services.v3_shadow import ShadowPrediction
            return ShadowPrediction("xgboost_v3_market_flow_12m", opened,
                                    0.4, 0.6, "UP", 7.0, 2.0)

    monkeypatch.setattr(module, "V3Shadow", Shadow)
    market = Market(candle(i) for i in range(119))
    predictions = Predictions()
    source = Source([candle(119)])
    with LiveCycle(Settings(_env_file=None, storage_provider="influxdb3", v3_shadow_enabled=True),
                   source, market, predictions, model_loader=lambda _: bundle(Estimator()),
                   clock=lambda: START + timedelta(minutes=120, seconds=5)) as cycle:
        actual = cycle.run(now=START + timedelta(minutes=120, seconds=5))
    assert actual.status == expected.status == "SUCCESS"
    assert actual.prediction_id == expected.prediction_id
    assert actual.prediction.prob_up == expected.prediction.prob_up
    assert actual.prediction.model_version == "xgboost_v1"
    assert len(predictions.rows) == len(baseline_predictions.rows) == 1
    assert predictions.save_calls == baseline_predictions.save_calls == 1
    assert calls == [("BTCUSDT", "1m", candle(119).open_time)]


def test_existing_xgboost_artifact_runs_without_training():
    settings = Settings(_env_file=None, storage_provider="influxdb3")
    market = Market(candle(i) for i in range(119))
    predictions = Predictions()
    runs = Runs()
    evaluations = Evaluations()
    with LiveCycle(settings, Source([candle(119)]), market, predictions,
                   runs=runs, evaluations=evaluations,
                   clock=lambda: START + timedelta(minutes=120, seconds=5)) as cycle:
        result = cycle.run()
    assert result.status == "SUCCESS"
    assert result.prediction.model_version == "xgboost_v1"
    assert result.prediction_id in predictions.rows
    assert runs.rows[0].prediction_id == result.prediction_id
    assert runs.rows[0].evaluation_ms is not None
    assert evaluations.rows == {}  # The target candle has not closed yet.


def test_empty_cloud_repository_bootstraps_closed_model_history():
    market = Market()
    source = Source(candle(i) for i in range(121))
    cycle, _, predictions, _, _, _ = make_cycle(market=market, source=source)
    with cycle:
        result = cycle.run(now=START + timedelta(minutes=120, seconds=5))
    assert result.status == "SUCCESS"
    assert result.candles_ingested == 120
    assert len(market.rows) == 120
    assert len(predictions.rows) == 1
    assert source.calls[0] == (START, START + timedelta(minutes=120))
    assert candle(120).open_time not in [row.open_time for row in market.rows.values()]


def test_bootstrap_never_uses_current_open_candle_at_minute_end():
    market = Market()
    source = Source(candle(i) for i in range(121))
    cycle, _, predictions, _, _, _ = make_cycle(market=market, source=source)
    with cycle:
        result = cycle.run(now=START + timedelta(minutes=120, seconds=59))
    assert result.candle.open_time == candle(119).open_time
    assert len(predictions.rows) == 1
    assert candle(120).open_time not in [row.open_time for row in market.rows.values()]


def test_existing_telemetry_stages_are_emitted(tmp_path):
    cycle, _, _, _, _, _ = make_cycle()
    telemetry = TelemetryStore(tmp_path)
    cycle._telemetry = telemetry
    with cycle:
        result = cycle.run()
    events = telemetry.read_events()
    assert [row["stage"] for row in events] == [
        "ingestion_started", "data_ready", "prediction_started", "prediction_ready"
    ]
    assert all(row["trace_id"] == result.trace_id for row in events)
    assert result.trace_id == result.run_id.replace("-", "")
    assert events[-1]["prediction_id"] == result.prediction_id


def test_no_closed_candle_is_normal():
    cycle, _, predictions, _, _, _ = make_cycle(market=Market(), source=Source())
    with cycle:
        result = cycle.run()
    assert result.status == "NO_NEW_CANDLE" and result.candle is None
    assert result.prediction_id is None and predictions.save_calls == 0


@pytest.mark.parametrize("second", [0, 5, 59])
def test_current_minute_candle_is_never_used(second):
    # At 12:00:00/05/59 the 12:00 candle remains open.
    source = Source([candle(119), candle(120)])
    cycle, market, _, _, _, _ = make_cycle(source=source)
    now = START + timedelta(minutes=120, seconds=second)
    with cycle:
        result = cycle.run(now=now)
    assert result.candle.open_time == candle(119).open_time
    assert candle(120).open_time not in [row.open_time for row in market.rows.values()]
    assert all(stop == START + timedelta(minutes=120) for _, stop in source.calls)


def test_open_candle_in_storage_is_skipped():
    market = Market(candle(i) for i in range(121))
    cycle, _, _, _, _, _ = make_cycle(market=market, source=Source())
    with cycle:
        result = cycle.run(now=START + timedelta(minutes=120, seconds=5))
    assert result.latest_stored_candle.open_time == candle(120).open_time
    assert result.candle.open_time == candle(119).open_time
    assert result.prediction.timestamp == candle(119).open_time


def test_insufficient_history_returns_explicit_status():
    cycle, _, predictions, _, _, _ = make_cycle(market=Market(candle(i) for i in range(80, 119)),
                                                 source=Source([candle(119)]))
    with cycle:
        result = cycle.run()
    assert result.status == "INSUFFICIENT_HISTORY"
    assert result.prediction is None and predictions.save_calls == 0


def test_gap_in_history_prevents_prediction():
    market = Market(candle(i) for i in range(119) if i != 60)
    cycle, _, predictions, _, _, _ = make_cycle(market=market)
    with cycle:
        result = cycle.run()
    assert result.status == "INSUFFICIENT_HISTORY"
    assert predictions.save_calls == 0


def test_binance_and_storage_errors_are_not_reported_as_success():
    source = Source([candle(119)])
    source.fail = True
    cycle, _, predictions, _, _, _ = make_cycle(source=source)
    with cycle, pytest.raises(MarketDataError):
        cycle.run()
    assert predictions.save_calls == 0
    market = Market(candle(i) for i in range(119))
    market.fail_upsert = True
    cycle, _, predictions, _, _, _ = make_cycle(market=market)
    with cycle, pytest.raises(StorageError):
        cycle.run()
    assert predictions.save_calls == 0


def test_invalid_model_never_saves_prediction():
    cycle, _, predictions, _, _, _ = make_cycle(estimator=Estimator((float("nan"), 0.7)))
    with cycle, pytest.raises(PredictionUnavailableError):
        cycle.run()
    assert predictions.save_calls == 0


def test_concurrent_duplicate_save_recovers_existing_prediction():
    predictions = Predictions()
    predictions.race = True
    cycle, _, _, _, _, _ = make_cycle(predictions=predictions)
    with cycle:
        result = cycle.run()
    assert result.status == "PREDICTION_ALREADY_EXISTS"
    assert not result.prediction_created and result.prediction_id in predictions.rows
    assert len(predictions.rows) == 1


def test_factory_supports_both_configured_providers(monkeypatch):
    from alpha_engine.services import live_cycle as module
    for provider in ("influxdb3", "supabase"):
        market = Market(candle(i) for i in range(119))
        predictions = Predictions()
        source = Source([candle(119)])
        observed = []
        monkeypatch.setattr(module, "create_market_repository",
                            lambda settings: observed.append(settings.storage_provider) or market)
        monkeypatch.setattr(module, "create_prediction_repository", lambda settings: predictions)
        monkeypatch.setattr(module, "create_pipeline_run_repository", lambda settings: Runs())
        monkeypatch.setattr(module, "create_live_evaluation_repository", lambda settings: Evaluations())
        monkeypatch.setattr(module, "BinanceKlineClient", lambda url: source)
        settings = Settings(_env_file=None, storage_provider=provider)
        with module.LiveCycle.from_settings(settings, model_loader=lambda path: bundle(Estimator())) as cycle:
            result = cycle.run(now=START + timedelta(minutes=120, seconds=5))
        assert result.status == "SUCCESS" and observed == [provider]
        assert market.closed and predictions.closed and source.closed


def test_one_shot_closes_resources(monkeypatch):
    from alpha_engine.services import live_cycle as module
    cycle, market, predictions, source, _, _ = make_cycle()
    monkeypatch.setattr(module.LiveCycle, "from_settings", lambda *args, **kwargs: cycle)
    result = run_live_cycle(now=START + timedelta(minutes=120, seconds=5))
    assert result.status == "SUCCESS"
    assert market.closed and predictions.closed and source.closed


def test_naive_clock_is_rejected():
    cycle, _, _, _, _, _ = make_cycle(clock=lambda: datetime(2026, 1, 1, 12, 0))
    with cycle, pytest.raises(ValueError, match="zona horaria"):
        cycle.run()


def test_offset_clock_is_normalized_to_utc():
    cycle, _, _, _, _, _ = make_cycle()
    lima = timezone(timedelta(hours=-5))
    now = (START + timedelta(minutes=120, seconds=5)).astimezone(lima)
    with cycle:
        result = cycle.run(now=now)
    assert result.as_of.tzinfo is timezone.utc
    assert result.as_of == START + timedelta(minutes=120, seconds=5)
    assert result.prediction.timestamp.tzinfo is timezone.utc


def test_pipeline_runs_record_success_and_no_new_data_with_unique_ids():
    runs = Runs()
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                  1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    cycle, _, _, _, _, _ = make_cycle(runs=runs, monotonic=lambda: next(ticks))
    with cycle:
        first = cycle.run()
        second = cycle.run()
    assert first.status == "SUCCESS" and second.status == "NO_NEW_CANDLE"
    assert first.run_id != second.run_id and len(runs.rows) == 2
    assert runs.rows[0].duration_ms == pytest.approx(500)
    assert runs.rows[0].market_data_ms == pytest.approx(100)
    assert runs.rows[0].prediction_ms == pytest.approx(100)
    assert runs.rows[0].new_candle and runs.rows[0].new_prediction
    assert runs.rows[0].prediction_id == first.prediction_id
    assert runs.rows[1].status == "NO_NEW_CANDLE" and not runs.rows[1].new_prediction
    assert runs.closed


def test_run_cycle_timing_logs_all_requested_phases(caplog):
    cycle, _, _, source, _, _ = make_cycle(runs=Runs())
    with caplog.at_level("INFO", logger="uvicorn.error"):
        with cycle:
            result = cycle.run(now=START + timedelta(minutes=120, seconds=5))
    timing = [record.message for record in caplog.records
              if record.message.startswith("RUN_CYCLE_TIMING")]
    assert result.status == "SUCCESS" and len(timing) == 1
    for name in ("total_ms", "binance_ms", "db_ms", "features_ms",
                 "prediction_ms", "evaluation_ms"):
        assert f"{name}=" in timing[0]
    assert source.calls


def test_wall_clock_timestamps_and_monotonic_duration_are_independent():
    runs = Runs()
    initial = START + timedelta(minutes=120, seconds=5)
    wall = iter([initial, initial + timedelta(seconds=2), initial + timedelta(seconds=4)])
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cycle, _, _, _, _, _ = make_cycle(
        runs=runs, clock=lambda: next(wall), monotonic=lambda: next(ticks)
    )
    with cycle:
        result = cycle.run()
    run = runs.rows[0]
    assert run.started_at == initial and run.finished_at == initial + timedelta(seconds=4)
    assert run.duration_ms == pytest.approx(500)
    assert result.prediction.generated_at == initial + timedelta(seconds=2)


def test_pipeline_run_records_insufficient_history():
    runs = Runs()
    cycle, _, _, _, _, _ = make_cycle(
        market=Market(candle(i) for i in range(80, 119)), runs=runs
    )
    with cycle:
        result = cycle.run()
    assert result.status == runs.rows[0].status == "INSUFFICIENT_HISTORY"
    assert runs.rows[0].candle_open_time == candle(119).open_time


def test_pipeline_run_records_invalid_clock_as_failure():
    runs = Runs()
    cycle, _, _, _, _, _ = make_cycle(runs=runs)
    with cycle, pytest.raises(ValueError, match="zona horaria"):
        cycle.run(now=datetime(2026, 1, 1, 12, 0))
    assert runs.rows[0].status == "FAILED"
    assert runs.rows[0].error_type == "ValueError"
    assert runs.rows[0].error_message == "Invalid cycle clock or timestamp"


def test_pipeline_run_records_sanitized_market_and_model_failures():
    runs = Runs()
    source = Source([candle(119)])
    source.fail = True
    cycle, _, _, _, _, _ = make_cycle(source=source, runs=runs)
    with cycle, pytest.raises(MarketDataError):
        cycle.run()
    assert runs.rows[0].status == "FAILED"
    assert runs.rows[0].error_type == "MarketDataError"
    assert runs.rows[0].error_message == "Market data request failed"
    assert runs.rows[0].prediction_id is None
    cycle, _, predictions, _, _, _ = make_cycle(
        estimator=Estimator((float("nan"), 0.7)), runs=runs
    )
    with cycle, pytest.raises(PredictionUnavailableError):
        cycle.run()
    assert runs.rows[1].status == "FAILED"
    assert runs.rows[1].error_type == "PredictionUnavailableError"
    assert predictions.save_calls == 0


def test_failed_storage_record_is_attempted_then_logged_when_store_is_down(caplog):
    class BrokenRuns(Runs):
        def save(self, row):
            raise RuntimeError("run store down")

    market = Market(candle(i) for i in range(119))
    market.fail_upsert = True
    cycle, _, _, _, _, _ = make_cycle(market=market, runs=BrokenRuns())
    with cycle, pytest.raises(StorageError):
        cycle.run()
    assert "No se pudo persistir el run fallido" in caplog.text


def test_secondary_evaluation_failure_does_not_erase_new_prediction():
    class BrokenEvaluations(Evaluations):
        def get_by_prediction(self, prediction_id):
            raise RuntimeError("evaluation store down")

    runs = Runs()
    cycle, _, predictions, _, _, _ = make_cycle(runs=runs, evaluations=BrokenEvaluations())
    with cycle:
        result = cycle.run()
    assert result.status == "SUCCESS" and result.prediction_id in predictions.rows
    assert runs.rows[0].evaluation_error_type == "RuntimeError"
    assert runs.rows[0].evaluation_error_message == "Historical evaluation failed"


def test_run_storage_failure_after_prediction_is_visible():
    class BrokenRuns(Runs):
        def save(self, row):
            raise RuntimeError("run store down")

    cycle, _, predictions, _, _, _ = make_cycle(runs=BrokenRuns())
    with cycle, pytest.raises(StorageError, match="pipeline"):
        cycle.run()
    assert len(predictions.rows) == 1


def test_system_status_reads_persisted_facts_without_health_guess(monkeypatch):
    runs = Runs()
    evaluations = Evaluations()
    cycle, market, predictions, _, _, _ = make_cycle(runs=runs, evaluations=evaluations)
    with cycle:
        result = cycle.run()
        monkeypatch.setattr(
            runs, "get_latest",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("recent runs already contain the latest successful run")
            ),
        )
        status = SystemStatusService(market, predictions, runs, evaluations).get("BTCUSDT")
    assert status.last_run.run_id.hex == result.run_id.replace("-", "")
    assert status.last_successful_run == status.last_run
    assert status.last_candle.open_time == candle(119).open_time
    assert status.last_prediction.prediction_id == result.prediction_id
    assert status.last_evaluation is None and status.recent_errors == []
    assert status.recent_evaluation_errors == []
    failed = PipelineRunRecord(
        run_id=uuid4(), started_at=status.last_run.started_at + timedelta(minutes=1),
        finished_at=status.last_run.finished_at + timedelta(minutes=1),
        duration_ms=15, status="FAILED", symbol="BTCUSDT", interval="1m",
        error_type="MarketDataError", error_message="Market data request failed",
    )
    runs.save(failed)
    updated = SystemStatusService(market, predictions, runs, evaluations).get("BTCUSDT")
    assert updated.last_run == failed
    assert updated.last_successful_run == status.last_run
    assert updated.recent_errors == [failed]


def test_system_status_finds_last_success_beyond_recent_failures():
    runs = Runs()
    evaluations = Evaluations()
    cycle, market, predictions, _, _, _ = make_cycle(runs=runs, evaluations=evaluations)
    with cycle:
        cycle.run()
    successful = runs.rows[0]
    for index in range(1, 22):
        runs.save(PipelineRunRecord(
            run_id=uuid4(), started_at=successful.started_at + timedelta(minutes=index),
            finished_at=successful.finished_at + timedelta(minutes=index),
            duration_ms=15, status="FAILED", symbol="BTCUSDT", interval="1m",
            error_type="MarketDataError", error_message="Market data request failed",
        ))

    status = SystemStatusService(market, predictions, runs, evaluations).get("BTCUSDT")

    assert status.last_run.status == "FAILED"
    assert status.last_successful_run == successful
