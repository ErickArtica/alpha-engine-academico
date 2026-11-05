from datetime import datetime, timezone

from alpha_engine.data.schemas import PredictionRecord
from alpha_engine.repositories.influx_prediction import InfluxPredictionRepository


TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def sample_prediction():
    return PredictionRecord(
        prediction_id="abc123", symbol="BTCUSDT", timestamp=TIMESTAMP,
        horizon="1m", prob_up=0.7, prob_down=0.3,
        predicted_class="UP", model_version="xgboost_v1",
    )


class FakeWriteApi:
    def __init__(self):
        self.calls = []

    def write(self, **kwargs):
        self.calls.append(kwargs)


class FakeRecord:
    def __init__(self, prediction):
        self.values = prediction.model_dump(exclude={"timestamp"})
        self._time = prediction.timestamp

    def get_time(self):
        return self._time


class FakeTable:
    def __init__(self, records):
        self.records = records


class FakeQueryApi:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeTable(self.records)]


class FakeClient:
    def __init__(self, records=()):
        self.writer = FakeWriteApi()
        self.querier = FakeQueryApi(records)
        self.closed = False

    def write_api(self, **kwargs):
        return self.writer

    def query_api(self):
        return self.querier

    def close(self):
        self.closed = True


def test_save_prediction_uses_stable_identity_and_timestamp():
    client = FakeClient()
    repo = InfluxPredictionRepository(client, "alpha-engine", "market")
    record = sample_prediction()
    repo.save(record)
    repo.save(record)
    first = client.writer.calls[0]
    second = client.writer.calls[1]
    assert first["bucket"] == "market"
    assert first["record"].to_line_protocol() == second["record"].to_line_protocol()
    assert "prediction_id=abc123" in first["record"].to_line_protocol()
    assert str(int(TIMESTAMP.timestamp() * 1000)) in first["record"].to_line_protocol()


def test_read_prediction_by_id_and_latest():
    prediction = sample_prediction()
    client = FakeClient([FakeRecord(prediction)])
    repo = InfluxPredictionRepository(client, "alpha-engine", "market")
    assert repo.get_by_id("abc123") == prediction
    assert repo.get_latest("BTCUSDT") == prediction
    first_query = client.querier.calls[0]["query"]
    latest_query = client.querier.calls[1]["query"]
    assert 'r.prediction_id == "abc123"' in first_query
    assert 'r.symbol == "BTCUSDT"' in latest_query
    assert 'group(columns: [])' in latest_query
    assert 'sort(columns: ["_time"], desc: true)' in latest_query
    repo.close()
    assert client.closed


def test_missing_prediction_returns_none():
    repo = InfluxPredictionRepository(FakeClient(), "alpha-engine", "market")
    assert repo.get_by_id("missing") is None

