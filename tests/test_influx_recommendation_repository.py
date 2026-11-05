from datetime import datetime, timezone

from alpha_engine.data.schemas import RecommendationRecord
from alpha_engine.repositories.influx_recommendation import InfluxRecommendationRepository


TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def sample_record():
    return RecommendationRecord(
        prediction_id="pred-1", prompt_version="recommendation_v1",
        llm_model="qwen3:4b-instruct", timestamp=TIMESTAMP,
        generated_at=TIMESTAMP, candidate_signal="BUY", llm_signal=None,
        final_signal="HOLD", risk_level="HIGH",
        context_json='{"market":{}}', response_json=None,
        guardrails_json='["HIGH_RISK"]',
    )


class FakeWriteApi:
    def __init__(self):
        self.calls = []

    def write(self, **kwargs):
        self.calls.append(kwargs)


class FakeRecord:
    def __init__(self, record):
        self.values = {
            **record.model_dump(exclude={"timestamp", "generated_at"}),
            "generated_at": record.generated_at.isoformat(),
            "llm_signal": record.llm_signal or "",
            "response_json": record.response_json or "",
        }
        self._time = record.timestamp

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


def test_recommendation_write_keeps_cache_and_adds_history_event():
    client = FakeClient()
    repo = InfluxRecommendationRepository(client, "alpha-engine", "market")
    record = sample_record()
    repo.save(record)
    repo.save(record)
    first, first_history = [point.to_line_protocol() for point in client.writer.calls[0]["record"]]
    second, second_history = [point.to_line_protocol() for point in client.writer.calls[1]["record"]]
    assert first == second
    assert first_history != second_history
    assert "recommendation_history" in first_history
    assert "candle_open_at=" in first_history
    assert "prediction_id=pred-1" in first
    assert "llm_signal=\"\"" in first
    assert str(int(TIMESTAMP.timestamp() * 1000)) in first


def test_recommendation_cache_read_reconstructs_optional_fields():
    record = sample_record()
    client = FakeClient([FakeRecord(record)])
    repo = InfluxRecommendationRepository(client, "alpha-engine", "market")
    assert repo.get_cached("pred-1", "recommendation_v1", "qwen3:4b-instruct") == record
    query = client.querier.calls[0]["query"]
    assert 'r.prediction_id == "pred-1"' in query
    assert 'r.prompt_version == "recommendation_v1"' in query
    assert 'group(columns: [])' in query
    repo.close()
    assert client.closed


def test_recommendation_cache_miss():
    repo = InfluxRecommendationRepository(FakeClient(), "alpha-engine", "market")
    assert repo.get_cached("none", "v1", "model") is None
