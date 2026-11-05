"""Read-only summaries of audited prescriptive requests."""

from dataclasses import dataclass

from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord
from alpha_engine.repositories.protocols import RecommendationRequestRepository


@dataclass(frozen=True)
class RecommendationObservability:
    last_attempt: RecommendationRequestRecord | None
    last_successful_attempt: RecommendationRequestRecord | None
    recent_attempts: list[RecommendationRequestRecord]
    timely_count: int
    late_count: int
    timeout_count: int
    provider_error_count: int
    average_llm_latency_ms: float | None


class RecommendationObservabilityService:
    def __init__(self, requests: RecommendationRequestRepository):
        self._requests = requests

    def get(self, symbol: str, limit: int = 1000) -> RecommendationObservability:
        records = self._requests.list_recent(symbol, limit=limit)
        successful = [record for record in records if record.status in ("SUCCESS", "CACHED")]
        latencies = [record.llm_latency_ms for record in records if record.llm_latency_ms is not None]
        return RecommendationObservability(
            last_attempt=records[0] if records else None,
            last_successful_attempt=successful[0] if successful else None,
            recent_attempts=records,
            timely_count=sum(record.timing_status == "TIMELY" for record in records),
            late_count=sum(record.timing_status == "LATE" for record in records),
            timeout_count=sum(record.status == "LLM_TIMEOUT" for record in records),
            provider_error_count=sum(record.status == "PROVIDER_ERROR" for record in records),
            average_llm_latency_ms=sum(latencies) / len(latencies) if latencies else None,
        )
