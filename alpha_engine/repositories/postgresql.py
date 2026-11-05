"""Small PostgreSQL adapter for the local Compose stack.

The domain services and cloud Supabase adapter are unchanged. Table and column
names are fixed here; every runtime value is bound as a SQL parameter.
"""

from datetime import datetime, timedelta
import json
from uuid import UUID

from alpha_engine.config.settings import Settings
from alpha_engine.data.live_schemas import PendingLiveEvaluation
from alpha_engine.repositories.supabase import (
    SupabaseMarketRepository as MarketCodec,
    SupabasePredictionRepository as PredictionCodec,
    SupabaseRecommendationRepository as RecommendationCodec,
    SupabaseRecommendationRequestRepository as RequestCodec,
    SupabasePipelineRunRepository as RunCodec,
    SupabaseLiveEvaluationRepository as EvaluationCodec,
)
from alpha_engine.repositories.supabase_news import NewsFetchRecord, marketaux_snapshot, marketaux_claim


class _PostgresRepository:
    table: str

    def __init__(self, connection):
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: Settings):
        url = settings.database_url.get_secret_value()
        if not url:
            raise ValueError("DATABASE_URL is required for local PostgreSQL")
        import psycopg2
        from psycopg2.extras import RealDictCursor, register_uuid

        register_uuid()
        connection = psycopg2.connect(url, cursor_factory=RealDictCursor)
        connection.autocommit = True
        return cls(connection)

    def close(self) -> None:
        self._connection.close()

    def _one(self, query: str, params=()):
        with self._connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def _many(self, query: str, params=()):
        with self._connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def _insert(self, row: dict, conflict: str = "DO NOTHING", *, table: str | None = None) -> bool:
        from psycopg2 import sql

        columns = list(row)
        statement = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT {} RETURNING 1").format(
            sql.Identifier(table or self.table),
            sql.SQL(",").join(map(sql.Identifier, columns)),
            sql.SQL(",").join(sql.Placeholder() for _ in columns),
            sql.SQL(conflict),
        )
        return self._one(statement, tuple(row.values())) is not None


class PostgresMarketRepository(_PostgresRepository):
    table = "candles"

    def upsert_candles(self, candles):
        for candle in candles:
            self._insert(MarketCodec._row(candle))
        return len(candles)

    def get_candles(self, symbol: str, interval: str, start: datetime, stop: datetime):
        if stop <= start:
            raise ValueError("stop must be after start")
        rows = self._many(
            "SELECT * FROM candles WHERE symbol=%s AND interval=%s AND open_time >= %s "
            "AND open_time < %s ORDER BY open_time", (symbol, interval, start, stop),
        )
        return [MarketCodec._candle(row) for row in rows]

    def get_latest_candle(self, symbol: str, interval: str):
        row = self._one(
            "SELECT * FROM candles WHERE symbol=%s AND interval=%s ORDER BY open_time DESC LIMIT 1",
            (symbol, interval),
        )
        return MarketCodec._candle(row) if row else None


class PostgresPredictionRepository(_PostgresRepository):
    table = "predictions"

    def save(self, prediction):
        return self._insert(PredictionCodec._row(prediction))

    def get_by_id(self, prediction_id: str):
        row = self._one("SELECT * FROM predictions WHERE prediction_id=%s", (prediction_id,))
        return PredictionCodec._prediction(row) if row else None

    def get_latest(self, symbol: str):
        row = self._one("SELECT * FROM predictions WHERE symbol=%s ORDER BY candle_open_time DESC LIMIT 1", (symbol,))
        return PredictionCodec._prediction(row) if row else None

    def list_recent(self, symbol: str, limit: int = 100, offset: int = 0):
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("invalid pagination")
        rows = self._many(
            "SELECT * FROM predictions WHERE symbol=%s ORDER BY candle_open_time DESC LIMIT %s OFFSET %s",
            (symbol, limit, offset),
        )
        return [PredictionCodec._prediction(row) for row in rows]

    def list_recent_before(self, symbol: str, before: datetime, limit: int = 100):
        if before.tzinfo is None or not 1 <= limit <= 1000:
            raise ValueError("invalid replay window")
        rows = self._many(
            "SELECT * FROM predictions WHERE symbol=%s AND candle_open_time <= %s "
            "ORDER BY candle_open_time DESC LIMIT %s", (symbol, before, limit),
        )
        return [PredictionCodec._prediction(row) for row in rows]

    def list_entry_signals(self, symbol: str, model_version: str,
                           minimum_up: float, limit: int = 100):
        if not 0.5 < minimum_up < 1 or not 1 <= limit <= 1000:
            raise ValueError("invalid entry search")
        rows = self._many(
            "SELECT * FROM predictions WHERE symbol=%s AND model_version=%s "
            "AND prob_up >= %s ORDER BY candle_open_time DESC LIMIT %s",
            (symbol, model_version, minimum_up, limit),
        )
        return [PredictionCodec._prediction(row) for row in rows]

    def list_evaluable_pending(self, symbol: str, now: datetime, limit: int = 100):
        if now.tzinfo is None or not 1 <= limit <= 1000:
            raise ValueError("now/limit invalid")
        rows = self._many(
            "SELECT p.prediction_id, p.symbol, p.predicted_class, "
            "source.close AS source_close, target.open_time AS target_open_time, "
            "target.close_time AS target_close_time, target.close AS target_close "
            "FROM predictions AS p "
            "JOIN candles AS source ON source.symbol=p.symbol AND source.interval=p.interval "
            "AND source.open_time=p.candle_open_time "
            "JOIN candles AS target ON target.symbol=p.symbol AND target.interval=p.interval "
            "AND target.open_time=p.candle_open_time + INTERVAL '1 minute' "
            "WHERE p.symbol=%s AND p.interval='1m' AND p.generated_at IS NOT NULL "
            "AND target.close_time < %s "
            "AND NOT EXISTS (SELECT 1 FROM live_evaluations AS e "
            "WHERE e.prediction_id=p.prediction_id) "
            "ORDER BY p.candle_open_time ASC, p.prediction_id ASC LIMIT %s",
            (symbol, now, limit),
        )
        return [PendingLiveEvaluation.model_validate(row) for row in rows]


class PostgresPipelineRunRepository(_PostgresRepository):
    table = "pipeline_runs"

    def save(self, run):
        self._insert(RunCodec._row(run))

    def get_latest(self, symbol: str, status=None):
        rows = self.list_recent(symbol, 1, status)
        return rows[0] if rows else None

    def list_recent(self, symbol: str, limit: int = 20, status=None):
        if not 1 <= limit <= 1000:
            raise ValueError("invalid limit")
        where = "symbol=%s" + (" AND status=%s" if status else "")
        params = (symbol, status, limit) if status else (symbol, limit)
        rows = self._many(f"SELECT * FROM pipeline_runs WHERE {where} ORDER BY started_at DESC LIMIT %s", params)
        return [RunCodec._record(row) for row in rows]


class PostgresLiveEvaluationRepository(_PostgresRepository):
    table = "live_evaluations"

    def save(self, evaluation):
        return self._insert(EvaluationCodec._row(evaluation))

    def save_batch(self, evaluations):
        if not evaluations:
            return set()
        from psycopg2.extras import execute_values

        columns = tuple(EvaluationCodec._row(evaluations[0]))
        statement = (
            "INSERT INTO live_evaluations (" + ",".join(columns) + ") VALUES %s "
            "ON CONFLICT (prediction_id) DO NOTHING RETURNING prediction_id"
        )
        inserted: set[str] = set()
        with self._connection.cursor() as cursor:
            for start in range(0, len(evaluations), 500):
                rows = [EvaluationCodec._row(row) for row in evaluations[start:start + 500]]
                values = [tuple(row[column] for column in columns) for row in rows]
                inserted.update(row["prediction_id"] for row in execute_values(
                    cursor, statement, values, page_size=500, fetch=True,
                ))
        return inserted

    def get_by_prediction(self, prediction_id: str):
        row = self._one("SELECT * FROM live_evaluations WHERE prediction_id=%s", (prediction_id,))
        return EvaluationCodec._record(row) if row else None

    def get_latest(self, symbol: str):
        rows = self.list_recent(symbol, 1)
        return rows[0] if rows else None

    def list_recent(self, symbol: str, limit: int = 100, offset: int = 0):
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("invalid pagination")
        rows = self._many(
            "SELECT * FROM live_evaluations WHERE symbol=%s ORDER BY evaluated_at DESC LIMIT %s OFFSET %s",
            (symbol, limit, offset),
        )
        return [EvaluationCodec._record(row) for row in rows]


class PostgresRecommendationRepository(_PostgresRepository):
    table = "recommendations"

    def save(self, recommendation):
        row = RecommendationCodec._row(recommendation)
        from psycopg2 import sql

        columns = list(row)
        statement = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT "
                            "(prediction_id,prompt_version,llm_model) DO UPDATE SET {} RETURNING 1").format(
            sql.Identifier(self.table),
            sql.SQL(",").join(map(sql.Identifier, columns)),
            sql.SQL(",").join(sql.Placeholder() for _ in columns),
            sql.SQL(",").join(
                sql.SQL("{}=EXCLUDED.{}").format(sql.Identifier(name), sql.Identifier(name))
                for name in columns if name not in {"prediction_id", "prompt_version", "llm_model"}
            ),
        )
        self._one(statement, tuple(row.values()))

    def save_if_absent(self, recommendation):
        return self._insert(RecommendationCodec._row(recommendation))

    def get_cached(self, prediction_id: str, prompt_version: str, llm_model: str):
        row = self._one(
            "SELECT * FROM recommendations WHERE prediction_id=%s AND prompt_version=%s AND llm_model=%s",
            (prediction_id, prompt_version, llm_model),
        )
        return RecommendationCodec._recommendation(row) if row else None

    def list_recent(self, limit: int = 20):
        if not 1 <= limit <= 100:
            raise ValueError("invalid limit")
        rows = self._many("SELECT * FROM recommendations ORDER BY generated_at DESC LIMIT %s", (limit,))
        return [RecommendationCodec._recommendation(row) for row in rows]


class PostgresRecommendationRequestRepository(_PostgresRepository):
    table = "recommendation_requests"

    def save(self, request):
        self._insert(RequestCodec._row(request))

    def get_latest(self, symbol: str, status=None):
        rows = self.list_recent(symbol, 1, status)
        return rows[0] if rows else None

    def list_recent(self, symbol: str, limit: int = 20, status=None):
        if not 1 <= limit <= 1000:
            raise ValueError("invalid limit")
        where = "symbol=%s" + (" AND status=%s" if status else "")
        params = (symbol, status, limit) if status else (symbol, limit)
        rows = self._many(f"SELECT * FROM recommendation_requests WHERE {where} ORDER BY requested_at DESC LIMIT %s", params)
        return [RequestCodec._record(row) for row in rows]


class PostgresPrescriptiveNewsRepository(_PostgresRepository):
    table = "prescriptive_news_fetches"

    def latest_successful(self):
        row = self._one(
            "SELECT snapshot_id,attempted_at,fetched_at,status FROM prescriptive_news_fetches "
            "WHERE provider='ALPHA_VANTAGE' AND status IN ('SUCCESS','EMPTY') ORDER BY fetched_at DESC LIMIT 1"
        )
        return NewsFetchRecord(**row) if row else None

    def claim_slot(self, slot_start: datetime, attempted_at: datetime, snapshot_id: UUID):
        return self._insert({
            "snapshot_id": snapshot_id, "slot_start": slot_start,
            "attempted_at": attempted_at, "status": "CLAIMED", "article_count": 0,
        })

    def finish(self, snapshot_id: UUID, status: str, fetched_at: datetime | None, article_count: int):
        self._one(
            "UPDATE prescriptive_news_fetches SET status=%s,fetched_at=%s,article_count=%s "
            "WHERE snapshot_id=%s RETURNING 1", (status, fetched_at, article_count, snapshot_id),
        )

    def save_articles(self, articles: list[dict]):
        for article in articles:
            self._insert(article, table="prescriptive_news_articles")

    def articles_at(self, fetched_at: datetime):
        return self._many(
            "SELECT published_at,first_observed_at,btc_sentiment_score,btc_relevance_score,source "
            "FROM prescriptive_news_articles WHERE provider='ALPHA_VANTAGE' AND published_at>%s AND published_at<=%s "
            "AND first_observed_at<=%s ORDER BY published_at DESC",
            (fetched_at - timedelta(minutes=60), fetched_at, fetched_at),
        )

    def latest_marketaux_snapshot(self):
        row = self._one("SELECT snapshot_id,attempted_at,fetched_at,marketaux_payload FROM prescriptive_news_fetches "
                        "WHERE provider='MARKETAUX' AND status IN ('SUCCESS','EMPTY') ORDER BY fetched_at DESC LIMIT 1")
        return marketaux_snapshot(row) if row else None

    def claim_marketaux(self, snapshot_id, daily_budget):
        row = self._one("SELECT * FROM claim_marketaux_news(%s,%s)", (snapshot_id, daily_budget))
        if not row:
            raise ValueError("INVALID_CLAIM")
        return marketaux_claim(row)

    def finish_marketaux(self, snapshot_id, articles, fetched_at):
        payload = json.dumps([a.model_dump(mode="json") for a in articles], allow_nan=False) if articles is not None else None
        row = self._one("UPDATE prescriptive_news_fetches SET status=%s,fetched_at=%s,marketaux_payload=%s::jsonb,article_count=%s "
                  "WHERE provider='MARKETAUX' AND snapshot_id=%s AND status='CLAIMED' RETURNING 1",
                  ("FAILED" if articles is None else "SUCCESS" if articles else "EMPTY", fetched_at,
                   payload, len(articles or []), snapshot_id))
        if not row:
            raise ValueError("SNAPSHOT_NOT_FINISHED")
