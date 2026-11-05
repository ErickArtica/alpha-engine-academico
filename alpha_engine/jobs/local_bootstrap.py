"""One-time causal replay of recent closed Binance candles for the local PMV."""

import logging
from datetime import datetime, timedelta, timezone

from alpha_engine.config.settings import get_settings
from alpha_engine.data.binance import BinanceKlineClient
from alpha_engine.repositories.factory import (
    create_live_evaluation_repository, create_market_repository, create_prediction_repository,
)
from alpha_engine.services.ingestion import MarketIngestionService, closed_boundary
from alpha_engine.services.live_evaluation import LiveEvaluationService
from alpha_engine.services.prediction import PredictionService

logger = logging.getLogger(__name__)
BOOTSTRAP_CANDLES = 360


def causal_source_pairs(candles, lookback: int = 120):
    """Only a contiguous closed source may be predicted; target is evaluation only."""
    step = timedelta(minutes=1)
    for index in range(lookback - 1, len(candles)):
        history = candles[index - lookback + 1:index + 1]
        if all(b.open_time - a.open_time == step for a, b in zip(history, history[1:])):
            yield candles[index]


def bootstrap(settings=None, *, now=None, source=None, market=None, predictions=None, evaluations=None):
    selected = settings or get_settings()
    if selected.storage_provider != "postgresql" or selected.app_env != "local":
        raise RuntimeError("Local bootstrap requires APP_ENV=local and PostgreSQL")
    from deploy.render.app import verify_model_artifact
    verify_model_artifact()
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("Bootstrap clock needs UTC timezone")
    own_market = market is None
    own_predictions = predictions is None
    own_evaluations = evaluations is None
    own_source = source is None
    market = market or create_market_repository(selected)
    predictions = predictions or create_prediction_repository(selected)
    evaluations = evaluations or create_live_evaluation_repository(selected)
    source = source or BinanceKlineClient(selected.binance_base_url)
    try:
        with market._connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM local_bootstrap_runs WHERE id=1")
            if cursor.fetchone():
                logger.info("LOCAL_BOOTSTRAP already completed; reusing persisted data")
                return {"status": "ALREADY_DONE"}
        stop = closed_boundary(instant, "1m")
        start = stop - timedelta(minutes=BOOTSTRAP_CANDLES)
        MarketIngestionService(source, market).ingest_range(selected.symbol, "1m", start, stop)
        candles = market.get_candles(selected.symbol, "1m", start, stop)
        if len(candles) < selected.prediction_lookback_candles + 2:
            raise RuntimeError("Not enough continuous closed Binance candles for local bootstrap")
        predictor = PredictionService(market, predictions, selected.model_path,
                                      lookback_candles=selected.prediction_lookback_candles)
        created = 0
        for candle in causal_source_pairs(candles, selected.prediction_lookback_candles):
            simulated_at = candle.close_time + timedelta(milliseconds=1)
            outcome = predictor.predict_candle_execution(
                candle, selected.symbol, "1m", now=simulated_at,
                generated_at_clock=lambda value=simulated_at: value,
            )
            created += int(outcome.created)
        # Evaluation reads the next stored close only now, after all predictions
        # were generated from their own 120-candle backward windows.
        batch = LiveEvaluationService(market, predictions, evaluations).evaluate_pending(
            selected.symbol, instant, limit=1000,
        )
        with market._connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO local_bootstrap_runs "
                "(id,completed_at,source_start,source_end,candle_count,prediction_count,evaluation_count) "
                "VALUES (1,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                (instant, candles[0].open_time, candles[-1].open_time,
                 len(candles), created, batch.created),
            )
        result = {"status": "LOCAL_BOOTSTRAP", "candles": len(candles),
                  "predictions": created, "evaluations": batch.created,
                  "source_start": candles[0].open_time.isoformat(),
                  "source_end": candles[-1].open_time.isoformat()}
        logger.info("LOCAL_BOOTSTRAP candles=%s predictions=%s evaluations=%s",
                    len(candles), created, batch.created)
        return result
    finally:
        if own_source:
            source.close()
        if own_evaluations:
            evaluations.close()
        if own_predictions:
            predictions.close()
        if own_market:
            market.close()


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        bootstrap()
    except Exception as exc:
        # The UI and scheduler remain available if Binance is temporarily down.
        logger.warning("LOCAL_BOOTSTRAP unavailable: %s", type(exc).__name__)


if __name__ == "__main__":
    main()
