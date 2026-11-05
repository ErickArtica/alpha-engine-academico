"""Synthetic, in-memory fixtures. This file never enters the runtime image."""
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx

PUBLIC_ENV = dict(NORTHFLANK_PUBLIC_READ_ONLY="true", NORTHFLANK_PUBLIC_DATA_APPROVED="true",
                  STORAGE_PROVIDER="supabase", SUPABASE_URL="https://synthetic.invalid",
                  SUPABASE_KEY="sb_publishable_synthetic_only", FRONTEND_PUBLIC_HOST="frontend.example.com",
                  MODEL_CONFIDENCE_THRESHOLD="0.60", PRESCRIPTIVE_NEWS_ENABLED="false", DASHBOARD_FALLBACK_ENABLED="false")


def synthetic_rows():
    now = datetime.now(timezone.utc)
    source = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
    candles = []
    for index in range(60):
        opened = source - timedelta(minutes=index)
        price = 50_000 - index
        candles.append(dict(symbol="BTCUSDT", interval="1m", open_time=opened.isoformat(),
                            close_time=(opened + timedelta(minutes=1, milliseconds=-1)).isoformat(),
                            open=price, high=price + 2, low=price - 1, close=price + 1, volume=100 + index))
    def prediction(index, identifier):
        opened = source - timedelta(minutes=index)
        return dict(prediction_id=identifier * 64, symbol="BTCUSDT", interval="1m", horizon="1m",
                    candle_open_time=opened.isoformat(), prob_up=0.7, prob_down=0.3, predicted_class="UP",
                    model_version="xgboost_v1", trained_at=None,
                    generated_at=(opened + timedelta(minutes=1)).isoformat(),
                    valid_until=(opened + timedelta(minutes=2)).isoformat(),
                    context_json="PRIVATE_CONTENT_MUST_NOT_SURFACE")
    evaluation = dict(evaluation_id=str(UUID(int=1)), prediction_id="e" * 64, symbol="BTCUSDT",
                      evaluated_at=now.isoformat(), source_close=100, target_close=101,
                      target_open_time=source.isoformat(),
                      target_close_time=(source + timedelta(minutes=1, milliseconds=-1)).isoformat(),
                      observed_move="UP", observed_class="UP", correct=True)
    return dict(candles=candles, predictions=[prediction(0, "f"), prediction(1, "e")], live_evaluations=[evaluation])


def mock_transport(rows, requests, *, unavailable=None):
    def handle(request):
        requests.append(request)
        table = request.url.path.rsplit("/", 1)[-1]
        if table == unavailable:
            return httpx.Response(503, json={"message": "synthetic unavailable"})
        return httpx.Response(200, json=rows[table][:int(request.url.params["limit"])])
    return httpx.MockTransport(handle)
