"""Read-only drift for V3 features reconstructible from persisted LIVE OHLCV.

Supabase production candles do not persist Binance trade_count or taker-buy
volume. Those two V3 features remain unavailable in this LIVE check.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

from alpha_engine.data.schemas import Candle
from alpha_engine.features import build_feature_frame
from scripts.build_v3_dataset import AUDIT as V3_AUDIT, OUT as V3_DATASET, V3
from scripts.live_model_diagnostic import _fetch_all, _publishable_key, _require_public_read_key
from scripts.train_compare_xgboost_v2 import ROOT, feature_drift

OUT = ROOT / "reports/xgboost_v3_market_flow_6m_live_drift.json"
AVAILABLE = tuple(name for name in V3 if name not in ("trades_zscore", "taker_buy_ratio"))


def replay_ohlcv(candles: list[Candle]) -> tuple[pd.DataFrame, int]:
    result = []
    omitted = 0
    for i in range(119, len(candles)):
        window = candles[i-119:i+1]
        if any((right.open_time-left.open_time).total_seconds() != 60
               for left, right in zip(window, window[1:])):
            omitted += 1
            continue
        frame = build_feature_frame(window)
        last = frame.iloc[-1]
        c = window[-1]
        close = c.close
        row = {"timestamp": c.open_time, "return_1m": float(last.return_1m),
               "rsi_14": float(last.rsi_14),
               "ema_distance": (close-float(last.ema_20))/close,
               "macd_relative": float(last.macd)/close,
               "macd_signal_relative": float(last.macd_signal)/close,
               "macd_histogram_relative": float(last.macd_histogram)/close,
               "rolling_volatility": float(last.rolling_volatility),
               "volume_change": float(last.volume_change),
               "candle_range": (c.high-c.low)/close,
               "body_return": (close-c.open)/c.open,
               "upper_wick": (c.high-max(c.open,close))/close,
               "lower_wick": (min(c.open,close)-c.low)/close}
        for lag in (3,5,15):
            row[f"return_{lag}m"] = close/window[-lag-1].close-1
        volumes = np.asarray([item.volume for item in window[-20:]], dtype=float)
        std = float(volumes.std(ddof=0))
        row["volume_zscore"] = (c.volume-float(volumes.mean()))/std if std else 0.0
        result.append(row)
    frame = pd.DataFrame(result)
    if frame.empty or not np.isfinite(frame[list(AVAILABLE)].to_numpy(dtype=float)).all():
        raise ValueError("No complete, finite LIVE OHLCV replay")
    return frame, omitted


def main() -> None:
    cutoff = datetime.now(timezone.utc)
    ref = os.getenv("SUPABASE_PROJECT_REF") or (ROOT/"supabase/.temp/project-ref").read_text(encoding="utf-8").strip()
    key = os.getenv("SUPABASE_KEY") or _publishable_key(ref)
    _require_public_read_key(key)
    with httpx.Client(timeout=60) as client:
        rows, count = _fetch_all(client, f"https://{ref}.supabase.co", key, "candles", [
            ("select", "symbol,interval,open_time,close_time,open,high,low,close,volume"),
            ("symbol", "eq.BTCUSDT"), ("interval", "eq.1m"),
            ("open_time", "gte.2026-09-21T22:00:00Z"),
            ("close_time", f"lte.{cutoff.isoformat()}"), ("order", "open_time.asc")])
    candles = [Candle.model_validate(row) for row in rows]
    if count != len(candles) or len({c.open_time for c in candles}) != count:
        raise ValueError("Incomplete or duplicate public LIVE candles")
    live, omitted = replay_ohlcv(candles)
    live = live.loc[live.timestamp >= datetime(2026,9,22,tzinfo=timezone.utc)]
    if live.empty:
        raise ValueError("No post-historical LIVE rows")
    historical = pd.read_parquet(V3_DATASET)
    from scripts.build_v2_dataset import temporal_splits
    train = temporal_splits(historical)[0]["TRAIN"]
    audit = json.loads(V3_AUDIT.read_text(encoding="utf-8"))
    report = {"cutoff_utc": cutoff.isoformat(), "source": "Supabase public OHLCV; read only",
              "v3_dataset_sha256": audit["v3_dataset"]["sha256"],
              "raw_candles": count, "usable_live_rows": len(live), "omitted_gap_windows": omitted,
              "first_source_utc": live.timestamp.iloc[0].isoformat(),
              "last_source_utc": live.timestamp.iloc[-1].isoformat(),
              "available_features": list(AVAILABLE),
              "unavailable_features": ["trades_zscore", "taker_buy_ratio"],
              "limitation": "Persisted production candles omit number_of_trades and taker_buy_base; no proxy or backfill is invented.",
              "drift": feature_drift(train, live, AVAILABLE)}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(f"LIVE V3 OHLCV drift: {len(live)} rows, {len(AVAILABLE)} features; 2 unavailable")


if __name__ == "__main__":
    main()
