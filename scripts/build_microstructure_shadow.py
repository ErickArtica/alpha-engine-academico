"""Build a small local shadow sample from existing quote telemetry and candles.

No network calls, training, worker changes or production writes. Quote times are
local receipt times; the sample has no historical trade flow and is unsuitable
for model training until completeness and coverage are established.
"""

from __future__ import annotations

import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import httpx

from alpha_engine.config.settings import Settings
from alpha_engine.experiments.microstructure import L1Quote, SourceMinute, aggregate_source_minute, trade_flow_from_klines
from alpha_engine.repositories.influx3 import Influx3MarketRepository

ROOT = Path(__file__).resolve().parents[1]
QUOTES = ROOT / "runtime/telemetry/quotes.jsonl"
SHADOW = ROOT / "data/experiments/microstructure/shadow_minutes.jsonl"


def _quote(row: dict) -> L1Quote:
    return L1Quote(symbol=row["symbol"], observed_at=datetime.fromisoformat(row["received_at"]),
                   bid_price=Decimal(str(row["bid_price"])), ask_price=Decimal(str(row["ask_price"])),
                   bid_qty=Decimal(str(row["bid_qty"])), ask_qty=Decimal(str(row["ask_qty"])))


def _json(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(type(value).__name__)


def build(quotes_path: Path = QUOTES, output: Path = SHADOW, *, with_kline_flow: bool = False) -> dict:
    rows = [json.loads(line) for line in quotes_path.read_text(encoding="utf-8").splitlines() if line.strip()] if quotes_path.exists() else []
    quotes = [_quote(row) for row in rows]
    groups = defaultdict(list)
    for quote in quotes:
        minute = quote.observed_at.replace(second=0, microsecond=0)
        groups[(quote.symbol, minute)].append(quote)
    matched = []
    unmatched = 0
    if groups:
        opens = [minute for _, minute in groups]
        repo = Influx3MarketRepository.from_settings(Settings())
        try:
            candles = repo.get_candles("BTCUSDT", "1m", min(opens), max(opens) + timedelta(minutes=1))
        finally:
            repo.close()
        by_key = {(c.symbol, c.open_time): c for c in candles}
        now = datetime.now(timezone.utc)
        for key in sorted(groups):
            candle = by_key.get(key)
            if candle is None or candle.close_time >= now:
                unmatched += 1
                continue
            source = SourceMinute(candle.symbol, candle.interval, candle.open_time, candle.close_time)
            feature = aggregate_source_minute(source, groups[key], [])
            matched.append(feature)
    kline_errors = 0
    if with_kline_flow:
        if len(matched) > 30:
            raise ValueError("Experimental Binance sample is limited to 30 closed minutes")
        with httpx.Client(base_url="https://api.binance.com", timeout=10.0) as client:
            for row in matched:
                open_time = row["source_candle_open_time"]
                close_time = row["source_candle_close_time"]
                response = client.get("/api/v3/klines", params={
                    "symbol": row["symbol"], "interval": "1m",
                    "startTime": int(open_time.timestamp() * 1000),
                    "endTime": int(close_time.timestamp() * 1000), "limit": 1})
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or len(payload) != 1 or len(payload[0]) < 10:
                    kline_errors += 1
                    continue
                candle = payload[0]
                if int(candle[0]) != int(open_time.timestamp() * 1000) or int(candle[6]) != int(close_time.timestamp() * 1000):
                    kline_errors += 1
                    continue
                row.update(trade_flow_from_klines(candle[5], candle[9]))
                # The complete source kline is known only after its close.
                row["feature_cutoff_time"] = close_time
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, default=_json, ensure_ascii=False, allow_nan=False) + "\n" for row in matched), encoding="utf-8")
    times = sorted(q.observed_at for q in quotes)
    gap_minutes = [(right - left).total_seconds()/60 for left, right in zip(times, times[1:])]
    minute_duplicates = sum(max(0, count-1) for count in Counter(k for k in groups for _ in groups[k]).values())
    return {"observations": len(quotes), "distinct_observed_minutes": len(groups),
            "start_utc": times[0].isoformat() if times else None,
            "end_utc": times[-1].isoformat() if times else None,
            "median_interarrival_minutes": float(np.median(gap_minutes)) if gap_minutes else None,
            "max_interarrival_minutes": max(gap_minutes) if gap_minutes else None,
            "multi_quote_minutes": sum(len(group)>1 for group in groups.values()),
            "additional_quotes_same_minute": minute_duplicates,
            "exact_duplicate_quotes": sum(row["quote_duplicates"] for row in matched),
            "shadow_rows_with_closed_candle": len(matched),
            "unmatched_minutes": unmatched,
            "complete_l1_and_trade_flow_rows": sum(row["spread_bps"] is not None and row["trade_flow_imbalance"] is not None for row in matched),
            "zero_total_trade_volume_rows": sum(row["aggressive_buy_volume"] is not None and row["trade_flow_imbalance"] is None for row in matched),
            "kline_lookup_errors": kline_errors,
            "feature_cutoff_violations": sum(row["feature_cutoff_time"] > row["source_candle_close_time"] for row in matched),
            "shadow_path": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-binance-kline-flow", action="store_true",
                        help="Fetch at most 30 matching closed historical klines; no writes to production")
    args = parser.parse_args()
    print(json.dumps(build(with_kline_flow=args.with_binance_kline_flow), ensure_ascii=False, indent=2))
