"""Read-only V1/V2 drift on persisted LIVE candles; never fits or selects a model.

Each LIVE row replays the production 120-consecutive-candle indicator window.
Only aggregated distributions are persisted; no raw Supabase rows or key.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame
from scripts.build_v2_dataset import AUDIT, EXPERIMENT, V1, V2, temporal_splits
from scripts.live_model_diagnostic import _fetch_all, _publishable_key, _require_public_read_key
from scripts.train_compare_xgboost_v2 import ROOT, feature_drift

OUT = ROOT / "reports/xgboost_v2_6m_live_drift.json"


def replay_features(candles: list[Candle]) -> tuple[pd.DataFrame, int]:
    rows = []
    omitted = 0
    for i in range(119, len(candles)):
        window = candles[i-119:i+1]
        if any((right.open_time-left.open_time).total_seconds() != 60
               for left, right in zip(window, window[1:])):
            omitted += 1
            continue
        feature = build_feature_frame(window).iloc[-1]
        close = window[-1].close
        item = {"timestamp": window[-1].open_time, "close": close}
        item.update({name: float(feature[name]) for name in FEATURE_COLUMNS})
        item["ema_distance"] = (close-item["ema_20"])/close
        for old, new in (("macd", "macd_relative"), ("macd_signal", "macd_signal_relative"),
                         ("macd_histogram", "macd_histogram_relative")):
            item[new] = item[old]/close
        rows.append(item)
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame[list(dict.fromkeys((*V1, *V2)))].to_numpy(dtype=float)).all():
        raise ValueError("No complete finite LIVE feature replay")
    return frame, omitted


def main() -> None:
    start = datetime.now(timezone.utc)
    reference = ROOT / "supabase/.temp/project-ref"
    project_ref = os.getenv("SUPABASE_PROJECT_REF") or reference.read_text(encoding="utf-8").strip()
    key = os.getenv("SUPABASE_KEY") or _publishable_key(project_ref)
    _require_public_read_key(key)
    url = f"https://{project_ref}.supabase.co"
    # Two hours of preceding candles permit exact 120-row replay from Sep 22.
    with httpx.Client(timeout=60) as client:
        rows, count = _fetch_all(client, url, key, "candles", [
            ("select", "symbol,interval,open_time,close_time,open,high,low,close,volume"),
            ("symbol", "eq.BTCUSDT"), ("interval", "eq.1m"),
            ("open_time", "gte.2026-09-21T22:00:00Z"),
            ("close_time", f"lte.{start.isoformat()}"), ("order", "open_time.asc")])
    candles = [Candle.model_validate(item) for item in rows]
    if len(candles) != count or len({c.open_time for c in candles}) != count:
        raise ValueError("Incomplete or duplicate LIVE candle cohort")
    frame, omitted = replay_features(candles)
    frame = frame.loc[frame.timestamp >= datetime(2026, 9, 22, tzinfo=timezone.utc)].copy()
    if frame.empty:
        raise ValueError("No post-historical LIVE feature rows")
    history = pd.read_parquet(EXPERIMENT)
    train = temporal_splits(history)[0]["TRAIN"]
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    report = {"cutoff_utc": start.isoformat(),
              "training_cohort_sha256": audit["cohort"]["sha256"],
              "source": "Supabase public candles; read only; production 120-candle feature replay",
              "raw_candles": count, "usable_live_rows": len(frame),
              "first_source_utc": frame.timestamp.iloc[0].isoformat(),
              "last_source_utc": frame.timestamp.iloc[-1].isoformat(),
              "omitted_for_missing_120_history": omitted,
              "drift": {"A": feature_drift(train, frame, V1),
                        "B": feature_drift(train, frame, V2)}}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(f"LIVE drift rows={len(frame)} omitted={omitted}; no labels or model fitting")


if __name__ == "__main__":
    main()
