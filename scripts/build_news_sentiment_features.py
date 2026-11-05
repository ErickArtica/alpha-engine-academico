"""Offline, publication-time-safe BTC sentiment features from private AV cache.

Historical publication timestamps are only a proxy for availability. No historical
first-seen timestamp is fabricated. Never used by the production prediction path.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.news.alpha_vantage import article_key, parse_article, parse_time
from scripts.analyze_hold_target import ROOT, write_json
from scripts.analyze_horizons import build_horizon_frames, common_frames
from scripts.fetch_alpha_vantage_news import MANIFEST, RAW, START, END

FEATURES = ("btc_sentiment_15m", "btc_sentiment_60m", "news_count_15m",
            "news_count_60m", "minutes_since_last_news")
PRIVATE = ROOT / "data/experiments/alpha_vantage_news"
AUDIT = ROOT / "reports/alpha_vantage_data_audit_v1.json"
FEATURE_REPORT = ROOT / "reports/news_sentiment_features_v1.json"


def load_articles() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    windows = [entry for entry in manifest["windows"].values() if entry["status"] == "complete"]
    saturated = [entry for entry in manifest["windows"].values() if entry["status"] == "saturated"]
    if not windows or (len(windows) == 1 and saturated):
        raise ValueError("No completed news windows")
    # A saturated parent must have complete child coverage; current one-window
    # backfill has no such parent. Reject partial manifests for this frozen run.
    if saturated:
        raise ValueError("Verify recursive child coverage before analysis")
    seen = {}
    duplicate_count = 0
    invalid = Counter()
    raw_count = 0
    for entry in windows:
        key = f"{entry['time_from']}_{entry['time_to']}"
        path = RAW / f"{key}.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["response_sha256"]:
            raise ValueError("Private news cache checksum mismatch")
        payload = json.loads(raw)
        feed = payload.get("feed")
        if not isinstance(feed, list) or len(feed) != entry["articles"]:
            raise ValueError("Manifest article count mismatch")
        for item in feed:
            raw_count += 1
            try:
                article = parse_article(item)
            except (ValueError, KeyError, TypeError) as exc:
                invalid[type(exc).__name__] += 1
                continue
            if article["key"] in seen:
                duplicate_count += 1
                continue
            seen[article["key"]] = article
    articles = pd.DataFrame(seen.values()).sort_values("published_at_utc").reset_index(drop=True)
    if articles.empty or articles["published_at_utc"].dt.tz is None:
        raise ValueError("No valid UTC BTC news")
    if articles["published_at_utc"].min().to_pydatetime() < START or articles["published_at_utc"].max().to_pydatetime() > END:
        raise ValueError("News outside requested historical window")
    by_day = articles["published_at_utc"].dt.strftime("%Y-%m-%d").value_counts().sort_index()
    days = pd.date_range(START.date(), END.date(), freq="D", tz="UTC")
    audit = {"raw_articles": raw_count, "unique_valid_articles": len(articles),
             "duplicate_articles": duplicate_count, "invalid_articles_by_error": dict(invalid),
             "first_published_utc": articles["published_at_utc"].min().isoformat(),
             "last_published_utc": articles["published_at_utc"].max().isoformat(),
             "days_with_news": int(len(by_day)),
             "days_without_news": [day.strftime("%Y-%m-%d") for day in days if day.strftime("%Y-%m-%d") not in by_day],
             "sources": articles["source"].value_counts().to_dict(),
             "daily_article_count": by_day.to_dict(),
             "btc_specific_score_coverage": 1.0,
             "btc_relevance_coverage": 1.0,
             "historical_first_observed_available": False,
             "availability_basis": "publication timestamp only; publication-to-provider delay unknown",
             "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
    return articles, audit


def build_features(source_close: pd.Series | pd.DatetimeIndex, articles: pd.DataFrame) -> pd.DataFrame:
    closes = pd.DatetimeIndex(source_close)
    if closes.tz is None or not closes.is_monotonic_increasing or not closes.is_unique:
        raise ValueError("Source close times must be ordered unique UTC")
    if any(timestamp.utcoffset() != timedelta(0) for timestamp in closes):
        raise ValueError("Source close timezone must be UTC")
    news = articles.sort_values("published_at_utc")
    times = pd.DatetimeIndex(news["published_at_utc"])
    if times.tz is None or any(timestamp.utcoffset() != timedelta(0) for timestamp in times):
        raise ValueError("News timestamps must be UTC")
    news_ns = times.asi8
    scores = news["score"].to_numpy(dtype=float)
    relevance = news["relevance"].to_numpy(dtype=float)
    values = {name: [] for name in FEATURES}
    cutoffs = []
    for close in closes:
        right = int(np.searchsorted(news_ns, close.value, side="right"))
        last = times[right - 1] if right else pd.NaT
        cutoffs.append(last)
        values["minutes_since_last_news"].append(float((close - last).total_seconds() / 60) if right else np.nan)
        for minutes in (15, 60):
            left = int(np.searchsorted(news_ns, close.value - minutes * 60 * 1_000_000_000, side="right"))
            window_scores = scores[left:right]
            window_relevance = relevance[left:right]
            values[f"news_count_{minutes}m"].append(int(right - left))
            values[f"btc_sentiment_{minutes}m"].append(
                float(np.average(window_scores, weights=window_relevance)) if right > left else np.nan)
    result = pd.DataFrame(values, index=closes)
    result["sentiment_cutoff_time"] = pd.to_datetime(cutoffs, utc=True)
    valid = result["sentiment_cutoff_time"].notna().to_numpy()
    if (result["sentiment_cutoff_time"].to_numpy()[valid] > closes.to_numpy()[valid]).any():
        raise ValueError("Future news leaked into sentiment features")
    return result


def main() -> None:
    articles, audit = load_articles()
    frames, _ = build_horizon_frames()
    common = common_frames(frames)
    source = common[1]
    features = build_features(source["source_candle_close_time"], articles)
    features.index = source.index
    if not features.index.equals(common[5].index):
        raise ValueError("H1/H5 news common cohort mismatch")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    path = PRIVATE / "sentiment_common_cohort.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for t, row in features.iterrows():
            item = {"source_candle_open_time": t.isoformat(),
                    "source_candle_close_time": source.at[t, "source_candle_close_time"].isoformat(),
                    "sentiment_cutoff_time": row["sentiment_cutoff_time"].isoformat() if pd.notna(row["sentiment_cutoff_time"]) else None,
                    "features": {name: (float(row[name]) if pd.notna(row[name]) else None) for name in FEATURES}}
            handle.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
    audit["source_rows"] = len(source)
    audit["news_density_per_source_window"] = {
        str(m): {"rows_with_news": int((features[f"news_count_{m}m"] > 0).sum()),
                 "fraction": float((features[f"news_count_{m}m"] > 0).mean())} for m in (15, 60)}
    audit["minutes_with_publication"] = int(articles["published_at_utc"].dt.floor("min").nunique())
    audit["feature_dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"features": {name: {"window_minutes": 15 if "15m" in name else 60 if "60m" in name else None,
                                   "coverage": float(features[name].notna().mean()),
                                   "missing_semantics": "NaN when no eligible news" if "sentiment" in name or name == "minutes_since_last_news" else "zero means no news",
                                   "causal_cutoff": "published_at <= source_candle_close_time"} for name in FEATURES},
              "rows": len(features), "dataset_sha256": audit["feature_dataset_sha256"],
              "no_news_60m_rows": int((features["news_count_60m"] == 0).sum()),
              "source_range_utc": [source.index[0].isoformat(), source.index[-1].isoformat()],
              "research_only": True}
    write_json(AUDIT, audit)
    write_json(FEATURE_REPORT, report)
    print(json.dumps({"articles": len(articles), "rows": len(features),
                      "coverage_15m": audit["news_density_per_source_window"]["15"]["fraction"],
                      "coverage_60m": audit["news_density_per_source_window"]["60"]["fraction"]}))


if __name__ == "__main__":
    main()
