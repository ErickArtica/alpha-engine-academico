"""Resumable private Alpha Vantage NEWS_SENTIMENT backfill for offline research.

No production imports or writes. Raw responses are gitignored. The public manifest
contains only query windows, counts and hashes; API keys and articles never enter it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values

from alpha_engine.news.alpha_vantage import validate_payload
from scripts.analyze_hold_target import ROOT

RAW = ROOT / "data/experiments/alpha_vantage_news/raw"
MANIFEST = ROOT / "reports/alpha_vantage_download_manifest_v1.json"
START = datetime(2026, 8, 18, 19, 0, tzinfo=timezone.utc)
END = datetime(2026, 9, 17, 21, 0, tzinfo=timezone.utc)
LIMIT = 1000
ENDPOINT = "https://www.alphavantage.co/query"


def stamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("Window must be UTC")
    return value.strftime("%Y%m%dT%H%M")


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"endpoint": "NEWS_SENTIMENT", "ticker": "CRYPTO:BTC", "limit": LIMIT,
            "start_utc": START.isoformat(), "end_utc": END.isoformat(),
            "windows": {}, "api_calls_by_utc_date": {},
            "note": "Only status/count/checksum; raw articles remain in gitignored cache."}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    pending = MANIFEST.with_suffix(".json.tmp")
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(MANIFEST)


def window_id(start: datetime, end: datetime) -> str:
    return f"{stamp(start)}_{stamp(end)}"


def download(start: datetime, end: datetime, key: str, manifest: dict, *, daily_cap: int) -> dict:
    identifier = window_id(start, end)
    cached = manifest["windows"].get(identifier)
    if cached and cached["status"] in ("complete", "saturated"):
        raw_path = RAW / f"{identifier}.json"
        if not raw_path.exists() or hashlib.sha256(raw_path.read_bytes()).hexdigest() != cached["response_sha256"]:
            raise ValueError("Manifest/cache checksum mismatch")
        return cached
    day = datetime.now(timezone.utc).date().isoformat()
    if manifest["api_calls_by_utc_date"].get(day, 0) >= daily_cap:
        raise RuntimeError("LOCAL_DAILY_CALL_BUDGET_EXHAUSTED")
    params = {"function": "NEWS_SENTIMENT", "tickers": "CRYPTO:BTC",
              "time_from": stamp(start), "time_to": stamp(end),
              "sort": "EARLIEST", "limit": LIMIT, "apikey": key}
    # Never log request URLs or httpx exception messages: they may contain apikey.
    manifest["api_calls_by_utc_date"][day] = manifest["api_calls_by_utc_date"].get(day, 0) + 1
    save_manifest(manifest)
    try:
        with httpx.Client(timeout=45.0, follow_redirects=False) as client:
            response = client.get(ENDPOINT, params=params)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP_STATUS_{response.status_code}")
            payload = response.json()
        feed = validate_payload(payload)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ALPHA_VANTAGE_REQUEST_FAILED_{type(exc).__name__}") from None
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{identifier}.json"
    raw_path.write_bytes(raw)
    entry = {"time_from": stamp(start), "time_to": stamp(end),
             "status": "saturated" if len(feed) >= LIMIT else "complete",
             "articles": len(feed), "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
             "response_sha256": hashlib.sha256(raw).hexdigest()}
    manifest["windows"][identifier] = entry
    save_manifest(manifest)
    return entry


def backfill(start: datetime, end: datetime, key: str, manifest: dict, *, daily_cap: int) -> None:
    entry = download(start, end, key, manifest, daily_cap=daily_cap)
    if entry["status"] == "complete":
        return
    midpoint = start + (end - start) / 2
    midpoint = midpoint.replace(second=0, microsecond=0)
    if midpoint <= start or midpoint >= end:
        raise RuntimeError("SATURATED_MINUTE_WINDOW")
    backfill(start, midpoint, key, manifest, daily_cap=daily_cap)
    backfill(midpoint, end, key, manifest, daily_cap=daily_cap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Consume API calls; default is read-only plan")
    parser.add_argument("--daily-cap", type=int, default=24, help="Conservative cap below free 25/day")
    args = parser.parse_args()
    manifest = load_manifest()
    if not args.execute:
        print(json.dumps({"range": [START.isoformat(), END.isoformat()], "limit": LIMIT,
                          "cached_windows": len(manifest["windows"]),
                          "estimated_first_call": 1, "if_saturated": "split recursively"}))
        return
    key = dotenv_values(ROOT / ".env").get("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise SystemExit("ALPHA_VANTAGE_API_KEY is not configured locally")
    try:
        backfill(START, END, key, manifest, daily_cap=args.daily_cap)
    except (RuntimeError, ValueError) as exc:
        print(f"BACKFILL_INCOMPLETE: {type(exc).__name__}: {exc}")
        raise SystemExit(2) from None
    print(json.dumps({"status": "complete", "windows": len(manifest["windows"]),
                      "api_calls_by_utc_date": manifest["api_calls_by_utc_date"]}))


if __name__ == "__main__":
    main()
