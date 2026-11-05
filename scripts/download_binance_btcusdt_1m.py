"""Download verified official Binance Spot BTCUSDT 1m ZIP archives.

2026-03 through 2026-08 use monthly archives; 2026-09-01 through 09-21
use daily archives. Existing verified ZIPs are reused. Raw files stay outside
Git; the small provenance manifest contains names, URLs, sizes and hashes.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/binance"
MANIFEST = ROOT / "reports/xgboost_v2_6m_source_manifest.json"
BASE = "https://data.binance.vision/data/spot"
CSV_COLUMNS = 12


@dataclass(frozen=True)
class Archive:
    granularity: str
    period: str
    filename: str
    url: str


def planned_archives() -> list[Archive]:
    archives = []
    for month in range(3, 9):
        period = f"2026-{month:02d}"
        filename = f"BTCUSDT-1m-{period}.zip"
        archives.append(Archive("monthly", period, filename,
                                f"{BASE}/monthly/klines/BTCUSDT/1m/{filename}"))
    day = date(2026, 9, 1)
    while day <= date(2026, 9, 21):
        period = day.isoformat()
        filename = f"BTCUSDT-1m-{period}.zip"
        archives.append(Archive("daily", period, filename,
                                f"{BASE}/daily/klines/BTCUSDT/1m/{filename}"))
        day += timedelta(days=1)
    return archives


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_checksum(text: str, filename: str) -> str:
    tokens = text.strip().split()
    if len(tokens) != 2 or len(tokens[0]) != 64 or tokens[1].lstrip("*") != filename:
        raise ValueError(f"Malformed official checksum for {filename}")
    value = tokens[0].lower()
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Invalid SHA-256 for {filename}")
    return value


def _validate_zip(path: Path, filename: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != [filename.removesuffix(".zip") + ".csv"]:
            raise ValueError(f"Unexpected ZIP member in {filename}")
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC mismatch in {filename}")
        with archive.open(names[0]) as member:
            first = next(csv.reader(io.TextIOWrapper(member, encoding="utf-8")), None)
        if first is None or len(first) != CSV_COLUMNS:
            raise ValueError(f"Unexpected Binance kline schema in {filename}")
        if not first[0].isdigit():
            raise ValueError(f"Unexpected timestamp in {filename}")


def _fetch_one(archive: Archive, raw: Path) -> dict:
    target = raw / archive.filename
    checksum_url = archive.url + ".CHECKSUM"
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for attempt in range(3):
            try:
                checksum_response = client.get(checksum_url)
                if checksum_response.status_code == 404:
                    return {**asdict(archive), "status": "UNAVAILABLE", "reason": "CHECKSUM_404"}
                checksum_response.raise_for_status()
                expected = _expected_checksum(checksum_response.text, archive.filename)
                if target.exists() and _sha256(target) == expected:
                    _validate_zip(target, archive.filename)
                    return {**asdict(archive), "status": "REUSED", "sha256": expected,
                            "bytes": target.stat().st_size, "checksum_url": checksum_url}
                temporary = target.with_suffix(".zip.part")
                with client.stream("GET", archive.url) as response:
                    if response.status_code == 404:
                        return {**asdict(archive), "status": "UNAVAILABLE", "reason": "ZIP_404"}
                    response.raise_for_status()
                    with temporary.open("wb") as handle:
                        for block in response.iter_bytes(chunk_size=1024 * 1024):
                            handle.write(block)
                actual = _sha256(temporary)
                if actual != expected:
                    temporary.unlink(missing_ok=True)
                    raise ValueError(f"Official checksum mismatch for {archive.filename}")
                _validate_zip(temporary, archive.filename)
                os.replace(temporary, target)
                return {**asdict(archive), "status": "DOWNLOADED", "sha256": actual,
                        "bytes": target.stat().st_size, "checksum_url": checksum_url}
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    raise AssertionError("Unreachable")


def download(raw: Path = RAW, manifest: Path = MANIFEST, workers: int = 4) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("Use 1 to 4 workers")
    raw.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, item, raw): item for item in planned_archives()}
        for future in as_completed(futures):
            item = futures[future]
            try:
                entries[item.filename] = future.result()
            except Exception as exc:
                entries[item.filename] = {**asdict(item), "status": "FAILED",
                                          "reason": type(exc).__name__}
            snapshot = {"provider": "Binance Public Data / Data Vision", "market": "SPOT",
                        "symbol": "BTCUSDT", "interval": "1m", "timestamp_unit": "microseconds",
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "files": [entries[name] for name in sorted(entries)]}
            manifest.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
            print(f"{item.filename}: {entries[item.filename]['status']}", flush=True)
    if len(entries) != len(planned_archives()):
        raise RuntimeError("Incomplete source manifest")
    if any(entry["status"] == "FAILED" for entry in entries.values()):
        raise RuntimeError("Some archives failed; rerun to reuse verified ZIPs")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = download(workers=args.workers)
    print(f"verified={sum('sha256' in item for item in result['files'])} unavailable={sum(item['status']=='UNAVAILABLE' for item in result['files'])}")


if __name__ == "__main__":
    main()
