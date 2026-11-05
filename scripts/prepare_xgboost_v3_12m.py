"""Prepare the final 12-month BTCUSDT Spot 1m robustness cohort.

Downloads only six missing official monthly ZIPs (2025-09 through 2026-02),
reuses verified 2026-03..09 Parquet/ZIPs, and preserves V2/V3 feature code.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_v2_dataset import (
    AUDIT as V2_AUDIT, CANONICAL as CANONICAL_6M, ROOT, _read_one,
    build_paired, temporal_splits, validate_canonical,
)
from scripts.build_v3_dataset import (
    AUDIT as V3_AUDIT, OUT as V3_6M, V3, build_v3_paired,
    load_extra_raw,
)
from scripts.download_binance_btcusdt_1m import (
    Archive, BASE, RAW, _fetch_one, _validate_zip,
)
from scripts.train_compare_xgboost_v2 import sha256

START = pd.Timestamp("2025-09-22T00:00:00Z")
END = pd.Timestamp("2026-09-21T23:59:00Z")
OLDER_END = pd.Timestamp("2026-03-01T00:00:00Z")
MANIFEST = ROOT / "reports/xgboost_v3_12m_source_manifest.json"
AUDIT = ROOT / "reports/xgboost_v3_12m_dataset_audit.json"
CANONICAL = ROOT / "data/processed/btcusdt_spot_1m_20250922_20260921.parquet"
PAIRED = ROOT / "data/processed/btcusdt_v3_12m_paired.parquet"
SPLIT_BOUNDS = {
    "TRAIN": ("2025-09-22", "2026-08-01"),
    "VALIDATION": ("2026-08-01", "2026-09-01"),
    "TEST": ("2026-09-01", "2026-09-16"),
    "FORWARD": ("2026-09-16", "2026-09-22"),
}


def missing_archives() -> list[Archive]:
    months = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
    return [Archive("monthly", month, f"BTCUSDT-1m-{month}.zip",
                    f"{BASE}/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{month}.zip")
            for month in months]


def download_missing() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    previous = {}
    if MANIFEST.exists():
        previous = {item["filename"]: item for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]}
    result = {}
    pending = []
    for archive in missing_archives():
        item = previous.get(archive.filename)
        path = RAW / archive.filename
        if item and item.get("sha256") and path.exists() and sha256(path) == item["sha256"]:
            _validate_zip(path, archive.filename)
            result[archive.filename] = {**item, "status": "REUSED_VERIFIED"}
        else:
            pending.append(archive)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_one, archive, RAW): archive for archive in pending}
        for future in as_completed(futures):
            archive = futures[future]
            result[archive.filename] = future.result()
            print(f"{archive.filename}: {result[archive.filename]['status']}", flush=True)
    entries = [result[archive.filename] for archive in missing_archives()]
    manifest = {"provider": "Binance Public Data / Data Vision", "market": "SPOT",
                "symbol": "BTCUSDT", "interval": "1m", "timestamp_unit": "microseconds",
                "requested_start_utc": START.isoformat(), "requested_end_utc": END.isoformat(),
                "note": "September 2025 monthly ZIP is cropped to Sep 22 onward; 2026-03..09 reused from the frozen 6m dataset.",
                "checked_at_utc": datetime.now(timezone.utc).isoformat(), "files": entries}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    if any("sha256" not in entry for entry in entries):
        raise RuntimeError("A required official archive was unavailable")
    return manifest


def build_dataset() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    old_v2 = json.loads(V2_AUDIT.read_text(encoding="utf-8"))
    old_v3 = json.loads(V3_AUDIT.read_text(encoding="utf-8"))
    if (sha256(CANONICAL_6M) != old_v2["canonical"]["sha256"] or
        sha256(V3_6M) != old_v3["v3_dataset"]["sha256"]):
        raise ValueError("Frozen six-month reference dataset changed")
    wanted = {archive.filename for archive in missing_archives()}
    entries = manifest["files"]
    if len(entries) != len(wanted) or {item["filename"] for item in entries} != wanted:
        raise ValueError("Incomplete older source manifest")
    older = pd.concat([_read_one(item, RAW) for item in entries], ignore_index=True)
    older = older.loc[(older.timestamp >= START) & (older.timestamp < OLDER_END)].reset_index(drop=True)
    recent = pd.read_parquet(CANONICAL_6M)
    canonical = pd.concat([older, recent], ignore_index=True)
    quality = validate_canonical(canonical)
    if (canonical.timestamp.iloc[0] != START or canonical.timestamp.iloc[-1] != END or
        quality["gap_count"] != 0 or quality["rows"] != 525_600):
        raise ValueError("12m source range, continuity or row count failed")
    extra_older, old_raw_quality = load_extra_raw(MANIFEST, RAW, expected_archives=missing_archives())
    extra_older = extra_older.loc[(extra_older.timestamp >= START) & (extra_older.timestamp < OLDER_END)].reset_index(drop=True)
    extra_recent, recent_raw_quality = load_extra_raw()
    extra = pd.concat([extra_older, extra_recent], ignore_index=True)
    v2, v2_cohort = build_paired(canonical)
    paired, v3_cohort = build_v3_paired(v2, canonical, extra)
    splits, split_audit = temporal_splits(paired, SPLIT_BOUNDS)
    old = pd.read_parquet(V3_6M)
    for split in ("VALIDATION", "TEST", "FORWARD"):
        bounds = old_v3["splits"][split]
        current = splits[split]
        reference = old.loc[(old.timestamp >= current.timestamp.iloc[0]) &
                            (old.timestamp <= current.timestamp.iloc[-1])]
        if (len(current) != bounds["rows"] or not current.timestamp.reset_index(drop=True).equals(reference.timestamp.reset_index(drop=True)) or
            not current.target.reset_index(drop=True).equals(reference.target.reset_index(drop=True))):
            raise ValueError(f"Recent {split} cohort differs from 6m")
    shared = paired.loc[paired.timestamp >= pd.Timestamp("2026-03-01T00:33:00Z")]
    old_aligned = old.loc[old.timestamp.isin(shared.timestamp)]
    if not shared.timestamp.reset_index(drop=True).equals(old_aligned.timestamp.reset_index(drop=True)):
        raise ValueError("Recent 6m timestamp alignment failed")
    feature_parity = {name: {"mean_absolute_difference": float(np.mean(np.abs(shared[name].to_numpy()-old_aligned[name].to_numpy()))),
                             "max_absolute_difference": float(np.max(np.abs(shared[name].to_numpy()-old_aligned[name].to_numpy())))}
                      for name in V3}
    CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(CANONICAL, index=False)
    paired.to_parquet(PAIRED, index=False)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": "official Binance Public Data Spot BTCUSDT 1m",
              "new_source_manifest": str(MANIFEST.relative_to(ROOT)),
              "reused_6m_source_manifest": "reports/xgboost_v2_6m_source_manifest.json",
              "new_archives": [item["filename"] for item in entries],
              "new_downloaded_count": sum(item["status"] == "DOWNLOADED" for item in entries),
              "new_source_zip_bytes": sum(item["bytes"] for item in entries),
              "reused_6m_canonical_sha256": old_v2["canonical"]["sha256"],
              "reused_6m_v3_sha256": old_v3["v3_dataset"]["sha256"],
              "canonical": {**quality, "path": str(CANONICAL.relative_to(ROOT)),
                            "sha256": sha256(CANONICAL), "bytes": CANONICAL.stat().st_size,
                            "exact_duplicates_removed": 0},
              "cohort": {**v3_cohort, "v2_rows": v2_cohort["labeled_rows"],
                         "path": str(PAIRED.relative_to(ROOT)),
                         "sha256": sha256(PAIRED), "bytes": PAIRED.stat().st_size},
              "splits": split_audit,
              "extra_quality": {"older": old_raw_quality, "recent": recent_raw_quality},
              "shared_6m_feature_parity": feature_parity,
              "recent_validation_test_forward_exactly_paired": True}
    AUDIT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(f"canonical={quality['rows']} cohort={len(paired)} gaps={quality['gap_count']} splits="+
          str({name: len(part) for name, part in splits.items()}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.download_only and args.build_only:
        raise ValueError("Choose only one action")
    if not args.build_only:
        download_missing()
    if not args.download_only:
        build_dataset()


if __name__ == "__main__":
    main()
