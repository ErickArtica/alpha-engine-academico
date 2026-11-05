"""Build an exactly paired V2/V3 H1 cohort from verified local Binance ZIPs.

The V2 Parquet, target, timestamps and split boundaries remain frozen. This
script only adds causal candle, activity and taker-flow columns for V3.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.features.v3 import V3_FEATURE_COLUMNS, _zscore, build_v3_feature_frame

from scripts.build_v2_dataset import AUDIT as V2_AUDIT, CANONICAL, EXPERIMENT as V2_DATASET, BINANCE_COLUMNS, ROOT, temporal_splits, validate_canonical
from scripts.download_binance_btcusdt_1m import Archive, MANIFEST, RAW, planned_archives
from scripts.train_compare_xgboost_v2 import sha256

OUT = ROOT / "data/processed/btcusdt_v3_market_flow_6m_paired.parquet"
AUDIT = ROOT / "reports/xgboost_v3_market_flow_6m_dataset_audit.json"
V3 = V3_FEATURE_COLUMNS
WINDOW = 20
FEATURE_DEFINITIONS = {
    "return_1m": ("close[t]/close[t-1]-1", "2 closes", "one-minute return"),
    "return_3m": ("close[t]/close[t-3]-1", "4 closes", "three-minute momentum"),
    "return_5m": ("close[t]/close[t-5]-1", "6 closes", "five-minute momentum"),
    "return_15m": ("close[t]/close[t-15]-1", "16 closes", "fifteen-minute momentum"),
    "rsi_14": ("production RSI(14)", "14 deltas", "relative price strength"),
    "ema_distance": ("(close[t]-ema_20[t])/close[t]", "EMA20", "price relative to EMA"),
    "macd_relative": ("macd[t]/close[t]", "EMA12/26", "relative MACD"),
    "macd_signal_relative": ("macd_signal[t]/close[t]", "MACD signal9", "relative MACD signal"),
    "macd_histogram_relative": ("macd_histogram[t]/close[t]", "MACD signal9", "relative MACD gap"),
    "candle_range": ("(high[t]-low[t])/close[t]", "current candle", "total intrabar range"),
    "body_return": ("(close[t]-open[t])/open[t]", "current candle", "signed candle body"),
    "upper_wick": ("(high[t]-max(open[t],close[t]))/close[t]", "current candle", "upper rejection"),
    "lower_wick": ("(min(open[t],close[t])-low[t])/close[t]", "current candle", "lower rejection"),
    "rolling_volatility": ("production std(return_1m,20,ddof=0)", "20 returns", "recent volatility"),
    "volume_change": ("volume[t]/volume[t-1]-1", "2 candles", "volume change"),
    "volume_zscore": ("(volume[t]-mean(volume[t-19:t+1]))/std(population)", "20 candles incl. t", "unusual base volume"),
    "trades_zscore": ("(trades[t]-mean(trades[t-19:t+1]))/std(population)", "20 candles incl. t", "unusual trade count"),
    "taker_buy_ratio": ("taker_buy_base[t]/volume[t]", "current candle", "share of base volume from taker buyers"),
}


def load_extra_raw(
    manifest_path: Path = MANIFEST, raw_dir: Path = RAW,
    expected_archives: list[Archive] | None = None,
) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    expected = {archive.filename for archive in (expected_archives if expected_archives is not None else planned_archives())}
    if len(entries) != len(expected) or {entry["filename"] for entry in entries} != expected:
        raise ValueError("V2 source archive set changed")
    frames = []
    column_counts = {}
    for entry in entries:
        path = raw_dir / entry["filename"]
        if not path.exists() or sha256(path) != entry.get("sha256"):
            raise ValueError(f"Missing or changed verified ZIP: {entry['filename']}")
        with zipfile.ZipFile(path) as archive:
            member_name = entry["filename"].removesuffix(".zip") + ".csv"
            if archive.namelist() != [member_name] or archive.testzip() is not None:
                raise ValueError(f"Invalid ZIP: {entry['filename']}")
            with archive.open(member_name) as member:
                first = next(csv.reader(io.TextIOWrapper(member, encoding="utf-8")), None)
            count = len(first) if first else 0
            column_counts[entry["filename"]] = count
            if count != len(BINANCE_COLUMNS):
                raise ValueError(f"Expected 12 Spot kline fields in {entry['filename']}; got {count}")
            with archive.open(member_name) as member:
                part = pd.read_csv(member, header=None, names=BINANCE_COLUMNS,
                                   usecols=[0, 5, 7, 8, 9],
                                   dtype={"open_time_us":"int64", "volume":"float64",
                                          "quote_volume":"float64", "trade_count":"int64",
                                          "taker_buy_base":"float64"})
        part["timestamp"] = pd.to_datetime(part.pop("open_time_us"), unit="us", utc=True)
        period = part.timestamp.dt.strftime("%Y-%m" if entry["granularity"] == "monthly" else "%Y-%m-%d")
        if not period.eq(entry["period"]).all():
            raise ValueError(f"Archive rows outside stated period: {entry['filename']}")
        frames.append(part)
    extra = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if extra.timestamp.duplicated().any() or not extra.timestamp.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted raw timestamps")
    numeric = extra[["volume", "quote_volume", "trade_count", "taker_buy_base"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (extra[["volume", "quote_volume", "trade_count", "taker_buy_base"]] < 0).any().any():
        raise ValueError("Nonfinite or negative raw activity")
    excess = extra.taker_buy_base - extra.volume
    if (excess > 1e-8).any():
        raise ValueError("Taker-buy base volume exceeds total base volume")
    return extra, {"raw_field_count_by_file": column_counts, "rows": len(extra),
                   "zero_volume": int(extra.volume.eq(0).sum()),
                   "zero_trade_count": int(extra.trade_count.eq(0).sum()),
                   "max_taker_buy_excess": float(excess.max()),
                   "available_raw_fields": list(BINANCE_COLUMNS)}


def make_v3_columns(canonical: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    validate_canonical(canonical)
    if len(canonical) != len(extra) or not canonical.timestamp.equals(extra.timestamp):
        raise ValueError("Raw additional fields do not align 1:1 with canonical candles")
    if not np.allclose(canonical.volume.to_numpy(), extra.volume.to_numpy(), atol=1e-9, rtol=1e-12):
        raise ValueError("Raw volume differs from V2 canonical volume")
    raw = canonical.assign(trade_count=extra.trade_count,
                           taker_buy_base=extra.taker_buy_base)
    features = build_v3_feature_frame(raw)
    result = pd.DataFrame({"timestamp": canonical.timestamp,
                           "trade_count": extra.trade_count,
                           "taker_buy_base": extra.taker_buy_base,
                           "quote_volume": extra.quote_volume})
    for name in V3:
        result[name] = features[name]
    return result


def build_v3_paired(v2: pd.DataFrame, canonical: pd.DataFrame, extra: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = make_v3_columns(canonical, extra)
    if len(features) != len(canonical) or features.timestamp.duplicated().any():
        raise ValueError("Nonunique V3 features")
    result = v2.drop(columns=list(set(V3).intersection(v2.columns))).merge(
        features, on="timestamp", how="left", validate="one_to_one", sort=False
    )
    if len(result) != len(v2) or not result.timestamp.equals(v2.timestamp):
        raise ValueError("V3 changed the V2 cohort")
    if not result.target.equals(v2.target) or not result.target_open_time.equals(v2.target_open_time):
        raise ValueError("V3 changed H1 target alignment")
    if not np.isfinite(result[list(V3)].to_numpy(dtype=float)).all():
        raise ValueError("V3 has unexpected NaN/Inf on frozen V2 cohort")
    if not result.taker_buy_ratio.between(0, 1 + 1e-12).all():
        raise ValueError("Taker buy ratio outside [0,1]")
    return result, {"rows": len(result), "first_source_utc": result.timestamp.iloc[0].isoformat(),
                    "last_source_utc": result.timestamp.iloc[-1].isoformat(),
                    "missing_v3_features": int(result[list(V3)].isna().sum().sum()),
                    "feature_set_v3": list(V3),
                    "same_cohort_as_v2": True}


def main() -> None:
    v2_audit = json.loads(V2_AUDIT.read_text(encoding="utf-8"))
    if sha256(CANONICAL) != v2_audit["canonical"]["sha256"] or sha256(V2_DATASET) != v2_audit["cohort"]["sha256"]:
        raise ValueError("V2 dataset fingerprints changed; refusing silent rebuild")
    canonical = pd.read_parquet(CANONICAL)
    v2 = pd.read_parquet(V2_DATASET)
    extra, raw_quality = load_extra_raw()
    result, cohort = build_v3_paired(v2, canonical, extra)
    splits, split_audit = temporal_splits(result)
    if split_audit != v2_audit["splits"]:
        raise ValueError("V3 changed frozen split boundaries or counts")
    quote_ratio = extra.quote_volume / (extra.volume * canonical.close).replace(0, np.nan)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT, index=False)
    audit = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
             "source_manifest": str(MANIFEST.relative_to(ROOT)),
             "v2_canonical_sha256": v2_audit["canonical"]["sha256"],
             "v2_cohort_sha256": v2_audit["cohort"]["sha256"],
             "v3_dataset": {**cohort, "path": str(OUT.relative_to(ROOT)),
                            "sha256": sha256(OUT), "bytes": OUT.stat().st_size},
             "splits": split_audit, "raw_quality": raw_quality,
             "quote_volume_to_base_times_close": {
                 "median": float(quote_ratio.median()),
                 "p05": float(quote_ratio.quantile(.05)),
                 "p95": float(quote_ratio.quantile(.95))},
             "redundancy": "taker_imbalance = 2*taker_buy_ratio - 1; omitted",
             "definitions": {name: {"formula": item[0], "window": item[1], "meaning": item[2]}
                             for name, item in FEATURE_DEFINITIONS.items()}}
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(f"v3_rows={len(result)} raw_rows={len(extra)} zero_volume={raw_quality['zero_volume']} split_counts="+
          str({name: len(part) for name, part in splits.items()}))


if __name__ == "__main__":
    main()
