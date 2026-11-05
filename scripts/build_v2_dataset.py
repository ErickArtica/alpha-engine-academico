"""Build the causal, paired V1/V2 H1 experiment cohort from verified Binance ZIPs.

The original indicator formulas are mirrored vectorially and checked against
build_feature_frame in tests. Gaps start a new indicator history; no candles
are synthesized. Parquet stays outside Git.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import zipfile

from alpha_engine.features import FEATURE_COLUMNS
from scripts.download_binance_btcusdt_1m import MANIFEST, RAW, planned_archives

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/processed/btcusdt_spot_1m_20260301_20260921.parquet"
EXPERIMENT = ROOT / "data/processed/btcusdt_v2_6m_paired.parquet"
AUDIT = ROOT / "reports/xgboost_v2_6m_dataset_audit.json"
V1 = tuple(FEATURE_COLUMNS)
V2 = ("return_1m", "rsi_14", "ema_distance", "macd_relative",
      "macd_signal_relative", "macd_histogram_relative",
      "rolling_volatility", "volume_change")
BINANCE_COLUMNS = ("open_time_us", "open", "high", "low", "close", "volume",
                   "close_time_us", "quote_volume", "trade_count",
                   "taker_buy_base", "taker_buy_quote", "ignore")
SPLIT_BOUNDS = {
    "TRAIN": ("2026-03-01", "2026-08-01"),
    "VALIDATION": ("2026-08-01", "2026-09-01"),
    "TEST": ("2026-09-01", "2026-09-16"),
    "FORWARD": ("2026-09-16", "2026-09-22"),
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_one(entry: dict, raw: Path) -> pd.DataFrame:
    path = raw / entry["filename"]
    if not path.exists() or _hash(path) != entry["sha256"]:
        raise ValueError(f"Missing or changed verified archive: {entry['filename']}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != [entry["filename"].removesuffix(".zip") + ".csv"]:
            raise ValueError(f"Unexpected archive member: {entry['filename']}")
        with archive.open(names[0]) as member:
            frame = pd.read_csv(member, header=None, names=BINANCE_COLUMNS,
                                usecols=list(range(7)), dtype={"open_time_us": "int64",
                                "close_time_us": "int64", "open": "float64",
                                "high": "float64", "low": "float64", "close": "float64",
                                "volume": "float64"})
    if frame.empty:
        raise ValueError(f"Empty archive: {entry['filename']}")
    frame["timestamp"] = pd.to_datetime(frame.pop("open_time_us"), unit="us", utc=True)
    frame["close_time"] = pd.to_datetime(frame.pop("close_time_us"), unit="us", utc=True)
    period = frame["timestamp"].dt.strftime("%Y-%m" if entry["granularity"] == "monthly" else "%Y-%m-%d")
    if not period.eq(entry["period"]).all():
        raise ValueError(f"Archive has rows outside its period: {entry['filename']}")
    return frame[["timestamp", "open", "high", "low", "close", "volume", "close_time"]]


def validate_canonical(frame: pd.DataFrame) -> dict:
    if frame.empty or not isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype):
        raise ValueError("UTC timestamp column required")
    if str(frame["timestamp"].dt.tz) != "UTC" or not isinstance(frame["close_time"].dtype, pd.DatetimeTZDtype) or str(frame["close_time"].dt.tz) != "UTC":
        raise ValueError("Open and close timestamps must be UTC")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("Canonical timestamps must be increasing and unique")
    if (frame["timestamp"] != frame["timestamp"].dt.floor("min")).any():
        raise ValueError("Open time must align to a UTC minute")
    if (frame["close_time"] - frame["timestamp"] != pd.Timedelta(minutes=1) - pd.Timedelta(microseconds=1)).any():
        raise ValueError("Unexpected Binance 1m close time")
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.isnan(values).any():
        raise ValueError("Nonfinite OHLCV")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("Invalid price or negative volume")
    if (frame.high < frame[["open", "low", "close"]].max(axis=1)).any() or (
        frame.low > frame[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC")
    gap_minutes = frame.timestamp.diff().dt.total_seconds().div(60).iloc[1:]
    gaps = gap_minutes[gap_minutes > 1]
    if (gap_minutes < 1).any():
        raise ValueError("Non-increasing timestamps")
    return {"rows": int(len(frame)), "first_utc": frame.timestamp.iloc[0].isoformat(),
            "last_utc": frame.timestamp.iloc[-1].isoformat(),
            "gap_count": int(len(gaps)), "missing_minutes": int((gaps - 1).sum()),
            "largest_gap_minutes": float(gaps.max()) if len(gaps) else 0.0}


def load_canonical(manifest_path: Path = MANIFEST, raw: Path = RAW) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned = {item.filename for item in planned_archives()}
    entries = manifest["files"]
    if {entry["filename"] for entry in entries} != planned:
        raise ValueError("Source manifest differs from planned archive range")
    missing = [entry["filename"] for entry in entries if "sha256" not in entry]
    if missing:
        raise ValueError(f"Official archives missing; no synthetic fill: {missing}")
    frames = [_read_one(entry, raw) for entry in sorted(entries,key=lambda row:row["filename"])]
    frame = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    original_rows = len(frame)
    frame = frame.drop_duplicates()
    duplicates_removed = original_rows - len(frame)
    if frame.timestamp.duplicated().any():
        raise ValueError("Conflicting duplicate Binance timestamps")
    frame = frame.reset_index(drop=True)
    validation = validate_canonical(frame)
    validation["exact_duplicates_removed"] = int(duplicates_removed)
    return frame, validation


def _indicators(segment: pd.DataFrame) -> pd.DataFrame:
    """Vectorized copy of production formulas; segment begins after any gap."""
    result = segment.copy()
    close = result.close.astype("float64")
    volume = result.volume.astype("float64")
    result["previous_close"] = close.shift(1)
    result["return_1m"] = close.pct_change(fill_method=None)
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    strength = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + strength)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    result["rsi_14"] = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    result["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = ema12 - ema26
    result["macd_signal"] = result.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_histogram"] = result.macd - result.macd_signal
    result["rolling_volatility"] = result.return_1m.rolling(window=20,min_periods=20).std(ddof=0)
    result["volume_change"] = volume.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan)
    result["ema_distance"] = (close - result.ema_20) / close
    for absolute, relative in (("macd","macd_relative"),
                               ("macd_signal","macd_signal_relative"),
                               ("macd_histogram","macd_histogram_relative")):
        result[relative] = result[absolute] / close
    result["target_open_time"] = result.timestamp.shift(-1)
    next_close = close.shift(-1)
    valid_next = result.target_open_time.eq(result.timestamp + pd.Timedelta(minutes=1))
    result["target"] = pd.Series(np.where(valid_next, (next_close > close).astype("int8"), -1),index=result.index,dtype="int8")
    return result


def build_paired(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_canonical(frame)
    segment_id = frame.timestamp.diff().ne(pd.Timedelta(minutes=1)).cumsum()
    enriched = pd.concat([_indicators(part) for _,part in frame.groupby(segment_id,sort=True)],ignore_index=True)
    valid = enriched.target.ge(0) & np.isfinite(enriched[list(dict.fromkeys((*V1,*V2)))].to_numpy(dtype=float)).all(axis=1)
    usable = enriched.loc[valid].copy()
    if usable.empty:
        raise ValueError("No causal labeled rows")
    usable = usable.reset_index(drop=True)
    if not usable.target_open_time.eq(usable.timestamp + pd.Timedelta(minutes=1)).all():
        raise ValueError("H1 target alignment failed")
    if not np.isfinite(usable[list(dict.fromkeys((*V1,*V2)))].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite model features")
    return usable, {"raw_rows":len(frame),"labeled_rows":len(usable),
                    "warmup_or_missing_target_rows":len(frame)-len(usable),
                    "first_source_utc":usable.timestamp.iloc[0].isoformat(),
                    "last_source_utc":usable.timestamp.iloc[-1].isoformat(),
                    "feature_set_v1":list(V1),"feature_set_v2":list(V2)}


def temporal_splits(
    frame: pd.DataFrame, bounds: dict[str, tuple[str, str]] | None = None,
) -> tuple[dict[str,pd.DataFrame],dict]:
    if not frame.timestamp.is_monotonic_increasing or frame.timestamp.duplicated().any():
        raise ValueError("Ordered, unique cohort required")
    splits = {}
    for name,(lower,upper) in (bounds if bounds is not None else SPLIT_BOUNDS).items():
        start = pd.Timestamp(lower,tz="UTC")
        end = pd.Timestamp(upper,tz="UTC")
        part = frame.loc[(frame.timestamp >= start)&(frame.timestamp < end)].copy()
        if part.empty:
            raise ValueError(f"Empty {name} split")
        splits[name] = part
    for left,right in (("TRAIN","VALIDATION"),("VALIDATION","TEST"),("TEST","FORWARD")):
        if splits[left].target_open_time.iloc[-1] >= splits[right].timestamp.iloc[0]:
            splits[left] = splits[left].iloc[:-1].copy()
        if splits[left].empty or splits[left].target_open_time.max() >= splits[right].timestamp.min():
            raise ValueError(f"H1 label leakage at {left}/{right} boundary")
    if sum(len(part) for part in splits.values()) > len(frame):
        raise ValueError("Overlapping splits")
    audit = {name:{"rows":int(len(part)),"first_source_utc":part.timestamp.iloc[0].isoformat(),
                   "last_source_utc":part.timestamp.iloc[-1].isoformat(),
                   "up_fraction":float(part.target.mean()),"down_fraction":float(1-part.target.mean())}
             for name,part in splits.items()}
    return splits,audit


def main() -> None:
    canonical, quality = load_canonical()
    paired, cohort = build_paired(canonical)
    splits, split_audit = temporal_splits(paired)
    CANONICAL.parent.mkdir(parents=True,exist_ok=True)
    canonical.to_parquet(CANONICAL,index=False)
    paired.to_parquet(EXPERIMENT,index=False)
    report = {"created_at_utc":datetime.now(timezone.utc).isoformat(),
              "source_manifest":str(MANIFEST.relative_to(ROOT)),
              "official_source":"https://data.binance.vision/data/spot/",
              "canonical":{**quality,"path":str(CANONICAL.relative_to(ROOT)),
                           "sha256":_hash(CANONICAL),"bytes":CANONICAL.stat().st_size},
              "cohort":{**cohort,"path":str(EXPERIMENT.relative_to(ROOT)),
                        "sha256":_hash(EXPERIMENT),"bytes":EXPERIMENT.stat().st_size},
              "splits":split_audit,
              "definitions":{"target":"UP iff close(t+1m)>close(t); ties DOWN",
                             "ema_distance":"(close-ema_20)/close",
                             "macd_relative":"macd/close",
                             "macd_signal_relative":"macd_signal/close",
                             "macd_histogram_relative":"macd_histogram/close"},
              "library_versions":{"pandas":pd.__version__,"pyarrow":pyarrow.__version__}}
    AUDIT.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(f"canonical={quality['rows']} cohort={cohort['labeled_rows']} gaps={quality['gap_count']} duplicates={quality['exact_duplicates_removed']}")
    print("splits="+str({name:len(part) for name,part in splits.items()}))


if __name__ == "__main__":
    main()
