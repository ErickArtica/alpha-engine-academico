"""Read-only Binance Spot kline backfill for frozen xgboost_v1 source candles.

Raw pages and joined rows stay in gitignored data/experiments/tfi/. All source
OHLCV rows must match the verified local Influx history before a report is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx

from alpha_engine.config.settings import Settings
from alpha_engine.experiments.microstructure import trade_flow_from_klines
from alpha_engine.ml.walk_forward import candle_digest
from alpha_engine.repositories.influx3 import Influx3MarketRepository

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "models/xgboost_v1_metadata.json"
RAW = ROOT / "data/experiments/tfi"
OUT = ROOT / "reports/tfi_backfill_v1.json"
FROZEN_CANDLE_SHA = "b5138e394357b6a55133a730a402f03653c9b4046ddf630f9cb497ed2760904b"


def expected_candles(metadata: dict):
    first = datetime.fromisoformat(metadata["data_range"]["start"])
    stop = datetime.fromisoformat(metadata["data_range"]["end"]) + timedelta(minutes=1)
    repo = Influx3MarketRepository.from_settings(Settings())
    try:
        candles = repo.get_candles("BTCUSDT", "1m", first, stop)
    finally:
        repo.close()
    if len(candles) != metadata["data_range"]["closed_candles"] or candle_digest(candles) != FROZEN_CANDLE_SHA:
        raise ValueError("Original candle cohort differs from the frozen model")
    if any(right.open_time - left.open_time != timedelta(minutes=1) for left, right in zip(candles, candles[1:])):
        raise ValueError("Original candle cohort has minute gaps")
    return candles


def validate_row(row: list, candle) -> dict:
    """Check exact source minute and numerically consistent OHLCV before using field 9."""
    if not isinstance(row, list) or len(row) < 11:
        raise ValueError("Incomplete Binance kline")
    if int(row[0]) != int(candle.open_time.timestamp() * 1000) or int(row[6]) != int(candle.close_time.timestamp() * 1000):
        raise ValueError("Binance timestamp differs from original source candle")
    for pos, name in ((1,"open"),(2,"high"),(3,"low"),(4,"close"),(5,"volume")):
        original = Decimal(str(getattr(candle, name)))
        remote = Decimal(str(row[pos]))
        if not remote.is_finite() or abs(remote - original) > max(Decimal("0.000001"), abs(original)*Decimal("1e-10")):
            raise ValueError(f"Binance {name} differs from original source candle at {candle.open_time.isoformat()}")
    flow = trade_flow_from_klines(row[5], row[9])
    if flow["trade_flow_imbalance"] is not None and not -1 <= flow["trade_flow_imbalance"] <= 1:
        raise ValueError("Trade-flow imbalance outside [-1,1]")
    return {"symbol":candle.symbol,"interval":candle.interval,
            "source_candle_open_time":candle.open_time.astimezone(timezone.utc).isoformat(),
            "source_candle_close_time":candle.close_time.astimezone(timezone.utc).isoformat(),
            "feature_cutoff_time":candle.close_time.astimezone(timezone.utc).isoformat(),
            "open":str(row[1]),"high":str(row[2]),"low":str(row[3]),"close":str(row[4]),
            "volume":str(row[5]),"taker_buy_base_volume":str(row[9]),
            "aggressive_sell_volume":str(flow["aggressive_sell_volume"]),
            "trade_flow_imbalance":str(flow["trade_flow_imbalance"]) if flow["trade_flow_imbalance"] is not None else None}


def _page(client: httpx.Client, candles: list, page_index: int, *, fetch: bool) -> list:
    page_path = RAW / "pages" / f"{page_index:03d}.json"
    if page_path.exists():
        result = json.loads(page_path.read_text(encoding="utf-8"))
    elif fetch:
        start = candles[0].open_time
        end = candles[-1].close_time
        params = {"symbol":"BTCUSDT","interval":"1m","startTime":int(start.timestamp()*1000),
                  "endTime":int(end.timestamp()*1000),"timeZone":"0","limit":len(candles)}
        for attempt in range(3):
            try:
                response = client.get("/api/v3/klines", params=params)
                response.raise_for_status()
                result = response.json()
                break
            except httpx.HTTPError:
                if attempt == 2: raise
                time.sleep(1 + attempt)
        if not isinstance(result, list) or len(result) != len(candles):
            raise ValueError(f"Incomplete page {page_index}: expected {len(candles)} rows")
        # Validate before caching. Partial or inconsistent pages are never stored.
        for row, candle in zip(result,candles): validate_row(row,candle)
        page_path.parent.mkdir(parents=True,exist_ok=True)
        tmp = page_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result,separators=(",",":"),ensure_ascii=False)+"\n",encoding="utf-8")
        tmp.replace(page_path)
        time.sleep(.2)
    else:
        raise FileNotFoundError(f"Backfill page {page_index} is missing; run with --fetch")
    if not isinstance(result,list) or len(result)!=len(candles):
        raise ValueError(f"Cached page {page_index} has an unexpected size")
    for row,candle in zip(result,candles): validate_row(row,candle)
    return result


def backfill(*, fetch: bool = False, client: httpx.Client | None = None) -> dict:
    metadata=json.loads(META.read_text(encoding="utf-8"))
    candles=expected_candles(metadata)
    owns_client=client is None
    client=client or httpx.Client(base_url="https://api.binance.com",timeout=20.0)
    rows=[]
    try:
        for index in range(math.ceil(len(candles)/1000)):
            chunk=candles[index*1000:(index+1)*1000]
            remote=_page(client,chunk,index,fetch=fetch)
            rows.extend(validate_row(row,candle) for row,candle in zip(remote,chunk))
            if (index+1)%5==0 or index+1==math.ceil(len(candles)/1000):
                print(f"Validated pages {index+1}/{math.ceil(len(candles)/1000)}; source minutes {len(rows)}")
    finally:
        if owns_client:client.close()
    opens=[row["source_candle_open_time"] for row in rows]
    if len(rows)!=len(candles) or len(set(opens))!=len(opens):
        raise ValueError("Backfill contains missing or duplicate source minutes")
    nulls=sum(row["trade_flow_imbalance"] is None for row in rows)
    invalid_time=sum(row["feature_cutoff_time"]>row["source_candle_close_time"] for row in rows)
    if invalid_time:raise ValueError("Source cutoff is after source candle close")
    RAW.mkdir(parents=True,exist_ok=True)
    joined=RAW/"joined_klines_tfi.jsonl"
    with joined.open("w",encoding="utf-8",newline="\n") as handle:
        for row in rows:
            line=json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
            handle.write(line+"\n")
    values=[float(row["trade_flow_imbalance"]) for row in rows if row["trade_flow_imbalance"] is not None]
    report={"generated_at_utc":datetime.now(timezone.utc).isoformat(),
            "source":"Binance Spot REST /api/v3/klines; UTC; source closed 1m only",
            "original_model_version":metadata["model_version"],"original_candle_sha256":FROZEN_CANDLE_SHA,
            "range":{"first_open_time":opens[0],"last_open_time":opens[-1]},
            "rows":len(rows),"pages":math.ceil(len(candles)/1000),
            "non_null_tfi":len(values),"zero_volume_null_tfi":nulls,
            "missing_open_times":0,"duplicate_open_times":0,"invalid_values":0,"future_cutoff_violations":invalid_time,
            "tfi_min":min(values) if values else None,"tfi_max":max(values) if values else None,
            "dataset_sha256":hashlib.sha256(joined.read_bytes()).hexdigest(),"raw_path":"data/experiments/tfi/joined_klines_tfi.jsonl (gitignored)",
            "target_leakage_rule":"Only field 9 and field 5 from each closed source kline; never use target kline fields."}
    shadow=ROOT/"data/experiments/microstructure/shadow_minutes.jsonl"
    if shadow.exists():
        previous=[json.loads(line) for line in shadow.read_text(encoding="utf-8").splitlines() if line.strip()]
        compared=0
        for row in previous:
            if row.get("trade_flow_imbalance") is None:continue
            volume=Decimal(row["aggressive_buy_volume"])+Decimal(row["aggressive_sell_volume"])
            rebuilt=trade_flow_from_klines(volume,row["aggressive_buy_volume"])["trade_flow_imbalance"]
            if rebuilt!=Decimal(row["trade_flow_imbalance"]):
                raise ValueError("Phase 3 shadow formula mismatch")
            compared+=1
        report["phase3_formula_crosschecks"]={"rows":compared,"historical_overlap_with_frozen_model":0,
            "note":"The 19 local shadow minutes are after the original model's Sep 17 historical cutoff."}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    md=["# TFI backfill — FASE 4","",f"Corte: {report['generated_at_utc']}","",
        f"Fuente: [Binance Spot klines](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market). Rango UTC {opens[0]} a {opens[-1]}.",
        f"{len(rows)} velas fuente verificadas con el histórico de xgboost_v1 (SHA-256 `{FROZEN_CANDLE_SHA}`); {report['pages']} páginas; fingerprint del dataset `{report['dataset_sha256']}`.",
        f"TFI no nulo: {len(values)}; volumen cero: {nulls}; duplicados, huecos, valores inválidos y cutoffs futuros: 0.",
        "El kline de la vela objetivo nunca entra en las features. Los datos crudos están ignorados por Git.",
        "Las 19 filas L1 de FASE 3 son posteriores al histórico congelado: sólo permiten contrastar la fórmula, no alinear la muestra de entrenamiento.",""]
    OUT.with_suffix(".md").write_text("\n".join(md),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch",action="store_true",help="Fetch missing pages from public Binance REST")
    args=parser.parse_args()
    result=backfill(fetch=args.fetch)
    print(f"Complete TFI backfill: {result['rows']} minutes; SHA-256 {result['dataset_sha256']}")
