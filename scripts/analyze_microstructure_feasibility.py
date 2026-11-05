"""Summarize the ignored local shadow sample without exposing raw market rows."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import median

import numpy as np

from scripts.build_microstructure_shadow import QUOTES, ROOT, SHADOW

REPORT = ROOT / "reports/microstructure_feasibility_v1.json"


def analyze(quotes_path: Path = QUOTES, shadow_path: Path = SHADOW) -> dict:
    quotes = [json.loads(line) for line in quotes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in shadow_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    receipt = sorted(datetime.fromisoformat(row["received_at"]).astimezone(timezone.utc) for row in quotes)
    minutes = Counter(t.replace(second=0, microsecond=0) for t in receipt)
    gaps = sorted((b-a).total_seconds()/60 for a,b in zip(receipt,receipt[1:]))
    span_minutes = int((max(minutes)-min(minutes)).total_seconds()/60)+1 if minutes else 0
    cutoff_failures = sum(datetime.fromisoformat(row["feature_cutoff_time"]) > datetime.fromisoformat(row["source_candle_close_time"]) for row in rows)
    duplicate_keys = len(rows)-len({(row["symbol"],row["interval"],row["source_candle_open_time"]) for row in rows})
    complete = sum(all(row.get(name) is not None for name in (
        "spread_bps", "l1_order_book_imbalance", "microprice_offset_bps", "trade_flow_imbalance")) for row in rows)
    measures = {}
    for name in ("spread_bps", "l1_order_book_imbalance", "microprice_offset_bps", "trade_flow_imbalance"):
        values = np.array([float(row[name]) for row in rows if row.get(name) is not None])
        measures[name] = {"n":len(values), "min":float(values.min()) if len(values) else None,
                          "median":float(np.median(values)) if len(values) else None,
                          "p95":float(np.quantile(values,.95)) if len(values) else None,
                          "max":float(values.max()) if len(values) else None}
    invalid_physical_ranges = sum(
        Decimal(row["spread_bps"]) < 0 or
        abs(Decimal(row["l1_order_book_imbalance"])) > 1 or
        abs(Decimal(row["trade_flow_imbalance"])) > 1 or
        abs(Decimal(row["microprice_offset_bps"])) > Decimal(row["spread_bps"])/2
        for row in rows if all(row.get(name) is not None for name in measures))
    return {"analyzed_at_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"Local experimental sample only; no production collection or training",
            "quote_capture":{"count":len(quotes),"first_received_at":receipt[0].isoformat() if receipt else None,
                "last_received_at":receipt[-1].isoformat() if receipt else None,
                "minutes_with_quotes":len(minutes),"span_minutes":span_minutes,
                "coverage_over_span":len(minutes)/span_minutes if span_minutes else None,
                "median_interarrival_minutes":median(gaps) if gaps else None,
                "largest_interarrival_minutes":max(gaps) if gaps else None,
                "minutes_with_multiple_quotes":sum(n>1 for n in minutes.values()),
                "has_exchange_event_time":False},
            "shadow":{"closed_source_minutes":len(rows),"complete_four_feature_minutes":complete,
                "missing_l1":sum(row["spread_bps"] is None for row in rows),
                "missing_trade_flow":sum(row["trade_flow_imbalance"] is None for row in rows),
                "zero_l1_denominator":sum(row["spread_bps"] is not None and row["l1_order_book_imbalance"] is None for row in rows),
                "zero_trade_denominator":sum(row["aggressive_buy_volume"] is not None and row["trade_flow_imbalance"] is None for row in rows),
                "duplicate_source_keys":duplicate_keys,"future_cutoff_violations":cutoff_failures,
                "rows_from_kline_taker_buy":sum(row["aggressive_buy_volume"] is not None for row in rows),
                "invalid_physical_ranges":invalid_physical_ranges,"feature_ranges":measures,
                "storage":"data/experiments/microstructure/shadow_minutes.jsonl (gitignored)"},
            "candidate_features":[
                {"feature":"spread_bps","historical_backfill_possible":False,"source":"Spot bookTicker L1 capture","resolution":"local REST snapshots; future WebSocket L1 updates","limits":"No exchange event time in bookTicker, no official historical L1 archive for this period"},
                {"feature":"l1_order_book_imbalance","historical_backfill_possible":False,"source":"best bid/ask quantities from bookTicker","resolution":"L1 only","limits":"Queue snapshot, not deep-book imbalance; sparse local observations"},
                {"feature":"microprice_offset_bps","historical_backfill_possible":False,"source":"best bid/ask and quantities","resolution":"L1 only","limits":"Only prospective capture; no historical book reconstruction"},
                {"feature":"trade_flow_imbalance","historical_backfill_possible":True,"source":"Spot closed kline total base volume and taker-buy base volume (fields 5 and 9); alternatively aggTrades","resolution":"1m aggregated base-asset volume","limits":"Historical klines available for same source minute; aggTrades are much larger and need dedup/pagination"}],
            "historical_flow_backfill_estimate":{"frozen_historical_candles":43199,
                "kline_page_limit":1000,"estimated_pages":math.ceil(43199/1000),
                "status":"Not downloaded; source candles must be matched by open_time and close_time and checked for completeness"},
            "temporal_rule":"All quote receipt times and trade event times must lie within the source candle; feature_cutoff_time <= source_candle_close_time. Kline trade flow uses only the fully closed source kline.",
            "source_urls":[
                "https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market",
                "https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-streams/~",
                "https://github.com/binance/binance-public-data",
                "https://render.com/docs/free"]}


def markdown(report: dict) -> str:
    q,s=report["quote_capture"],report["shadow"]
    lines=["# Microstructure feasibility — FASE 3","",
           f"Corte UTC: {report['analyzed_at_utc']}. Sólo investigación local; producción sin cambios.","",
           "## Datos presentes","",
           f"BookTicker REST local: {q['count']} observaciones entre {q['first_received_at']} y {q['last_received_at']}; {q['minutes_with_quotes']} minutos distintos en un tramo de {q['span_minutes']} minutos ({q['coverage_over_span']:.2%} de minutos con al menos una observación).",
           f"Hueco máximo: {q['largest_interarrival_minutes']:.1f} min. {q['minutes_with_multiple_quotes']} minutos tienen más de un snapshot. bookTicker no ofrece hora de evento; `received_at` es el reloj local.",
           f"Shadow local: {s['closed_source_minutes']} minutos con vela cerrada y snapshot; {s['complete_four_feature_minutes']} se enriquecieron con volumen taker-buy del kline cerrado correspondiente. Es una muestra demasiado dispersa para entrenar o estudiar correlaciones robustas.","",
           "## Definiciones experimentales","",
           "* `spread_bps = (ask-bid)/mid*10000`, `mid=(bid+ask)/2`.",
           "* `l1_order_book_imbalance = (bid_qty-ask_qty)/(bid_qty+ask_qty)`; denominador cero ⇒ null.",
           "* `microprice = (ask*bid_qty+bid*ask_qty)/(bid_qty+ask_qty)`; `microprice_offset_bps=(microprice-mid)/mid*10000`.",
           "* `trade_flow_imbalance = (aggressive_buy_base-aggressive_sell_base)/total_base`; kline índice 9 es volumen taker-buy y sell=volumen total (índice 5) menos buy. Para aggTrade, `m=true` significa buyer maker ⇒ venta agresiva.",
           "Se usa el último L1 observado dentro de la vela fuente y el volumen agresor agregado sobre esa misma vela. No se rellena una fuente sin observaciones.","",
           "## Disponibilidad histórica","",
           "| Feature | Backfill | Fuente y limitación |","|---|---|---|" ]
    for item in report["candidate_features"]:
        lines.append(f"| {item['feature']} | {'Sí' if item['historical_backfill_possible'] else 'No'} | {item['source']}; {item['limits']} |")
    estimate=report["historical_flow_backfill_estimate"]
    lines += ["",f"Para las {estimate['frozen_historical_candles']} velas históricas del modelo, el flujo basado en klines requeriría aproximadamente {estimate['estimated_pages']} páginas REST de hasta {estimate['kline_page_limit']} filas; no se descargó ese backfill completo. El coste en aggTrades no se infiere sin medir su densidad real."]
    lines += ["", "## Temporalidad y calidad", "",
              f"Cutoff futuro: {s['future_cutoff_violations']}; claves duplicadas: {s['duplicate_source_keys']}; rangos físicos inválidos: {s['invalid_physical_ranges']}; L1 faltante en filas shadow: {s['missing_l1']}; flujo faltante: {s['missing_trade_flow']}. La cobertura completa dentro del tramo es sólo {s['complete_four_feature_minutes']}/{q['span_minutes']} minutos.",
              "Mínimo, mediana, P95 y máximo de cada señal aparecen en el JSON. Con 19 filas no se define un umbral estadístico robusto de outliers.",
              "Las 25 capturas son locales e históricas; el worker Render actual no recolecta L1. Los snapshots se capturaron tras predicción y no pueden adjudicarse a la vela anterior. Se asignan sólo al minuto en que fueron recibidos, si existe la vela cerrada.",
              "Para construir agregados L1 fiables se requieren eventos o muestreo intra-minuto continuos, UTC y con manejo de reinicios/huecos. Un WebSocket en el Web Service actual no añadiría otra instancia, pero Render Free puede reiniciarlo y su disco es efímero; un segundo servicio activo sí consumiría más horas. Con el Cron actual de cada minuto, una instancia despierta todo el mes consumiría unas 720–744 de las 750 horas mensuales publicadas, sujeto al uso real y a los límites vigentes. No se desplegó proceso nuevo.","",
              "## Fuentes oficiales", ""]
    lines += [f"- {url}" for url in report["source_urls"]]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    report=analyze()
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    REPORT.with_suffix(".md").write_text(markdown(report),encoding="utf-8")
    print(f"Microstructure report: quotes={report['quote_capture']['count']} shadow={report['shadow']['closed_source_minutes']} complete={report['shadow']['complete_four_feature_minutes']}")
