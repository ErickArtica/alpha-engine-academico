"""Ejecutar: python -m alpha_engine.ml.validate_temporal [--days N] [--windows N]."""

import argparse
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path

from alpha_engine.config.settings import PROJECT_ROOT, get_settings
from alpha_engine.ml.walk_forward import evaluate_walk_forward, save_walk_forward_report
from alpha_engine.repositories.factory import create_market_repository
from alpha_engine.services.ingestion import closed_boundary
from alpha_engine.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _utc_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use fecha ISO 8601 con zona horaria") from exc
    if timestamp.tzinfo is None:
        raise argparse.ArgumentTypeError("La fecha debe incluir zona horaria")
    return timestamp.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa XGBoost en ventanas temporales fuera de muestra")
    parser.add_argument("--days", type=int, help="Días previos al fin; por defecto BACKFILL_DAYS")
    parser.add_argument("--start", type=_utc_timestamp, help="Inicio fijo ISO 8601 con zona; requiere --end")
    parser.add_argument("--end", type=_utc_timestamp, help="Fin exclusivo ISO 8601 con zona; por defecto última vela cerrada")
    parser.add_argument("--windows", type=int, default=4, help="Ventanas de prueba disjuntas (2 a 8)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "walk_forward_v1.json")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    if settings.interval != "1m":
        parser.error("La evaluación requiere INTERVAL=1m")
    if args.start is not None and args.end is None:
        parser.error("--start requiere --end para fijar el periodo reproducible")
    if args.start is not None and args.days is not None:
        parser.error("Use --start o --days, no ambos")
    days = args.days if args.days is not None else settings.backfill_days
    if days < 1:
        parser.error("--days debe ser >= 1")
    if not 2 <= args.windows <= 8:
        parser.error("--windows debe estar entre 2 y 8")
    stop = closed_boundary(args.end or datetime.now(timezone.utc), settings.interval)
    start = args.start or stop - timedelta(days=days)
    if start >= stop:
        parser.error("El inicio debe ser anterior al fin")
    repository = create_market_repository(settings)
    try:
        candles = repository.get_candles(settings.symbol, settings.interval, start, stop)
    finally:
        repository.close()
    logger.info("Velas cerradas leídas: count=%s start=%s end=%s", len(candles), start, stop)
    report = evaluate_walk_forward(candles, windows=args.windows)
    path = save_walk_forward_report(report, args.output)
    logger.info("Evaluación guardada: %s", path)
    logger.info("Ventanas con mayor accuracy que baseline: %s/%s", report["summary"]["windows_beating_baseline_accuracy"], args.windows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
