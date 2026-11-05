"""Comprueba que los repositorios del dominio se conectan a InfluxDB 3."""

import httpx
from datetime import datetime, timedelta, timezone

from alpha_engine.config.settings import Settings
from alpha_engine.repositories.factory import (
    create_market_repository, create_prediction_repository, create_recommendation_repository,
)


def main() -> None:
    settings = Settings()
    for name, repository in (
        ("market", create_market_repository(settings)),
        ("prediction", create_prediction_repository(settings)),
        ("recommendation", create_recommendation_repository(settings)),
    ):
        try:
            if name == "market":
                print(name, repository.get_latest_candle(settings.symbol, settings.interval))
                now = datetime.now(timezone.utc)
                print("market range", len(repository.get_candles(settings.symbol, settings.interval, now - timedelta(minutes=2), now)))
            elif name == "prediction":
                print(name, repository.get_latest(settings.symbol))
            else:
                print(name, repository.get_cached("smoke", "v1", "smoke"))
        except httpx.HTTPStatusError as exc:
            print(name, exc.response.status_code, exc.response.text)
            raise
        finally:
            repository.close()


if __name__ == "__main__":
    main()
