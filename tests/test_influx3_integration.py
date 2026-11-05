"""Prueba real opcional; los datos ficticios van a una tabla separada."""

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.repositories.influx3_connection import Influx3Connection


@pytest.mark.skipif(os.getenv("ALPHA_INTEGRATION") != "1", reason="requiere InfluxDB 3 local")
def test_market_test_record_roundtrip() -> None:
    settings = Settings()
    assert settings.storage_provider == "influxdb3"
    run_id = uuid4().hex
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    connection = Influx3Connection.from_settings(settings)
    try:
        connection.write(
            f"alpha_market_test,symbol=BTCUSDT,data_kind=test,run_id={run_id} "
            f"price=100000,volume=10 {timestamp_ms}"
        )
        rows = connection.query(
            "SELECT symbol, data_kind, price, volume FROM alpha_market_test WHERE run_id = $run_id",
            {"run_id": run_id},
        )
        assert len(rows) == 1
        assert rows[0]["symbol"] == "BTCUSDT"
        assert rows[0]["data_kind"] == "test"
        assert rows[0]["price"] == 100000
        assert rows[0]["volume"] == 10
    finally:
        connection.close()
