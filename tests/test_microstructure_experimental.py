"""Temporal and numerical invariants for research-only microstructure features."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from alpha_engine.experiments.microstructure import (
    AggTrade, L1Quote, SourceMinute, aggregate_source_minute, l1_features,
    trade_flow_from_klines,
)


T = datetime(2026, 9, 23, 10, 4, tzinfo=timezone.utc)
SOURCE = SourceMinute("BTCUSDT", "1m", T, T + timedelta(minutes=1, milliseconds=-1))


def quote(at, bid="100", ask="102", bid_qty="3", ask_qty="1"):
    return L1Quote("BTCUSDT", at, Decimal(bid), Decimal(ask), Decimal(bid_qty), Decimal(ask_qty))


def test_l1_numeric_spread_obi_microprice():
    result = l1_features(quote(T))
    assert result["spread_abs"] == Decimal("2")
    assert result["spread_bps"] == Decimal("2") / Decimal("101") * 10_000
    assert result["l1_order_book_imbalance"] == Decimal("0.5")
    # ask is weighted by bid quantity, bid by ask quantity: microprice=101.5.
    assert result["microprice_offset_bps"] == Decimal("0.5") / Decimal("101") * 10_000


def test_zero_depth_and_zero_volume_are_missing_not_fake_zero():
    result = l1_features(quote(T, bid_qty="0", ask_qty="0"))
    assert result["l1_order_book_imbalance"] is None
    assert result["microprice_offset_bps"] is None
    assert trade_flow_from_klines("0", "0")["trade_flow_imbalance"] is None


def test_kline_taker_buy_volume_flow_and_bad_inputs():
    result = trade_flow_from_klines("10", "7")
    assert result["aggressive_sell_volume"] == Decimal("3")
    assert result["trade_flow_imbalance"] == Decimal("0.4")
    with pytest.raises(ValueError):
        trade_flow_from_klines("10", "11")


def test_quote_aggregation_last_snapshot_and_trade_deduplication():
    early = quote(T + timedelta(seconds=1), bid_qty="1", ask_qty="3")
    late = quote(T + timedelta(seconds=59, milliseconds=999), bid_qty="3", ask_qty="1")
    buy = AggTrade("BTCUSDT", 1, T + timedelta(seconds=10), Decimal("7"), False)
    sell = AggTrade("BTCUSDT", 2, T + timedelta(seconds=50), Decimal("3"), True)
    row = aggregate_source_minute(SOURCE, [early, late, late], [buy, sell, buy])
    assert row["quote_count"] == 2 and row["quote_duplicates"] == 1
    assert row["trade_count"] == 2 and row["trade_duplicates"] == 1
    assert row["l1_order_book_imbalance"] == Decimal("0.5")
    assert row["trade_flow_imbalance"] == Decimal("0.4")
    assert row["feature_cutoff_time"] == SOURCE.close_time


def test_no_target_minute_leakage_even_if_prediction_is_at_100505():
    target_time = T + timedelta(minutes=1, seconds=5)
    with pytest.raises(ValueError, match="outside source"):
        aggregate_source_minute(SOURCE, [quote(target_time)], [])
    with pytest.raises(ValueError, match="outside source"):
        aggregate_source_minute(SOURCE, [], [AggTrade("BTCUSDT", 1, target_time, Decimal("1"), False)])


def test_closed_boundary_and_timezone_validation():
    row = aggregate_source_minute(SOURCE, [quote(SOURCE.close_time)], [])
    assert row["feature_cutoff_time"] == SOURCE.close_time
    assert row["trade_flow_imbalance"] is None
    pet = timezone(timedelta(hours=-5))
    local_time = datetime(2026, 9, 23, 5, 4, tzinfo=pet)
    assert quote(local_time).observed_at == SOURCE.open_time
    with pytest.raises(ValueError, match="UTC-aware"):
        quote(datetime(2026, 9, 23, 10, 4))


def test_conflicting_trade_id_and_invalid_quote_rejected():
    a = AggTrade("BTCUSDT", 1, T, Decimal("1"), False)
    b = AggTrade("BTCUSDT", 1, T, Decimal("2"), False)
    with pytest.raises(ValueError, match="Conflicting"):
        aggregate_source_minute(SOURCE, [], [a, b])
    with pytest.raises(ValueError, match="Invalid best"):
        quote(T, bid="103", ask="102")
