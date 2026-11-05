"""Causal and read-only V1 replay with small, explicit unit fixtures."""

from datetime import datetime, timedelta, timezone

import pytest

from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.repositories.postgresql import PostgresPredictionRepository
from alpha_engine.services.historical_replay import (
    GATE, ReplayConfig, build_replay, find_event_window, load_window,
)


T0 = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)


def prediction(minute: int, up: float, *, model="xgboost_v1", late=False):
    source = T0 + timedelta(minutes=minute)
    generated = source + timedelta(minutes=2, seconds=1) if late else source + timedelta(minutes=1, seconds=5)
    return PredictionRecord(prediction_id=f"{minute + 1:064x}", symbol="BTCUSDT",
                            timestamp=source, horizon="1m", prob_up=up,
                            prob_down=1-up, predicted_class="UP" if up > .5 else "DOWN",
                            model_version=model, generated_at=generated,
                            valid_until=source + timedelta(minutes=2) - timedelta(milliseconds=1))


def candle(minute: int, price: float):
    start = T0 + timedelta(minutes=minute)
    return Candle(symbol="BTCUSDT", interval="1m", open_time=start,
                  close_time=start+timedelta(minutes=1)-timedelta(milliseconds=1),
                  open=price, high=price, low=price, close=price, volume=1)


def test_wait_buy_hold_sell_and_accounting_are_causal():
    predictions = [prediction(0, .55), prediction(1, .70),
                   prediction(2, .65), prediction(3, .30)]
    prices = [candle(2, 100), candle(3, 100), candle(4, 110), candle(5, 120)]
    replay = build_replay(predictions, prices)
    assert [row["action"] for row in replay.steps] == ["WAIT", "BUY", "HOLD", "SELL"]
    assert replay.steps[1]["timestamp"] == T0 + timedelta(minutes=3)
    assert replay.steps[1]["generated_at"] < replay.steps[1]["timestamp"]
    assert [row["action"] for row in replay.ledger] == ["BUY", "SELL"]
    assert all(row["model_version"] == "xgboost_v1" for row in replay.ledger)
    end = replay.at(4)
    debit = 200.0
    buy_notional = debit / 1.001
    qty = buy_notional / (100 * 1.0002)
    buy_fee = buy_notional * .001
    proceeds = qty * 120 * (1-.0002)
    sell_fee = proceeds * .001
    expected_pnl = proceeds - sell_fee - debit
    assert end["cash"] == pytest.approx(1000 + expected_pnl)
    assert end["capital_final"] == pytest.approx(1000 + expected_pnl)
    assert end["realized_pnl"] == pytest.approx(expected_pnl)
    assert end["fees"] == pytest.approx(buy_fee + sell_fee)
    assert end["btc"] == 0
    assert end["BUY"] == end["SELL"] == end["HOLD"] == end["WAIT"] == 1
    assert end["closed_trades"] == end["wins"] == 1
    assert end["win_rate_pct"] == 100


def test_reset_is_prefix_zero_and_no_trade_period_is_coherent():
    replay = build_replay([prediction(0, .4), prediction(1, .55)],
                          [candle(2, 100), candle(3, 101)])
    assert replay.at(2)["WAIT"] == 2
    assert replay.at(2)["BUY"] == 0
    assert replay.at(2)["capital_final"] == 1000
    assert replay.at(2)["win_rate_pct"] is None
    reset = replay.at(0)
    assert reset["capital_final"] == 1000
    assert reset["WAIT"] == reset["BUY"] == 0
    assert reset["ledger"] == ()


def test_causality_rejects_future_or_missing_fill_without_lookahead():
    replay = build_replay([prediction(0, .70, late=True), prediction(1, .70)],
                          [candle(2, 100)])
    assert replay.omitted == 2
    assert replay.steps == ()
    assert replay.ledger == ()


def test_expired_prediction_does_not_buy():
    p = prediction(0, .70)
    p.valid_until = p.generated_at
    replay = build_replay([p], [candle(2, 100)])
    assert replay.at(1)["WAIT"] == 1
    assert replay.at(1)["BUY"] == 0


def test_v3_is_never_replayed_as_productive_v1():
    replay = build_replay([prediction(0, .9, model="xgboost_v3")], [candle(2, 100)])
    assert replay.steps == ()


class ReadOnlyPredictions:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    def list_recent(self, symbol, limit, offset=0):
        self.calls.append((symbol, limit, offset))
        return self.rows[offset:offset+limit]

    def list_recent_before(self, symbol, before, limit):
        self.calls.append((symbol, before, limit))
        return [p for p in self.rows if p.timestamp <= before][:limit]

    def list_entry_signals(self, symbol, model_version, minimum_up, limit):
        self.calls.append((symbol, model_version, minimum_up, limit))
        return [p for p in self.rows if p.model_version == model_version
                and p.prob_up >= minimum_up][:limit]


class ReadOnlyCandles:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get_candles(self, symbol, interval, start, stop):
        self.calls.append((symbol, interval, start, stop))
        return [row for row in self.rows if start <= row.open_time < stop]


def test_loader_uses_two_reads_for_500_rows_and_never_touches_live_wallet():
    rows = [prediction(i, .4) for i in range(500)][::-1]
    prices = [candle(i, 100) for i in range(502)]
    predictions_repo, market_repo = ReadOnlyPredictions(rows), ReadOnlyCandles(prices)
    selected, candles = load_window(predictions_repo, market_repo, "BTCUSDT", "1m", 360)
    replay = build_replay(selected, candles)
    assert len(selected) == len(replay.steps) == 360
    assert len(predictions_repo.calls) == len(market_repo.calls) == 1
    assert market_repo.calls[0][2] == selected[0].timestamp + timedelta(minutes=2)
    assert market_repo.calls[0][3] == selected[-1].timestamp + timedelta(minutes=3)
    assert all(row["action"] == "WAIT" for row in replay.steps)


def test_event_search_uses_one_filtered_read_and_does_not_change_probabilities():
    rows = [prediction(i, up) for i, up in enumerate([.7, .65, .3, .5])][::-1]
    repository = ReadOnlyPredictions(rows)
    probabilities_before = [p.prob_up for p in rows]
    before, description, scanned = find_event_window(repository, "BTCUSDT", 60)
    assert before == T0 + timedelta(minutes=46)
    assert "BUY" in description
    assert scanned == 2
    assert len(repository.calls) == 1
    assert repository.calls[0][1:3] == ("xgboost_v1", .60)
    assert [p.prob_up for p in rows] == probabilities_before


def test_event_search_reports_absence_without_lowering_gate():
    repository = ReadOnlyPredictions([prediction(i, .59) for i in range(60)][::-1])
    before, description, scanned = find_event_window(repository, "BTCUSDT", 60)
    assert before is None and scanned == 0
    assert "No se encontró" in description
    assert GATE == .60


def test_event_search_skips_late_up_signal():
    repository = ReadOnlyPredictions([prediction(0, .7, late=True)])
    before, _, scanned = find_event_window(repository, "BTCUSDT", 60)
    assert before is None and scanned == 1


def test_only_one_position_can_exist_and_open_long_is_not_force_sold():
    replay = build_replay([prediction(0, .7), prediction(1, .8)],
                          [candle(2, 100), candle(3, 110)])
    assert [row["action"] for row in replay.steps] == ["BUY", "HOLD"]
    end = replay.at(2)
    assert end["SELL"] == 0 and end["btc"] > 0
    assert end["realized_pnl"] == 0 and end["unrealized_pnl"] > 0


def test_anchor_loads_old_event_window_in_two_reads():
    rows = [prediction(i, .7 if i == 10 else .4) for i in range(200)][::-1]
    predictions_repo = ReadOnlyPredictions(rows)
    market_repo = ReadOnlyCandles([candle(i, 100) for i in range(202)])
    before, _, _ = find_event_window(predictions_repo, "BTCUSDT", 60)
    selected, prices = load_window(predictions_repo, market_repo, "BTCUSDT", "1m", 60, before)
    assert any(p.timestamp == T0 + timedelta(minutes=10) for p in selected)
    result = build_replay(selected, prices)
    assert result.at(len(result.steps))["BUY"] == 1
    assert len(predictions_repo.calls) == 2
    assert len(market_repo.calls) == 1


class _PgCursor:
    def __init__(self):
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, query, params):
        self.query, self.params = query, params

    def fetchall(self):
        return []


def test_postgres_replay_reads_filter_in_sql():
    cursor = _PgCursor()
    repo = PostgresPredictionRepository(type("Connection", (), {"cursor": lambda self: cursor})())
    assert repo.list_entry_signals("BTCUSDT", "xgboost_v1", .6, 100) == []
    assert "model_version=%s" in cursor.query and "prob_up >= %s" in cursor.query
    assert cursor.params == ("BTCUSDT", "xgboost_v1", .6, 100)
    assert repo.list_recent_before("BTCUSDT", T0, 60) == []
    assert "candle_open_time <= %s" in cursor.query
    assert cursor.params == ("BTCUSDT", T0, 60)
