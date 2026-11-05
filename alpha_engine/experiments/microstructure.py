"""Causal, source-candle microstructure features for offline shadow research.

L1 quotes have a local observation time because Spot bookTicker carries no
exchange event timestamp. This module never extrapolates a quote into a minute
for which it was not actually observed before the source candle closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterable


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("UTC-aware timestamp required")
    return value.astimezone(timezone.utc)


def _d(value: Decimal | str | int | float) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Finite decimal required")
    return result


@dataclass(frozen=True)
class SourceMinute:
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime

    def __post_init__(self) -> None:
        opened, closed = _utc(self.open_time), _utc(self.close_time)
        if self.interval != "1m" or closed != opened + timedelta(minutes=1, milliseconds=-1):
            raise ValueError("Expected a complete 1m source candle with inclusive close")
        object.__setattr__(self, "open_time", opened)
        object.__setattr__(self, "close_time", closed)


@dataclass(frozen=True)
class L1Quote:
    symbol: str
    observed_at: datetime
    bid_price: Decimal
    ask_price: Decimal
    bid_qty: Decimal
    ask_qty: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _utc(self.observed_at))
        for field in ("bid_price", "ask_price", "bid_qty", "ask_qty"):
            object.__setattr__(self, field, _d(getattr(self, field)))
        if self.bid_price <= 0 or self.ask_price <= 0 or self.ask_price < self.bid_price:
            raise ValueError("Invalid best bid/ask")
        if self.bid_qty < 0 or self.ask_qty < 0:
            raise ValueError("Negative L1 quantity")


@dataclass(frozen=True)
class AggTrade:
    symbol: str
    trade_id: int
    trade_time: datetime
    quantity: Decimal
    buyer_was_maker: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "trade_time", _utc(self.trade_time))
        object.__setattr__(self, "quantity", _d(self.quantity))
        if self.trade_id < 0 or self.quantity < 0:
            raise ValueError("Invalid aggregate trade")


def l1_features(quote: L1Quote) -> dict[str, Decimal | None]:
    """Last observed L1 snapshot: spread, queue imbalance and microprice offset."""
    mid = (quote.bid_price + quote.ask_price) / 2
    spread = quote.ask_price - quote.bid_price
    depth = quote.bid_qty + quote.ask_qty
    if depth:
        obi = (quote.bid_qty - quote.ask_qty) / depth
        microprice = (quote.ask_price * quote.bid_qty + quote.bid_price * quote.ask_qty) / depth
        offset = (microprice - mid) / mid * 10_000
    else:
        obi = offset = None
    return {"spread_abs": spread, "spread_bps": spread / mid * 10_000,
            "l1_order_book_imbalance": obi, "microprice_offset_bps": offset}


def trade_flow_from_klines(base_volume: Decimal | str | float,
                           taker_buy_base_volume: Decimal | str | float) -> dict[str, Decimal | None]:
    """Official kline fields 5 and 9: all base volume and aggressive-buy base volume."""
    total, buy = _d(base_volume), _d(taker_buy_base_volume)
    if total < 0 or buy < 0 or buy > total:
        raise ValueError("Invalid kline taker-buy volume")
    sell = total - buy
    return {"aggressive_buy_volume": buy, "aggressive_sell_volume": sell,
            "trade_flow_imbalance": (buy - sell) / total if total else None}


def aggregate_source_minute(source: SourceMinute, quotes: Iterable[L1Quote],
                            trades: Iterable[AggTrade]) -> dict:
    """Use only observations within the closed source minute, never target-minute data.

    Quotes are snapshots, represented by the last locally observed L1 state.
    Aggregate trades are deduplicated by aggregate-trade ID before summing.
    Missing inputs remain None instead of being forward filled.
    """
    quotes = list(quotes)
    trades = list(trades)
    for item in (*quotes, *trades):
        time = item.observed_at if isinstance(item, L1Quote) else item.trade_time
        if item.symbol != source.symbol:
            raise ValueError("Symbol mismatch")
        if time < source.open_time or time > source.close_time:
            raise ValueError("Microstructure observation falls outside source candle")
    distinct_quotes = {(q.observed_at, q.bid_price, q.ask_price, q.bid_qty, q.ask_qty): q for q in quotes}
    ordered_quotes = sorted(distinct_quotes.values(), key=lambda q: q.observed_at)
    by_id: dict[int, AggTrade] = {}
    for trade in trades:
        if trade.trade_id in by_id and by_id[trade.trade_id] != trade:
            raise ValueError("Conflicting aggregate trade ID")
        by_id[trade.trade_id] = trade
    deduped_trades = list(by_id.values())
    buy = sum((t.quantity for t in deduped_trades if not t.buyer_was_maker), Decimal(0))
    sell = sum((t.quantity for t in deduped_trades if t.buyer_was_maker), Decimal(0))
    flow = (buy - sell) / (buy + sell) if buy + sell else None
    quote_features = l1_features(ordered_quotes[-1]) if ordered_quotes else {
        "spread_abs": None, "spread_bps": None,
        "l1_order_book_imbalance": None, "microprice_offset_bps": None}
    cutoff = max([q.observed_at for q in ordered_quotes] + [t.trade_time for t in deduped_trades], default=None)
    if cutoff is not None and cutoff > source.close_time:
        raise AssertionError("Feature cutoff exceeds source close")
    return {"symbol": source.symbol, "interval": source.interval,
            "source_candle_open_time": source.open_time, "source_candle_close_time": source.close_time,
            "feature_cutoff_time": cutoff, **quote_features,
            "aggressive_buy_volume": buy if deduped_trades else None,
            "aggressive_sell_volume": sell if deduped_trades else None,
            "trade_flow_imbalance": flow,
            "quote_count": len(ordered_quotes), "trade_count": len(deduped_trades),
            "quote_duplicates": len(quotes) - len(ordered_quotes),
            "trade_duplicates": len(trades) - len(deduped_trades)}
