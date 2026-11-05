"""Virtual-only policy, accounting path, idempotence and HTTP boundary."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.services.paper_api import PaperApiClient
from alpha_engine.data.book_ticker import BinanceBookTickerClient
from alpha_engine.services.paper_policy import paper_decision
from alpha_engine.services.paper_trading import PaperTradingService, handle_paper_request


NOW = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)


def prediction(up, number=1):
    return SimpleNamespace(prediction_id=f"{number:064x}", symbol="BTCUSDT",
                           prob_up=up, prob_down=1-up,
                           predicted_class="UP" if up > .5 else "DOWN",
                           model_version="xgboost_v1", generated_at=NOW-timedelta(seconds=3),
                           valid_until=NOW+timedelta(seconds=40))


@pytest.mark.parametrize("up,long,action,executable", [
    (.55, False, "WAIT", False), (.30, False, "WAIT", False),
    (.70, False, "BUY", True), (.70, True, "HOLD", False),
    (.45, True, "HOLD", False), (.30, True, "SELL", True),
])
def test_position_aware_rule(up, long, action, executable):
    result = paper_decision(prediction(up), has_long=long, now=NOW)
    assert (result.action, result.executable) == (action, executable)


def test_expired_prediction_abstains_and_never_calls_llm():
    past = prediction(.8)
    past.valid_until = NOW
    assert paper_decision(past, has_long=False, now=NOW).action == "WAIT"
    assert paper_decision(past, has_long=True, now=NOW).action == "HOLD"


class Accounts:
    def __init__(self):
        self.row = dict(initial_capital=1000, cash=1000, btc=0, entry_price=None,
                        entry_cost=None, realized_pnl=0, fees_paid=0, closed_trades=0,
                        winning_trades=0, generation=0, version=0,
                        operation_count=0,
                        allocation=.2, fee_rate=.001, slippage_rate=.0002)
        self.trades = {}
        self.reset_ids = set()

    def close(self):
        pass

    def create(self, portfolio_id, settings):
        self.portfolio_id = portfolio_id

    def get(self, portfolio_id):
        return self.row if portfolio_id == self.portfolio_id else None

    def ledger(self, portfolio_id, generation):
        return [row for row in reversed(list(self.trades.values())) if row["generation"] == generation]

    def trade_by_id(self, portfolio_id, trade_id):
        return self.trades.get(trade_id)

    def apply(self, portfolio_id, trade_id, prediction_id, action, reference_price,
              model_version, signal_probability, expected_version, executed_at):
        a = self.row
        if trade_id in self.trades:
            return "DUPLICATE"
        if a["version"] != expected_version:
            return "CONFLICT"
        if any(row["prediction_id"] == prediction_id and row["generation"] == a["generation"]
               for row in self.trades.values()):
            return "DUPLICATE_PREDICTION"
        price = reference_price * (1 + a["slippage_rate"] if action == "BUY" else 1 - a["slippage_rate"])
        if action == "BUY":
            value = a["cash"] * a["allocation"] / (1 + a["fee_rate"])
            fee = value * a["fee_rate"]
            quantity = value / price
            a["cash"] -= value + fee
            a["btc"] = quantity
            a["entry_price"] = price
            a["entry_cost"] = value + fee
            pnl = None
        elif action == "SELL":
            quantity = a["btc"]
            value = quantity * price
            fee = value * a["fee_rate"]
            pnl = value - fee - a["entry_cost"]
            a["cash"] += value - fee
            a["btc"] = 0
            a["entry_price"] = a["entry_cost"] = None
            a["realized_pnl"] += pnl
            a["closed_trades"] += 1
            a["winning_trades"] += int(pnl > 0)
        else:
            raise AssertionError("Service sent non-trading action")
        a["fees_paid"] += fee
        a["operation_count"] += 1
        a["version"] += 1
        self.trades[trade_id] = dict(trade_id=trade_id, portfolio_id=portfolio_id,
            generation=a["generation"], prediction_id=prediction_id, executed_at=executed_at,
            action=action, reference_price=reference_price, price=price,
            quantity_btc=quantity, value_usd=value, fee=fee,
            cash_after=a["cash"], btc_after=a["btc"], realized_pnl=pnl,
            model_version=model_version, signal_probability=signal_probability)
        return "APPLIED"

    def reset(self, portfolio_id, reset_id, expected_version, reset_at):
        if reset_id in self.reset_ids:
            return "DUPLICATE"
        if self.row["version"] != expected_version:
            return "CONFLICT"
        self.reset_ids.add(reset_id)
        self.row.update(cash=1000, btc=0, entry_price=None, entry_cost=None,
                        realized_pnl=0, fees_paid=0, closed_trades=0, winning_trades=0,
                        operation_count=0, generation=self.row["generation"]+1,
                        version=self.row["version"]+1)
        return "RESET"


class Predictions:
    def __init__(self, current):
        self.current = current

    def get_latest(self, symbol):
        assert symbol == "BTCUSDT"
        return self.current

    def close(self):
        pass


class Quotes:
    def __init__(self):
        self.paths = []

    def fetch(self, symbol):
        self.paths.append("/api/v3/ticker/bookTicker")
        return SimpleNamespace(symbol=symbol, bid_price=99.9, ask_price=100.1,
                               requested_at=NOW, received_at=NOW+timedelta(milliseconds=20))

    def close(self):
        pass


def service():
    accounts = Accounts()
    predictions = Predictions(prediction(.7))
    quotes = Quotes()
    settings = Settings(_env_file=None, storage_provider="postgresql")
    result = PaperTradingService(settings, accounts, predictions, quotes, clock=lambda: NOW)
    token = uuid4()
    accounts.portfolio_id = token
    return result, token, accounts, predictions, quotes


def test_manual_buy_hold_sell_fees_pnl_reset_and_idempotence():
    paper, token, accounts, predictions, quotes = service()
    first = paper.snapshot(token)
    assert first["suggestion"]["action"] == "BUY" and first["cash"] == 1000
    buy_id = uuid4()
    status, after_buy = paper.apply(token, buy_id, 0, predictions.current.prediction_id)
    assert status == "APPLIED" and after_buy["position"] == "LONG"
    assert after_buy["cash"] == pytest.approx(800)
    buy_fee = 200 / 1.001 * .001
    assert after_buy["fees_paid"] == pytest.approx(buy_fee)
    assert after_buy["ledger"][0]["model_version"] == "xgboost_v1"
    assert after_buy["ledger"][0]["signal_probability"] == pytest.approx(.7)
    assert paper.apply(token, buy_id, 0, predictions.current.prediction_id)[0] == "DUPLICATE"
    assert len(accounts.trades) == 1
    assert paper.snapshot(token)["suggestion"]["action"] == "HOLD"
    predictions.current = prediction(.3, 2)
    sell_id = uuid4()
    status, after_sell = paper.apply(token, sell_id, 1, predictions.current.prediction_id)
    assert status == "APPLIED" and after_sell["position"] == "SIN_POSICION"
    quantity = (200 / 1.001) / (100.1 * 1.0002)
    proceeds = quantity * 99.9 * (1-.0002)
    sell_fee = proceeds * .001
    expected_pnl = proceeds - sell_fee - 200
    assert after_sell["realized_pnl"] == pytest.approx(expected_pnl)
    assert after_sell["total_pnl"] == pytest.approx(expected_pnl)
    assert after_sell["cash"] == pytest.approx(1000 + expected_pnl)
    assert after_sell["fees_paid"] == pytest.approx(buy_fee + sell_fee)
    assert after_sell["closed_trades"] == 1
    assert after_sell["operation_count"] == 2
    assert after_sell["win_rate_pct"] == 0
    assert set(quotes.paths) == {"/api/v3/ticker/bookTicker"}
    reset_id = uuid4()
    status, reset = paper.reset(token, reset_id, 2)
    assert status == "RESET" and reset["cash"] == 1000 and reset["btc"] == 0
    assert reset["ledger"] == [] and reset["fees_paid"] == 0
    assert reset["operation_count"] == 0
    assert paper.reset(token, reset_id, 2)[0] == "DUPLICATE"


def test_apply_rechecks_position_prediction_and_version():
    paper, token, _, predictions, _ = service()
    assert paper.apply(token, uuid4(), 1, predictions.current.prediction_id)[0] == "CONFLICT"
    assert paper.apply(token, uuid4(), 0, "f" * 64)[0] == "STALE_PREDICTION"
    predictions.current = prediction(.3)
    assert paper.apply(token, uuid4(), 0, predictions.current.prediction_id)[0] == "NO_ACTION"


def test_transport_rejects_client_supplied_action_price_and_invalid_tokens():
    settings = Settings(_env_file=None, storage_provider="postgresql")
    assert handle_paper_request({"op": "apply", "action": "BUY", "price": 1}, settings).status_code == 400
    assert handle_paper_request({"op": "read", "portfolio_id": "guess"}, settings).status_code == 400


def test_paper_http_uses_only_virtual_endpoint_and_backend_auth():
    observed = []
    def respond(request):
        observed.append((request.url.path, request.headers.get("authorization"), request.json if hasattr(request, "json") else request.content))
        return httpx.Response(200, json={"status": "OK", "portfolio": {}})
    settings = Settings(_env_file=None, storage_provider="supabase", prescriptive_backend="http",
                        recommendation_api_url="https://example.invalid/api/recommendation",
                        recommendation_api_key="server-to-server-test", recommendation_api_auth_scheme="bearer")
    client = PaperApiClient(settings, transport=httpx.MockTransport(respond))
    assert client.request({"op": "read", "portfolio_id": str(uuid4())})["status"] == "OK"
    assert observed[0][0] == "/api/paper"
    assert observed[0][1] == "Bearer server-to-server-test"


def test_public_quote_transport_never_uses_trading_endpoint_or_api_key():
    seen = []
    def respond(request):
        seen.append(request)
        return httpx.Response(200, json={"symbol": "BTCUSDT", "bidPrice": "99.9",
                                         "bidQty": "2", "askPrice": "100.1", "askQty": "2"})
    with httpx.Client(transport=httpx.MockTransport(respond), base_url="https://api.binance.com") as http:
        quote = BinanceBookTickerClient(client=http).fetch("BTCUSDT")
    assert quote.bid_price == 99.9
    assert len(seen) == 1
    assert seen[0].method == "GET" and seen[0].url.path == "/api/v3/ticker/bookTicker"
    assert "x-mbx-apikey" not in seen[0].headers
