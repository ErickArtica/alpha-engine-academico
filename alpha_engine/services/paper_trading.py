"""Server-side manual paper trading. Public bookTicker only; no order API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from alpha_engine.config.settings import Settings
from alpha_engine.data.book_ticker import BinanceBookTickerClient
from alpha_engine.repositories.factory import create_prediction_repository
from alpha_engine.repositories.paper import create_paper_repository
from alpha_engine.services.paper_policy import paper_decision

logger = logging.getLogger(__name__)


def _uuid(value):
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid virtual portfolio/request token") from exc


def _trade_view(row):
    return {"timestamp": str(row["executed_at"]), "action": row["action"],
            "price": float(row["price"]), "reference_price": float(row["reference_price"]),
            "quantity_btc": float(row["quantity_btc"]), "value_usd": float(row["value_usd"]),
            "fee": float(row["fee"]), "cash_after": float(row["cash_after"]),
            "btc_after": float(row["btc_after"]),
            "realized_pnl": float(row["realized_pnl"]) if row["realized_pnl"] is not None else None,
            "model_version": row["model_version"],
            "signal_probability": float(row["signal_probability"]),
            "prediction_id": row["prediction_id"], "trade_id": str(row["trade_id"])}


class PaperTradingService:
    def __init__(self, settings: Settings, accounts, predictions, quote_client,
                 *, clock=lambda: datetime.now(timezone.utc)):
        self.settings = settings
        self.accounts = accounts
        self.predictions = predictions
        self.quote_client = quote_client
        self.clock = clock

    @classmethod
    def from_settings(cls, settings: Settings):
        accounts = create_paper_repository(settings)
        try:
            predictions = create_prediction_repository(settings)
        except Exception:
            accounts.close()
            raise
        return cls(settings, accounts, predictions,
                   BinanceBookTickerClient(settings.binance_base_url))

    def close(self):
        self.quote_client.close()
        self.predictions.close()
        self.accounts.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _now(self):
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("Paper clock must be timezone aware")
        return value.astimezone(timezone.utc)

    def create(self):
        token = uuid4()
        self.accounts.create(token, self.settings)
        return self.snapshot(token)

    def snapshot(self, portfolio_id: UUID, *, quote=None):
        account = self.accounts.get(portfolio_id)
        if account is None:
            return None
        now = self._now()
        prediction = self.predictions.get_latest(self.settings.symbol)
        decision = paper_decision(prediction, has_long=float(account["btc"]) > 0,
                                  threshold=self.settings.model_confidence_threshold, now=now)
        if quote is None:
            try:
                quote = self.quote_client.fetch(self.settings.symbol)
            except Exception:
                logger.warning("Public paper quote unavailable")
        current_price = (quote.bid_price + quote.ask_price) / 2 if quote else None
        cash, btc = float(account["cash"]), float(account["btc"])
        entry_cost = float(account["entry_cost"]) if account["entry_cost"] is not None else None
        unrealized = btc * current_price - entry_cost if btc and current_price is not None else (0.0 if not btc else None)
        realized = float(account["realized_pnl"])
        current_capital = cash + btc * current_price if current_price is not None else (cash if not btc else None)
        ledger = [_trade_view(row) for row in self.accounts.ledger(portfolio_id, int(account["generation"]))]
        closed = int(account["closed_trades"])
        return {
            "portfolio_id": str(portfolio_id), "mode": "MANUAL_ASSISTED", "virtual_only": True,
            "initial_capital": float(account["initial_capital"]),
            "current_capital": current_capital, "cash": cash, "btc": btc,
            "position": "LONG" if btc else "SIN_POSICION",
            "entry_price": float(account["entry_price"]) if account["entry_price"] is not None else None,
            "current_price": current_price,
            "quote_received_at": quote.received_at.isoformat() if quote else None,
            "realized_pnl": realized, "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized if unrealized is not None else None,
            "return_pct": 100 * (current_capital / float(account["initial_capital"]) - 1)
                          if current_capital is not None else None,
            "closed_trades": closed,
            "operation_count": int(account["operation_count"]),
            "win_rate_pct": 100 * int(account["winning_trades"]) / closed if closed else None,
            "fees_paid": float(account["fees_paid"]),
            "allocation": float(account["allocation"]), "fee_rate": float(account["fee_rate"]),
            "slippage_rate": float(account["slippage_rate"]),
            "generation": int(account["generation"]), "version": int(account["version"]),
            "suggestion": {"action": decision.action, "executable": decision.executable,
                           "reason": decision.reason, "confidence": decision.confidence,
                           "signal_probability": decision.signal_probability},
            "prediction": ({"prediction_id": prediction.prediction_id,
                            "predicted_class": prediction.predicted_class,
                            "prob_up": prediction.prob_up, "prob_down": prediction.prob_down,
                            "model_version": prediction.model_version,
                            "generated_at": prediction.generated_at.isoformat() if prediction.generated_at else None,
                            "valid_until": prediction.valid_until.isoformat() if prediction.valid_until else None}
                           if prediction else None),
            "ledger": ledger,
        }

    def apply(self, portfolio_id: UUID, trade_id: UUID, expected_version: int,
              prediction_id: str):
        if self.accounts.trade_by_id(portfolio_id, trade_id) is not None:
            return "DUPLICATE", self.snapshot(portfolio_id)
        account = self.accounts.get(portfolio_id)
        if account is None:
            return "NOT_FOUND", None
        if int(account["version"]) != expected_version:
            return "CONFLICT", self.snapshot(portfolio_id)
        prediction = self.predictions.get_latest(self.settings.symbol)
        if prediction is None or prediction.prediction_id != prediction_id:
            return "STALE_PREDICTION", self.snapshot(portfolio_id)
        decision = paper_decision(prediction, has_long=float(account["btc"]) > 0,
                                  threshold=self.settings.model_confidence_threshold, now=self._now())
        if not decision.executable:
            return "NO_ACTION", self.snapshot(portfolio_id)
        quote = self.quote_client.fetch(self.settings.symbol)
        if quote.requested_at < prediction.generated_at or quote.received_at >= prediction.valid_until:
            return "STALE_QUOTE", self.snapshot(portfolio_id, quote=quote)
        reference_price = quote.ask_price if decision.action == "BUY" else quote.bid_price
        status = self.accounts.apply(
            portfolio_id, trade_id, prediction.prediction_id, decision.action,
            reference_price, prediction.model_version, decision.signal_probability,
            expected_version, quote.received_at,
        )
        return status, self.snapshot(portfolio_id, quote=quote)

    def reset(self, portfolio_id: UUID, reset_id: UUID, expected_version: int):
        status = self.accounts.reset(portfolio_id, reset_id, expected_version, self._now())
        return status, self.snapshot(portfolio_id)


def handle_paper_request(payload: object, settings: Settings, *, service_factory=None):
    """Narrow authenticated transport handler; action and fill price are server computed."""
    from alpha_engine.services.recommendation_api import ApiReply

    if not isinstance(payload, dict) or not isinstance(payload.get("op"), str):
        return ApiReply(400, {"status": "INVALID_REQUEST"})
    op = payload["op"]
    expected_keys = {
        "create": {"op"}, "read": {"op", "portfolio_id"},
        "apply": {"op", "portfolio_id", "request_id", "expected_version", "prediction_id"},
        "reset": {"op", "portfolio_id", "request_id", "expected_version"},
    }
    if op not in expected_keys or set(payload) != expected_keys[op]:
        return ApiReply(400, {"status": "INVALID_REQUEST"})
    try:
        portfolio_id = _uuid(payload["portfolio_id"]) if op != "create" else None
        request_id = _uuid(payload["request_id"]) if op in {"apply", "reset"} else None
        expected_version = payload.get("expected_version")
        if op in {"apply", "reset"} and (type(expected_version) is not int or expected_version < 0):
            raise ValueError("Invalid version")
        if op == "apply" and (not isinstance(payload["prediction_id"], str)
                               or len(payload["prediction_id"]) != 64
                               or any(ch not in "0123456789abcdef" for ch in payload["prediction_id"])):
            raise ValueError("Invalid prediction ID")
    except ValueError:
        return ApiReply(400, {"status": "INVALID_REQUEST"})
    try:
        factory = service_factory or PaperTradingService.from_settings
        with factory(settings) as service:
            if op == "create":
                return ApiReply(200, {"status": "CREATED", "portfolio": service.create()})
            if op == "read":
                snapshot = service.snapshot(portfolio_id)
                return ApiReply(200 if snapshot else 404,
                                {"status": "OK" if snapshot else "NOT_FOUND", "portfolio": snapshot})
            if op == "apply":
                status, snapshot = service.apply(portfolio_id, request_id, expected_version,
                                                 payload["prediction_id"])
            else:
                status, snapshot = service.reset(portfolio_id, request_id, expected_version)
            code = 200 if status in {"APPLIED", "RESET", "DUPLICATE"} else (
                404 if status == "NOT_FOUND" else 409)
            return ApiReply(code, {"status": status, "portfolio": snapshot})
    except Exception as exc:
        logger.warning("Virtual portfolio request failed: %s", type(exc).__name__)
        return ApiReply(503, {"status": "TEMPORARILY_UNAVAILABLE"})
