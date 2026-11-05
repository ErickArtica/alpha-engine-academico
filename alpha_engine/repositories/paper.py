"""Private persistence for virtual accounts; never touches trading APIs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from alpha_engine.config.settings import Settings


class SupabasePaperRepository:
    def __init__(self, client, *, owns_client: bool = False):
        self.client = client
        self.owns_client = owns_client

    @classmethod
    def from_settings(cls, settings: Settings):
        from alpha_engine.repositories.supabase import _client_from_settings
        return cls(_client_from_settings(settings), owns_client=True)

    def close(self):
        if self.owns_client:
            self.client.postgrest.aclose()
            self.owns_client = False

    def create(self, portfolio_id: UUID, settings: Settings):
        self.client.table("paper_accounts").insert({
            "portfolio_id": str(portfolio_id), "initial_capital": str(settings.paper_initial_capital),
            "cash": str(settings.paper_initial_capital), "allocation": str(settings.paper_allocation),
            "fee_rate": str(settings.paper_fee_rate), "slippage_rate": str(settings.paper_slippage_rate),
        }).execute()

    def get(self, portfolio_id: UUID):
        response = (self.client.table("paper_accounts").select("*")
                    .eq("portfolio_id", str(portfolio_id)).limit(1).execute())
        return response.data[0] if response.data else None

    def ledger(self, portfolio_id: UUID, generation: int):
        response = (self.client.table("paper_trades").select("*")
                    .eq("portfolio_id", str(portfolio_id)).eq("generation", generation)
                    .order("executed_at", desc=True).limit(500).execute())
        return response.data or []

    def trade_by_id(self, portfolio_id: UUID, trade_id: UUID):
        response = (self.client.table("paper_trades").select("*")
                    .eq("portfolio_id", str(portfolio_id)).eq("trade_id", str(trade_id))
                    .limit(1).execute())
        return response.data[0] if response.data else None

    def apply(self, portfolio_id: UUID, trade_id: UUID, prediction_id: str,
              action: str, reference_price: float, model_version: str,
              signal_probability: float, expected_version: int, executed_at: datetime):
        return self.client.rpc("paper_apply", {
            "p_portfolio_id": str(portfolio_id), "p_trade_id": str(trade_id),
            "p_prediction_id": prediction_id, "p_action": action,
            "p_reference_price": str(reference_price), "p_model_version": model_version,
            "p_signal_probability": str(signal_probability),
            "p_expected_version": expected_version, "p_executed_at": executed_at.isoformat(),
        }).execute().data

    def reset(self, portfolio_id: UUID, reset_id: UUID, expected_version: int, reset_at: datetime):
        return self.client.rpc("paper_reset", {
            "p_portfolio_id": str(portfolio_id), "p_reset_id": str(reset_id),
            "p_expected_version": expected_version, "p_reset_at": reset_at.isoformat(),
        }).execute().data


class PostgresPaperRepository:
    def __init__(self, connection):
        self.connection = connection

    @classmethod
    def from_settings(cls, settings: Settings):
        from alpha_engine.repositories.postgresql import _PostgresRepository
        helper = _PostgresRepository.from_settings(settings)
        return cls(helper._connection)

    def close(self):
        self.connection.close()

    def _one(self, query, params):
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def create(self, portfolio_id: UUID, settings: Settings):
        self._one(
            "INSERT INTO paper_accounts (portfolio_id,initial_capital,cash,allocation,fee_rate,slippage_rate) "
            "VALUES (%s,%s,%s,%s,%s,%s) RETURNING portfolio_id",
            (portfolio_id, settings.paper_initial_capital, settings.paper_initial_capital,
             settings.paper_allocation, settings.paper_fee_rate, settings.paper_slippage_rate),
        )

    def get(self, portfolio_id: UUID):
        return self._one("SELECT * FROM paper_accounts WHERE portfolio_id=%s", (portfolio_id,))

    def ledger(self, portfolio_id: UUID, generation: int):
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT * FROM paper_trades WHERE portfolio_id=%s AND generation=%s "
                           "ORDER BY executed_at DESC LIMIT 500", (portfolio_id, generation))
            return cursor.fetchall()

    def trade_by_id(self, portfolio_id: UUID, trade_id: UUID):
        return self._one("SELECT * FROM paper_trades WHERE portfolio_id=%s AND trade_id=%s",
                         (portfolio_id, trade_id))

    def apply(self, portfolio_id: UUID, trade_id: UUID, prediction_id: str,
              action: str, reference_price: float, model_version: str,
              signal_probability: float, expected_version: int, executed_at: datetime):
        row = self._one("SELECT public.paper_apply(%s,%s,%s,%s,%s,%s,%s,%s,%s) AS status",
                        (portfolio_id, trade_id, prediction_id, action, reference_price,
                         model_version, signal_probability, expected_version, executed_at))
        return row["status"]

    def reset(self, portfolio_id: UUID, reset_id: UUID, expected_version: int, reset_at: datetime):
        row = self._one("SELECT public.paper_reset(%s,%s,%s,%s) AS status",
                        (portfolio_id, reset_id, expected_version, reset_at))
        return row["status"]


def create_paper_repository(settings: Settings):
    if settings.storage_provider == "supabase":
        return SupabasePaperRepository.from_settings(settings)
    if settings.storage_provider == "postgresql":
        return PostgresPaperRepository.from_settings(settings)
    raise ValueError("Paper trading requires Supabase or PostgreSQL")
