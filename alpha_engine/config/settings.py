"""Configuración única para ejecución local y futuros adaptadores."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "local"
    symbol: str = Field(default="BTCUSDT", pattern=r"^[A-Z0-9]{3,20}$")
    interval: str = Field(default="1m", pattern=r"^[1-9][0-9]*[mhd]$")
    binance_base_url: str = "https://api.binance.com"
    storage_provider: Literal["influxdb", "influxdb3", "postgresql", "supabase"] = "influxdb"
    influx_url: str = "http://localhost:8086"
    influx_org: str = "alpha-engine"
    influx_bucket: str = "market"
    influx_token: SecretStr = SecretStr("")
    influx_username: str = "alpha-admin"
    influx_password: SecretStr = SecretStr("")
    influx3_url: str = "http://127.0.0.1:8181"
    influx3_database: str = "alpha_engine"
    influx3_token: SecretStr = SecretStr("")
    supabase_url: str = ""
    supabase_key: SecretStr = SecretStr("")
    database_url: SecretStr = SecretStr("")
    local_demo_mode: bool = False
    prescriptive_backend: Literal["local", "http"] = "local"
    recommendation_api_url: str = ""
    recommendation_api_key: SecretStr = SecretStr("")
    recommendation_api_auth_scheme: Literal["azure_function", "bearer"] = "azure_function"
    recommendation_api_timeout_seconds: int = Field(default=50, ge=1, le=180)
    recommendation_max_attempts_per_minute: int = Field(default=3, ge=1, le=20)
    llm_provider: Literal["ollama", "nvidia", "deepseek", "gemini"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_num_ctx: int = Field(default=4096, ge=512)
    ollama_timeout_seconds: int = Field(default=180, ge=5)
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "google/gemma-4-31b-it"
    nvidia_api_key: SecretStr = SecretStr("")
    nvidia_reasoning_effort: Literal["low"] | None = None
    alpha_vantage_api_key: SecretStr = SecretStr("")
    prescriptive_news_enabled: bool = False
    prescriptive_news_provider: Literal["alpha_vantage", "marketaux"] = "alpha_vantage"
    marketaux_timeout_seconds: float = Field(default=6.0, ge=1, le=10)
    marketaux_daily_budget: int = Field(default=50, ge=1, le=50)
    alpha_vantage_timeout_seconds: float = Field(default=6.0, ge=1, le=15)
    nvidia_timeout_seconds: int = Field(default=90, ge=5)
    # Bound an interactive LIVE request even when a provider's general timeout
    # is much longer than the one-minute prediction horizon.
    llm_live_timeout_seconds: int = Field(default=45, ge=1)
    llm_deadline_margin_seconds: float = Field(default=1.0, ge=0, lt=30)
    # The Streamlit UI is read-only by default.  A reproducible snapshot can be
    # explicitly enabled for an offline demonstration, never silently used as LIVE.
    dashboard_fallback_enabled: bool = False
    dashboard_fallback_snapshot: Path = Path("public_demo/data/snapshot.json")
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    max_candle_age_seconds: int = Field(default=120, ge=1)
    model_confidence_threshold: float = Field(default=0.60, ge=0, le=1)
    paper_initial_capital: float = Field(default=1000.0, gt=0, le=1_000_000)
    paper_allocation: float = Field(default=0.20, gt=0, le=1)
    paper_fee_rate: float = Field(default=0.001, ge=0, lt=1)
    paper_slippage_rate: float = Field(default=0.0002, ge=0, lt=1)
    backfill_days: int = Field(default=30, ge=1)
    model_path: Path = Path("models/xgboost_v1.joblib")
    v3_shadow_enabled: bool = False
    prediction_lookback_candles: int = Field(default=120, ge=35)
    risk_volatility_medium: float = Field(default=0.002, ge=0)
    risk_volatility_high: float = Field(default=0.005, ge=0)
    risk_rsi_medium_low: float = Field(default=35.0, ge=0, le=100)
    risk_rsi_medium_high: float = Field(default=65.0, ge=0, le=100)
    risk_rsi_high_low: float = Field(default=25.0, ge=0, le=100)
    risk_rsi_high_high: float = Field(default=75.0, ge=0, le=100)
    risk_min_stop_pct: float = Field(default=0.002, gt=0, lt=1)
    risk_stop_vol_multiplier: float = Field(default=2.0, gt=0)
    risk_stop_cap_pct: float = Field(default=0.10, gt=0, lt=1)
    risk_reward_ratio: float = Field(default=1.5, gt=0)

    @property
    def marketaux_api_token(self) -> SecretStr:
        # Deliberately NOT a Settings field: no .env, constructor override or alias.
        return SecretStr(os.environ.get("MARKETAUX_API_TOKEN", "").strip())

    @property
    def llm_model_name(self) -> str:
        return self.nvidia_model if self.llm_provider == "nvidia" else self.ollama_model

    @field_validator("nvidia_reasoning_effort", mode="before")
    @classmethod
    def normalize_nvidia_reasoning_effort(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def validate_risk_thresholds(self) -> "Settings":
        if self.risk_volatility_medium >= self.risk_volatility_high:
            raise ValueError("RISK_VOLATILITY_MEDIUM debe ser menor que HIGH")
        if not (
            self.risk_rsi_high_low
            < self.risk_rsi_medium_low
            < self.risk_rsi_medium_high
            < self.risk_rsi_high_high
        ):
            raise ValueError("Los límites RSI de riesgo deben estar ordenados")
        if self.risk_min_stop_pct > self.risk_stop_cap_pct:
            raise ValueError("RISK_MIN_STOP_PCT no puede superar RISK_STOP_CAP_PCT")
        if self.risk_stop_cap_pct * self.risk_reward_ratio >= 1:
            raise ValueError("El take profit de una posición corta debe permanecer positivo")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
