"""Fail-closed configuration for the Northflank public reader; no dotenv or I/O."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
import re
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    pass


def public_hosts(env: dict[str, str]) -> list[str]:
    candidates = [env.get("FRONTEND_PUBLIC_HOST", ""), *env.get("NF_HOSTS", "").split(",")]
    hosts = []
    pattern = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*"
    for candidate in candidates:
        host = candidate.strip().lower()
        if not host:
            continue
        if len(host) > 253 or re.fullmatch(pattern, host) is None:
            raise ConfigurationError("Use public hostnames without scheme, port or path")
        if host not in hosts:
            hosts.append(host)
    if not hosts:
        raise ConfigurationError("NF_HOSTS or FRONTEND_PUBLIC_HOST is required")
    return hosts


@dataclass(frozen=True)
class PublicConfig:
    supabase_url: str
    supabase_key: str = field(repr=False)
    hosts: tuple[str, ...]
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    confidence_threshold: float = 0.60

    @classmethod
    def from_environment(cls, environment: dict[str, str]) -> "PublicConfig":
        env = {}
        for key, value in environment.items():
            name = key.upper()
            if name in env and env[name] != value:
                raise ConfigurationError("Conflicting environment variable case")
            env[name] = value
        if env.get("NORTHFLANK_PUBLIC_READ_ONLY", "").lower() != "true":
            raise ConfigurationError("NORTHFLANK_PUBLIC_READ_ONLY=true is mandatory; no full-dashboard fallback")
        if env.get("NORTHFLANK_PUBLIC_DATA_APPROVED", "").lower() != "true":
            raise ConfigurationError("Review public table/column permissions before setting NORTHFLANK_PUBLIC_DATA_APPROVED=true")
        if env.get("STORAGE_PROVIDER") != "supabase":
            raise ConfigurationError("The public reader requires STORAGE_PROVIDER=supabase")
        for name in ("PRESCRIPTIVE_NEWS_ENABLED", "DASHBOARD_FALLBACK_ENABLED", "LOCAL_DEMO_MODE", "V3_SHADOW_ENABLED"):
            if env.get(name, "false").lower() != "false":
                raise ConfigurationError(name + " must remain false")
        forbidden = ("RECOMMENDATION_API_URL", "RECOMMENDATION_API_KEY", "RECOMMENDATION_API_AUTH_SCHEME",
                     "NVIDIA_API_KEY", "MARKETAUX_API_TOKEN", "ALPHA_VANTAGE_API_KEY", "OPENAI_API_KEY",
                     "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "SUPABASE_ACCESS_TOKEN",
                     "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SECRET_KEY", "INTERNAL_CRON_SECRET", "DATABASE_URL",
                     "INFLUX_TOKEN", "INFLUX3_TOKEN", "INFLUX_PASSWORD", "OLLAMA_BASE_URL")
        credential_names = {name for name in env if name not in {"SUPABASE_KEY", "GPG_KEY"} and (
            name.endswith(("_KEY", "_TOKEN", "_SECRET", "_DSN")) or "_SECRET_" in name or "PASSWORD" in name)}
        if any(env.get(name) for name in set(forbidden) | credential_names):
            raise ConfigurationError("Action/backend/provider/database credentials and endpoints are forbidden in this image")
        if any(value for name, value in env.items() if name.startswith("ACADEMIC_DEMO_")
               and name not in {"ACADEMIC_DEMO_ENABLED", "ACADEMIC_DEMO_API_URL"}):
            raise ConfigurationError("DEMO server configuration belongs exclusively on the backend")
        from deploy.northflank.academic_client import endpoint
        try:
            endpoint(env)
        except (ValueError, RuntimeError):
            raise ConfigurationError("Invalid DEMO backend origin") from None
        try:
            gate = Decimal(env.get("MODEL_CONFIDENCE_THRESHOLD", "0.60"))
        except InvalidOperation:
            raise ConfigurationError("MODEL_CONFIDENCE_THRESHOLD must be 0.60") from None
        if not gate.is_finite() or gate != Decimal("0.60"):
            raise ConfigurationError("MODEL_CONFIDENCE_THRESHOLD must be 0.60")
        url = env.get("SUPABASE_URL", "")
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.port not in (None, 443)
                or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
            raise ConfigurationError("SUPABASE_URL must be a plain HTTPS project URL")
        key = env.get("SUPABASE_KEY", "")
        if not key.startswith("sb_publishable_"):
            try:
                header, claims, signature = key.split(".")
                role = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4))).get("role")
            except (ValueError, UnicodeError, AttributeError):
                role = None
            if role != "anon":
                raise ConfigurationError("Use a publishable or legacy anon key; service/admin keys are forbidden")
        symbol = env.get("SYMBOL", "BTCUSDT")
        if re.fullmatch(r"[A-Z0-9]{3,20}", symbol) is None or env.get("INTERVAL", "1m") != "1m":
            raise ConfigurationError("Use an alphanumeric SYMBOL and INTERVAL=1m")
        return cls(url.rstrip("/"), key, tuple(public_hosts(env)), symbol)
