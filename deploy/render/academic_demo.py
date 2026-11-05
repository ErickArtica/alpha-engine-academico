"""Explicit classroom API: backend authorization, short sessions and durable budgets.

No existing service bearer, paper handler or scheduler is reachable through here.
Disabled by default; providers need separate explicit enablement and rights approval.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
import secrets
from threading import Lock

from deploy.render.academic_state import StateUnavailable, SupabaseState

PREFIX = "/api/academic/"
PATHS = {PREFIX + name for name in ("login", "logout", "context", "explain", "news", "refresh-news")}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


def b64(value: bytes):
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def password_hash(password: str, *, salt: bytes | None = None):
    if not isinstance(password, str) or not 12 <= len(password) <= 256:
        raise ValueError("Use an individual password of 12 to 256 characters")
    salt = salt or secrets.token_bytes(24)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600000)
    return "pbkdf2_sha256$600000$" + b64(salt) + "$" + b64(digest)


def valid_password(password, encoded):
    try:
        algorithm, iterations, salt, expected = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or iterations != "600000":
            return False
        salt_bytes = base64.urlsafe_b64decode(salt + "=" * (-len(salt) % 4))
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt_bytes, 600000)
        return hmac.compare_digest(b64(digest), expected)
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class DemoConfig:
    users: dict = field(repr=False)
    signing_key: str = field(repr=False)
    expires_at: int
    llm: bool = False
    news: bool = False
    news_rights: bool = False

    @classmethod
    def load(cls, env, now):
        if env.get("ACADEMIC_DEMO_ENABLED", "false").lower() != "true":
            return None
        end = datetime.fromisoformat(env.get("ACADEMIC_DEMO_EXPIRES_AT", "").replace("Z", "+00:00"))
        if end.tzinfo is None or not now < end.timestamp() <= now + 48 * 3600:
            raise ValueError("An explicit DEMO window of at most 48 hours is required")
        users = json.loads(env.get("ACADEMIC_DEMO_USERS_JSON", "{}"))
        if not isinstance(users, dict) or not 1 <= len(users) <= 25:
            raise ValueError("Authorized DEMO subjects required")
        for name, encoded in users.items():
            if (not re.fullmatch(r"[a-z0-9_-]{3,40}", name)
                    or not isinstance(encoded, str)
                    or not re.fullmatch(r"pbkdf2_sha256\$600000\$[A-Za-z0-9_-]{32}\$[A-Za-z0-9_-]{43}", encoded)):
                raise ValueError("Use salted password hashes, not plaintext credentials")
        key = env.get("ACADEMIC_DEMO_SESSION_SECRET", "")
        if len(key) < 48 or any(hmac.compare_digest(key, env.get(n, "")) for n in
                ("INTERNAL_CRON_SECRET", "RECOMMENDATION_API_SECRET", "NVIDIA_API_KEY", "SUPABASE_KEY")):
            raise ValueError("Independent DEMO session key required")
        return cls(users, key, int(end.timestamp()),
                   env.get("ACADEMIC_DEMO_LLM_ENABLED", "false").lower() == "true",
                   env.get("ACADEMIC_DEMO_NEWS_ENABLED", "false").lower() == "true",
                   env.get("ACADEMIC_DEMO_NEWS_RIGHTS_APPROVED", "false").lower() == "true")


class Quotas:
    """Reservation and cross-instance leases commit before external provider I/O."""
    def __init__(self, store, clock):
        self.store, self.clock = store, clock

    def reserve(self, kind, subject, session="", *, lease=False):
        now = int(self.clock())
        user, sid = fingerprint(subject), fingerprint(session)
        # Minute/hour/day counters and user limits survive process restarts.
        limits = {"login": [("global", 60, 5), ("global", 86400, 40), (user, 3600, 5)],
                  "api": [("global", 60, 120), (user, 60, 30)],
                  "llm": [("global", 86400, 10), (user, 86400, 3), (sid, 86400, 3), (user, 60, 1)],
                  "news": [("global", 86400, 48), (user, 86400, 10), (sid, 86400, 5)]}[kind]
        ticket = secrets.token_hex(16)
        def operation(state):
            counters = state.setdefault("counters", {})
            for key in list(counters):
                if counters[key][1] <= now:
                    del counters[key]
            leases = state.setdefault("leases", {})
            if lease and leases.get(kind, {}).get("until", 0) > now:
                return None
            keys = [(f"{kind}:{scope}:{period}:{now // period}", maximum, (now // period + 1) * period)
                    for scope, period, maximum in limits]
            if any(counters.get(key, [0])[0] >= maximum for key, maximum, _ in keys):
                return None
            for key, _, until in keys:
                counters[key] = [counters.get(key, [0])[0] + 1, until]
            if lease:
                leases[kind] = {"ticket": ticket, "until": now + 120}
            return ticket
        return self.store.update(operation)

    def finish(self, kind, ticket, subject, status, context_hash=None):
        now = int(self.clock())
        def operation(state):
            if state.get("leases", {}).get(kind, {}).get("ticket") == ticket:
                del state["leases"][kind]
            receipts = state.setdefault("receipts", [])
            receipts.append(dict(at=now, subject=fingerprint(subject), kind=kind, status=status,
                                 context_hash=context_hash))
            state["receipts"] = receipts[-50:]
        self.store.update(operation)


class AcademicDemo:
    def __init__(self, environment, settings_factory, *, store=None, service=None, clock=None):
        import time
        self.env, self.settings_factory = environment, settings_factory
        self.clock = clock or time.time
        self.store, self.service = store, service
        self._ingress_lock = Lock()
        self._ingress_window, self._ingress_count = -1, 0

    def _session(self, token, cfg):
        try:
            if len(token) > 1500:
                return None
            message, signature = token.split(".")
            expected = b64(hmac.digest(cfg.signing_key.encode(), message.encode(), "sha256"))
            if not hmac.compare_digest(signature, expected):
                return None
            claims = json.loads(base64.urlsafe_b64decode(message + "=" * (-len(message) % 4)))
            now = int(self.clock())
            if (set(claims) != {"sub", "sid", "iat", "exp", "aud"}
                    or claims["aud"] != "alpha-engine-academic"
                    or claims["sub"] not in cfg.users or not re.fullmatch(r"[0-9a-f]{32}", claims["sid"])
                    or type(claims["iat"]) is not int or type(claims["exp"]) is not int
                    or not claims["iat"] <= now < claims["exp"] <= min(cfg.expires_at, claims["iat"] + 900)):
                return None
            _, state = self.store.read()
            if state.get("revoked", {}).get(claims["sid"], 0) > now:
                return None
            return claims
        except (ValueError, TypeError, KeyError):
            return None

    async def handle(self, scope, read_json):
        if scope.get("path") not in PATHS:
            return 404, {"status": "NOT_FOUND"}
        if scope.get("method") != "POST":
            return 405, {"status": "METHOD_NOT_ALLOWED"}
        try:
            cfg = DemoConfig.load(self.env, self.clock())
            if cfg is None:
                return 404, {"status": "NOT_FOUND"}
            # Cheap process guard before database I/O, including failed logins.
            # Durable quotas still enforce provider limits across workers/restarts.
            with self._ingress_lock:
                window = int(self.clock()) // 60
                if self._ingress_window != window:
                    self._ingress_window, self._ingress_count = window, 0
                if self._ingress_count >= 60:
                    return 429, {"status": "RATE_LIMIT"}
                self._ingress_count += 1
            if self.store is None:
                self.store = SupabaseState(self.settings_factory())
            quotas = Quotas(self.store, self.clock)
            headers = scope.get("headers", [])
            ctype = [v for k, v in headers if k.lower() == b"content-type"]
            if len(ctype) != 1 or ctype[0].split(b";", 1)[0].strip() != b"application/json":
                return 415, {"status": "UNSUPPORTED_MEDIA_TYPE"}
            path = scope["path"].removeprefix(PREFIX)
            claims = None
            if path != "login":
                auth = [v for k, v in headers if k.lower() == b"authorization"]
                if len(auth) == 1 and auth[0].startswith(b"Bearer "):
                    claims = await asyncio.to_thread(self._session, auth[0][7:].decode("ascii"), cfg)
                if claims is None:
                    return 401, {"status": "UNAUTHORIZED"}
                if not await asyncio.to_thread(quotas.reserve, "api", claims["sub"], claims["sid"]):
                    return 429, {"status": "RATE_LIMIT"}
            payload = await read_json()
            if not isinstance(payload, dict):
                return 400, {"status": "INVALID_REQUEST"}
            if path == "login":
                if (set(payload) != {"user", "password"} or not isinstance(payload["user"], str)
                        or not re.fullmatch(r"[a-z0-9_-]{3,40}", payload["user"])
                        or not isinstance(payload["password"], str) or not 12 <= len(payload["password"]) <= 256):
                    return 401, {"status": "UNAUTHORIZED"}
                if not await asyncio.to_thread(quotas.reserve, "login", payload["user"]):
                    return 429, {"status": "RATE_LIMIT"}
                encoded = cfg.users.get(payload["user"], next(iter(cfg.users.values())))
                valid = await asyncio.to_thread(valid_password, payload["password"], encoded)
                if not valid or payload["user"] not in cfg.users:
                    return 401, {"status": "UNAUTHORIZED"}
                now = int(self.clock())
                claims = dict(sub=payload["user"], sid=secrets.token_hex(16), iat=now,
                              exp=min(now + 900, cfg.expires_at), aud="alpha-engine-academic")
                message = b64(canonical(claims).encode())
                token = message + "." + b64(hmac.digest(cfg.signing_key.encode(), message.encode(), "sha256"))
                return 200, {"status": "AUTHORIZED", "access_token": token, "expires_at": claims["exp"]}
            if path == "logout":
                if payload:
                    return 400, {"status": "INVALID_REQUEST"}
                def revoke(state):
                    now = self.clock()
                    state["revoked"] = {k: v for k, v in state.get("revoked", {}).items() if v > now}
                    state["revoked"][claims["sid"]] = claims["exp"]
                await asyncio.to_thread(self.store.update, revoke)
                return 200, {"status": "SIGNED_OUT"}
            if self.service is None:
                from deploy.render.academic_service import AcademicService
                self.service = AcademicService(self.settings_factory(), self.store, self.clock)
            return await asyncio.to_thread(self.service.handle, path, payload, claims, cfg, quotas)
        except OverflowError:
            return 413, {"status": "REQUEST_TOO_LARGE"}
        except (UnicodeError, ValueError, TypeError):
            return 400, {"status": "INVALID_REQUEST"}
        except Exception:
            # No raw exception, credential, prompt, password or provider response in logs.
            return 503, {"status": "DEMO_UNAVAILABLE"}
