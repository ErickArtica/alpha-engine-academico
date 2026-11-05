"""Private atomic DEMO state. No market tables, user credentials or auto migrations."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Callable

import httpx


class StateUnavailable(RuntimeError):
    pass


class AtomicState:
    def update(self, operation: Callable):
        for _ in range(8):
            version, state = self.read()
            before = json.dumps(state, sort_keys=True, allow_nan=False)
            result = operation(state)
            if json.dumps(state, sort_keys=True, allow_nan=False) == before:
                return result
            if self.cas(version, state):
                return result
        raise StateUnavailable("Concurrent DEMO requests; retry later")


class SQLiteState(AtomicState):
    """Local tests only. Hosted wiring never selects this ephemeral store."""
    def __init__(self, path: str | Path):
        self.path = str(path)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS demo_state (id INTEGER PRIMARY KEY, version INTEGER, payload TEXT)")
            db.execute("INSERT OR IGNORE INTO demo_state VALUES (1,0,'{}')")

    def read(self):
        with closing(sqlite3.connect(self.path)) as db:
            version, payload = db.execute("SELECT version,payload FROM demo_state WHERE id=1").fetchone()
        return version, json.loads(payload)

    def cas(self, version, state):
        payload = json.dumps(state, allow_nan=False, separators=(",", ":"))
        if len(payload.encode()) > 262144:
            raise StateUnavailable("DEMO state capacity")
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute("UPDATE demo_state SET version=version+1,payload=? WHERE id=1 AND version=?",
                              (payload, version)).rowcount == 1


class SupabaseState(AtomicState):
    """Two narrow service-only RPCs, provisioned separately after SQL review."""
    def __init__(self, settings, *, transport=None):
        from urllib.parse import urlsplit
        url, key = settings.supabase_url.rstrip("/"), settings.supabase_key.get_secret_value()
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment or not key):
            raise StateUnavailable("Private state not configured")
        self.url, self.key, self.transport = url, key, transport

    def _rpc(self, name, body):
        try:
            headers = {"apikey": self.key}
            if not self.key.startswith("sb_secret_"):
                headers["Authorization"] = "Bearer " + self.key
            with httpx.Client(transport=self.transport, timeout=5, trust_env=False,
                              follow_redirects=False) as client:
                response = client.post(self.url + "/rest/v1/rpc/" + name, headers=headers, json=body)
                response.raise_for_status()
                return response.json()
        except Exception:
            raise StateUnavailable("Private DEMO state unavailable") from None

    def read(self):
        data = self._rpc("academic_demo_state_read", {})
        if (not isinstance(data, dict) or type(data.get("version")) is not int
                or not isinstance(data.get("payload"), dict)):
            raise StateUnavailable("Private DEMO state invalid")
        return data["version"], data["payload"]

    def cas(self, version, state):
        if len(json.dumps(state, allow_nan=False).encode()) > 262144:
            raise StateUnavailable("Private DEMO state capacity")
        result = self._rpc("academic_demo_state_cas", {"p_version": version, "p_payload": state})
        if type(result) is not bool:
            raise StateUnavailable("Private DEMO state invalid")
        return result
