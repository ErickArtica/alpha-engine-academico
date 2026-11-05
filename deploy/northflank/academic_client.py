"""Session-scoped server-to-server DEMO transport. No service/provider credential."""
from urllib.parse import urlsplit

import httpx


class DemoUnavailable(RuntimeError):
    pass


def endpoint(environment):
    if environment.get("ACADEMIC_DEMO_ENABLED", "false").lower() != "true":
        return None
    value = environment.get("ACADEMIC_DEMO_API_URL", "").rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
            or "." not in parsed.hostname):
        raise DemoUnavailable("Configure the HTTPS DEMO backend origin")
    return value


class DemoClient:
    def __init__(self, origin, *, transport=None):
        self.origin = endpoint({"ACADEMIC_DEMO_ENABLED": "true", "ACADEMIC_DEMO_API_URL": origin})
        self.transport = transport

    def post(self, action, payload, token=None):
        if action not in {"login", "logout", "context", "explain", "news", "refresh-news"}:
            raise DemoUnavailable("Unknown DEMO action")
        if action != "login" and not token:
            return 401, {"status": "UNAUTHORIZED"}
        headers = {"Authorization": "Bearer " + token} if token else {}
        try:
            with httpx.Client(transport=self.transport, timeout=50, trust_env=False,
                              follow_redirects=False) as client:
                with client.stream("POST", self.origin + "/api/academic/" + action,
                                   json=payload, headers=headers) as response:
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        data.extend(chunk)
                        if len(data) > 32000:
                            raise DemoUnavailable("DEMO response capacity")
                    import json
                    result = json.loads(data)
                    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                        raise ValueError("Invalid DEMO response")
                    return response.status_code, result
        except Exception:
            raise DemoUnavailable("El servicio DEMO no está disponible; no se reintentó la operación.") from None
