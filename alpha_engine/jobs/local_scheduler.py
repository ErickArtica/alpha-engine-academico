"""Compose clock: one authenticated POST per minute, no business logic."""

import logging
import os
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)


def seconds_to_next_tick(now: datetime) -> float:
    if now.tzinfo is None:
        raise ValueError("Scheduler clock must be timezone aware")
    return 65.0 - (now.timestamp() % 60)


def run_once(client: httpx.Client, api_url: str, secret: str) -> int:
    response = client.post(
        api_url.rstrip("/") + "/internal/run-cycle",
        headers={"Authorization": f"Bearer {secret}"}, json={},
    )
    return response.status_code


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    secret = os.environ["INTERNAL_CRON_SECRET"]
    api_url = os.environ.get("LOCAL_API_URL", "http://api:8000")
    heartbeat = "/tmp/alpha_scheduler_heartbeat"
    with open(heartbeat, "w", encoding="utf-8") as handle:
        handle.write(str(time.time()))
    with httpx.Client(timeout=35) as client:
        while True:
            time.sleep(seconds_to_next_tick(datetime.now(timezone.utc)))
            with open(heartbeat, "w", encoding="utf-8") as handle:
                handle.write(str(time.time()))
            try:
                code = run_once(client, api_url, secret)
                logger.info("LOCAL_SCHEDULER run-cycle HTTP %s", code)
            except httpx.HTTPError as exc:
                logger.warning("LOCAL_SCHEDULER request failed: %s", type(exc).__name__)


if __name__ == "__main__":
    main()
