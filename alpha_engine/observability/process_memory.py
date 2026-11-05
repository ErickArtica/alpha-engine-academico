"""Opt-in, low-volume Linux RSS diagnostics for the Render process."""

from __future__ import annotations

import logging
import os
import threading


_ENABLED = os.environ.get("ALPHA_MEMORY_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}
_LOGGER = logging.getLogger("uvicorn.error")
_CURRENT_RUN = threading.local()


def memory_diagnostics_enabled() -> bool:
    return _ENABLED


def memory_event(event: str, *, run_id: str | None = None) -> None:
    """Log process current/peak resident MiB; never affect the work being measured."""
    if not _ENABLED:
        return
    if event == "cycle_begin":
        _CURRENT_RUN.run_id = run_id
    selected_run_id = run_id or getattr(_CURRENT_RUN, "run_id", None)
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            values = {}
            for line in status:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, amount, unit = line.split()
                    if unit == "kB":
                        values[key.rstrip(":")] = int(amount) / 1024
        rss = values.get("VmRSS")
        peak = values.get("VmHWM")
        if rss is None or peak is None:
            return
        _LOGGER.info(
            "PROCESS_MEMORY event=%s pid=%d ppid=%d tid=%d rss_mb=%.2f "
            "peak_mb=%.2f thread_count=%d%s",
            event, os.getpid(), os.getppid(), threading.get_ident(), rss, peak,
            threading.active_count(),
            f" run_id={selected_run_id}" if selected_run_id else "",
        )
    except (OSError, ValueError):
        return
    finally:
        if event == "cycle_end" and hasattr(_CURRENT_RUN, "run_id"):
            del _CURRENT_RUN.run_id
