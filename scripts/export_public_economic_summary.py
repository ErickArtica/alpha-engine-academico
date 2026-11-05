"""Derive aggregate simulation metrics without per-trade market observations."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "economic_check_v1.json"
TARGET = ROOT / "models" / "economic_summary_v1.json"


def _without_trades(value):
    if isinstance(value, dict):
        return {key: _without_trades(item) for key, item in value.items()
                if key != "trades"}
    if isinstance(value, list):
        return [_without_trades(item) for item in value]
    return value


def summarize() -> Path:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    summary = _without_trades(report)
    TARGET.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return TARGET


if __name__ == "__main__":
    print(summarize())
