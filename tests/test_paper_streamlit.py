"""Manual portfolio page cannot write until a button and reset confirmation."""

from pathlib import Path
from uuid import uuid4

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "streamlit_app/pages/5_Simulacion.py"


def snapshot(token, *, long=False, version=0):
    return {
        "portfolio_id": token, "initial_capital": 1000.0,
        "current_capital": 1000.0, "cash": 800.0 if long else 1000.0,
        "btc": 2.0 if long else 0.0, "position": "LONG" if long else "SIN_POSICION",
        "entry_price": 100.0 if long else None, "current_price": 100.0,
        "quote_received_at": "2026-09-26T12:00:00+00:00", "realized_pnl": 0.0,
        "unrealized_pnl": 0.0, "total_pnl": 0.0, "return_pct": 0.0,
        "closed_trades": 0, "win_rate_pct": None, "fees_paid": 0.0,
        "operation_count": 1 if long else 0,
        "version": version, "ledger": [],
        "prediction": {"prediction_id": "a" * 64, "predicted_class": "UP",
                       "prob_up": .7, "prob_down": .3,
                       "model_version": "xgboost_v1", "valid_until": "2026-09-26T12:01:00+00:00"},
        "suggestion": {"action": "HOLD" if long else "BUY", "executable": not long,
                       "reason": "Regla determinista.", "confidence": .7,
                       "signal_probability": .7},
    }


def test_create_apply_and_confirmed_reset_are_explicit(monkeypatch):
    token = str(uuid4())
    state = {"long": False, "version": 0, "calls": []}

    def fake_request(_settings, payload):
        state["calls"].append(payload)
        op = payload["op"]
        if op == "create":
            return {"status": "CREATED", "portfolio": snapshot(token)}
        if op == "apply":
            state["long"] = True
            state["version"] += 1
            return {"status": "APPLIED", "portfolio": snapshot(token, long=True, version=state["version"])}
        if op == "reset":
            state["long"] = False
            state["version"] += 1
            return {"status": "RESET", "portfolio": snapshot(token, version=state["version"])}
        return {"status": "OK", "portfolio": snapshot(token, long=state["long"], version=state["version"])}

    monkeypatch.setattr("alpha_engine.services.paper_api.paper_request", fake_request)
    app = AppTest.from_file(PAGE, default_timeout=20).run()
    assert not app.exception
    assert state["calls"] == []
    create = next(button for button in app.button if button.label == "Crear cartera virtual")
    app = create.click().run()
    assert not app.exception and state["calls"][0] == {"op": "create"}
    apply = next(button for button in app.button if button.label == "Aplicar operación simulada")
    assert not apply.disabled
    app = apply.click().run()
    assert not app.exception and state["long"]
    assert state["calls"][-2]["op"] == "apply" or state["calls"][-1]["op"] == "apply"
    reset_button = next(button for button in app.button if button.label == "Reiniciar simulación")
    assert reset_button.disabled
    confirmed = next(box for box in app.checkbox if box.label.startswith("Confirmo"))
    app = confirmed.check().run()
    reset_button = next(button for button in app.button if button.label == "Reiniciar simulación")
    assert not reset_button.disabled
    app = reset_button.click().run()
    assert not app.exception and not state["long"]
    assert sum(item["op"] == "reset" for item in state["calls"]) == 1
