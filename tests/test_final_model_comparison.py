"""Safeguard final evidence synthesis; no cloud calls or model fitting."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts import build_final_model_comparison as final


def test_candidate_registry_preserves_task_specific_controls() -> None:
    data = final.load_inputs()
    candidates = final.candidate_registry(data)
    assert [candidate["id"] for candidate in candidates] == [f"P{index}" for index in range(8)]
    assert candidates[0]["status"] == "PRODUCTION_REFERENCE"
    assert candidates[2]["task"] == candidates[3]["task"] == "ternary H1"
    assert "ternary" in candidates[2]["control"]
    assert candidates[4]["test"] is None  # H3 was not locked for final TEST.
    assert candidates[6]["control"] == "H1 no-news on identical Phase 7 rows"
    assert candidates[7]["control"] == "H5 no-news on identical Phase 7 rows"
    scorecard = final.scorecard(data)
    assert [row["candidate"] for row in scorecard] == [f"P{index}" for index in range(8)]
    assert all("leakage_risk" in row and "operational_complexity" in row for row in scorecard)
    assert scorecard[2]["result"] == "NOT_SUPPORTED"


def test_synthesis_rejects_changed_product_model_or_baseline() -> None:
    data = final.load_inputs()
    with pytest.raises(ValueError, match="SHA-256"):
        final.validate_inputs(data, "0" * 64)
    data["tfi"]["baseline_reproduction"]["test_class_flips"] = 1
    with pytest.raises(ValueError, match="reproduce classes"):
        final.validate_inputs(data, final.EXPECTED_SHA)


def test_final_reports_have_no_private_payload_fields() -> None:
    for filename in ("final_model_comparison_v1.json", "pmv_readiness_v1.json"):
        report = json.loads((final.REPORTS / filename).read_text(encoding="utf-8"))
        rendered = json.dumps(report, ensure_ascii=False).lower()
        for private_field in ("context_json", "raw_response", "service_role_key", "api_key", "authorization: bearer"):
            assert private_field not in rendered
    model = final.MODEL.read_bytes()
    assert hashlib.sha256(model).hexdigest() == final.EXPECTED_SHA
