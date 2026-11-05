"""Known LIVE cohorts for the public presentation metrics."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from alpha_engine.data.live_schemas import LiveEvaluationRecord
from alpha_engine.data.schemas import PredictionRecord
from alpha_engine.services.live_evaluation import LiveEvaluationService, _roc_auc_up
from alpha_engine.services.public_dashboard import PublicLiveMetrics


START = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


class Rows:
    def __init__(self, values):
        self.values = values

    def list_recent(self, symbol, limit=500, offset=0):
        assert symbol == "BTCUSDT"
        return self.values[offset:offset + limit]


def cohort(classes=("DOWN", "UP", "DOWN", "UP"),
           predictions=("DOWN", "DOWN", "UP", "UP"),
           probabilities=(.1, .4, .6, .9), version="xgboost_v1"):
    predicted_rows, evaluated_rows = [], []
    for index, (actual, predicted, prob_up) in enumerate(zip(classes, predictions, probabilities, strict=True)):
        source = START + timedelta(minutes=index)
        identifier = f"prediction-{index}"
        predicted_rows.append(PredictionRecord(
            prediction_id=identifier, symbol="BTCUSDT", timestamp=source,
            horizon="1m", prob_up=prob_up, prob_down=1 - prob_up,
            predicted_class=predicted, model_version=version,
            generated_at=source + timedelta(minutes=1, seconds=5),
            valid_until=source + timedelta(minutes=2) - timedelta(milliseconds=1),
        ))
        evaluated_rows.append(LiveEvaluationRecord(
            evaluation_id=uuid4(), prediction_id=identifier, symbol="BTCUSDT",
            evaluated_at=source + timedelta(minutes=2), source_close=100,
            target_open_time=source + timedelta(minutes=1),
            target_close_time=source + timedelta(minutes=2) - timedelta(milliseconds=1),
            target_close=101 if actual == "UP" else 99,
            observed_move=actual, observed_class=actual, correct=predicted == actual,
        ))
    return predicted_rows, evaluated_rows


def test_confusion_orientation_and_metrics_use_same_four_rows():
    predictions, evaluations = cohort()
    metrics = LiveEvaluationService.metrics_for(Rows(predictions), Rows(evaluations), "BTCUSDT")
    assert (metrics.evaluated, metrics.correct, metrics.incorrect) == (4, 2, 2)
    assert (metrics.pred_down_actual_down, metrics.pred_down_actual_up,
            metrics.pred_up_actual_down, metrics.pred_up_actual_up) == (1, 1, 1, 1)
    assert metrics.accuracy == metrics.balanced_accuracy == metrics.macro_f1 == .5
    assert metrics.precision_up == metrics.recall_up == metrics.f1_up == .5
    assert metrics.precision_down == metrics.recall_down == metrics.f1_down == .5
    assert metrics.always_down_accuracy == .5
    assert (metrics.predicted_up_fraction, metrics.predicted_down_fraction) == (.5, .5)
    assert (metrics.actual_up_fraction, metrics.actual_down_fraction) == (.5, .5)
    assert metrics.roc_auc_up == .75
    assert metrics.source_start == START
    assert metrics.source_end == START + timedelta(minutes=3)
    assert metrics.inconsistent_correct_rows == 0
    assert not {"context_json", "raw_response", "prompt", "service_role"} & set(PublicLiveMetrics.model_fields)


def test_single_class_and_tied_probabilities_do_not_invent_discrimination():
    predictions, evaluations = cohort(classes=("DOWN",), predictions=("DOWN",), probabilities=(.5,))
    metrics = LiveEvaluationService.metrics_for(Rows(predictions), Rows(evaluations), "BTCUSDT")
    assert metrics.roc_auc_up is None and metrics.balanced_accuracy is None
    assert metrics.always_down_accuracy == 1
    assert _roc_auc_up([(.5, False), (.5, True)]) == .5


def test_zero_true_positives_have_zero_f1_when_both_classes_present():
    predictions, evaluations = cohort(
        classes=("UP", "DOWN"), predictions=("DOWN", "UP"), probabilities=(.2, .8),
    )
    metrics = LiveEvaluationService.metrics_for(Rows(predictions), Rows(evaluations), "BTCUSDT")
    assert metrics.f1_up == metrics.f1_down == metrics.macro_f1 == 0.0
    assert metrics.balanced_accuracy == 0.0


def test_model_filter_keeps_a_single_version_and_period():
    predictions, evaluations = cohort()
    other_predictions, other_evaluations = cohort(classes=("UP",), predictions=("DOWN",), probabilities=(.2,), version="other")
    other_predictions[0] = other_predictions[0].model_copy(update={"prediction_id": "other-0"})
    other_evaluations[0] = other_evaluations[0].model_copy(update={"prediction_id": "other-0"})
    metrics = LiveEvaluationService.metrics_for(
        Rows(predictions + other_predictions), Rows(evaluations + other_evaluations),
        "BTCUSDT", "xgboost_v1",
    )
    assert metrics.evaluated == 4 and metrics.correct == 2
    assert metrics.source_end == START + timedelta(minutes=3)


def test_public_pages_separate_probability_accuracy_and_explain_experimental_gate():
    root = Path(__file__).resolve().parents[1] / "streamlit_app/pages"
    predictive = (root / "2_Predictivo.py").read_text(encoding="utf-8")
    prescriptive = (root / "3_Prescriptivo.py").read_text(encoding="utf-8")
    assert 'b.metric("P(SUBE)"' in predictive and 'c.metric("P(BAJA)"' in predictive
    assert '"Accuracy LIVE"' in predictive
    assert "No son accuracy ni rentabilidad esperada" in predictive
    assert "Gate prescriptivo experimental" in prescriptive
    assert "precisión histórica validada" in prescriptive
    assert 'if st.button("Generar recomendación"' in prescriptive
