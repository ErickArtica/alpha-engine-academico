from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scripts.analyze_live_down_bias import (
    contribution_aggregate, confusion_predicted_rows, excess_down_bias,
    feature_range, probability_bins, temporal_subset,
)


def test_excess_down_bias_uses_same_cohort():
    assert excess_down_bias(8, 5, 10) == pytest.approx(.3)
    with pytest.raises(ValueError):
        excess_down_bias(0, 0, 0)


def test_probability_bins_cover_edges_without_double_counting():
    probabilities = [.399, .4, .449, .45, .479, .48, .499, .5, .519, .52, .549, .55, .599, .6]
    bins = probability_bins(probabilities, [0, 1] * 7)
    assert [row["n"] for row in bins] == [1, 2, 2, 2, 2, 2, 2, 1]
    assert sum(row["n"] for row in bins) == len(probabilities)
    assert bins[0]["actual_up_rate"] == 0


def test_confusion_rows_are_predicted_columns_are_actual():
    actual = [0, 0, 1, 1, 1]
    predicted = [0, 1, 0, 0, 1]
    matrix = confusion_predicted_rows(actual, predicted)
    assert matrix == {"DOWN": {"DOWN": 1, "UP": 2}, "UP": {"DOWN": 1, "UP": 1}}


def test_feature_range_reports_extreme_and_quantile_exceedance():
    result = feature_range(np.arange(100), [-1, 0, 50, 99, 100])
    assert result["below_train_min"] == .2
    assert result["above_train_max"] == .2
    assert result["below_train_p01"] == .4
    assert result["above_train_p99"] == .4


def test_temporal_subset_uses_aware_strict_lower_boundary():
    end = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
    times = [end - timedelta(hours=1), end - timedelta(minutes=59), end,
             end + timedelta(seconds=1)]
    assert temporal_subset(times, end, timedelta(hours=1)).tolist() == [False, True, True, False]
    with pytest.raises(ValueError):
        temporal_subset(times, end.replace(tzinfo=None), timedelta(hours=1))


def test_contribution_aggregate_preserves_signed_push():
    result = contribution_aggregate([[-.2, .1], [-.4, -.1]], ["ema_20", "macd"])
    assert result["ema_20"]["mean"] == pytest.approx(-.3)
    assert result["ema_20"]["mean_absolute"] == pytest.approx(.3)
    assert result["macd"]["mean"] == pytest.approx(0)
