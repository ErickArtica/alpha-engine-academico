"""Fixed 12-month range, H1 purges and six monthly development folds."""

import pandas as pd
import pytest

from scripts.build_v2_dataset import temporal_splits
from scripts.prepare_xgboost_v3_12m import END, SPLIT_BOUNDS, START, missing_archives
from scripts.run_xgboost_v3_12m_robustness import MONTHS, monthly_expanding_folds


def test_requested_calendar_and_missing_archives_are_fixed():
    assert START.isoformat() == "2025-09-22T00:00:00+00:00"
    assert END.isoformat() == "2026-09-21T23:59:00+00:00"
    assert len(pd.date_range(START, END, freq="min")) == 525_600
    assert [archive.period for archive in missing_archives()] == [
        "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
    assert len(MONTHS) == 6


def test_12m_split_boundary_purges_last_h1_label():
    stamps = pd.to_datetime([
        "2025-09-22T00:33Z", "2026-07-31T23:58Z", "2026-07-31T23:59Z",
        "2026-08-01T00:00Z", "2026-08-31T23:58Z", "2026-08-31T23:59Z",
        "2026-09-01T00:00Z", "2026-09-15T23:58Z", "2026-09-15T23:59Z",
        "2026-09-16T00:00Z", "2026-09-21T23:58Z"], utc=True)
    frame = pd.DataFrame({"timestamp": stamps,
                          "target_open_time": stamps + pd.offsets.Minute(1),
                          "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})
    splits, audit = temporal_splits(frame, SPLIT_BOUNDS)
    assert [audit[name]["rows"] for name in SPLIT_BOUNDS] == [2, 2, 2, 2]
    for left, right in (("TRAIN", "VALIDATION"), ("VALIDATION", "TEST"), ("TEST", "FORWARD")):
        assert splits[left].target_open_time.max() < splits[right].timestamp.min()
        assert splits[left].timestamp.max() == splits[right].timestamp.min() - pd.offsets.Minute(2)


def test_monthly_walk_forward_is_expanding_and_never_uses_test():
    stamps = [START]
    for first, after_last in MONTHS:
        start, stop = pd.Timestamp(first, tz="UTC"), pd.Timestamp(after_last, tz="UTC")
        stamps += [start - pd.offsets.Minute(2), start - pd.offsets.Minute(1),
                   start, stop - pd.offsets.Minute(2), stop - pd.offsets.Minute(1)]
    frame = pd.DataFrame({"timestamp": sorted(set(stamps))})
    frame["target_open_time"] = frame.timestamp + pd.offsets.Minute(1)
    folds = list(monthly_expanding_folds(frame))
    assert len(folds) == 6
    for number, train, valid in folds:
        assert valid.timestamp.min() == pd.Timestamp(MONTHS[number - 1][0], tz="UTC")
        assert train.timestamp.max() == valid.timestamp.min() - pd.offsets.Minute(2)
        assert train.target_open_time.max() < valid.timestamp.min()
        assert valid.timestamp.max() < pd.Timestamp("2026-09-01", tz="UTC")
        assert valid.target_open_time.max() < pd.Timestamp("2026-09-01", tz="UTC")
    assert folds[-1][2].timestamp.max() == pd.Timestamp("2026-08-31T23:58:00Z")
    assert all(len(folds[i][1]) < len(folds[i + 1][1]) for i in range(5))


def test_monthly_walk_forward_rejects_out_of_order_sources():
    stamps = pd.to_datetime(["2026-03-01T00:01Z", "2026-03-01T00:00Z"], utc=True)
    frame = pd.DataFrame({"timestamp": stamps, "target_open_time": stamps + pd.offsets.Minute(1)})
    with pytest.raises(ValueError, match="ordered unique"):
        list(monthly_expanding_folds(frame))
