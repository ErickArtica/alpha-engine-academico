"""Offline tests for the causal historical TFI experiment."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_engine.data.schemas import Candle
from alpha_engine.experiments.microstructure import trade_flow_from_klines
from alpha_engine.features import FEATURE_COLUMNS
from scripts.backfill_trade_flow import validate_row
from scripts.run_tfi_ablation import (
    A_FEATURES, B_FEATURES, C_FEATURES, EXPECTED_PROD_SHA, PROD,
    expanding_folds, metrics, paired_block_bootstrap, read_backfill,
    reproduction_gate,
)


T = datetime(2026, 8, 18, 20, 24, tzinfo=timezone.utc)


def candle(at=T, volume=10.0):
    return Candle(symbol="BTCUSDT",interval="1m",open_time=at,
        close_time=at+timedelta(minutes=1,milliseconds=-1),
        open=100,high=102,low=99,close=101,volume=volume)


def kline(at=T, volume="10", buy="7"):
    return [int(at.timestamp()*1000),"100","102","99","101",volume,
            int((at+timedelta(minutes=1,milliseconds=-1)).timestamp()*1000),"1000",10,buy,"700","0"]


def test_kline_tfi_uses_only_closed_source_and_derives_buy_sell():
    row=validate_row(kline(),candle())
    assert row["trade_flow_imbalance"]=="0.4"
    assert row["aggressive_sell_volume"]=="3"
    assert row["feature_cutoff_time"]==candle().close_time.isoformat()
    with pytest.raises(ValueError,match="timestamp"):
        validate_row(kline(T+timedelta(minutes=1)),candle())


def test_zero_volume_is_missing_and_bad_buy_or_price_rejected():
    row=validate_row(kline(volume="0",buy="0"),candle(volume=0))
    assert row["trade_flow_imbalance"] is None
    with pytest.raises(ValueError):
        trade_flow_from_klines("10","11")
    bad=kline();bad[4]="103"
    with pytest.raises(ValueError,match="close"):
        validate_row(bad,candle())


def test_exact_dataset_join_rejects_missing_duplicates_future(tmp_path,monkeypatch):
    import scripts.run_tfi_ablation as ablation
    backfill=tmp_path/"backfill.jsonl";report=tmp_path/"report.json"
    row=validate_row(kline(),candle())
    backfill.write_text(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    report.write_text(json.dumps({"dataset_sha256":hashlib.sha256(backfill.read_bytes()).hexdigest(),"rows":1}))
    monkeypatch.setattr(ablation,"BACKFILL",backfill)
    monkeypatch.setattr(ablation,"BACKFILL_REPORT",report)
    frame=pd.DataFrame(index=pd.DatetimeIndex([T]))
    assert read_backfill(frame).iloc[0]==pytest.approx(.4)
    with pytest.raises(ValueError,match="join"):
        read_backfill(pd.DataFrame(index=pd.DatetimeIndex([T,T+timedelta(minutes=1)])))
    row["feature_cutoff_time"]=(T+timedelta(minutes=1)).isoformat()
    backfill.write_text(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    report.write_text(json.dumps({"dataset_sha256":hashlib.sha256(backfill.read_bytes()).hexdigest(),"rows":1}))
    with pytest.raises(ValueError,match="alignment"):
        read_backfill(frame)


def test_expanding_folds_purge_and_never_include_future_rows():
    times=pd.date_range(T,periods=1000,freq="1min")
    frame=pd.DataFrame({"target":np.arange(1000)%2},index=times)
    folds=list(expanding_folds(frame,count=4))
    assert len(folds)==4
    for number,train,validation in folds:
        assert number>=1
        assert train.index.max()+timedelta(minutes=2)==validation.index.min()
        assert train.index.max()<validation.index.min()
    assert folds[-1][2].index.max()==frame.index.max()


def test_metrics_and_block_bootstrap_identical_models():
    y=np.array([0,1]*120)
    p=np.tile([.4,.6],120)
    score=metrics(y,p)
    assert score["accuracy"]==1
    assert score["confusion_real_rows_down_up_pred_columns_down_up"]==[[120,0],[0,120]]
    paired=paired_block_bootstrap(y,p,p,block=20,repeats=100,seed=7)
    assert all(v["lower_95"]==v["upper_95"]==0 for v in paired["differences_B_minus_A"].values())


def test_feature_order_and_original_model_are_immutable():
    assert A_FEATURES==list(FEATURE_COLUMNS)
    assert B_FEATURES==list(FEATURE_COLUMNS)+["trade_flow_imbalance"]
    assert C_FEATURES==["trade_flow_imbalance"]
    assert hashlib.sha256(PROD.read_bytes()).hexdigest()==EXPECTED_PROD_SHA


def test_baseline_gate_requires_matching_frozen_probabilities():
    assert reproduction_gate(np.array([.40,.60]),np.array([.40,.60]),label="VALIDATION") == {
        "max_abs_prob_difference":0.0,"class_flips":0}
    with pytest.raises(RuntimeError,match="reproduction gate"):
        reproduction_gate(np.array([.40,.51]),np.array([.40,.49]),label="VALIDATION")
    with pytest.raises(ValueError,match="unaligned"):
        reproduction_gate(np.array([.40]),np.array([.40,.60]),label="VALIDATION")
