"""Causal Alpha Vantage parsing/features and frozen-product safety."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_hold_target import PRODUCT_SHA, ROOT, sha
from scripts.build_news_sentiment_features import (FEATURES, article_key, build_features,
                                                    parse_article, parse_time)
from scripts.fetch_alpha_vantage_news import stamp, validate_payload
from scripts.run_news_sentiment_ablation import paired_bootstrap, x_for


def sample_article(time="20260917T100459", score="0.3", relevance="0.7"):
    return {"url": "https://example.org/news?id=1", "time_published": time,
            "source": "Test", "ticker_sentiment": [{"ticker": "CRYPTO:BTC",
            "ticker_sentiment_score": score, "relevance_score": relevance,
            "ticker_sentiment_label": "Somewhat_Bullish"}],
            "overall_sentiment_score": -0.9}


def test_parse_btc_specific_not_overall_and_utc():
    parsed = parse_article(sample_article())
    assert parsed["score"] == pytest.approx(.3)
    assert parsed["relevance"] == pytest.approx(.7)
    assert parsed["published_at_utc"] == datetime(2026, 9, 17, 10, 4, 59, tzinfo=timezone.utc)
    assert article_key(sample_article()) == "https://example.org/news"
    assert parse_time("20260917T100459").utcoffset().total_seconds() == 0
    with pytest.raises(ValueError):
        parse_time("2026-09-17")
    with pytest.raises(ValueError):
        stamp(datetime(2026, 9, 17, 10, 5))


def test_schema_rate_limit_and_api_error_are_not_feed():
    assert validate_payload({"feed": [sample_article()]})
    with pytest.raises(RuntimeError, match="RATE_OR_PLAN"):
        validate_payload({"Information": "quota"})
    with pytest.raises(ValueError):
        validate_payload({"Error Message": "bad input"})
    with pytest.raises(ValueError):
        validate_payload({"feed": "bad"})


def test_publication_cutoff_and_backward_windows():
    articles = pd.DataFrame([parse_article(sample_article("20260917T100459", ".4", ".25")),
                             parse_article(sample_article("20260917T100500", "-.4", ".75"))])
    closes = pd.DatetimeIndex(["2026-09-17T10:04:59+00:00", "2026-09-17T10:05:00+00:00"])
    result = build_features(closes, articles)
    assert result.iloc[0]["news_count_15m"] == 1
    assert result.iloc[0]["btc_sentiment_15m"] == pytest.approx(.4)
    assert result.iloc[0]["sentiment_cutoff_time"] == closes[0]
    assert result.iloc[1]["news_count_15m"] == 2
    assert result.iloc[1]["btc_sentiment_15m"] == pytest.approx(-.2)
    assert result.iloc[1]["sentiment_cutoff_time"] == closes[1]


def test_no_news_is_explicit_not_neutral():
    article = pd.DataFrame([parse_article(sample_article("20260917T100500"))])
    result = build_features(pd.DatetimeIndex(["2026-09-17T10:04:59+00:00"]), article)
    assert result.iloc[0]["news_count_15m"] == 0
    assert result.iloc[0]["news_count_60m"] == 0
    assert np.isnan(result.iloc[0]["btc_sentiment_15m"])
    assert np.isnan(result.iloc[0]["minutes_since_last_news"])
    assert pd.isna(result.iloc[0]["sentiment_cutoff_time"])


def test_window_left_boundary_excluded():
    article = pd.DataFrame([parse_article(sample_article("20260917T095000"))])
    closes = pd.DatetimeIndex(["2026-09-17T10:05:00+00:00"])
    result = build_features(closes, article)
    assert result.iloc[0]["news_count_15m"] == 0
    assert result.iloc[0]["news_count_60m"] == 1


def test_malformed_btc_and_relevance_rejected():
    wrong = sample_article()
    wrong["ticker_sentiment"][0]["ticker"] = "CRYPTO:ETH"
    with pytest.raises(ValueError):
        parse_article(wrong)
    with pytest.raises(ValueError):
        parse_article(sample_article(relevance="0"))


def test_exact_feature_order_and_no_future_column():
    frame = pd.DataFrame({"return_1m": [.1], "btc_sentiment_15m": [np.nan], "target": [1]})
    x = x_for(frame, ("return_1m", "btc_sentiment_15m"))
    assert list(x.columns) == ["return_1m", "btc_sentiment_15m"]
    with pytest.raises(ValueError):
        x_for(frame, ("return_1m", "target"))


def test_pair_bootstrap_requires_same_rows():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(250), np.zeros(249), np.zeros(250))


def test_product_model_immutable_and_no_secret_in_public_manifest():
    assert sha(ROOT / "models/xgboost_v1.joblib") == PRODUCT_SHA
    text = (ROOT / "reports/alpha_vantage_download_manifest_v1.json").read_text(encoding="utf-8")
    assert "apikey" not in text.lower()
    assert "ALPHA_VANTAGE_API_KEY" not in text
    assert "data/experiments/alpha_vantage_news/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
