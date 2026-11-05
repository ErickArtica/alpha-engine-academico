"""Generate aggregate-only FASE 7 reports from private cached research inputs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.analyze_hold_target import ROOT, write_json
from scripts.analyze_horizons import build_horizon_frames, common_frames, common_splits
from scripts.build_news_sentiment_features import AUDIT, FEATURE_REPORT, load_articles
from scripts.run_news_sentiment_ablation import DEVELOPMENT, FINAL, LOCK, WALK


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows) + "\n"


def metric_rows(models: dict, names: tuple[str, ...]) -> list[list[object]]:
    metrics = ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc_up", "average_precision_up",
               "precision_up", "recall_up", "f1_up", "precision_down", "recall_down",
               "f1_down", "macro_f1", "brier", "log_loss", "predicted_up_fraction")
    return [[key] + [f"{models[name]['metrics'][key]:.4f}" for name in names] for key in metrics]


def event_study(articles: pd.DataFrame) -> dict:
    frames, _ = build_horizon_frames()
    common = common_frames(frames)
    closes = pd.DatetimeIndex(common[1]["source_candle_close_time"])
    out = {h: {label: [] for label in ("negative", "neutral", "positive")} for h in (1, 5)}
    for article in articles.itertuples(index=False):
        published = pd.Timestamp(article.published_at_utc)
        idx = int(np.searchsorted(closes.asi8, published.value, side="left"))
        if idx >= len(closes) or published < common[1].index[0]:
            continue
        label = "negative" if article.score <= -.15 else "positive" if article.score >= .15 else "neutral"
        for h in (1, 5):
            out[h][label].append(float(common[h].iloc[idx]["future_return_bps"]))
    return {f"H{h}": {label: {"articles": len(values),
                               "median_return_bps": float(np.median(values)) if values else None,
                               "up_fraction": float(np.mean(np.asarray(values) > 0)) if values else None}
                     for label, values in bins.items()} for h, bins in out.items()}


def density(articles: pd.DataFrame) -> dict:
    start = articles["published_at_utc"].min().floor("D")
    end = articles["published_at_utc"].max().ceil("D")
    result = {}
    for minutes in (1, 5, 15, 60, 1440):
        grouped = articles.groupby(articles["published_at_utc"].dt.floor(f"{minutes}min")).size()
        total_bins = int((end - start).total_seconds() // (minutes * 60))
        result[f"{minutes}m"] = {"bins_with_news": len(grouped), "total_bins": total_bins,
                                 "fraction_with_news": len(grouped) / total_bins,
                                 "mean_articles_when_nonempty": float(grouped.mean()),
                                 "max_articles_in_bin": int(grouped.max())}
    return result


def relevance_study(articles: pd.DataFrame) -> dict:
    frames, _ = build_horizon_frames()
    common = common_frames(frames)
    splits = common_splits(common)
    train_end = splits[1].train["source_candle_close_time"].iloc[-1]
    train_articles = articles.loc[articles["published_at_utc"] <= train_end]
    threshold = float(train_articles["relevance"].median())
    closes = pd.DatetimeIndex(common[1]["source_candle_close_time"])
    groups = {h: {"below_train_median": [], "at_or_above_train_median": []} for h in (1, 5)}
    for article in articles.itertuples(index=False):
        published = pd.Timestamp(article.published_at_utc)
        idx = int(np.searchsorted(closes.asi8, published.value, side="left"))
        if idx >= len(closes) or published < common[1].index[0]:
            continue
        group = "at_or_above_train_median" if article.relevance >= threshold else "below_train_median"
        for h in (1, 5):
            groups[h][group].append(float(common[h].iloc[idx]["future_return_bps"]))
    return {"train_only_median_relevance_cutoff": threshold,
            "train_articles_for_cutoff": len(train_articles),
            "groups": {f"H{h}": {name: {"articles": len(values),
                                          "median_future_bps": float(np.median(values)) if values else None,
                                          "up_fraction": float(np.mean(np.asarray(values) > 0)) if values else None}
                                for name, values in group.items()} for h, group in groups.items()},
            "interpretation": "Descriptive post-hoc; threshold fixed using TRAIN articles only; not a model selector"}


def main() -> None:
    articles, audit = load_articles()
    frames, _ = build_horizon_frames()
    source = common_frames(frames)[1]
    closes_ns = pd.DatetimeIndex(source["source_candle_close_time"]).asi8
    news_ns = pd.DatetimeIndex(articles["published_at_utc"]).asi8
    audit["source_rows"] = len(source)
    audit["news_density_per_source_window"] = {}
    for minutes in (5, 15, 60):
        right = np.searchsorted(news_ns, closes_ns, side="right")
        left = np.searchsorted(news_ns, closes_ns - minutes * 60 * 1_000_000_000, side="right")
        nonempty = right > left
        audit["news_density_per_source_window"][str(minutes)] = {
            "rows_with_news": int(nonempty.sum()), "fraction": float(nonempty.mean())}
    audit["bin_density"] = density(articles)
    audit["event_study_post_hoc_descriptive"] = event_study(articles)
    audit["relevance_study_post_hoc"] = relevance_study(articles)
    audit["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(AUDIT, audit)
    features = json.loads(FEATURE_REPORT.read_text(encoding="utf-8"))
    for name, item in features["features"].items():
        item["definition"] = ("BTC ticker_sentiment_score relevance-weighted mean in causal window"
                              if name.startswith("btc_sentiment") else
                              "Number of unique BTC articles in causal window" if name.startswith("news_count") else
                              "Minutes from source close to last publication at/before close")
    write_json(FEATURE_REPORT, features)
    dev = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    report = ROOT / "reports"

    audit_md = ["# Alpha Vantage BTC news data audit — FASE 7", "",
                "Source: [official NEWS_SENTIMENT documentation](https://www.alphavantage.co/documentation/).",
                "Terms: [Alpha Vantage Terms of Service](https://www.alphavantage.co/terms_of_service/).",
                "Only aggregates are versioned; raw responses remain in gitignored private cache.", "",
                f"Articles: {audit['unique_valid_articles']} unique / {audit['raw_articles']} raw; duplicates {audit['duplicate_articles']}.",
                f"Publication range UTC: {audit['first_published_utc']} to {audit['last_published_utc']}.",
                f"Days with news: {audit['days_with_news']}; days without news: {len(audit['days_without_news'])}.",
                "BTC-specific ticker_sentiment_score and relevance_score were used; not overall article sentiment.",
                "Historical first-seen/availability time is unavailable. Results are publication-time backtests only.",
                "The API returned fewer than 1000 items for the full requested window, but the provider gives no independent completeness proof.", "",
                "## Density", "",
                table(["Bin", "Bins with news", "All bins", "Fraction", "Mean articles/nonempty"],
                      [[key, value["bins_with_news"], value["total_bins"], f"{value['fraction_with_news']:.2%}",
                        f"{value['mean_articles_when_nonempty']:.2f}"] for key, value in audit["bin_density"].items()]),
                "## Source distribution", "",
                table(["Source", "Articles"], [[name, count] for name, count in audit["sources"].items()]),
                "## Event study: descriptive, articles are not independent events", "",
                "Each article maps to the first source candle closing at or after its publication. H1/H5 returns follow that source candle. No causal inference or trading claim.", "",
                table(["Horizon", "BTC sentiment", "Articles", "Median future bps", "UP fraction"],
                      [[h, label, value["articles"],
                        f"{value['median_return_bps']:.3f}" if value["median_return_bps"] is not None else "N/A",
                        f"{value['up_fraction']:.2%}" if value["up_fraction"] is not None else "N/A"]
                       for h, bins in audit["event_study_post_hoc_descriptive"].items() for label, value in bins.items()]),
                "## BTC relevance, descriptive only", "",
                f"High/low cutoff is the TRAIN-only median relevance: {audit['relevance_study_post_hoc']['train_only_median_relevance_cutoff']:.4f}.",
                table(["Horizon", "Relevance", "Articles", "Median future bps", "UP fraction"],
                      [[h, name, value["articles"],
                        f"{value['median_future_bps']:.3f}" if value["median_future_bps"] is not None else "N/A",
                        f"{value['up_fraction']:.2%}" if value["up_fraction"] is not None else "N/A"]
                       for h, groups in audit["relevance_study_post_hoc"]["groups"].items() for name, value in groups.items()])]
    (report / "alpha_vantage_data_audit_v1.md").write_text("\n".join(audit_md), encoding="utf-8")

    feature_md = ["# NEWS_SENTIMENT causal features — FASE 7", "",
                  "For every source candle, include only news with publication time <= source candle close (UTC).",
                  "Historical AV publication time does not prove actual provider availability at that instant.",
                  "BTC-specific sentiment is relevance-weighted. Missing sentiment is NaN; counts are zero. XGBoost handles NaN explicitly; absence is never labeled neutral.", "",
                  table(["Feature", "Window", "Coverage", "No-news meaning"],
                        [[name, value["window_minutes"] or "all prior", f"{value['coverage']:.2%}",
                          value["missing_semantics"]] for name, value in features["features"].items()]),
                  f"Common H1/H5 rows: {features['rows']}; private feature fingerprint: `{features['dataset_sha256']}`.",
                  "News 5m coverage is diagnostic only; it is not included in the model.",
                  "Prospective design: save article key, published_at, fetched_at and first_observed_at on each successful fetch; training should require first_observed_at <= source close. No deployment in FASE 7.",
                  "Free-plan 25 calls/day is incompatible with minute polling. Hourly polling is <=24/day, subject to other calls and actual plan limits."]
    (report / "news_sentiment_features_v1.md").write_text("\n".join(feature_md), encoding="utf-8")

    ablation = ["# NEWS_SENTIMENT paired ablation — FASE 7", "",
                f"Candidate lock: {lock['locked_at_utc']} UTC; candidates H1 and H5; TEST not scored at lock.",
                "A = original 8 technical features; B = same 8 + five causal news features. Same timestamps, target, split, XGBoost parameters and threshold within each pair.",
                "C = sentiment-only diagnostic, N = technical plus news presence/recency. C and N were scored on VALIDATION only, not TEST.",
                "Publication-time availability proxy is an important unresolved limitation. No production use.", ""]
    for h in ("H1", "H5"):
        val = dev["validation"][h]
        test = final["test"][h]
        ablation += [f"## {h} VALIDATION ({val['paired_rows']} rows)", "",
                     table(["Metric", "A", "B", "C", "N"], metric_rows(val, ("A_control", "B_sentiment", "C_sentiment_only", "N_presence_only"))),
                     f"## {h} TEST ({test['rows']} rows; locked configurations only)", "",
                     table(["Metric", "A", "B"], metric_rows(test, ("A", "B"))),
                     "Confusion matrices use REAL rows [DOWN,UP] and PRED columns [DOWN,UP].", "",
                     f"A: {test['A']['metrics']['confusion_real_rows_down_up_pred_columns_down_up']}",
                     f"B: {test['B']['metrics']['confusion_real_rows_down_up_pred_columns_down_up']}", ""]
    ablation += ["## Interpretation", "",
                 "H1 VALIDATION improvement is small and its block-bootstrap intervals cross zero; TEST deteriorates. H5 worsens on VALIDATION and TEST ROC AUC/balanced accuracy.",
                 "The sentiment-only models have ROC AUC near 0.5 in VALIDATION. News-presence control does not establish a durable advantage for sentiment scores.",
                 "No model was promoted. TEST price period had already been viewed in earlier phases, limiting its independence."]
    (report / "news_sentiment_ablation_v1.md").write_text("\n".join(ablation), encoding="utf-8")

    walk_rows = []
    for fold in dev["walk_forward"]:
        for h in ("H1", "H5"):
            a = fold["horizons"][h]["A"]
            b = fold["horizons"][h]["B"]
            walk_rows.append([fold["fold"], h, fold["validation_start"][:16],
                              f"{a['roc_auc']:.4f}", f"{b['roc_auc']:.4f}", f"{b['roc_auc']-a['roc_auc']:+.4f}",
                              f"{b['balanced_accuracy']-a['balanced_accuracy']:+.4f}",
                              f"{b['macro_f1']-a['macro_f1']:+.4f}"])
    walk_md = ["# NEWS_SENTIMENT expanding walk-forward — FASE 7", "",
               "Four sequential TRAIN+VALIDATION folds with a five-row horizon-aware purge; paired timestamps in each A/B comparison.", "",
               table(["Fold", "Horizon", "Start UTC", "A ROC", "B ROC", "Δ ROC", "Δ BalAcc", "Δ MacroF1"], walk_rows),
               "No TEST or LIVE observation was used in these folds. Consecutive news windows and H5 labels are correlated; fold rows are not independent."]
    (report / "news_sentiment_walk_forward_v1.md").write_text("\n".join(walk_md), encoding="utf-8")
    print(json.dumps({"reports": 4, "articles": len(articles), "event_study_groups": 6}))


if __name__ == "__main__":
    main()
