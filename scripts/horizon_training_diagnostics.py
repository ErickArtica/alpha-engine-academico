"""Post-lock descriptive TRAIN predictions for every horizon; no TEST scoring."""

import json
from datetime import datetime, timezone

from scripts.analyze_hold_target import ROOT, sha, write_json
from scripts.analyze_horizons import HORIZONS, build_horizon_frames, common_frames, common_splits
from scripts.run_horizon_experiment import DEVELOPMENT, LOCK, fit, probability, score


def run():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    dev = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if sha(DEVELOPMENT) != lock["development_report_sha256"]:
        raise ValueError("Development changed after candidate lock")
    frames, _ = build_horizon_frames()
    splits = common_splits(common_frames(frames))
    results = {}
    for horizon in HORIZONS:
        train = splits[horizon].train
        model = fit(train)
        measured = score(train, probability(model, train))
        results[f"H{horizon}"] = {"n": measured["n"],
                                   "predicted_up_fraction": measured["predicted_up_fraction"],
                                   "predicted_down_fraction": 1 - measured["predicted_up_fraction"],
                                   "train_accuracy": measured["accuracy"]}
    output = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_lock_sha256": sha(LOCK), "common_TRAIN": results,
              "note": "Only common TRAIN model outputs; H3 TEST remains unscored by design."}
    write_json(ROOT / "reports/horizon_training_diagnostics_v1.json", output)
    return output


if __name__ == "__main__":
    print(run()["common_TRAIN"])
