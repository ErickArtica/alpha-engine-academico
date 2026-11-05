"""Post-lock TEST uncertainty for the two horizons selected before TEST."""

import json
from datetime import datetime, timezone

import joblib

from scripts.analyze_hold_target import ROOT, sha, write_json
from scripts.analyze_horizons import build_horizon_frames, common_frames, common_splits
from scripts.run_horizon_experiment import LOCK, REPORT, compare_bootstrap, probability


def run():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["candidates_minutes"] != [1, 5] or lock["test_metrics_seen_at_lock"] is not False:
        raise ValueError("Expected frozen H1/H5 candidate lock")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report["candidate_lock"] != lock:
        raise ValueError("Horizon report and candidate lock differ")
    frames, _ = build_horizon_frames()
    splits = common_splits(common_frames(frames))
    probabilities = {}
    model_hashes = {}
    for h in (1, 5):
        item = report["final_test"]["common_TEST_results"][f"H{h}"]["experimental_artifact"]
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError("Experimental model checksum differs")
        bundle = joblib.load(path)
        if bundle["horizon"] != f"{h}m" or bundle["experimental_only"] is not True:
            raise ValueError("Wrong experimental model")
        probabilities[h] = probability(bundle["estimator"], splits[h].test)
        model_hashes[f"H{h}"] = item["sha256"]
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_lock_sha256": sha(LOCK), "model_sha256": model_hashes,
              "common_TEST_n": len(splits[1].test),
              "H5_minus_H1": compare_bootstrap(
                  splits[1].test["target"].to_numpy(dtype=int), probabilities[1],
                  splits[5].test["target"].to_numpy(dtype=int), probabilities[5],
                  block=60, repetitions=1000),
              "limitation": "Different binary targets; paired source minutes and 60-minute blocks account partly for overlapping H5 labels, but this is descriptive uncertainty."}
    write_json(ROOT / "reports/horizon_test_uncertainty_v1.json", result)
    return result


if __name__ == "__main__":
    output = run()
    print(output["common_TEST_n"], output["H5_minus_H1"]["difference"])
