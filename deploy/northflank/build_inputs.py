"""Reviewed runtime files. Never expand this list from an entire local directory."""
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FILES = {
    "Dockerfile.frontend.northflank",
    "alpha_engine/__init__.py",
    "alpha_engine/data/__init__.py", "alpha_engine/data/schemas.py", "alpha_engine/data/live_schemas.py",
    "alpha_engine/data/prescriptive_schemas.py",
    "alpha_engine/features/__init__.py", "alpha_engine/features/indicators.py",
    "alpha_engine/features/split.py", "alpha_engine/features/target.py",
    "alpha_engine/repositories/__init__.py", "alpha_engine/repositories/protocols.py",
    "alpha_engine/services/__init__.py", "alpha_engine/services/live_evaluation.py",
    "alpha_engine/services/paper_policy.py",
    "alpha_engine/services/historical_replay.py",
    "models/xgboost_v1.joblib", "models/xgboost_v1.sha256", "models/xgboost_v1_metadata.json",
    "models/baseline_v1.json", "models/economic_summary_v1.json",
    "deploy/northflank/entrypoint.py", "deploy/northflank/public_config.py",
    "deploy/northflank/public_data.py", "deploy/northflank/public_views.py", "deploy/northflank/public_app.py",
    "deploy/northflank/public_presentation.py", "deploy/northflank/public_replay.py",
    "deploy/northflank/academic_client.py", "deploy/northflank/academic_panel.py",
    "deploy/northflank/requirements.frontend.txt", "deploy/northflank/streamlit.toml",
}


def ignore_rules() -> list[str]:
    directories = {str(p) + "/" for f in RUNTIME_FILES for p in PurePosixPath(f).parents if str(p) != "."}
    rules = ["**"]
    for directory in sorted(directories):
        rules.extend(["!" + directory, directory + "**"])
    rules.extend("!" + name for name in sorted(RUNTIME_FILES))
    return rules


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-ignore", action="store_true")
    args = parser.parse_args()
    if args.write_ignore:
        (ROOT / "Dockerfile.frontend.northflank.dockerignore").write_text(
            "# Generated from deploy/northflank/build_inputs.py; exact files only.\n" + "\n".join(ignore_rules()) + "\n", encoding="utf-8")
    print(f"Reviewed runtime files: {len(RUNTIME_FILES)}")
