"""Offline audit of the exact allowlist; does not emulate Docker's matcher."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tomllib
from build_inputs import RUNTIME_FILES, ignore_rules

ROOT = Path(__file__).resolve().parents[2]
BASE = "b4cf58d96cb9bc1e8f037cc8c69abe9cc865741d"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def audit() -> dict:
    if git("branch", "--show-current") != "feature/northflank-frontend":
        raise ValueError("Unexpected branch; report discrepancy before proceeding")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", "a8b40af", "HEAD"], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "alpha_engine", "streamlit_app", "models",
                    "requirements-streamlit.txt", "pyproject.toml"], cwd=ROOT, check=True)
    expected = RUNTIME_FILES
    rules = [line.strip() for line in (ROOT / "Dockerfile.frontend.northflank.dockerignore").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    # Reopening a directory alone also includes its descendants in Docker.
    # Immediately exclude its contents, then reopen nested directories/files.
    expected_rules = ignore_rules()
    if rules != expected_rules:
        raise ValueError("Allowlist differs from reviewed source/configuration manifest")
    allowed = expected
    secret = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
                        rb"AKIA[0-9A-Z]{16}|sb_(?:secret|publishable)_[A-Za-z0-9_-]{24,}|"
                        rb"eyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}|"
                        rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}")
    manifest = []
    for name in sorted(allowed):
        path = ROOT / name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != ROOT and ROOT in p.parents):
            raise ValueError("Symlink forbidden: " + name)
        data = path.read_bytes()
        if len(data) > 2_000_000:
            raise ValueError("Unexpected large file: " + name)
        if secret.search(data):
            raise ValueError("Possible credential in " + name + " (value suppressed)")
        if name.endswith(".py"):
            ast.parse(data, filename=name)
        manifest.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    model = ROOT / "models/xgboost_v1.joblib"
    if hashlib.sha256(model.read_bytes()).hexdigest() != model.with_suffix(".sha256").read_text().strip():
        raise ValueError("Published V1 checksum mismatch")
    tomllib.loads((ROOT / "deploy/northflank/streamlit.toml").read_text())
    return dict(head=git("rev-parse", "HEAD"), base=BASE, ancestor_a8b40af=True, published_files_unchanged=True,
                file_count=len(manifest), total_bytes=sum(f["bytes"] for f in manifest), files=manifest,
                credential_scan="No high-confidence patterns found; not a proof against arbitrary embedded secrets",
                docker_matcher_verified=False)


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
