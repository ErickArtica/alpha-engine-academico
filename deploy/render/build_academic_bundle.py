"""Export reviewed backend sources without Git history, env, tests or local data."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def build():
    files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "alpha_engine").rglob("*")
             if p.is_file() and p.suffix in {".py", ".txt"} and "__pycache__" not in p.parts}
    files.update("deploy/render/" + name for name in (
        "__init__.py", "app.py", "academic_only.py", "academic_demo.py", "academic_state.py", "academic_service.py", "academic_nvidia.py",
        "academic_prompt.txt", "muse_diagnostic.py", "muse_diagnostic_worker.py", "fixtures/muse3_context.json",
        "requirements.txt"))
    files.update("models/" + name for name in ("xgboost_v1.joblib", "xgboost_v1.sha256", "xgboost_v1_metadata.json"))
    pattern = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:ghp_|github_pat_|nvapi-)[A-Za-z0-9_-]{24,}|sb_secret_[A-Za-z0-9_-]{24,}|eyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}")
    entries = []
    for name in sorted(files):
        p = ROOT / name
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT) or any(parent.is_symlink() for parent in p.parents if parent != ROOT and ROOT in parent.parents):
            raise ValueError("Backend source path escape")
        data = p.read_bytes()
        if pattern.search(data) or len(data) > 2_000_000:
            raise ValueError("Backend source requires security review: " + name)
        if p.suffix == ".py":
            ast.parse(data, filename=name)
        entries.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    out = ROOT / "deploy/northflank/validation/academic-implementation"
    out.mkdir(parents=True, exist_ok=True)
    inventory = json.dumps(entries, sort_keys=True).encode()
    version = hashlib.sha256(inventory).hexdigest()[:12]
    target = out / ("academic-backend-" + version + ".zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(ROOT / entry["path"], entry["path"])
    summary = {"bundle": str(target.relative_to(ROOT)), "source_fingerprint": version,
               "zip_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "files": entries, "secrets_pattern_matches": 0,
               "no_git_history_env_or_sql": True, "published": False}
    (out / "backend-bundle.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    build()
