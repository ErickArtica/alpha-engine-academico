"""Audit the exact proposed Git additions without staging or reading .env files."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from build_inputs import ROOT, RUNTIME_FILES


PATTERNS = [
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    rb"(?:AKIA|ASIA)[0-9A-Z]{16}",
    rb"sb_(?:secret|publishable)_[A-Za-z0-9_-]{24,}",
    rb"eyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}",
    rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}",
    rb"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}",
    rb"nvapi-[A-Za-z0-9_-]{24,}",
    rb"(?:postgres(?:ql)?|https?)://[^\s/:]+:[^\s@]+@",
]


def git_paths(*args) -> set[str]:
    output = subprocess.check_output(["git", *args, "-z"], cwd=ROOT)
    return {p.decode("utf-8") for p in output.split(b"\0") if p}


def audit_publication() -> dict:
    manifest_path = ROOT / "deploy/northflank/PUBLICATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposed = set(manifest["new_files"])
    additional = set(manifest.get("additional_publication_files", []))
    academic_paths = {"deploy/render/" + name for name in (
        "app.py", "academic_only.py", "academic_demo.py", "academic_state.py", "academic_service.py", "academic_nvidia.py", "academic_prompt.txt",
        "build_academic_bundle.py", "ACADEMIC_DEMO.md", "sql/academic_demo_state.sql",
        "sql/academic_demo_rollback.sql", "sql/academic_dependencies_inspect.sql")}
    academic_paths.add("tests/test_academic_demo.py")
    academic_paths.update({"alpha_engine/news/marketaux.py", "tests/test_academic_provider_reliability.py"})
    academic_paths.update({
        '.env.local.example',
        '.gitignore',
        'README.md',
        'deploy/local/Dockerfile.api.dockerignore',
        'deploy/local/Dockerfile.streamlit.dockerignore',
        'docker-compose.yml',
        'docs/01_INSTALACION_LOCAL.md',
        'docs/02_DESPLIEGUE_NUBE.md',
        'docs/03_VARIABLES_Y_CREDENCIALES.md',
        'docs/04_PRUEBAS_DE_RECONSTRUCCION.md',
        'scripts/configure_academic_users.py',
        'scripts/configure_local.py',
        'scripts/probe_academic_provider.py',
        'scripts/verify_local.py',
        'tests/test_marketaux_diagnostics.py',
        'tests/test_reconstruction_helpers.py',
    })
    academic_paths.update({'tests/test_marketaux_prescriptive_integration.py', 'docs/05_EVOLUCION_POST_ENTREGA.md'})
    if additional - academic_paths:
        raise ValueError("Unreviewed academic backend publication path")
    proposed |= additional
    tracked = git_paths("ls-files")
    untracked = git_paths("ls-files", "--others", "--exclude-standard")
    modified = git_paths("diff", "--name-only", "HEAD")
    if untracked - proposed or proposed - (tracked | untracked):
        raise ValueError("Publication inventory mismatch: " + json.dumps(dict(
            unexpected=sorted(untracked - proposed), missing=sorted(proposed - (tracked | untracked)))))
    if modified - proposed:
        raise ValueError("Unexpected modifications outside the proposed Northflank files")
    if not RUNTIME_FILES <= proposed | tracked:
        raise ValueError("A required Docker input is missing from the publication inventory")
    patterns = [re.compile(p) for p in PATTERNS]
    files = []
    for name in sorted(proposed):
        parts = PurePosixPath(name).parts
        if ".." in parts or PurePosixPath(name).is_absolute() or not (
                name.startswith("deploy/northflank/") or name in additional
                or name in {"Dockerfile.frontend.northflank", "Dockerfile.frontend.northflank.dockerignore"}):
            raise ValueError("Out-of-scope publication path")
        path = ROOT / name
        if path.is_symlink() or path.resolve() != ROOT.joinpath(*parts):
            raise ValueError("Symlink or path escape in publication set")
        if ((path.name.startswith(".env") and path.name != ".env.local.example") or path.name in {"secrets.toml", "config.json"}
                or path.suffix.lower() in {".pem", ".key", ".pfx", ".p12", ".zip", ".parquet"}
                or any(p in {"validation", "runtime", ".git", ".venv", "__pycache__"} for p in parts)):
            raise ValueError("Private/generated file in publication set: " + name)
        content = path.read_bytes()
        if len(content) > 1_000_000 or any(pattern.search(content) for pattern in patterns):
            raise ValueError("Oversized file or possible credential in " + name + " (value suppressed)")
        if path.suffix == ".py":
            ast.parse(content, filename=name)
        files.append(dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    # This records what was reviewed; it cannot prove absence of every possible secret.
    return dict(base=manifest["base"], head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                inventoried_files=len(files), runtime_files=len(RUNTIME_FILES), unexpected_files=[],
                credential_patterns="No matches; limited pattern scan plus review of synthetic fixtures/templates",
                files=files)


if __name__ == "__main__":
    result = audit_publication()
    local = ROOT / "deploy/northflank/validation"
    local.mkdir(exist_ok=True)
    (local / "publication-audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
