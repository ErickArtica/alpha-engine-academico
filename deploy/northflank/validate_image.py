"""Build and smoke-test in isolation. Requires an already running Linux daemon.

Never starts Docker Desktop, reads real credentials, or alters existing resources.
Only publishes 127.0.0.1:18501. Health checks do not exercise Streamlit pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import uuid

from audit_context import ROOT, audit

NETWORK_GUARD = '''"""Validation-only socket restrictions; never shipped in the image."""
import ipaddress
import sys

def _local(host):
    if host in (None, "localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

def _guard(event, args):
    if event == "socket.getaddrinfo" and not _local(args[0]):
        raise PermissionError("Validation forbids external DNS")
    if event in ("socket.connect", "socket.sendto"):
        address = args[-1]
        if isinstance(address, tuple) and not _local(address[0]):
            raise PermissionError("Validation forbids external sockets")

sys.addaudithook(_guard)
ACTIVE = True
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-host", default="npipe:////./pipe/dockerDesktopLinuxEngine" if os.name == "nt" else "unix:///var/run/docker.sock")
    parser.add_argument("--evidence-dir", type=Path, help="Keep this revision's evidence under validation/ without replacing previous evidence")
    parser.add_argument("--image-tag", help="New local candidate tag; never the previously published tag")
    options = parser.parse_args()
    validation_root = (ROOT / "deploy/northflank/validation").resolve()
    evidence = (options.evidence_dir or validation_root).resolve()
    if not evidence.is_relative_to(validation_root):
        parser.error("Evidence must stay inside this worktree's validation directory")
    evidence.mkdir(parents=True, exist_ok=True)
    report = dict(status="INCOMPLETE", built=False, container_created=False, http_health_verified=False,
                  integrations_tested=False, cleanup_complete=True)
    token = uuid.uuid4().hex[:12]
    name = "alpha-nf-validation-" + token
    tag = "alpha-engine-frontend:northflank-validation-" + token
    if options.image_tag:
        import re
        if not re.fullmatch(r"alpha-engine-frontend:(?:ui|academic)-f28acc0c-[a-f0-9]{12}", options.image_tag):
            parser.error("Use a distinct local tag identifying the audited UI content")
        tag = options.image_tag
    report["image_tag"] = tag
    container_id = None
    network_id = None
    with tempfile.TemporaryDirectory(prefix=".validation-", dir=ROOT / "deploy/northflank") as temp:
        temp = Path(temp)
        config = temp / "docker-config"
        config.mkdir()
        (config / "config.json").write_text("{}")
        guard = temp / "network-guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(NETWORK_GUARD, encoding="utf-8")
        offline = temp / "offline-tests"
        offline.mkdir()
        for filename in ("test_prepare.py", "test_public_policy.py", "test_public_views.py", "test_public_visual_parity.py", "test_academic_frontend.py", "public_test_support.py"):
            shutil.copyfile(ROOT / "deploy/northflank" / filename, offline / filename)
        docker = ["docker", "--config", str(config), "--host", options.docker_host]

        def run(*args: str) -> str:
            print("Docker: " + " ".join(args[:4]), flush=True)
            process = subprocess.run([*docker, *args], cwd=ROOT, text=True,
                                     encoding="utf-8", errors="replace",
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with (evidence / "docker.log").open("a", encoding="utf-8") as log:
                log.write("\n$ docker " + " ".join(args) + "\n" + process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError("Docker command failed (see validation/docker.log): " + " ".join(args[:3]))
            return process.stdout.strip()

        (evidence / "docker.log").write_text("", encoding="utf-8")
        try:
            manifest = audit()
            (evidence / "context-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            server = json.loads(run("info", "--format", "{{json .OSType}}"))
            if server != "linux":
                raise ValueError("A Linux Docker daemon is required")
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 18501))
            context = temp / "context"
            run("buildx", "build", "--platform", "linux/amd64", "--progress", "plain", "--file",
                "Dockerfile.frontend.northflank", "--target", "northflank-context", "--output",
                "type=local,dest=" + str(context), ".")
            actual = {p.relative_to(context).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in context.rglob("*") if p.is_file()}
            expected = {f["path"]: f["sha256"] for f in manifest["files"]}
            # BuildKit may withhold its Dockerfile/ignore control files from COPY.
            actual.pop("Dockerfile.frontend.northflank.dockerignore", None)
            expected.pop("Dockerfile.frontend.northflank", None)
            actual.pop("Dockerfile.frontend.northflank", None)
            if actual != expected:
                (evidence / "context-diff.json").write_text(json.dumps(dict(
                    missing=sorted(expected.keys() - actual.keys()),
                    unexpected=sorted(actual.keys() - expected.keys()),
                    changed=sorted(p for p in actual.keys() & expected.keys() if actual[p] != expected[p])
                ), indent=2), encoding="utf-8")
                raise ValueError("BuildKit context differs from the audited manifest")
            manifest["docker_matcher_verified"] = True
            (evidence / "context-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            run("buildx", "build", "--platform", "linux/amd64", "--progress", "plain", "--load", "--file",
                "Dockerfile.frontend.northflank", "--target", "frontend", "--tag", tag, ".")
            image = json.loads(run("image", "inspect", tag))[0]
            report.update(built=True, image_tag=tag, image_id=image["Id"], image_bytes=image["Size"],
                          image_os=image["Os"], image_arch=image["Architecture"])
            if image["Os"] != "linux" or image["Config"]["User"] != "app:app":
                raise ValueError("Unexpected image OS/user")
            # Docker Desktop suppresses published ports on an internal network.
            # Use a dedicated bridge and a validation-only Python socket guard.
            network_id = run("network", "create", "--label", "alpha.nf.validation=" + token, name)
            report["cleanup_complete"] = False
            container_id = run("create", "--name", name, "--label", "alpha.nf.validation=" + token,
                               "--network", network_id, "--publish", "127.0.0.1:18501:8501", "--read-only",
                               "--mount", "type=bind,source=" + str(guard) + ",target=/validation-guard,readonly",
                               "--mount", "type=bind,source=" + str(offline) + ",target=/offline-tests,readonly",
                               "--env", "PYTHONPATH=/validation-guard:/app",
                               "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "--tmpfs", "/home/app:rw,nosuid,nodev,size=16m,uid=10001,gid=10001",
                               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "768m", "--cpus", "1",
                               "--env", "FRONTEND_PUBLIC_HOST=localhost", "--env", "SUPABASE_URL=https://synthetic.invalid",
                               "--env", "SUPABASE_KEY=sb_publishable_synthetic_only",
                               "--env", "NORTHFLANK_PUBLIC_READ_ONLY=true",
                               "--env", "NORTHFLANK_PUBLIC_DATA_APPROVED=true", tag)
            report.update(container_created=True, cleanup_complete=False)
            run("start", container_id)
            inspected = json.loads(run("inspect", container_id))[0]
            if inspected["HostConfig"]["PortBindings"] != {"8501/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18501"}]}:
                raise ValueError("Unexpected port bindings")
            network = json.loads(run("network", "inspect", network_id))[0]
            if network["Labels"].get("alpha.nf.validation") != token:
                raise ValueError("Unexpected validation network")
            run("exec", container_id, "python", "-c",
                "import sitecustomize,socket; assert sitecustomize.ACTIVE; "
                "exec(\"try:\\n socket.getaddrinfo('backend.invalid',443)\\nexcept PermissionError:\\n print('External DNS blocked before resolution')\\nelse:\\n raise AssertionError('Network guard inactive')\")")
            report["python_external_network_guard_verified"] = True
            run("exec", container_id, "python", "-m", "pip", "check")
            packages = json.loads(run("exec", container_id, "python", "-m", "pip", "list", "--format=json"))
            (evidence / "packages.json").write_text(json.dumps(packages, indent=2), encoding="utf-8")
            streamlit_requires = json.loads(run("exec", container_id, "python", "-c",
                "import importlib.metadata as m,json; print(json.dumps(m.requires('streamlit')))") )
            (evidence / "streamlit-requires.json").write_text(json.dumps(streamlit_requires, indent=2), encoding="utf-8")
            forbidden = {"xgboost", "scikit-learn", "apscheduler", "fastapi", "influxdb-client", "psycopg", "openai", "anthropic", "supabase", "joblib", "pydantic-settings"}
            # Streamlit 1.66 uses Uvicorn itself; allow only with metadata evidence.
            if not any(r.lower().startswith("uvicorn") for r in streamlit_requires):
                forbidden.add("uvicorn")
            if forbidden & {p["name"].lower().replace("_", "-") for p in packages}:
                raise ValueError("Worker/backend/LLM dependency found")
            run("exec", container_id, "python", "-B", "-c",
                "import platform; import deploy.northflank.public_views; from zoneinfo import ZoneInfo; "
                "assert platform.system()=='Linux'; ZoneInfo('America/Lima'); print('Linux frontend imports OK')")
            run("exec", container_id, "python", "-B", "-c",
                "from pathlib import Path; forbidden=['streamlit_app','alpha_engine/jobs','alpha_engine/ai',"
                "'alpha_engine/services/prediction.py','alpha_engine/repositories/supabase.py']; "
                "assert all(not Path('/app',p).exists() for p in forbidden); print('Action/inference modules absent')")
            output = run("exec", container_id, "python", "-B", "-c",
                "import unittest,sys,json; suite=unittest.defaultTestLoader.discover('/offline-tests',pattern='test_*.py'); "
                "result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite); "
                "print('OFFLINE_RESULT='+json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors)))); "
                "sys.exit(not result.wasSuccessful())")
            tests = json.loads(output.rsplit("OFFLINE_RESULT=", 1)[1])
            (evidence / "public-tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
            report.update(public_profile=True, offline_tests=tests)
            run("exec", container_id, "python", "-c",
                "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3); "
                "assert r.status==200 and r.read().strip()==b'ok'; print('Container HTTP health OK')")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for attempt in range(30):
                try:
                    with opener.open("http://127.0.0.1:18501/_stcore/health", timeout=2) as response:
                        body = response.read().decode()
                        if response.status == 200 and body.strip() == "ok":
                            report.update(http_health_verified=True, http_status=200, health_body=body)
                            break
                except OSError:
                    pass
                time.sleep(1)
            if not report["http_health_verified"]:
                raise RuntimeError("Local HTTP health check failed")
            deadline = time.monotonic() + 45
            while True:
                report["docker_health"] = json.loads(run("inspect", container_id))[0]["State"]["Health"]["Status"]
                if report["docker_health"] == "healthy":
                    break
                if report["docker_health"] == "unhealthy" or time.monotonic() >= deadline:
                    raise RuntimeError("Docker HEALTHCHECK did not become healthy")
                time.sleep(2)
            report["status"] = "BUILT_LINUX_HTTP_HEALTH_OK_INTEGRATIONS_NOT_TESTED"
        except Exception as exc:
            report.update(status="BLOCKED_OR_FAILED", error=str(exc))
        finally:
            if container_id:
                try:
                    run("logs", container_id)
                except RuntimeError:
                    pass
                try:
                    run("container", "rm", "--force", container_id)
                    report["cleanup_complete"] = True
                except RuntimeError as exc:
                    report["cleanup_error"] = str(exc)
            if network_id:
                try:
                    run("network", "rm", network_id)
                    if not report.get("cleanup_error"):
                        report["cleanup_complete"] = True
                except RuntimeError as exc:
                    report.update(cleanup_complete=False, cleanup_error=str(exc))
            (evidence / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if (report["status"] == "BUILT_LINUX_HTTP_HEALTH_OK_INTEGRATIONS_NOT_TESTED"
                 and report["http_health_verified"] and report["cleanup_complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
