"""Launch only the Northflank public reader, after validating runtime settings."""
from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.northflank.public_config import ConfigurationError, PublicConfig, public_hosts


def validate_environment(env: dict[str, str]) -> list[str]:
    return list(PublicConfig.from_environment(env).hosts)


def command(env: dict[str, str]) -> list[str]:
    hosts = validate_environment(env)
    args = [sys.executable, "-m", "streamlit", "run", "deploy/northflank/public_app.py",
            "--global.developmentMode=false", "--server.address=0.0.0.0", "--server.port=8501",
            "--server.headless=true", "--server.enableCORS=true", "--server.enableXsrfProtection=true",
            "--server.enableStaticServing=false", "--server.fileWatcherType=none", "--server.baseUrlPath=",
            "--browser.gatherUsageStats=false", "--browser.serverAddress=" + hosts[0], "--browser.serverPort=443"]
    args.extend("--server.corsAllowedOrigins=https://" + host for host in hosts)
    return args


def main() -> None:
    try:
        args = command(dict(os.environ))
    except (ConfigurationError, ValueError):
        # Never print credential-bearing inputs or low-level URL exceptions.
        print("Northflank public reader: invalid runtime configuration; review README.", file=sys.stderr)
        raise SystemExit(2) from None
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
