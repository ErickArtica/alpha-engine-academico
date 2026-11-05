"""Create a NEW local-only .env without reading any external credentials."""
import argparse
import os
from pathlib import Path
import re
import secrets

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='alpha-profesor')
    parser.add_argument('--api-port', type=int, default=18000)
    parser.add_argument('--frontend-port', type=int, default=18501)
    args = parser.parse_args()
    if (not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,50}', args.project)
            or not 1024 <= args.api_port <= 65535 or not 1024 <= args.frontend_port <= 65535
            or args.api_port == args.frontend_port):
        parser.error('Use an independent project name and two distinct ports between 1024 and 65535')
    target = ROOT / '.env'
    if target.exists():
        parser.error('.env already exists; no file or secret was overwritten')
    changes = {'COMPOSE_PROJECT_NAME': args.project, 'LOCAL_API_PORT': str(args.api_port),
               'LOCAL_STREAMLIT_PORT': str(args.frontend_port),
               **{key: secrets.token_hex(32) for key in ('POSTGRES_PASSWORD', 'INTERNAL_CRON_SECRET', 'RECOMMENDATION_API_SECRET')}}
    template = (ROOT / '.env.local.example').read_text(encoding='utf-8')
    lines = [line.split('=', 1)[0] + '=' + changes[line.split('=', 1)[0]]
             if '=' in line and line.split('=', 1)[0] in changes else line for line in template.splitlines()]
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as output:
        output.write('\n'.join(lines) + '\n')
    print('Created local .env with independent random secrets; values suppressed. No provider enabled.')


if __name__ == '__main__':
    main()
