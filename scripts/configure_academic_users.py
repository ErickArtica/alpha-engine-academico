"""Prepare NEW own-installation user hashes, privately; no remote connections."""
import argparse
from getpass import getpass
import json
import os
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.render.academic_demo import password_hash

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users',nargs='+',required=True)
    args=parser.parse_args()
    if len(args.users)!=len(set(args.users)) or not 1<=len(args.users)<=25 or any(not re.fullmatch('[a-z0-9_-]{3,40}',u) for u in args.users):
        parser.error('Use 1 to 25 distinct authorized user names')
    target=ROOT/'.academic-demo-users.json'
    if target.exists():parser.error('Private users file exists; nothing overwritten')
    users={}
    for user in args.users:
        password=getpass('New private password for '+user+': ')
        if password!=getpass('Repeat private password: '):parser.error('Passwords do not match; nothing saved')
        users[user]=password_hash(password)
        del password
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as output:json.dump(users,output)
    print('Private local file created with salted hashes only; do not commit or display it publicly.')

if __name__=='__main__':main()
