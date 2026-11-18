"""Connection settings shared by the two historical HPC3 result watchers."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = '/data/share/jzhou945/cchen099/yoco'
SKILL = Path('/hpc2hdd/home/jzhou945/cchen099/.codex-home/skills/hpc3')

ENV = os.environ.copy()
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    ENV.pop(key, None)
ENV.update(NO_PROXY='*', no_proxy='*', SSH_ASKPASS=str(SKILL/'scripts/askpass.py'),
           SSH_ASKPASS_REQUIRE='force', DISPLAY='hpc3-auth')
