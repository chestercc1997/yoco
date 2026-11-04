"""Resume downloads from a chosen endpoint and verify the committed inventory."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--connections', type=int, default=1,
                   help='Connections per file (2-16 requires aria2c); content hashes remain mandatory')
    a = p.parse_args()
    if a.workers < 1:
        p.error('--workers must be positive')
    if not 1 <= a.connections <= 16:
        p.error('--connections must be between 1 and 16')
    if a.connections > 1 and not shutil.which('aria2c'):
        p.error('Multiple connections require aria2c; use --connections 1 for curl')
    lock = json.loads((ROOT/'provenance/fineweb_candidate.json').read_text())
    endpoint = a.endpoint.rstrip('/')
    tree_url = f"{endpoint}/api/datasets/{lock['id']}/tree/{lock['sha']}/sample/10BT?limit=100"
    tree = json.loads(subprocess.check_output([
        'curl', '--fail', '--silent', '--show-error', '--location',
        '--connect-timeout', '20', '--max-time', '60', '--retry', '3', tree_url]))
    entries = {x['path']: x for x in tree if x['type'] == 'file'}
    inventory = [{'path': f, 'bytes': entries[f]['size'], 'sha256': entries[f]['lfs']['oid']} for f in lock['files']]
    if inventory != lock['inventory']:
        raise RuntimeError('Endpoint inventory differs from the pinned source')
    base = ROOT/'data/fineweb_edu_10bt/raw'
    logdir = ROOT/'logs/setup/downloads'
    logdir.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        out = base/entry['path']
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if out.stat().st_size == entry['bytes'] and digest(out) == entry['sha256']:
                print(json.dumps({'event': 'verified_existing', 'file': entry['path']}), flush=True)
                return
            raise RuntimeError(f'Existing final file failed verification: {out}')
        partial = out.with_suffix(out.suffix+'.partial')
        url = f"{endpoint}/datasets/{lock['id']}/resolve/{lock['sha']}/{entry['path']}?download=true"
        print(json.dumps({'event': 'downloading', 'file': entry['path']}), flush=True)
        with (logdir/(out.name+'.log')).open('a') as log:
            if a.connections > 1:
                command = ['aria2c', '--continue=true', '--auto-file-renaming=false',
                           '--file-allocation=none', '--allow-overwrite=true',
                           '--max-connection-per-server='+str(a.connections), '--split='+str(a.connections),
                           '--min-split-size=8M', '--connect-timeout=20', '--timeout=120',
                           '--max-tries=8', '--retry-wait=5', '--summary-interval=30',
                           '--console-log-level=warn', '--dir='+str(partial.parent),
                           '--out='+partial.name, url]
            else:
                command = ['curl', '--fail', '--location', '--connect-timeout', '20',
                           '--retry', '8', '--retry-delay', '5', '--retry-all-errors',
                           '--speed-limit', '1024', '--speed-time', '120',
                           '--continue-at', '-', '--output', str(partial), url]
            subprocess.run(command, stdout=log, stderr=log, check=True)
        if partial.stat().st_size != entry['bytes'] or digest(partial) != entry['sha256']:
            raise RuntimeError(f'Download failed size/SHA256 verification: {partial}')
        partial.replace(out)
        print(json.dumps({'event': 'verified_download', 'file': entry['path'], 'bytes': entry['bytes']}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(fetch, inventory))
    print(json.dumps({'event': 'all_downloads_verified', 'files': len(inventory)}), flush=True)


if __name__ == '__main__':
    main()
