"""Verify configured corpora, hashing shared manifests' shards once per invocation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.train import sha256


def verify_configs(paths):
    verified = {}
    for path in paths:
        config = json.loads(path.read_text())
        data = config['data']
        packing = data.get('packing', 'stream')
        # Legacy stream configs live alongside the frozen release's configs.
        base = ROOT if packing == 'yoco_samples' else path.resolve().parents[1]
        manifest_path = (base / data['manifest']).resolve()
        if manifest_path in verified:
            manifest = verified[manifest_path]
        else:
            manifest = json.loads(manifest_path.read_text())

        if not manifest.get('ready') or not manifest.get('shards'):
            raise ValueError(f'{manifest_path} has no ready corpus')
        for key in ('dataset', 'subset', 'revision'):
            if manifest[key] != data[key]:
                raise ValueError(f'{path}: manifest {key} differs from config')
        if packing not in ('stream', 'yoco_samples') or manifest.get('packing', 'stream') != packing:
            raise ValueError(f'{path}: data packing differs from config')
        length = config['training']['sequence_length']
        if packing == 'yoco_samples' and manifest.get('sequence_length') != length:
            raise ValueError(f'{path}: packed record length differs from config')

        if manifest_path not in verified:
            totals = {'train': 0, 'validation': 0}
            for row in manifest['shards']:
                for split in totals:
                    shard = row[split]
                    file = Path(shard['path'])
                    count = shard['tokens']
                    if packing == 'yoco_samples' and count % (length + 1):
                        raise ValueError(f'Incomplete packed record in {file}')
                    if file.stat().st_size != count * 4 or sha256(file) != shard['sha256']:
                        raise ValueError(f'Dataset size or hash mismatch: {file}')
                    totals[split] += count
            if totals != manifest['totals'] or not all(totals.values()):
                raise ValueError(f'{manifest_path}: invalid split totals')
            verified[manifest_path] = manifest
        print(json.dumps({'verified': True, 'config': str(path), 'dataset': manifest['dataset'],
                          'revision': manifest['revision'], 'totals': manifest['totals']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, nargs='+', required=True)
    verify_configs(parser.parse_args().config)
