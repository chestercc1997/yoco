"""Audit an isolated fusion release against the previously verified HPC3 corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--config', default='configs/yoco_160m_fusion_token_2gpu.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root.parents[1]
    expected = json.loads((root/'release_manifest.json').read_text())
    for name, sha in expected.items():
        assert digest(root/name) == sha, name
    old = json.loads((base/'runs/setup/preflight-hpc3.json').read_text())
    manifest_path = root/'data/fineweb_edu_10bt/manifest.json'
    assert old['passed'] and old['data_verified']
    assert digest(manifest_path) == old['manifest_sha256']
    assert digest(root/'runtime/tokenizer.py') == old['tokenizer_source_sha256']
    manifest = json.loads(manifest_path.read_text())
    c = json.loads((root/args.config).read_text())
    baseline = json.loads((base/'configs/yoco_160m_4gpu.json').read_text())
    assert digest(base/'configs/yoco_160m_4gpu.json') == old['configs']['yoco']['config_sha256']
    assert manifest['ready'] and manifest['revision'] == c['data']['revision']
    for key in ('architecture', 'model', 'seed', 'tied_embeddings'):
        assert c[key] == baseline[key], key
    for key in ('approved', 'dataset', 'subset', 'revision'):
        assert c['data'][key] == baseline['data'][key], key
    t, b = c['training'], baseline['training']
    for key in b:
        if key not in ('world_size', 'gradient_accumulation', 'validation_batches'):
            assert t[key] == b[key], key
    assert t['world_size'] == 2 and t['microbatch'] == 4
    assert t['world_size']*t['microbatch']*t['gradient_accumulation']*t['sequence_length'] == 262144
    assert t['world_size']*t['microbatch']*t['validation_batches'] == 2048
    for step in (0, 249, 374, 999, 39999):
        def indices(v):
            return sorted(((step*v['gradient_accumulation']+j)*v['world_size']+r)*v['microbatch']+k
                          for j in range(v['gradient_accumulation']) for r in range(v['world_size']) for k in range(v['microbatch']))
        assert indices(t) == indices(b)
    prior = json.loads((root/'preflight.json').read_text()) if args.check_only else None
    if prior is not None:
        assert prior['config_sha256'] == digest(root/args.config)
    files = {}
    for row in manifest['shards']:
        for split in ('train', 'validation'):
            info = row[split]; path = Path(info['path']); stat = path.stat()
            assert stat.st_size == info['tokens']*4, str(path)
            record = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=info['sha256'])
            if args.check_only:
                assert record == prior['token_files'][str(path)], str(path)
            else:
                assert digest(path) == info['sha256'], str(path)
                print('VERIFIED ' + path.name, flush=True)
            files[str(path)] = record
    if not args.check_only:
        result = dict(passed=True, verified_unix=time.time(), token_files=files,
                      config=args.config, config_sha256=digest(root/args.config),
                      manifest_sha256=digest(manifest_path), code_files=len(expected),
                      world_size=2, training_samples_per_update=128, validation_samples=2048,
                      train_indices_match=True, stop_after=1000)
        (root/'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS isolated release, corpus integrity and paired two-rank configuration', flush=True)


if __name__ == '__main__':
    main()
