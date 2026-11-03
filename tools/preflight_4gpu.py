"""Fail closed on corpus integrity and 32-to-4 rank comparison changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_configs():
    reports = {}
    for arch in ('yoco', 'transformer'):
        base = json.loads((ROOT / f'configs/{arch}_160m.json').read_text())
        run = json.loads((ROOT / f'configs/{arch}_160m_4gpu.json').read_text())
        for key in ('architecture', 'model', 'tied_embeddings', 'seed'):
            assert run[key] == base[key], (arch, key)
        for key in ('approved', 'dataset', 'subset', 'revision'):
            assert run['data'][key] == base['data'][key], (arch, key)
        b, r = base['training'], run['training']
        for key in b:
            if key not in ('world_size', 'gradient_accumulation', 'validation_batches'):
                assert r[key] == b[key], (arch, key)
        assert r['world_size'] == 4 and r['microbatch'] == 4
        assert r['world_size'] * r['microbatch'] * r['gradient_accumulation'] * r['sequence_length'] == r['global_batch_tokens'] == 262144
        assert r['world_size'] * r['microbatch'] * r['validation_batches'] == b['world_size'] * b['microbatch'] * b['validation_batches'] == 2048
        # The exact global indices reaching the stateless sampler must match.
        def indices(t, step):
            return sorted(((step*t['gradient_accumulation']+j)*t['world_size']+rank)*t['microbatch']+k
                          for j in range(t['gradient_accumulation']) for rank in range(t['world_size']) for k in range(t['microbatch']))
        for step in (0, 249, 374, 999, 39999):
            assert indices(b, step) == indices(r, step)
        reports[arch] = {'config_sha256': digest(ROOT / f'configs/{arch}_160m_4gpu.json'),
                         'training_samples_per_update': 128, 'validation_samples': 2048,
                         'global_sample_indices_match': True}
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {'configs': check_configs(), 'scope': 'repository public-data reference, not Microsoft original training'}
    if not a.configs_only:
        import numpy as np
        import pyarrow.parquet as pq
        import importlib.metadata as metadata
        from runtime.model import tokenizer
        candidate = json.loads((ROOT / 'provenance/fineweb_candidate.json').read_text())
        manifest_path = ROOT / 'data/fineweb_edu_10bt/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['ready'] and manifest['dataset'] == candidate['id']
        assert manifest['revision'] == candidate['sha'] and manifest['subset'] == 'sample-10BT'
        assert manifest['dtype'] == 'little_endian_uint32'
        assert len(manifest['shards']) == len(candidate['inventory']) == 14
        tok = tokenizer()
        assert tok.n_words == 100288 and tok.eos_id == 2
        totals = dict(train=0, validation=0)
        for source, row in zip(candidate['inventory'], manifest['shards']):
            raw = ROOT / 'data/fineweb_edu_10bt/raw' / source['path']
            assert Path(row['source_file']).resolve() == raw.resolve()
            assert raw.stat().st_size == source['bytes']
            assert digest(raw) == row['source_sha256'] == source['sha256']
            assert row['revision'] == candidate['sha']
            parquet = pq.ParquetFile(raw)
            assert {'id', 'text'} <= set(parquet.schema.names)
            assert sum(row[s]['documents'] for s in totals) + row['skipped_empty_documents'] == parquet.metadata.num_rows
            for split in totals:
                record = row[split]
                path = Path(record['path'])
                assert path.stat().st_size == record['tokens'] * 4
                h = hashlib.sha256()
                with path.open('rb') as f:
                    for chunk in iter(lambda: f.read(8 << 20), b''):
                        h.update(chunk)
                        ids = np.frombuffer(chunk, dtype='<u4')
                        assert ids.size and ids.max() < tok.n_words and ids.min() >= 2
                    f.seek(-4, 2)
                    assert int.from_bytes(f.read(4), 'little') == tok.eos_id
                assert h.hexdigest() == record['sha256']
                totals[split] += record['tokens']
            print(json.dumps({'verified_shard': source['path']}), flush=True)
        assert totals == manifest['totals']
        for arch in result['configs']:
            c = json.loads((ROOT / f'configs/{arch}_160m_4gpu.json').read_text())
            path = ROOT / c['data']['manifest']
            assert path.resolve() == manifest_path.resolve()
        result.update(data_verified=True, totals=totals, manifest_sha256=digest(manifest_path),
                      tokenizer_source_sha256=digest(ROOT / 'runtime/tokenizer.py'),
                      versions={k: metadata.version(k) for k in ('torch', 'numpy', 'tiktoken', 'pyarrow')})
        assert {k:v.split('+')[0] for k,v in result['versions'].items()} == {'torch': '2.7.1', 'numpy': '2.1.2', 'tiktoken': '0.11.0', 'pyarrow': '20.0.0'}
        result['runtime_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'runtime').rglob('*.py'))}
    result['passed'] = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
