import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.model import build_model, tokenizer
from runtime.train import Tokens
from tools import prepare_data, verify_data
from tools.yoco_samples import SamplePacker, document_segments


class AppendixDBaselineTests(unittest.TestCase):
    def test_offline_preparation_and_sampler_share_the_new_format(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / 'data/fineweb_edu_10bt'
            key = 'sample/10BT/example.parquet'
            raw = base / 'raw' / key
            raw.parent.mkdir(parents=True)
            val_id = next(str(i) for i in range(10000)
                          if int.from_bytes(hashlib.sha256(str(i).encode()).digest()[:8], 'big') % 1000 == 0)
            docs = [{'id': i, 'text': 'a b c d e f g h i'} for i in ('training', val_id)]
            pq.write_table(pa.Table.from_pylist(docs), raw)
            inventory = [dict(path=key, bytes=raw.stat().st_size, sha256=prepare_data.sha256(raw))]
            (root / 'provenance').mkdir()
            (root / 'provenance/fineweb_candidate.json').write_text(json.dumps(
                dict(id='test', sha='v1', inventory=inventory)))
            output = base / 'yoco_samples_4/manifest.json'
            config = dict(training=dict(sequence_length=4), data=dict(dataset='test', revision='v1',
                          approved=True, subset='sample-10BT', manifest=str(output), packing='yoco_samples'))
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            with patch.object(prepare_data, 'ROOT', root), patch.object(sys, 'argv', [
                'prepare_data', '--offline', '--workers', '1', '--config', str(path)
            ]), patch('shutil.disk_usage', return_value=SimpleNamespace(free=100 << 30)):
                prepare_data.main()
            self.assertFalse((base / 'manifest.json').exists())
            self.assertEqual(json.loads(path.read_text()), config)
            manifest = json.loads(output.read_text())
            with patch.object(verify_data, 'sha256', wraps=prepare_data.sha256) as digest:
                verify_data.verify_configs([path, path])
                self.assertEqual(digest.call_count, 2)
            incompatible = root / 'other.json'
            for key, value in [('revision', 'wrong'), ('packing', 'stream')]:
                other = dict(config, data=dict(config['data'], **{key: value}))
                incompatible.write_text(json.dumps(other))
                with self.assertRaisesRegex(ValueError, key):
                    verify_data.verify_configs([path, incompatible])
            for split in ('train', 'validation'):
                ds = Tokens(manifest, split, 4, 42)
                self.assertEqual(ds.sample(0).tolist(), [0] + tokenizer().encode(docs[0]['text'])[:4])

    def test_shapes_and_unshared_weights_for_all_six_variants(self):
        # Paper appendix D, table 7: model width, number of layers, attention heads.
        for size, (width, layers, heads) in {
            '160m': (768, 12, 12), '400m': (1024, 24, 16), '830m': (1536, 24, 12)
        }.items():
            for variant in ('gret', 'swa'):
                with self.subTest(size=size, variant=variant):
                    path = ROOT / 'configs' / 'appendix_d' / f'{size}_{variant}.json'
                    config = json.loads(path.read_text())
                    with torch.device('meta'):
                        model = build_model(config)
                    self.assertIsNot(model.embed_tokens.weight, model.output_projection.weight)
                    self.assertEqual(model.embed_tokens.weight.shape, (100288, width))
                    self.assertEqual(config['model']['n_attn_heads'], heads)
                    self.assertEqual(config['model']['n_self_heads'], width // 256 if variant == 'gret' else heads)
                    self.assertEqual(config['model'].get('sliding_window'), 1024 if variant == 'swa' else None)
                    self.assertEqual(config['data']['packing'], 'yoco_samples')
                    for decoder in (model.self_decoder, model.cross_decoder):
                        self.assertEqual(len(decoder.layers), layers // 2)
                        for layer in decoder.layers:
                            for proj in (layer.ffn.fc1, layer.ffn.gate):
                                self.assertEqual(proj.weight.shape, (3 * width, width))
                            self.assertEqual(layer.ffn.fc2.weight.shape, (width, 3 * width))

    def test_pinned_line_and_record_boundaries(self):
        tok = tokenizer()
        self.assertEqual(document_segments('one\ntwo\n\nthree', tok),
                         [tok.encode('one\n'), tok.encode('two\n\n'), tok.encode('three') + [tok.eos_id]])
        self.assertEqual(document_segments('\n\n' + 'x' * 200001, tok), [])
        p = SamplePacker(tok, 3)
        self.assertEqual(list(p.add([[5, 6, 7]])), [])  # exact fit is still pending
        self.assertEqual(list(p.add([[100, 101]])), [[tok.bos_id, 5, 6, 7]])
        self.assertEqual(list(p.add([[8], [9, 10, 11, 12], [13]])), [[tok.bos_id, 8, 9, 10]])

    def test_sampler_reads_whole_records_and_rejects_wrong_length(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'data.bin'
            rows = np.array([[0, 3, 4, 5], [0, 6, 7, 8]], dtype='<u4')
            rows.tofile(path)
            manifest = dict(packing='yoco_samples', sequence_length=3,
                            shards=[{'train': {'path': str(path), 'tokens': rows.size}}])
            data = Tokens(manifest, 'train', 3, seed=42)
            self.assertCountEqual([data.sample(i).tolist() for i in range(data.count)], rows.tolist())
            with self.assertRaises(ValueError):
                Tokens(manifest, 'train', 2, seed=42)
            manifest['shards'][0]['train']['tokens'] = 7
            rows.ravel()[:7].tofile(path)
            with self.assertRaises(ValueError):
                Tokens(manifest, 'train', 3, seed=42)


if __name__ == '__main__':
    unittest.main()
