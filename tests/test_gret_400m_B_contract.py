"""The gRet B supervisor must retain the immutable recipe and correct arm path."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools import watch_gret_400m_B as guard


class GretBContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root = Path(self.temp.name)
        self.release = self.root / 'release';(self.release / 'tools').mkdir(parents=True)
        (self.root / 'tools').mkdir();(self.root / 'configs').mkdir()
        self.data = self.root / 'data.json';guard.write(self.data, {'ready': True})
        a = dict(architecture='yoco', model={'dim':1024, 'n_layers':24}, seed=42, tied_embeddings=True,
                 training={'lr':.00015, 'precision':'fp16_mixed_fp32_params'},
                 data={'manifest':str(self.data)}, execution={'compile':False},
                 memory_fusion={'mode':'learned_kv', 'source_layers':list(range(1,13)), 'temperature':1.})
        self.cfg = copy.deepcopy(a);self.cfg['memory_fusion']['mode'] = 'conditional_kv'
        guard.write(self.release / 'configs/A-reference.json', a)
        guard.write(self.release / 'configs/B.json', self.cfg)
        guard.write(self.root / 'configs/yoco_400m_plan_B_8gpu.json', self.cfg)
        for parent in (self.root, self.release):(parent / 'tools/run_gret_400m_B.py').write_text('# frozen controller\n')
        files = {str(p.relative_to(self.release)):guard.sha(p) for p in self.release.rglob('*') if p.is_file()}
        guard.write(self.root / 'release-manifest.json', dict(architecture='YOCO-gRet', files=files,
                    data_manifest_sha256=guard.sha(self.data)))

    def tearDown(self):self.temp.cleanup()

    def test_recipe_changes_cannot_be_used_for_recovery(self):
        with patch.object(guard, 'ROOT', self.root):
            contract = guard.verify_contract(self.root)
            self.assertTrue(contract['no_training_setting_changes'])
            for key,value in (('lr',.00001),('precision','bf16')):
                changed = copy.deepcopy(self.cfg);changed['training'][key] = value
                guard.write(self.root / 'configs/yoco_400m_plan_B_8gpu.json', changed)
                with self.assertRaisesRegex(AssertionError, 'setting changed'):guard.verify_contract(self.root)
            guard.write(self.root / 'configs/yoco_400m_plan_B_8gpu.json', self.cfg)
            guard.verify_contract(self.root)

    def test_frozen_and_root_controller_changes_are_rejected(self):
        with patch.object(guard, 'ROOT', self.root):
            guard.verify_contract(self.root)
            root_source = self.root / 'tools/run_gret_400m_B.py';root_source.write_text('# changed\n')
            with self.assertRaisesRegex(AssertionError, 'Root source changed'):guard.verify_contract(self.root)
            root_source.write_text('# frozen controller\n')
            (self.release / 'tools/run_gret_400m_B.py').write_text('# changed\n')
            with self.assertRaisesRegex(AssertionError, 'Frozen release changed'):guard.verify_contract(self.root)

    def test_B_rollback_preserves_full_state_and_failure_evidence(self):
        target = self.root / 'B';target.mkdir()
        identity = {'model':self.cfg['model'], 'training':self.cfg['training']}
        fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        guard.write(target / 'resolved_config.json', {'identity':identity})
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.arange(8,dtype=torch.uint8),
                   python=random.getstate(), numpy=np.random.get_state())
        for step in (5000,10000):
            state = dict(step=step, world_size=8, fingerprint=fingerprint, model={'weight':torch.ones(1)},
                         optimizer={'state':{0:{'step':torch.tensor(float(step)), 'exp_avg':torch.ones(1)}},
                                    'param_groups':[{'lr':.00015}]},
                         scaler={'scale':128., '_growth_tracker':0}, rngs=[copy.deepcopy(rng) for _ in range(8)])
            path = target / f'checkpoint_{step:08d}.pt';torch.save(state,path)
            guard.write(path.with_suffix('.json'), dict(file=path.name,step=step,fingerprint=fingerprint,sha256=guard.sha(path)))
        before = guard.sha(target / 'checkpoint_00005000.pt')
        for rank in range(8):
            rows=[dict(event='train',step=5000,loss=3.),dict(event='train',step=10000,loss=3.1),
                  dict(event='nonfinite_forward',step=10000,attempted_update=10001)]
            (target/f'rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        record = guard.recover(self.root,'B',dict(kind='nonfinite_forward',step=10001,rollback_before=10000),1)
        self.assertEqual(record['source']['step'],5000)
        self.assertEqual(guard.sha(target/'checkpoint_00005000.pt'),before)
        self.assertTrue((target/'recovery-attempts/001/checkpoint_00010000.pt').exists())
        self.assertEqual([x['step'] for x in guard.events(target/'rank7.jsonl')],[5000])
        saved=torch.load(target/'checkpoint_00005000.pt',weights_only=False)
        self.assertEqual(saved['optimizer']['param_groups'][0]['lr'],.00015)
        self.assertTrue(torch.equal(saved['rngs'][7]['cpu'],rng['cpu']))


if __name__=='__main__':unittest.main()
