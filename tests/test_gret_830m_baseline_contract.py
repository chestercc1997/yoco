"""Immutable SWA400M contracts and full-state rollback for the baseline."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools import watch_gret_830m_baseline as guard

ROOT=Path(__file__).resolve().parents[1]


class Gret830mBaselineContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.release=self.root/'release';(self.release/'configs').mkdir(parents=True)
        (self.root/'configs').mkdir();(self.root/'tools').mkdir();(self.release/'tools').mkdir()
        data=self.root/'data.json';guard.write(data,{'ready':True})
        self.configs={}
        for arm in guard.ARMS:
            cfg=guard.read(ROOT/f'configs/yoco_830m_gret_{arm}_8gpu.json')
            cfg['data']['manifest']=str(data);self.configs[arm]=cfg
            guard.write(self.root/f'configs/yoco_830m_gret_{arm}_8gpu.json',cfg)
            guard.write(self.release/f'configs/{arm}.json',cfg)
        for filename,source in [('SWA400M-reference.json','yoco_400m_swa_baseline_8gpu.json'),('gRet400M-reference.json','yoco_400m_gret_baseline_8gpu.json')]:
            reference=guard.read(ROOT/'configs'/source);reference['data']['manifest']=str(data)
            guard.write(self.release/'configs'/filename,reference)
        for parent in (self.root,self.release):(parent/'tools/run_gret_830m_baseline.py').write_text('# frozen controller\n')
        files={str(p.relative_to(self.release)):guard.sha(p) for p in self.release.rglob('*') if p.is_file()}
        guard.write(self.root/'release-manifest.json',dict(architecture='YOCO-gRet',scale='830M',files=files,data_manifest_sha256=guard.sha(data)))

    def tearDown(self):self.temp.cleanup()

    def test_all_arm_training_or_architecture_changes_rejected(self):
        with patch.object(guard,'ROOT',self.root):
            self.assertEqual(guard.verify_contract(self.root)['arm_order'],['baseline'])
            for arm in guard.ARMS:
                path=self.root/f'configs/yoco_830m_gret_{arm}_8gpu.json'
                for section,key,value in [('training','lr',1e-5),('training','precision','bf16'),('model','sliding_window',512),('execution','compile',True)]:
                    changed=copy.deepcopy(self.configs[arm]);changed[section][key]=value;guard.write(path,changed)
                    with self.assertRaisesRegex(AssertionError,'setting changed'):guard.verify_contract(self.root)
                guard.write(path,self.configs[arm])
            guard.verify_contract(self.root)

    def test_root_and_frozen_sources_rejected_after_change(self):
        with patch.object(guard,'ROOT',self.root):
            guard.verify_contract(self.root)
            (self.root/'tools/run_gret_830m_baseline.py').write_text('# changed\n')
            with self.assertRaisesRegex(AssertionError,'Root source changed'):guard.verify_contract(self.root)
            (self.root/'tools/run_gret_830m_baseline.py').write_text('# frozen controller\n')
            (self.release/'tools/run_gret_830m_baseline.py').write_text('# changed\n')
            with self.assertRaisesRegex(AssertionError,'Frozen release changed'):guard.verify_contract(self.root)

    def test_rollback_keeps_full_state_and_failed_suffix_for_each_arm(self):
        for arm in guard.ARMS:
            with self.subTest(arm=arm):
                target=self.root/arm;target.mkdir();cfg=self.configs[arm]
                identity={'model':cfg['model'],'training':cfg['training']}
                fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                guard.write(target/'resolved_config.json',{'identity':identity})
                rng=dict(cpu=torch.get_rng_state(),cuda=torch.arange(8,dtype=torch.uint8),python=random.getstate(),numpy=np.random.get_state())
                for step in (5000,10000):
                    state=dict(step=step,world_size=8,fingerprint=fingerprint,model={'weight':torch.ones(1)},
                        optimizer={'state':{0:{'step':torch.tensor(float(step)),'exp_avg':torch.ones(1)}},'param_groups':[{'lr':.00015}]},
                        scaler={'scale':128.,'_growth_tracker':0},rngs=[copy.deepcopy(rng) for _ in range(8)])
                    path=target/f'checkpoint_{step:08d}.pt';torch.save(state,path)
                    guard.write(path.with_suffix('.json'),dict(file=path.name,step=step,fingerprint=fingerprint,sha256=guard.sha(path)))
                before=guard.sha(target/'checkpoint_00005000.pt')
                for rank in range(8):
                    rows=[dict(event='train',step=5000,loss=3.),dict(event='train',step=10000,loss=3.1),dict(event='nonfinite_forward',step=10000,attempted_update=10001)]
                    (target/f'rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                record=guard.recover(self.root,arm,dict(kind='nonfinite_forward',step=10001,rollback_before=10000),1)
                self.assertEqual(record['source']['step'],5000)
                self.assertEqual(guard.sha(target/'checkpoint_00005000.pt'),before)
                self.assertTrue((target/'recovery-attempts/001/checkpoint_00010000.pt').exists())
                self.assertEqual([x['step'] for x in guard.events(target/'rank7.jsonl')],[5000])
                saved=torch.load(target/'checkpoint_00005000.pt',weights_only=False)
                self.assertEqual(saved['optimizer']['param_groups'][0]['lr'],.00015)
                self.assertTrue(torch.equal(saved['rngs'][7]['cpu'],rng['cpu']))


if __name__=='__main__':unittest.main()
