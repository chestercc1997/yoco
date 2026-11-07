"""Fault-injection checks for healthy-state selection and lossless rollback."""
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools.watch_swa_training import checkpoint_healthy,events,owned_tree,process_identity,recover,same_process,select_checkpoint,sha,stop_tree,unhealthy,verify_contract,write


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.target=self.root/'baseline';self.target.mkdir()
        self.identity={'seed':42,'training':{'lr':.00015,'precision':'fp16_mixed_fp32_params'}}
        self.fingerprint=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()
        self.rng={'cpu':torch.get_rng_state(),'cuda':torch.arange(10,dtype=torch.uint8),
                  'python':random.getstate(),'numpy':np.random.get_state()}

    def tearDown(self):self.temp.cleanup()

    def checkpoint(self,step,folder=None,bad=False):
        folder=folder or self.target;folder.mkdir(parents=True,exist_ok=True)
        write(folder/'resolved_config.json',{'identity':self.identity})
        path=folder/f'checkpoint_{step:08d}.pt'
        tensor=torch.tensor([float('nan') if bad else step],dtype=torch.float32)
        state={'step':step,'world_size':8,'fingerprint':self.fingerprint,'model':{'weight':tensor},
            'optimizer':{'state':{0:{'step':torch.tensor(float(step)),'exp_avg':torch.ones(1),'exp_avg_sq':torch.ones(1)}},
                         'param_groups':[{'lr':.00015,'betas':(.9,.98),'eps':1e-8}]},
            'scaler':{'scale':128.,'_growth_tracker':step},'rngs':[copy.deepcopy(self.rng) for _ in range(8)]}
        torch.save(state,path)
        write(path.with_suffix('.json'),{'file':path.name,'step':step,'fingerprint':self.fingerprint,'sha256':sha(path)})
        return path

    def test_transient_noise_and_amp_overflow_do_not_roll_back(self):
        rows=[{'event':'validation','step':250,'loss':3.},
              {'event':'train','step':251,'loss':25.,'grad_norm':40.},
              {'event':'overflow','step':251,'scale':64.},
              {'event':'validation','step':500,'loss':3.3},
              {'event':'validation','step':750,'loss':2.99}]
        self.assertIsNone(unhealthy(rows))

    def test_sustained_validation_and_nonfinite_detection(self):
        rows=[{'event':'validation','step':i*250,'loss':value} for i,value in enumerate((3.,3.2,3.3,3.25),1)]
        self.assertEqual(unhealthy(rows)['rollback_before'],500)
        self.assertEqual(unhealthy([{'event':'nonfinite_forward','step':123,'attempted_update':124}])['step'],124)
        self.assertEqual(unhealthy([{'event':'validation','step':500,'loss':float('nan')}])['kind'],'nonfinite_validation')

    def test_corrupt_checksum_and_nonfinite_checkpoint_fall_back(self):
        good=self.checkpoint(5000);bad=self.checkpoint(10000,bad=True);corrupt=self.checkpoint(15000)
        with corrupt.open('ab') as output:output.write(b'corruption')
        selected,rejected=select_checkpoint(self.target,{'rollback_before':20000})
        self.assertEqual(selected['path'],str(good));self.assertEqual(len(rejected),2)
        with self.assertRaises(AssertionError):checkpoint_healthy(bad)

    def test_checkpoint_identity_is_not_relaxed(self):
        path=self.checkpoint(5000)
        with self.assertRaises(AssertionError):checkpoint_healthy(path,'different')

    def test_learning_rate_or_frozen_source_change_is_rejected(self):
        release=self.root/'release';(release/'tools').mkdir(parents=True)
        (self.root/'tools').mkdir();(self.root/'configs').mkdir()
        data=self.root/'data.json';write(data,{'ready':True})
        files={}
        for arm in ('baseline','A','B'):
            config={'training':{'lr':.00015,'precision':'fp16_mixed_fp32_params'},'data':{'manifest':str(data)}}
            write(release/f'configs/{arm}.json',config)
            write(self.root/f'configs/yoco_160m_swa_{arm}_8gpu.json',config)
            files[f'configs/{arm}.json']=sha(release/f'configs/{arm}.json')
        for folder in (release,self.root):(folder/'tools/run_swa_ab.py').write_text('# frozen controller\n')
        files['tools/run_swa_ab.py']=sha(release/'tools/run_swa_ab.py')
        write(self.root/'release-manifest.json',{'files':files,'data_manifest_sha256':sha(data)})
        with patch('tools.watch_swa_training.ROOT',self.root):
            verify_contract(self.root)
            path=self.root/'configs/yoco_160m_swa_A_8gpu.json';original=path.read_text()
            changed=json.loads(original);changed['training']['lr']=.00001;write(path,changed)
            with self.assertRaisesRegex(AssertionError,'setting changed'):verify_contract(self.root)
            path.write_text(original)
            (release/'tools/run_swa_ab.py').write_text('# changed code\n')
            with self.assertRaisesRegex(AssertionError,'Frozen release changed'):verify_contract(self.root)

    def test_rollback_preserves_failed_evidence_and_exact_full_state(self):
        source=self.checkpoint(5000);source_sha=sha(source);self.checkpoint(10000)
        write(self.target/'latest.json',json.loads((self.target/'checkpoint_00010000.json').read_text()))
        for rank in range(8):
            rows=[{'event':'started','step':4},{'event':'train','step':5000,'loss':3.},
                  {'event':'validation','step':5000,'loss':3.},
                  {'event':'train','step':10000,'loss':3.1},
                  {'event':'nonfinite_forward','step':10000,'attempted_update':10001}]
            (self.target/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        snapshot=self.target/'nonfinite_forward_00010001_rank0.pt';snapshot.write_bytes(b'failure evidence')
        write(self.target/'stage_complete.json',{'formal_complete':False,'step':10000})
        record=recover(self.root,'baseline',{'kind':'nonfinite_forward','step':10001,'rollback_before':10000},1)
        self.assertEqual(record['source']['step'],5000);self.assertEqual(sha(source),source_sha)
        archived=self.target/'recovery-attempts/001'
        self.assertTrue((archived/'checkpoint_00010000.pt').exists())
        self.assertEqual((archived/snapshot.name).read_bytes(),b'failure evidence')
        for rank in range(8):
            self.assertEqual(len(events(archived/f'rank{rank}.jsonl')),5)
            self.assertEqual([r['step'] for r in events(self.target/f'rank{rank}.jsonl')],[4,5000,5000])
        self.assertEqual(json.loads((self.target/'latest.json').read_text())['step'],5000)
        # A crash while writing status can repeat the transaction safely.
        recover(self.root,'baseline',record['issue'],1)
        self.assertEqual(sha(source),source_sha)
        restored=torch.load(source,weights_only=False)
        self.assertEqual(restored['optimizer']['param_groups'][0]['lr'],.00015)
        self.assertEqual(restored['scaler']['_growth_tracker'],5000)
        self.assertTrue(torch.equal(restored['rngs'][7]['cpu'],self.rng['cpu']))

    def test_only_owned_process_is_stopped(self):
        worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
        identity=process_identity(worker.pid);self.assertTrue(same_process(identity))
        stop_tree(identity);worker.wait(timeout=5);self.assertFalse(same_process(identity))
        self.assertIsNotNone(process_identity(__import__('os').getpid()))

    def test_orphaned_worker_is_stopped_after_controller_dies(self):
        child_pid=self.root/'child.pid'
        script=('import pathlib,signal,subprocess,sys; '
                'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                f'pathlib.Path({str(child_pid)!r}).write_text(str(child.pid)); signal.pause()')
        parent=subprocess.Popen([sys.executable,'-c',script],start_new_session=True)
        unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
        worker=None
        try:
            deadline=time.monotonic()+5
            while not child_pid.exists() and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(child_pid.exists())
            identity=process_identity(parent.pid);known=owned_tree(identity)
            worker=process_identity(int(child_pid.read_text()))
            self.assertIn(worker['pid'],[item['pid'] for item in known])
            parent.terminate();parent.wait(timeout=5)
            self.assertFalse(same_process(identity));self.assertTrue(same_process(worker))
            stop_tree(identity,known)
            self.assertFalse(same_process(worker));self.assertIsNone(unrelated.poll())
        finally:
            if parent.poll() is None:parent.kill();parent.wait(timeout=5)
            if worker and same_process(worker):stop_tree(worker)
            unrelated.terminate();unrelated.wait(timeout=5)


if __name__=='__main__':unittest.main()
