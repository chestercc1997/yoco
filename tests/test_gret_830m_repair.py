"""Regression tests for within-chunk overflow and repaired-lineage reporting."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from runtime.yoco_official.backend import chunk_gate_retention
from runtime.yoco_official.rms_norm import RMSNorm
from tools.gret830_recovery_support import lineage_events,checkpoint_path

class RetentionRepairTests(unittest.TestCase):
 def test_only_out_of_range_inner_sums_are_widened(self):
  q=torch.full((1,1,256,16),8.,dtype=torch.float16,requires_grad=True)
  k=q.detach().clone().requires_grad_();v=q.detach().clone().requires_grad_();g=torch.zeros((1,1,256),dtype=torch.float16,requires_grad=True)
  output=chunk_gate_retention(q,k,v,g)
  self.assertTrue(torch.isfinite(output).all());self.assertEqual(output.dtype,torch.float32)
  torch.testing.assert_close(output[0,0,:,0],torch.arange(1,257)*2048.,rtol=0,atol=0)
  normalized=RMSNorm(16,elementwise_affine=False)(output)
  normalized.square().mean().backward()
  for x in [q,k,v,g]:self.assertTrue(torch.isfinite(x.grad).all())
 @unittest.skipUnless((ROOT/'runs/yoco-gret-830m-baseline-8gpu-20261113T183312Z/release/runtime/yoco_official/backend.py').exists(), 'Historical frozen runtime fixture is local to the reproduction host')
 def test_normal_path_exactly_matches_old_forward_and_gradients(self):
  spec=importlib.util.spec_from_file_location('legacy',ROOT/'runs/yoco-gret-830m-baseline-8gpu-20261113T183312Z/release/runtime/yoco_official/backend.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  torch.manual_seed(9)
  q,k,v=[torch.randn(1,2,16,8,requires_grad=True) for _ in range(3)];g=(-torch.rand(1,2,16)).requires_grad_()
  a=old.chunk_gate_retention(q,k,v,g,chunk_size=4);b=chunk_gate_retention(q,k,v,g,chunk_size=4)
  self.assertTrue(torch.equal(a,b));ga=torch.autograd.grad(a.square().sum(),[q,k,v,g],retain_graph=True);gb=torch.autograd.grad(b.square().sum(),[q,k,v,g])
  for x,y in zip(ga,gb):self.assertTrue(torch.equal(x,y))
 def test_lineage_joins_frozen_prefix_and_new_suffix(self):
  with tempfile.TemporaryDirectory() as name:
   run=Path(name);(run/'release/history').mkdir(parents=True);(run/'baseline/checks/resumed').mkdir(parents=True)
   (run/'release/history/rank0.jsonl').write_text(json.dumps(dict(event='train',step=5000,loss=3.))+'\n')
   (run/'baseline/checks/resumed/rank0.jsonl').write_text(json.dumps(dict(event='train',step=5001,loss=2.9))+'\n')
   (run/'recovery-source.json').write_text(json.dumps(dict(source_checkpoint='/source/checkpoint_00005000.pt')))
   self.assertEqual([e['step'] for e in lineage_events(run,'baseline',0)],[5000,5001])
   self.assertEqual(str(checkpoint_path(run,'baseline',5000)),'/source/checkpoint_00005000.pt')
   self.assertEqual(checkpoint_path(run,'baseline',10000),run/'baseline/checkpoint_00010000.pt')
 def test_guard_can_recover_before_first_new_scheduled_checkpoint(self):
  from tools import watch_gret_830m_repaired as guard
  with tempfile.TemporaryDirectory() as name:
   target=Path(name);(target/'checks/resumed').mkdir(parents=True)
   fallback=target/'checks/resumed/checkpoint_00005004.pt';fallback.touch()
   (target/'resolved_config.json').write_text(json.dumps(dict(identity={'setting':'unchanged'})))
   with patch.object(guard,'_original_select',side_effect=RuntimeError('No new formal checkpoint yet')),patch.object(guard,'checkpoint_healthy',return_value=dict(step=5004)) as verify:
    proof,_=guard.repaired_select(target,dict(rollback_before=5900))
    self.assertEqual(proof['step'],5004);self.assertEqual(verify.call_args.args[0],fallback)
    with self.assertRaises(AssertionError):guard.repaired_select(target,dict(rollback_before=5004))
if __name__=='__main__':unittest.main()
