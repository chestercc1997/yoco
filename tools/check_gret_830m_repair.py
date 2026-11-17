"""Validate the captured gRet830M overflow repair without updating training weights."""
import argparse,importlib.util,json,math,sys,hashlib,gc,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from runtime.model import build_model
from runtime.precision import configure,autocast
from runtime.yoco_official import backend
import runtime.yoco_official.gate_retention as gr

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
 torch.set_num_threads(2);configure('fp16_mixed_fp32_params');torch.backends.cuda.matmul.allow_tf32=True
 torch.use_deterministic_algorithms(True)
 oldrun=ROOT/'runs/yoco-gret-830m-baseline-8gpu-20261113T183312Z';folder=oldrun/'baseline'
 spec=importlib.util.spec_from_file_location('legacy',oldrun/'release/runtime/yoco_official/backend.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 fixed=backend.chunk_gate_retention;norm=gr.RMSNorm(256,elementwise_affine=False,eps=1e-5).cuda()
 # Float64 independent token recurrence checks the equation and all gradients.
 torch.manual_seed(7)
 vals=[torch.randn(1,2,16,8,device='cuda',dtype=torch.float64,requires_grad=True) for _ in range(3)]
 gate=(-torch.rand(1,2,16,device='cuda',dtype=torch.float64)).requires_grad_();initial=torch.randn(1,2,8,8,device='cuda',dtype=torch.float64,requires_grad=True)
 q,k,v=vals;actual=fixed(q,k,v,gate,chunk_size=4,last_hidden_state=initial);state=initial;parts=[]
 for i in range(16):
  state=state*gate[:,:,i].exp()[...,None,None]+k[:,:,i,:,None]*v[:,:,i,None,:]/math.sqrt(8)
  parts.append((q[:,:,i,None,:]@state).squeeze(-2))
 expected=torch.stack(parts,2);torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
 ga=torch.autograd.grad(actual.square().sum(),(*vals,gate,initial),retain_graph=True);gb=torch.autograd.grad(expected.square().sum(),(*vals,gate,initial))
 for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)
 print('PASS independent float64 recurrence and gradients',flush=True)
 snapshots=sorted(folder.glob('nonfinite*.pt'))+sorted(folder.glob('recovery-attempts/*/nonfinite*.pt'))
 records=[];model=None
 for snap in snapshots:
  saved=torch.load(snap,map_location='cpu',mmap=True,weights_only=False)
  if model is None:model=build_model(saved['config']).cuda().eval()
  model.load_state_dict(saved['model']);x=saved['input_tokens'].cuda();target=saved['targets'].cuda()
  gr.chunk_gate_retention=old.chunk_gate_retention
  with torch.no_grad(),autocast('fp16_mixed_fp32_params'):bad=model(x)[0]
  assert not torch.isfinite(bad).all();del bad
  gr.chunk_gate_retention=fixed;model.zero_grad(set_to_none=True)
  with autocast('fp16_mixed_fp32_params'):
   logits=model(x)[0];loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten())
  assert torch.isfinite(logits).all() and torch.isfinite(loss)
  loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  value=float(loss.detach());model.zero_grad(set_to_none=True);del logits,loss
  with torch.no_grad(),autocast('fp32'):
   ref_logits=model(x)[0];ref_loss=float(torch.nn.functional.cross_entropy(ref_logits.flatten(0,1),target.flatten()))
  # Reference check is diagnostic only; formal precision remains original AMP.
  assert abs(value-ref_loss)<.01,(value,ref_loss)
  records.append(dict(snapshot=str(snap),sha256=sha(snap),failure_update=saved['failure_update'],old_nonfinite_reproduced=True,fixed_loss=value,fp32_reference_loss=ref_loss,loss_abs_difference=abs(value-ref_loss),all_model_gradients_finite=True))
  print('PASS failure',records[-1],flush=True);del ref_logits,saved;gc.collect();torch.cuda.empty_cache()
 # Normal-range full-model loss and gradient agreement: same failed weights,
 # ordinary input with safe forward, preserving the original AMP interface.
 x=x[:,:256];target=target[:,:256];grads={};results={}
 for label,func in [('legacy',old.chunk_gate_retention),('fixed',fixed)]:
  gr.chunk_gate_retention=func;torch.use_deterministic_algorithms(label=='fixed');model.zero_grad(set_to_none=True)
  with autocast('fp16_mixed_fp32_params'):
   logits=model(x)[0];loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten())
  assert torch.isfinite(loss);loss.backward();results[label]=float(loss.detach())
  if label=='legacy':
   grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
  else:
   delta=total=0.
   for n,p in model.named_parameters():
    a=p.grad.detach().cpu().double();b=grads.pop(n).double();delta+=float((a-b).square().sum());total+=float(b.square().sum())
   rel=math.sqrt(delta/total);print('normal_range',results,'grad_relative_l2',rel,flush=True);assert rel<.005,rel
  del logits,loss
 assert abs(results['fixed']-results['legacy'])<.001
 checkpoint=folder/'checkpoint_00005000.pt';meta=json.load(open(checkpoint.with_suffix('.json')))
 identity=json.load(open(folder/'resolved_config.json'))['identity'];changes={}
 for name,h in identity['model_source_sha256'].items():
  new=sha(ROOT/'runtime'/name)
  if new!=h and name!='train.py':changes[name]=dict(before=h,after=new)
 assert set(changes)=={'yoco_official/backend.py','yoco_official/gate_retention.py'}
 report=dict(passed=True,source_checkpoint=str(checkpoint),source_checkpoint_sha256=meta['sha256'],model_source_changes=changes,failures=records,independent_float64_forward_backward_passed=True,normal_range=dict(losses=results,gradient_relative_l2=rel,tolerances=dict(loss_abs=.001,gradient_relative_l2=.005)),repair='Keep original finite AMP path exactly; recompute only out-of-range within-chunk sums in existing FP32 state dtype before unchanged RMSNorm/FP16 interface.',training_setting_changes={},internal_arithmetic_change_explicit=True,no_training_updates_performed=True)
 report.update(execution_change={'deterministic_algorithms':True,'ddp_find_unused_parameters':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8'},trainer_sha256=sha(ROOT/'runtime/train.py'))
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print('PASSED',args.output,flush=True)
if __name__=='__main__':main()
