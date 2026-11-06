"""Reproduce cross-chunk FP16 overflow using finite synthetic inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime.yoco_official.backend import chunk_gate_retention
from runtime.yoco_official.rms_norm import RMSNorm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-backend',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    spec=importlib.util.spec_from_file_location('old_retention_backend',args.original_backend)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Each causal token contributes 108: the final output is 110592, while
    # the within-chunk term is at most 27648. All inputs/states are finite.
    q=torch.full((1,1,1024,16),3.,device='cuda')
    k=q.clone()
    v=torch.full_like(q,3.,dtype=torch.float16)
    g=torch.zeros((1,1,1024),device='cuda',dtype=torch.float16)
    with torch.autocast('cuda',dtype=torch.float16):
        old=module.chunk_gate_retention(q,k,v,g,chunk_size=256)
        fixed=chunk_gate_retention(q,k,v,g,chunk_size=256)
        normalized=RMSNorm(16,elementwise_affine=False,eps=1e-5).cuda()(fixed).to(v.dtype)
        small_old=module.chunk_gate_retention(q[:,:,:256],k[:,:,:256],v[:,:,:256],g[:,:,:256],chunk_size=256)
        small_fixed=chunk_gate_retention(q[:,:,:256],k[:,:,:256],v[:,:,:256],g[:,:,:256],chunk_size=256)
    assert not torch.isfinite(old).all(),'Original overflow was not reproduced'
    assert torch.isfinite(fixed).all() and fixed.dtype==torch.float32
    expected=(torch.arange(1024,device='cuda')+1)*108
    maximum_error=float((fixed[0,0,:,0]-expected).abs().max())
    assert maximum_error<=8,'Stable result differs beyond within-chunk FP16 rounding'
    assert torch.equal(normalized,torch.ones_like(normalized)) and normalized.dtype==torch.float16
    assert torch.equal(small_old.float(),small_fixed)
    report={'passed':True,'original_nonfinite':int((~torch.isfinite(old)).sum()),
        'fixed_nonfinite':int((~torch.isfinite(fixed)).sum()),'fixed_maximum':float(fixed.max()),
        'closed_form_expected_maximum':110592,'maximum_closed_form_error':maximum_error,
        'normalized_activation_dtype':str(normalized.dtype),'normal_range_output_exact':True,
        'precision_scope':'Only accumulation/sum before existing FP32 RMSNorm; original FP16 activation interface retained.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
