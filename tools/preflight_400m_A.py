"""Check the scaled baseline/A common backbone and save the uniform A initialization."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from runtime.model import build_model
from runtime.train import sha256,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initialization',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    baseline=json.loads((ROOT/'configs/yoco_400m_baseline_8gpu.json').read_text())
    config=json.loads((ROOT/'configs/yoco_400m_plan_A_8gpu.json').read_text())
    old=json.loads((ROOT/'configs/yoco_160m_baseline_8gpu_matched.json').read_text())
    for key in ('architecture','model','seed','tied_embeddings','training','data'):
        assert config[key]==baseline[key],key
    assert config['training']==old['training'] and config['data']==old['data']
    assert config['memory_fusion']==dict(mode='learned_kv',source_layers=list(range(1,13)),temperature=1.)
    torch.manual_seed(config['seed']);base=build_model(baseline).eval()
    torch.manual_seed(config['seed']);model=build_model(config).eval()
    common=base.state_dict();state=model.state_dict()
    assert all(torch.equal(value,state[name]) for name,value in common.items())
    assert set(state)-set(common)=={'memory_fusion.logits'}
    params=sum(p.numel() for p in model.parameters());base_params=sum(p.numel() for p in base.parameters())
    assert params-base_params==12
    ids=torch.tensor([[10,20,30,40,50,60,70,80]])
    with torch.no_grad():
        weights=model.memory_fusion.conditional_weights([torch.ones(1,8,1024)]*12)
        torch.testing.assert_close(weights,torch.full_like(weights,1/12))
    args.initialization.parent.mkdir(parents=True,exist_ok=True)
    if args.initialization.exists():
        saved=torch.load(args.initialization,map_location='cpu',weights_only=False,mmap=True)
        assert all(torch.equal(value,saved['model'][name]) for name,value in state.items())
        del saved
    else:
        temp=args.initialization.with_suffix('.tmp')
        torch.save(dict(model=state,model_config=config['model'],tied_embeddings=config['tied_embeddings'],
                        memory_fusion=config['memory_fusion'],seed=config['seed']),temp)
        temp.replace(args.initialization)
    with torch.no_grad():
        expected=base(ids)[0]
        model.memory_fusion.logits.fill_(-1000);model.memory_fusion.logits[-1]=0
        actual=model(ids)[0]
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
    record={'passed':True,'baseline_parameters':base_params,'A_parameters':params,'extra_parameters':12,
        'common_state_tensors_verified':len(common),'common_weights_bitwise_equal':True,
        'last_source_limit_matches_baseline':True,'initial_weights':[1/12]*12,
        'training_and_data_match_completed_160m_recipe':True,'precision':config['training']['precision'],
        'model':config['model'],'memory_fusion':config['memory_fusion'],
        'initialization':str(args.initialization),'initialization_sha256':sha256(args.initialization),
        'note':'400M is a new 400.59M-parameter YOCO configuration derived from the same backbone family, not an official released 400M checkpoint.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
