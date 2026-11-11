"""Audit the new SWA 400M common initialization, fusion limits, locality and caches."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from runtime.model import build_model
from runtime.train import atomic_json,sha256
from runtime.yoco_official.backend import flash_attn_func
from runtime.yoco_official.sliding_window_attention import SlidingWindowAttention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initialization-dir',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    configs={arm:json.loads((ROOT/f'configs/yoco_400m_swa_{arm}_8gpu.json').read_text()) for arm in ('baseline','A','B')}
    reference=json.loads((ROOT/'configs/yoco_400m_gret_baseline_8gpu.json').read_text())
    base_config=configs['baseline']
    assert base_config['training']==reference['training'] and base_config['data']==reference['data']
    assert base_config['model']['sliding_window']==1024
    assert base_config['execution']==json.loads((ROOT/'configs/yoco_160m_swa_baseline_8gpu.json').read_text())['execution']
    assert base_config['model']['dim']==1024 and base_config['model']['n_layers']==24
    assert base_config['model']['hidden_dim']==2816
    for arm in ('A','B'):assert configs[arm]['memory_fusion']==dict(mode='learned_kv' if arm=='A' else 'conditional_kv',source_layers=list(range(1,13)),temperature=1.)
    assert base_config['model']['n_self_heads']==base_config['model']['n_attn_heads']==16
    for arm,c in configs.items():
        for key in ('architecture','seed','model','tied_embeddings','training','data','execution'):
            assert c[key]==base_config[key],(arm,key)
    models={}
    for arm,c in configs.items():
        torch.manual_seed(c['seed']);models[arm]=build_model(c).eval()
        assert all(isinstance(layer.mixer,SlidingWindowAttention) for layer in models[arm].self_decoder.layers)
    base=models['baseline']; common=base.state_dict()
    extras={'A':{'memory_fusion.logits'},'B':{'memory_fusion.score','memory_fusion.layer_bias'}}
    initializations={};counts={}
    args.initialization_dir.mkdir(parents=True,exist_ok=True)
    for arm,model in models.items():
        if arm!='baseline':
            missing=model.load_state_dict(common,strict=False)
            assert not missing.unexpected_keys and set(missing.missing_keys)==extras[arm]
        assert all(torch.equal(value,model.state_dict()[name]) for name,value in common.items())
        counts[arm]=sum(p.numel() for p in model.parameters())
        path=args.initialization_dir/f'{arm}.pt'
        state=model.state_dict()
        if path.exists():
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            assert state.keys()==saved['model'].keys()
            assert all(torch.equal(value,saved['model'][name]) for name,value in state.items())
        else:
            temp=path.with_suffix('.tmp')
            torch.save({'model':state,'model_config':configs[arm]['model'],
                'tied_embeddings':configs[arm]['tied_embeddings'],'seed':configs[arm]['seed'],
                'memory_fusion':configs[arm].get('memory_fusion'),'arm':arm},temp)
            temp.replace(path)
        initializations[arm]={'path':str(path.resolve()),'sha256':sha256(path)}
    assert counts['A']-counts['baseline']==12 and counts['B']-counts['baseline']==1036
    ids=torch.tensor([[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]])
    changed=ids.clone();changed[:,8:]+=31
    limits={};causality={};uniform_outputs={}
    for arm,model in models.items():
        with torch.no_grad():
            output=model(ids)[0]
            altered=model(changed)[0]
            causal_error=float((output[:,:8]-altered[:,:8]).abs().max())
            assert causal_error<2e-5
            causality[arm]=causal_error
            if arm!='baseline':
                uniform_outputs[arm]=output.clone()
                original=model.memory_fusion.conditional_weights
                def last_only(states):
                    weights=states[-1].new_zeros(*states[-1].shape[:-1],12)
                    weights[...,-1]=1
                    return weights
                model.memory_fusion.conditional_weights=last_only
                actual=model(ids)[0];expected=base(ids)[0]
                limits[arm]=float((actual-expected).abs().max())
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
                model.memory_fusion.conditional_weights=original
                weights=model.memory_fusion.conditional_weights([torch.ones(1,16,1024)]*12)
                torch.testing.assert_close(weights,torch.full_like(weights,1/12),rtol=0,atol=0)
    torch.testing.assert_close(uniform_outputs['A'],uniform_outputs['B'],rtol=0,atol=0)
    gradients={}
    for arm in ('A','B'):
        model=models[arm];model.zero_grad(set_to_none=True)
        logits=model(ids[:,:8])[0]
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),ids[:,1:9].flatten())
        loss.backward()
        values={name:float(parameter.grad.norm()) for name,parameter in model.memory_fusion.named_parameters()}
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(value>0 for value in values.values()),values
        gradients[arm]=values
        model.zero_grad(set_to_none=True)
    # Closed-form locality check: zero logits give a mean over the causal window.
    length=1030;window=1024
    q=torch.zeros(1,length,1,4);k=q.clone();v=q.clone();v[:,0]=1
    with torch.no_grad():local=flash_attn_func(q,k,v,causal=True,window_size=(window-1,0))
    torch.testing.assert_close(local[0,1023],torch.full((1,4),1/1024),rtol=1e-5,atol=1e-7)
    assert torch.equal(local[0,1024:],torch.zeros_like(local[0,1024:]))
    cache_results={}
    for arm,c in configs.items():
        tiny=copy.deepcopy(c)
        tiny['model'].update(dim=32,n_layers=4,hidden_dim=64,n_self_heads=4,
            n_attn_heads=4,n_attn_kv_heads=4,vocab_size=256,max_seq_len=32,sliding_window=8)
        if 'memory_fusion' in tiny:tiny['memory_fusion']['source_layers']=[1,2]
        torch.manual_seed(42);model=build_model(tiny).eval()
        tokens=torch.arange(10,30).repeat(2,1)
        with torch.no_grad():
            full=model(tokens)[0]
            cache={};parts=[]
            for pos in range(tokens.shape[1]):
                parts.append(model(tokens[:,pos:pos+1],start_pos=pos,incremental_state=cache,is_prefilling=False)[0])
            tokenwise=torch.cat(parts,1)
            cache={};parts=[]
            for start,end in ((0,11),(11,16),(16,20)):
                parts.append(model(tokens[:,start:end],start_pos=start,incremental_state=cache,is_prefilling=start==0)[0])
            chunkwise=torch.cat(parts,1)
        torch.testing.assert_close(full,tokenwise,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(full,chunkwise,rtol=2e-5,atol=2e-5)
        assert cache[0]['prev_key'].shape==(2,8,4,8)
        cache_results[arm]={'tokenwise_max_abs':float((full-tokenwise).abs().max()),
            'chunkwise_max_abs':float((full-chunkwise).abs().max()),'cache_window':8,'default_max_batch_size_zero_supported':True}
    record={'passed':True,'architecture':'YOCO-SWA','official_window':1024,
        'official_setting_source':'https://arxiv.org/html/2405.05254v2#S4.SS2',
        'model':base_config['model'],'parameters':counts,'common_state_tensors':len(common),
        'common_weights_bitwise_equal':True,'training_and_data_match_existing_400m_and_swa160m':True,
        'all_three_execution_settings_equal':True,'uniform_A_B_outputs_bitwise_equal':True,
        'last_source_limit_max_abs':limits,'causality_prefix_max_abs':causality,
        'full_size_fusion_gradients':gradients,'window_boundary_1024_passed':True,
        'incremental_cache_checks':cache_results,'initializations':initializations,
        'precision':base_config['training']['precision'],'initial_weights':[1/12]*12,
        'note':'New 400M SWA tier: dim1024,24 layers,FFN2816,16 heads of dim64,window1024. Three arms start from one new random SWA backbone, not trained gRet weights. Actual parameter counts are reported.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,record)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
