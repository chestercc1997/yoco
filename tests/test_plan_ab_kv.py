"""Six-source learned and conditional KV aggregation contracts."""
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from runtime.model import build_model
from runtime.yoco_official.yoco import apply_rotary_emb


def config(mode=None):
    c=dict(architecture='yoco',tied_embeddings=True,model=dict(dim=32,n_layers=12,hidden_dim=96,
        n_self_heads=2,n_attn_heads=4,n_attn_kv_heads=4,vocab_size=97,max_seq_len=512,norm_eps=1e-5))
    if mode: c['memory_fusion']=dict(mode=mode,source_layers=[1,2,3,4,5,6],temperature=1.)
    return c


class PlanABTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_common_initialization_and_uniform_output(self):
        models=[]
        for mode in (None,'learned_kv','conditional_kv'):
            torch.manual_seed(42); models.append(build_model(config(mode)).eval())
        for name,p in models[0].state_dict().items():
            for model in models[1:]: torch.testing.assert_close(p,model.state_dict()[name],rtol=0,atol=0)
        ids=torch.randint(0,97,(2,9))
        torch.testing.assert_close(models[1](ids)[0],models[2](ids)[0],rtol=0,atol=0)
        states=[torch.randn(2,9,32) for _ in range(6)]
        for model in models[1:]:
            torch.testing.assert_close(model.memory_fusion.conditional_weights(states),torch.full((2,9,6),1/6))
        self.assertEqual(sum(p.numel() for p in models[1].memory_fusion.parameters()),6)
        self.assertEqual(sum(p.numel() for p in models[2].memory_fusion.parameters()),38)

    def test_explicit_kv_formula_and_reference_gradients(self):
        for mode in ('learned_kv','conditional_kv'):
            with self.subTest(mode=mode):
                model=build_model(config(mode)); f=model.memory_fusion; d=model.cross_decoder
                with torch.no_grad():
                    for p in f.parameters(): p.normal_()
                states=[torch.randn(2,9,32,requires_grad=True) for _ in range(6)]
                if mode=='learned_kv':
                    expected=f.logits.softmax(-1).expand(2,9,-1)
                else:
                    norm=torch.stack([x*torch.rsqrt(x.square().mean(-1,keepdim=True)+f.eps) for x in states],-2)
                    expected=((norm*f.score).sum(-1)/32**.5+f.layer_bias).softmax(-1)
                weights=f.conditional_weights(states)
                torch.testing.assert_close(weights,expected)
                if mode=='learned_kv':
                    torch.testing.assert_close(weights[:,0],weights[:,1],rtol=0,atol=0)
                    changed=f.conditional_weights([x+1 for x in states])
                    torch.testing.assert_close(weights,changed,rtol=0,atol=0)
                else: self.assertGreater((weights[:,0]-weights[:,1]).abs().max().item(),1e-4)
                captured=[]
                handles=[layer.register_forward_pre_hook(lambda m,a,kw:captured.append((kw['key'],kw['value'])),with_kwargs=True) for layer in d.layers]
                try: d(states[-1],memory_sources=states,memory_weights=weights)
                finally:
                    for h in handles: h.remove()
                self.assertEqual(len(captured),6)
                self.assertTrue(all(k is captured[0][0] and v is captured[0][1] for k,v in captured))
                k=sum(d.k_proj(d.kv_layer_norm(x))*expected[...,i,None] for i,x in enumerate(states)).view(2,9,4,8)
                v=sum(d.v_proj(d.kv_layer_norm(x))*expected[...,i,None] for i,x in enumerate(states)).view(2,9,4,8)
                k=apply_rotary_emb(k,*d.build_rel_pos(states[-1],0),interleaved=True)
                torch.testing.assert_close(captured[0][0],k); torch.testing.assert_close(captured[0][1],v)
                variables=states+list(f.parameters())
                actual_grads=torch.autograd.grad(sum(x.square().mean() for x in captured[0]),variables,retain_graph=True)
                ref_grads=torch.autograd.grad(k.square().mean()+v.square().mean(),variables)
                for actual,ref in zip(actual_grads,ref_grads):
                    self.assertTrue(torch.isfinite(actual).all() and actual.abs().sum()>0)
                    torch.testing.assert_close(actual,ref,atol=2e-7,rtol=2e-5)

    def test_last_source_limit_matches_baseline(self):
        ids=torch.randint(0,97,(2,9))
        torch.manual_seed(42); base=build_model(config()).eval()
        for mode in ('learned_kv','conditional_kv'):
            torch.manual_seed(42); model=build_model(config(mode)).eval()
            f=model.memory_fusion
            with torch.no_grad():
                bias=f.logits if mode=='learned_kv' else f.layer_bias
                bias.fill_(-1000); bias[-1]=0
                torch.testing.assert_close(base(ids)[0],model(ids)[0],atol=2e-5,rtol=2e-5)

    def test_causality_and_shared_incremental_cache(self):
        ids=torch.randint(0,97,(2,9)); changed=ids.clone(); changed[:,5:]=(changed[:,5:]+1)%97
        for mode in ('learned_kv','conditional_kv'):
            with self.subTest(mode=mode),torch.no_grad():
                model=build_model(config(mode)).eval()
                for p in model.memory_fusion.parameters(): p.normal_()
                full=model(ids)[0]
                torch.testing.assert_close(full[:,:5],model(changed)[0][:,:5],atol=2e-5,rtol=2e-5)
                state={}
                pieces=[model(ids[:,i:i+1],start_pos=i,incremental_state=state,is_prefilling=False)[0] for i in range(9)]
                torch.testing.assert_close(full,torch.cat(pieces,1),atol=2e-5,rtol=2e-5)
                self.assertEqual(set(state)&{'prev_key','prev_value'},{'prev_key','prev_value'})
                self.assertEqual(state['prev_key'].shape,(2,512,4,8))

    def test_activation_checkpoint_and_optimizer_resume(self):
        ids=torch.randint(0,97,(2,9))
        for mode in ('learned_kv','conditional_kv'):
            with self.subTest(mode=mode):
                model=build_model(config(mode)); checked=build_model(config(mode),checkpoint_activations=True)
                checked.load_state_dict(model.state_dict())
                for m in (model,checked): m(ids)[0].square().mean().backward()
                for (name,p),(_,q) in zip(model.named_parameters(),checked.named_parameters()):
                    self.assertIsNotNone(q.grad,name)
                    torch.testing.assert_close(p.grad,q.grad,atol=5e-6,rtol=1e-4)
                opt=torch.optim.AdamW(model.parameters(),lr=1e-3); opt.step(); opt.zero_grad()
                stream=io.BytesIO(); torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),stream)
                stream.seek(0); saved=torch.load(stream,weights_only=True)
                restored=build_model(config(mode)); restored.load_state_dict(saved['model'])
                opt2=torch.optim.AdamW(restored.parameters(),lr=1e-3); opt2.load_state_dict(saved['optimizer'])
                for m,o in ((model,opt),(restored,opt2)): m(ids)[0].square().mean().backward(); o.step()
                for p,q in zip(model.parameters(),restored.parameters()): torch.testing.assert_close(p,q,atol=0,rtol=0)


if __name__=='__main__': unittest.main()
