"""Constructed CPU contract fixtures, never a learned-model query."""
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from train import ROOT,read,sha
from readout import generate
import pi0247_u_rollout_overlay as overlay
from pi0260_r3_physical import install as bind_physical
from temporal import self_check as temporal_check
from unclamped_fixture import check as rotation_check


def main():
    assert not torch.cuda.is_available();torch.set_num_threads(4)
    bind_physical();assert overlay.CLAMP_NORMALIZED_STATE is False
    tree=ast.parse(inspect.getsource(generate))
    assert not any(isinstance(n,ast.Name) and n.id in ('expert','action_input','label','target') for n in ast.walk(tree))
    assert list(inspect.signature(generate).parameters)==['head','backbone','state','embodiment','execution_phase','checkpointed','audit']
    class Encoder(torch.nn.Module):
        def forward(self,x,*unused):return x
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3));self.dropout=torch.nn.Dropout(.5)
        def forward(self,*,hidden_states,encoder_hidden_states,timestep,**kw):
            assert torch.count_nonzero(timestep)==0 and not self.training
            return self.dropout(hidden_states+encoder_hidden_states.mean())*self.weight
    head=torch.nn.Module();head.num_inference_timesteps=4;head.action_horizon=40;head.action_dim=132
    head.config=SimpleNamespace(add_pos_embed=False,use_alternate_vl_dit=True)
    head.state_encoder=Encoder();head.action_encoder=Encoder();head.action_decoder=Encoder();head.model=Toy();head.train()
    phase_results=[]
    for phase,dtype in (('training',torch.float32),('quality',torch.bfloat16)):
        head.model.weight.grad=None
        features=SimpleNamespace(backbone_features=torch.ones(1,2,132,dtype=dtype),image_mask=None,backbone_attention_mask=None)
        state=torch.full((1,1,132),.7,dtype=dtype);emb=torch.zeros(1,dtype=torch.long)
        modes=[m.training for m in head.modules()];rng=torch.get_rng_state().clone()
        a,q=generate(head,features,state,emb,execution_phase=phase,checkpointed=True)
        b,_=generate(head,features,state,emb,execution_phase=phase)
        assert a.shape==(1,40,132) and q.dtype==dtype and not q.any()
        assert torch.equal(a,b) and a.abs().sum()>0 and torch.equal(rng,torch.get_rng_state())
        a.float().square().mean().backward()
        assert torch.isfinite(head.model.weight.grad) and head.model.weight.grad.abs()>0
        assert modes==[m.training for m in head.modules()] and torch.equal(rng,torch.get_rng_state())
        wrong='quality' if phase=='training' else 'training'
        try:generate(head,features,state,emb,execution_phase=wrong)
        except AssertionError:pass
        else:raise AssertionError('Wrong explicit phase accepted')
        phase_results.append(dict(phase=phase,query_dtype=str(q.dtype),output_dtype=str(a.dtype),wrong_phase_rejected=True))
    prior=read(ROOT.with_name('PI0269')/'evidence/HOPPER_PHYSICAL_FIXTURES.json')
    assert prior['status']=='PASS_ROT_CONTRACT_PHYSICAL_AND_WEIGHTED_FIXTURES'
    result=dict(status='PASS_PI0272_CONSTRUCTED_DIRECT_AND_UNCHANGED_PHYSICAL_CPU_FIXTURES',
        direct_zero_latent_not_zero_output=True,one_generated_node=True,label_free_boundary=True,
        mode_and_rng_restored=True,checkpoint_gradient=True,temporal=temporal_check(),
        noncommuting_rotation=rotation_check(),inherited_actual_roundtrip_receipt_sha256=sha(ROOT.with_name('PI0269')/'evidence/HOPPER_PHYSICAL_FIXTURES.json'),
        phase_fixtures=phase_results,model_queries=0,learned_gradients=0,optimizer_updates=0)
    (ROOT/'evidence/CPU_FIXTURES_R1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(result['status'])


if __name__=='__main__':main()
