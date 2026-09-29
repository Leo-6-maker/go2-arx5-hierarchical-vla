"""Read-only, pre-toleranced comparison of the two complete R3 recovery paths."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0260/R3')


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def max_delta(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    if left.is_floating_point():
        return float((left.float()-right.float()).abs().max()) if left.numel() else 0.
    assert torch.equal(left,right)
    return 0.


def model_delta(left,right):
    a=json.loads((left/'model.safetensors.index.json').read_text())['weight_map']
    b=json.loads((right/'model.safetensors.index.json').read_text())['weight_map']
    assert a==b
    maximum=0.
    for shard in sorted(set(a.values())):
        with safe_open(left/shard,framework='pt',device='cpu') as x, safe_open(right/shard,framework='pt',device='cpu') as y:
            assert x.keys()==y.keys()
            for key in x.keys():maximum=max(maximum,max_delta(x.get_tensor(key),y.get_tensor(key)))
    return maximum,len(a)


def optimizer_delta(left,right):
    a=torch.load(left/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
    b=torch.load(right/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
    assert a['param_groups']==b['param_groups'] and a['state'].keys()==b['state'].keys()
    maximum=0.;tensors=0
    for key in a['state']:
        assert a['state'][key].keys()==b['state'][key].keys()
        for field,value in a['state'][key].items():
            other=b['state'][key][field]
            if isinstance(value,torch.Tensor):
                maximum=max(maximum,max_delta(value,other));tensors+=1
            else:assert value==other
    return maximum,tensors


def compare(arm, continuous="canary-cont", split="canary-split"):
    a=ROOT/arm/continuous/'training'
    b=ROOT/arm/split/'training'
    x,y=a/'checkpoint-8',b/'checkpoint-8'
    for p in (x,y):
        state=json.loads((p/'trainer_state.json').read_text())
        assert state['global_step']==8 and state['max_steps']==6000
        assert all((p/f).is_file() for f in ('optimizer.pt','scheduler.pt','rng_state.pth'))
    for name,fields in [('sample_audit.jsonl',('update','task','command','sequence','index','condition_sha256')),
                        ('step_rng.jsonl',('step','seed','condition_sha256','mode_restored')),
                        ('S4_audit.jsonl',('forward_index','native_epsilon_sha256','native_tau','stock_initial_noise')),
                        ('full_loss.jsonl',('step',))]:
        ra,rb=rows(a/name),rows(b/name)
        assert len(ra)==len(rb)==8,(name,len(ra),len(rb))
        for i,(left,right) in enumerate(zip(ra,rb),1):
            assert all(left[k]==right[k] for k in fields),(name,i)
    for p in (a,b):
        assert json.loads((p/'S4_parity.json').read_text())['status']=='PASS_PI0247_U_S4_PARITY'
        grads=json.loads((p/'S4_first_update_gradients.json').read_text())
        assert set(grads)=={'1','2','3','4'} and all(float(v)>0 for v in grads.values())
    model,tensors=model_delta(x,y)
    optimizer,states=optimizer_delta(x,y)
    scheduler_a=torch.load(x/'scheduler.pt',map_location='cpu',weights_only=False)
    scheduler_b=torch.load(y/'scheduler.pt',map_location='cpu',weights_only=False)
    assert scheduler_a.keys()==scheduler_b.keys()
    assert scheduler_a['last_epoch']==scheduler_b['last_epoch']==8
    lr_delta=max(abs(float(u)-float(v)) for u,v in zip(scheduler_a['_last_lr'],scheduler_b['_last_lr']))
    assert model<=1e-6 and optimizer<=1e-6 and lr_delta<=1e-12,(model,optimizer,lr_delta)
    result=dict(status='PASS_R3_REAL_FULL_TRAINER_CANARY',arm=arm,steps=8,
                model_max_abs=model,model_tensors=tensors,optimizer_max_abs=optimizer,
                optimizer_tensors=states,scheduler_lr_abs=lr_delta,
                condition_epsilon_tau_equal=True,mode_restored_each_step=True)
    out=ROOT/arm/('canary_compare.json' if continuous=='canary-cont' else 'canary_samegpu_compare.json');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


def demo():
    assert max_delta(torch.tensor([1.,2.]),torch.tensor([1.,2.]))==0.
    assert max_delta(torch.tensor([1.,2.]),torch.tensor([1.,3.]))==1.
    print('PASS_CANARY_COMPARISON_DEMO')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=('Base','G'))
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--continuous',default='canary-cont')
    parser.add_argument('--split',default='canary-split')
    args=parser.parse_args()
    demo() if args.demo else compare(args.arm,args.continuous,args.split)
