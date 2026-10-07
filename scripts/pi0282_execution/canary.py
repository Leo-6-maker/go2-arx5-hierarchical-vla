"""Read-only 8 versus fresh-process 4+4 complete-state comparison."""
import argparse,json,hashlib
from pathlib import Path
import torch
import numpy as np
from safetensors import safe_open
from learning import tensor_tree


def equal(a,b,tolerance=False):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype
        assert np.array_equal(a,b),'CANARY_NUMPY_RNG_MISMATCH'
        return True
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        bitwise=torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))
        numerical=bitwise or tolerance and a.dtype==torch.float32 and torch.allclose(a,b,atol=1e-6,rtol=1e-5)
        assert numerical,'CANARY_TENSOR_MISMATCH'
        return bitwise
    if isinstance(a,dict):
        assert set(a)==set(b)
        return all([equal(a[k],b[k],tolerance) for k in a])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return all([equal(x,y,tolerance) for x,y in zip(a,b)])
    assert a==b
    return True


def model_compare(a,b):
    def index(root):
        result={}
        for f in Path(root).glob('*.safetensors'):
            with safe_open(f,framework='pt',device='cpu') as h:
                for key in h.keys():assert key not in result;result[key]=f
        return result
    ia,ib=index(a),index(b);assert set(ia)==set(ib)
    exact=True;manifest={}
    for key in ia:
        with safe_open(ia[key],framework='pt',device='cpu') as fa,safe_open(ib[key],framework='pt',device='cpu') as fb:
            ta,tb=fa.get_tensor(key),fb.get_tensor(key)
            # Only authorized changed FP32 values may use published tolerance;
            # all other scalars, including packed non10 rows, stay exact.
            allowed=key.startswith(('action_head.model.','action_head.state_encoder.','action_head.action_encoder.','action_head.action_decoder.'))
            same=equal(ta,tb,allowed)
            if key.startswith(('action_head.state_encoder.','action_head.action_encoder.','action_head.action_decoder.')):
                equal(ta[:10],tb[:10]);equal(ta[11:],tb[11:])
            exact=exact and same;manifest[key]=dict(a=tensor_tree(ta),b=tensor_tree(tb),bitwise_equal=same)
    return exact,manifest


def compare(root,arm):
    root=Path(root);base=root/'learning'/arm
    continuous=base/'continuous/checkpoint-8';split=base/'split/checkpoint-8'
    assert (base/'split/checkpoint-4/PI0282_RECOVERY.json').exists()
    processes=[json.loads(p.read_text()) for p in (base/'continuous/PROCESS_resume0_stop8.json',base/'split/PROCESS_resume0_stop4.json',base/'split/PROCESS_resume4_stop8.json')]
    assert len({p['pid'] for p in processes})==3,'CANARY_MUST_USE_THREE_FRESH_PROCESSES'
    for checkpoint in (continuous,split):assert json.loads((checkpoint/'PI0282_RECOVERY.json').read_text())['cursor']==64
    exact,manifest=model_compare(continuous,split)
    states={}
    for name in ('optimizer.pt','scheduler.pt','rng_state.pth'):
        a=torch.load(continuous/name,map_location='cpu',weights_only=False);b=torch.load(split/name,map_location='cpu',weights_only=False)
        if name=='optimizer.pt':
            for state in (a,b):
                assert state['param_groups'][0]['weight_decay']==0.
                assert len(state['param_groups'][0]['params'])==14
                for index in state['param_groups'][0]['params']:
                    for key in ('exp_avg','exp_avg_sq'):
                        moment=state['state'][index][key]
                        assert moment.shape[0]==32
                        assert not torch.count_nonzero(moment[:10]) and not torch.count_nonzero(moment[11:]),'FROZEN_OPTIMIZER_ROWS_NONZERO'
        states[name]=dict(bitwise_equal=equal(a,b,tolerance=name=='optimizer.pt'),a=tensor_tree(a),b=tensor_tree(b))
    rows=[]
    for phase in ('continuous','split'):
        p=base/phase/'consumption.jsonl';v=[json.loads(x) for x in p.read_text().splitlines()]
        micro=len(v[0]['visits']);assert micro in (1,2,4)
        assert len(v)==64//micro and sum(len(x['visits']) for x in v)==64
        rows.append([{k:x[k] for k in ('step','micro','cursor','visits','inputs','native_noise_sha256','native_time_sha256','native_time')} for x in v])
    assert rows[0]==rows[1],'CONSUMED_INPUT_OR_NATIVE_RNG_DIFFERENCE'
    # Owner evaluation supplies fixed native S4 outputs and original gate receipt,
    # charged against quality-query caps; this CPU comparator performs no query.
    ca=json.loads((base/'continuous/CANARY_PREDICTIONS.json').read_text());cb=json.loads((base/'split/CANARY_PREDICTIONS.json').read_text())
    assert ca==cb,'PREDICTION_OR_GATE_DISCREPANCY'
    receipt=dict(status='PASS_FULL_RESUME_CANARY',arm=arm,model_bitwise_equal=exact,updated_FP32_allclose_atol=1e-6,
      updated_FP32_allclose_rtol=1e-5,model_tensors=manifest,states=states,consumed_chunks=64,completed_discarded_updates=16,
      inputs_native_noise_time_exact=True,fresh_process_restore=processes,predictions_gates_exact=True)
    (base/'CANARY_COMPARISON.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('model_tensors','states')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',choices=('ACTION2048','INTERFACE2048'),required=True)
    a=p.parse_args();compare(a.root,a.arm)
