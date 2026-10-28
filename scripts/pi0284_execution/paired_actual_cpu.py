"""Compare only completed actual consumption; preserve original event-transition definition."""
import json,os,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from prepare_core import sha,write

def run(root):
    root=Path(root);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    arms=('CTRL2048','LR5X2048');all_rows=[];all_modes=[];sources={}
    for arm in arms:
        folder=root/'learning'/arm/'main';assert (folder/'JOURNAL.json').exists()
        paths=[folder/'consumption.jsonl',folder/'native_loss_modes_LR.jsonl'];sources.update({str(p):sha(p) for p in paths})
        rows=[json.loads(x) for x in paths[0].read_text().splitlines()];modes=[json.loads(x) for x in paths[1].read_text().splitlines()]
        assert len(rows)==len(modes)==4096;all_rows.append(rows);all_modes.append(modes)
    keys=('step','micro','cursor','visits','inputs','native_noise_sha256','native_time_sha256','native_time')
    for a,b,ma,mb in zip(*all_rows,*all_modes):
        assert {k:a[k] for k in keys}=={k:b[k] for k in keys},a['step']
        assert ma['actual_LRs'][0]==mb['actual_LRs'][0]
        assert np.isclose(ma['actual_LRs'][1]*5,mb['actual_LRs'][1],rtol=1e-12,atol=1e-18)
    plan=[json.loads(x) for x in (root/'data/PAIRED_PLAN16384.jsonl').read_text().splitlines()]
    actual=[v for r in all_rows[0] for v in r['visits']];assert actual==plan
    events=defaultdict(Counter);commands={}
    for v in actual:
        if v['seed'] not in commands:commands[v['seed']]=np.load(root/'data/episodes'/f"TRAIN-{v['seed']}"/'action.npy',mmap_mode='r')[:,9]
        c=commands[v['seed']];a=v['anchor'];changes=np.diff(c,prepend=c[0]);prefix=changes[a:a+8];tail=changes[a+8:a+40]
        for name,sign in (('open',1),('close',-1)):
            if (prefix*sign>.1).any():events[str(v['phase'])][name+'_first8']+=1
            elif (tail*sign>.1).any():events[str(v['phase'])][name+'_tail_only']+=1
        if not (np.abs(changes[a:a+40])>.1).any():events[str(v['phase'])]['stable_open' if np.mean(c[a:a+8])>.9 else 'stable_closed' if np.mean(c[a:a+8])<.1 else 'stable_intermediate']+=1
    result=dict(status='PASS_FULL_FORMAL_ACTUAL_PAIRING_AND_ONLY_DIT_LR_SCALE',microbatches_each=4096,chunks_each=16384,sources=sources,common_plan_sha256=sha(root/'data/PAIRED_PLAN16384.jsonl'),events_by_phase={k:dict(v) for k,v in events.items()},event_definition='Full episode gripper diff including transition into anchor sample0; original first8/tail threshold .1. Preparatory opening_first8_draws excluded that transition and is not numerically interchangeable.',new_model_queries=0,new_optimizer_updates=0,new_physics=0,selected='NONE',deployment='NONE')
    write(root/'runtime/FORMAL_PAIRED_ACTUAL_CPU.json',result);print(json.dumps(result))

if __name__=='__main__':run(sys.argv[1])
