"""Read existing consumption/commands without a neural-model query."""
import json,hashlib,resource,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def run(old,output,stage='ACTION2048'):
    began=time.monotonic();old=Path(old);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    assert stage in ('ACTION2048','MORE','EVENT')
    total=8192 if stage=='EVENT' else 16384
    path=old/'learning'/stage/'main/consumption.jsonl'
    logs=[json.loads(x) for x in path.read_text().splitlines()]
    assert len(logs)==total//4 and sum(len(r['visits']) for r in logs)==total
    plan=[json.loads(x) for x in (old/'data'/('PLAN16384.jsonl' if stage=='ACTION2048' else stage+'_PLAN.jsonl')).read_text().splitlines()]
    actual=[v for r in logs for v in r['visits']];assert actual==plan
    commands={s:np.load(old/f'data/episodes/TRAIN-{s}/action.npy',mmap_mode='r') for s in {v['seed'] for v in actual}}
    counts=Counter();events=defaultdict(Counter)
    for v in actual:
        a=v['anchor'];c=commands[v['seed']][:,9];changes=np.diff(c,prepend=c[0])
        prefix=changes[a:a+8];tail=changes[a+8:a+40]
        kinds=[]
        for name,sign in [('open',1),('close',-1)]:
            kinds.extend([name+'_first8'] if (prefix*sign>.1).any() else [name+'_tail_only'] if (tail*sign>.1).any() else [])
        if not (np.abs(changes[a:a+40])>.1).any():
            kinds.append('stable_open' if np.mean(c[a:a+8])>.9 else 'stable_closed' if np.mean(c[a:a+8])<.1 else 'stable_intermediate')
        for kind in kinds:counts[kind]+=1;events[str(v['phase'])][kind]+=1
    history=json.loads((old/'learning'/stage/'main'/('checkpoint-1024' if stage=='EVENT' else 'checkpoint-2048')/'trainer_state.json').read_text())['log_history']
    losses=np.asarray([r['loss'] for r in logs],float);assert np.isfinite(losses).all()
    detailed=[r['gradient'] for r in logs if 'FINITE' in r['gradient'].get('status','') or 'finite' in r['gradient'].get('status','').lower()]
    value=dict(status='PASS_EXISTING_CONSUMPTION_UID_EVENT_CPU_AUDIT',source_consumption_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actual_microbatches=len(logs),actual_draws=len(actual),unique_uids=len({v['uid'] for v in actual}),repeated_draws=len(actual)-len({v['uid'] for v in actual}),episode_phase_counts={f'{s}:{p}':n for (s,p),n in Counter((v['seed'],v['phase']) for v in actual).items()},events=dict(counts),events_by_phase={k:dict(v) for k,v in events.items()},returned_microbatch_loss=dict(min=float(losses.min()),max=float(losses.max()),first16_mean=float(losses[:16].mean()),last16_mean=float(losses[-16:].mean())),trainer_log_history=history,group_learning_rate_history='Trainer logged group0 only; group1 historical per-step values UNOBSERVED; source schedule is not an actual log' if stage=='ACTION2048' else 'BOTH_GROUPS_ACTUAL_NATIVE_LOSS_MODES_LR_JOURNAL',historical_mode='UNOBSERVED' if stage=='ACTION2048' else 'ACTUAL_NATIVE_MODE_JOURNAL_PRESERVED',native_forward_loss_before_accumulation='UNOBSERVED; new proof captures actual native loss and returned loss' if stage=='ACTION2048' else 'DIRECT_NATIVE_LOSS_MODES_LR_JOURNAL',detailed_gradient_entries=sum('entries' in r['gradient'] or len(r['gradient'])>1 for r in logs),gradient_first=logs[0]['gradient'],constant_dimensions=json.loads((old/'data/DATA_CONTRACT.json').read_text())['constant_dimensions'],new_model_queries=0,new_optimizer=0,new_physics=0,elapsed_s=time.monotonic()-began,CPU_s=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)
    (output/('PARENT_NATIVE_EXPOSURE.json' if stage=='ACTION2048' else stage+'_NATIVE_EXPOSURE.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k not in ('trainer_log_history','episode_phase_counts','gradient_first','constant_dimensions')}))


if __name__=='__main__':
    import sys
    run(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 'ACTION2048')
