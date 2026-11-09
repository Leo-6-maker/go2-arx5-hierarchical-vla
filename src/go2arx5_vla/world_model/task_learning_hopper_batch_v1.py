"""Read actual finite microbatch trials; never infer 4096 simulator capacity."""
import json,math,statistics
from pathlib import Path
from .target_gr00t_profile_v1 import need


def summarize_trial(root,microbatch):
    root=Path(root);s=json.loads((root/'EXECUTION_SPEC.json').read_bytes())
    r=json.loads((root/'RESULT.json').read_bytes());lease=json.loads((root/'control/LEASE.json').read_bytes())
    w=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    updates=[json.loads(v)for v in(root/'OPTIMIZER_STEPS.jsonl').read_text().splitlines()]
    memory=json.loads((root/'GPU_MEMORY.json').read_bytes())['measurements']
    need(microbatch in(1,2,4,8)and s['recipe']['microbatch']==microbatch
        and s['recipe']['effective_batch']==32 and s['recipe']['accumulation']==32//microbatch
        and s['recipe']['variant']=='FULL_ACTION_SIDE'and s['recipe']['mode']=='RESTORE4'
        and lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==w.get('exit_code')==0
        and w.get('failure')is None and r.get('global_step')==4 and r.get('sample_visits')==128
        and r.get('real_action_parameter_changed')is True and r.get('backbone_frozen')is True
        and[r['step']for r in updates]==[1,2,3,4]
        and all(math.isfinite(v['optimizer_update_wall_s'])and v['optimizer_update_wall_s']>0 for v in updates),
        'ACTUAL_CLOSED_NONZERO_FOUR_UPDATE_MICROBATCH_TRIAL')
    steady=statistics.median(v['optimizer_update_wall_s']for v in updates[1:])
    return dict(microbatch=microbatch,effective_batch=32,optimizer_calls=4,sample_visits=128,
        steady_update_wall_s=steady,maximum_update_wall_s=max(v['optimizer_update_wall_s']for v in updates),
        TRAIN_visits_per_s=32/steady,worker_wall_s=w['wall_s'],
        peak_allocated_GPU_bytes=max(v['peak_allocated_bytes']for v in memory),
        peak_reserved_GPU_bytes=max(v['peak_reserved_bytes']for v in memory),
        minimum_free_GPU_bytes=min(v['free_bytes']for v in memory))


def select(trials):
    need(len(trials)==4 and[v['microbatch']for v in trials]==[1,2,4,8]
        and all(v['effective_batch']==32 and v['optimizer_calls']==4
            and v['sample_visits']==128 and v['minimum_free_GPU_bytes']>=2*1024**3
            and math.isfinite(v['steady_update_wall_s'])and v['steady_update_wall_s']>0 for v in trials),
        'COMPLETE_ORDERED_CAPACITY_SCAN_WITH_ORIGINAL_GPU_MARGIN')
    best=min(trials,key=lambda v:(v['steady_update_wall_s'],v['microbatch']))
    return dict(schema='TASK_LEARNING_SINGLE_H200_MICROBATCH_SELECTION_V1',
        selected_microbatch=best['microbatch'],effective_batch=32,
        measured_steady_update_speedup_over_micro1=trials[0]['steady_update_wall_s']/best['steady_update_wall_s'],
        selection='FASTEST_MEDIAN_UPDATES2_TO4; SMALLER_MICROBATCH_ON_EXACT_TIE',
        frozen_backbone=True,physics_environments=0,formal_training=False,
        larger_batch_is_not_claimed_bit_exact_to_micro1=True)
