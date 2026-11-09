"""Read actual finite microbatch trials; never infer 4096 simulator capacity."""
import json,math,statistics
from pathlib import Path
from .target_gr00t_profile_v1 import need

ROOT=Path('/scratch/e1192548/Projects/GR00T-WBC/runs/WM02')


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


def qualification(proof,spec):
    """Admit a chosen batch only after its real closed scan and full restore."""
    from .stable_task_scene_v1 import pinned
    from .storage import digest
    from .task_learning_training_admission_v1 import unchanged_training_math
    result=pinned(proof);root=Path(proof['path']).parent
    need(root.is_relative_to(ROOT)and Path(proof['path'])==root/'RESULT.private.json',
        'ACTUAL_OWN_HOPPER_BATCHSCAN_RESULT')
    def read(name):
        path=root/name
        need(path.resolve()==path and not path.is_symlink()and path.stat().st_size<16*1024**2,
            'BOUNDED_CANONICAL_BATCHSCAN_METADATA')
        return json.loads(path.read_bytes())
    c=read('ALLOCATION_CONFIG.json');lease=read('control/LEASE.json');worker=read('WORKER_EXIT.private.json')
    need(c.get('root')==str(root)and c.get('phase')=='BATCH_THROUGHPUT_SCAN'
        and c.get('pbs_job_name')=='gr00t_batchscan'
        and lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==worker.get('exit_code')==0
        and worker.get('failure')is None and bool(lease.get('job_id'))
        and lease.get('config_sha256')==digest(root/'ALLOCATION_CONFIG.json')
        and lease.get('result_sha256')==proof['sha256']
        and result.get('status')=='HOPPER_BATCH_THROUGHPUT_AND_SELECTED_FULL_RESTORE_SEALED'
        and result.get('optimizer_calls')==20 and result.get('effective_batch')==32
        and result.get('physics_integrations')==0 and result.get('formal_training')is False,
        'ACTUAL_CLOSED_ZERO_EXIT_BATCHSCAN_BEFORE_LARGER_MICROBATCH')
    actual=[summarize_trial(root/('batch'+str(m)+'-continuous4'),m)for m in(1,2,4,8)]
    chosen=select(actual);micro=chosen['selected_microbatch']
    need(result.get('trials')==read('BATCH_TRIALS_ACTUAL.json').get('trials')==actual
        and result.get('selection')==read('BATCH_SELECTION_ACTUAL.json')==chosen
        and result.get('selected_microbatch')==spec['recipe']['microbatch']==micro,
        'FORMAL_MICROBATCH_MATCHES_ACTUAL_FASTEST_QUALIFIED_SELECTION')
    expected=[('batch'+str(m)+'-continuous4',4)for m in(1,2,4,8)]+[
        ('selected-split2',2),('selected-resume4',4)]
    children=result.get('children')or[]
    need([r.get('name')for r in children]==[n for n,_ in expected],
        'ALL_SIX_ACTUAL_BATCHSCAN_CHILDREN_RETAINED')
    for row,(name,endpoint)in zip(children,expected):
        child=read(name+'/RESULT.json');closed=read(name+'/control/LEASE.json');w=read(name+'/WORKER_EXIT.private.json')
        need(row.get('exit_code')==closed.get('worker_exit_code')==w.get('exit_code')==0
            and closed.get('status')=='CLOSED'and w.get('failure')is None
            and row.get('result')==child and row.get('result_sha256')==digest(root/name/'RESULT.json')
            and child.get('global_step')==endpoint,
            'ACTUAL_TERMINAL_TRIAL_AND_SELECTED_RESTORE_CHILD_'+name)
    restored=read('SELECTED_FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json')
    checks={'target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth',
        'SAMPLER_STATE.json','PARAMETER_CONTRACT.json','FROZEN_PARENT_TENSORS.json'}
    need(restored.get('status')=='PASS_ACTUAL_UNINTERRUPTED_VERSUS_RESUMED_NEXT_UPDATE'
        and restored.get('comparison')=='CONTINUOUS4_VS_FRESH_PROCESS_2_PLUS_2'
        and restored.get('microbatch')==micro
        and restored.get('next_update_parameters_optimizer_scheduler_RNG_sampler_exact')is True
        and set(restored.get('checks',{}))==checks and all(v is True for v in restored['checks'].values())
        and result.get('selected_full_restore')=={k:v for k,v in restored.items()if k not in('comparison','microbatch')},
        'SELECTED_MICROBATCH_ACTUAL_SEVEN_STATE_CONTINUOUS4_VERSUS_FRESH2_PLUS2')
    need(c['python']==spec['python']and c['parent']==spec['parent']
        and c['pbs_server']==spec['pbs_server']
        and all(c['bindings'][k]==spec['bindings'][k]for k in('source','cosmos','cosmos_hub')),
        'SAME_ACTUALLY_BATCH_QUALIFIED_PARENT_RUNTIME_AND_BINDINGS')
    math=unchanged_training_math(c,spec)
    return dict(status='ACTUAL_SELECTED_SINGLE_H200_MICROBATCH_AND_FULL_RESTORE_ADMITTED',
        job_id=lease['job_id'],result_sha256=proof['sha256'],selected_microbatch=micro,
        effective_batch=32,measured_steady_update_speedup_over_micro1=chosen['measured_steady_update_speedup_over_micro1'],
        training_math=math,qualification_weights_used_as_formal_parent=False)
