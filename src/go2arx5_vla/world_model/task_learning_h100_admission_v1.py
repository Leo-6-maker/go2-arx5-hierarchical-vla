"""Bind new-task H100 training to its actual original-Trainer restore trial."""
import json
from pathlib import Path
from .target_gr00t_profile_v1 import need
from .stable_task_scene_v1 import pinned
from .storage import digest

ROOT=Path('/scratch/e1192548/Projects/GR00T-WBC/runs/WM02')
CHECKS={'target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth',
        'SAMPLER_STATE.json','PARAMETER_CONTRACT.json','FROZEN_PARENT_TENSORS.json'}


def qualification(proof,current):
    from .task_learning_hopper_v1 import actual_gpu
    from .task_learning_training_admission_v1 import unchanged_training_math
    result=pinned(proof);root=Path(proof['path']).parent
    need(root.is_relative_to(ROOT)and Path(proof['path'])==root/'RESULT.private.json',
        'ACTUAL_OWN_H100_RESTORE_RESULT')
    def read(name):
        p=root/name
        need(p.resolve()==p and not p.is_symlink()and p.stat().st_size<16*1024**2,
            'BOUNDED_CANONICAL_H100_RESTORE_METADATA')
        return json.loads(p.read_bytes())
    c=read('ALLOCATION_CONFIG.json');lease=read('control/LEASE.json');worker=read('WORKER_EXIT.private.json')
    need(c.get('root')==str(root)and c.get('phase')=='RESTORE_ONLY'
        and c.get('allocation_gpu_model')=='H100'and c.get('pbs_job_name')=='gr00t_h100_check'
        and lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==worker.get('exit_code')==0
        and worker.get('failure')is None and bool(lease.get('job_id'))
        and lease.get('config_sha256')==digest(root/'ALLOCATION_CONFIG.json')
        and lease.get('result_sha256')==proof['sha256']
        and result.get('status')=='HOPPER_ORIGINAL_TRAINER_RESTORE_ONLY_SEALED'
        and result.get('allocation_gpu_model')=='H100'and result.get('optimizer_calls')==8
        and result.get('physics_integrations')==0 and result.get('formal_training')is False,
        'ACTUAL_CLOSED_ZERO_EXIT_H100_RESTORE_BEFORE_FORMAL_TRAINING')
    actual_gpu(read('ACTUAL_GPU.private.json'),'H100')
    restored=read('FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json')
    need(restored.get('status')=='PASS_ACTUAL_UNINTERRUPTED_VERSUS_RESUMED_NEXT_UPDATE'
        and restored.get('comparison')=='CONTINUOUS4_VS_FRESH_PROCESS_2_PLUS_2'
        and restored.get('next_update_parameters_optimizer_scheduler_RNG_sampler_exact')is True
        and set(restored.get('checks',{}))==CHECKS and all(v is True for v in restored['checks'].values())
        and result.get('proof')=={k:v for k,v in restored.items()if k!='comparison'},
        'ACTUAL_H100_ALL_SEVEN_STATES_EXACT')
    children=result.get('children')or[]
    need([v.get('name')for v in children]==['continuous4','split2','resume4'],
        'ALL_THREE_ACTUAL_H100_RESTORE_CHILDREN')
    for row,(name,endpoint,cursor)in zip(children,[('continuous4',4,128),('split2',2,64),('resume4',4,128)]):
        child=read(name+'/RESULT.json');closed=read(name+'/control/LEASE.json');w=read(name+'/WORKER_EXIT.private.json')
        s=read(name+'/EXECUTION_SPEC.json');p=s.get('recipe')or{}
        contract=read(name+'/PARAMETER_WHITELIST.json')
        need(row.get('exit_code')==closed.get('worker_exit_code')==w.get('exit_code')==0
            and closed.get('status')=='CLOSED'and w.get('failure')is None
            and row.get('result')==child and row.get('result_sha256')==digest(root/name/'RESULT.json')
            and child.get('global_step')==endpoint and child.get('sample_visits')==cursor
            and child.get('real_action_parameter_changed')is True and child.get('backbone_frozen')is True,
            'ACTUAL_H100_FULL_ACTION_RESTORE_CHILD_'+name)
        need(s.get('registered_root')==str(root/name)and s.get('source_sha')==c['source_sha']
            and s.get('source_files')==c['source_files']and s.get('parent')==c['parent']
            and s.get('allocation_gpu_model')=='H100'and p.get('mode')=='RESTORE4'
            and p.get('variant')=='FULL_ACTION_SIDE'and p.get('effective_batch')==32
            and p.get('microbatch')==1 and p.get('endpoint')==endpoint
            and contract.get('trainable_numel')==1304913028
            and contract.get('visual_and_language_backbone_frozen')is True
            and contract.get('external_adapter_trainables')==0,
            'EXACT_H100_ORIGINAL_FULL_ACTION_PARAMETER_SCOPE_'+name)
    need(c['python']==current['python']and c['parent']==current['parent']
        and c['pbs_server']==current['pbs_server']
        and all(c['bindings'][k]==current['bindings'][k]for k in ('source','cosmos','cosmos_hub'))
        and current['recipe']['microbatch']==1,
        'SAME_ACTUALLY_H100_QUALIFIED_PARENT_RUNTIME_AND_MICRO1')
    math=unchanged_training_math(c,current)
    return dict(status='ACTUAL_H100_MICRO1_ORIGINAL_TRAINER_AND_FULL_RESTORE_ADMITTED',
        job_id=lease['job_id'],result_sha256=proof['sha256'],microbatch=1,effective_batch=32,
        training_math=math,qualification_weights_used_as_formal_parent=False)
