"""V5 successor's cold whole-package calculation using existing PV1 dimensions.

Pure proposals only. No registration, new cap, submission, model or physics.
Old ledgers and their scopes are inputs, never rewritten by this module.
"""
from .future_budget import _vector, FINAL
from .evaluation_work import MATRIX
from .evaluation_accounting import output_upper
from .evaluation_core_accounting import MODEL_BYTES
from .evaluation_admission import adopted_caps, resource_floor

TASK='V5_CD_AUTONOMOUS_SUCCESSOR_V1'
TASK_SOURCE='5b5de6f5d26ac90a6002c83ebcb60005acf06462'
ORDER=('TRAIN:C50:1024','EVAL:C50:N294101','TRAIN:D25:1024','EVAL:D25:N294101')
WINDOW_S=168*3600
TRAIN_STOP_MARGIN_S=24*3600
GIB=1024**3


def scope(t_cd):
    if type(t_cd) not in (int,float) or t_cd<=0:
        raise PermissionError('V5_ACTUAL_ONCE_REGISTRATION_EPOCH_REQUIRED')
    import math
    if not math.isfinite(t_cd):raise PermissionError('V5_FINITE_REGISTRATION_EPOCH')
    return dict(task=TASK,task_source=TASK_SOURCE,T_CD=t_cd,deadline_epoch=t_cd+WINDOW_S,
        stop_new_training_epoch=t_cd+WINDOW_S-TRAIN_STOP_MARGIN_S,order=list(ORDER),
        engineering_replacements_total=4,engineering_replacements_per_cell=1,
        same_root_cause_native_attempts_max=3,old_clocks_unchanged=True)


def whole_package():
    # Mature 70GB training output bound: full endpoint, one complete recovery
    # save plus bounded trace/temporary/control data. No checkpoint deletion.
    # Two panels have 24 fixed cells each; four bounded replacements are split
    # 2+2 for resource accounting only, never a right to replay valid failures.
    panels=2;conditions=52
    gpu_output=panels*(output_upper(24)+2*output_upper(1)+MODEL_BYTES)
    physical_output=conditions*768*1024**2+panels*512*1024**2
    return _vector(dict(H200_s=2*22560+2*28860,PBS_submissions=2,
        optimizer_updates=2048,episode_starts=conditions,neural_queries=2*29+conditions*80+2*17,
        physics_ticks=conditions*32000,engine_starts=conditions*81+4+30,
        Hopper_CPU_s=2*54000+2*1800+2*80000+2*7200+3000,
        Core_CPU_s=conditions*3600+600+1200,
        local_CPU_s=2*15+2*7200+600+300,
        transfer_bytes=gpu_output+panels*MODEL_BYTES+2*GIB+512*1024**2+64*1024**2,
        Hopper_storage_bytes=2*70000000000+gpu_output+256*1024**2,
        Core_storage_bytes=physical_output+panels*MODEL_BYTES+256*1024**2+64*1024**2,
        local_storage_bytes=256*1024**2))


def recompute(global_ledger,local_ledger,*,Core_storage_upper_bytes,local_storage_upper_bytes):
    used,held=resource_floor(global_ledger,local_ledger);caps=adopted_caps(global_ledger)
    for k,v in dict(Core_storage_bytes=Core_storage_upper_bytes,local_storage_bytes=local_storage_upper_bytes).items():
        if type(v) is not int or v<used[k]:raise PermissionError('V5_RETAIN_FULL_HISTORICAL_STORAGE_UPPER')
        used[k]=v
    if global_ledger['pending_updates'] or global_ledger['unknown_effects']:
        raise PermissionError('V5_NO_NEW_PACKAGE_WHILE_NATIVE_EFFECTS_UNKNOWN')
    future=whole_package();rows={}
    for k in caps:
        total=used[k]+held[k]+future[k]+MATRIX[k]+FINAL.get(k,0)
        rows[k]=dict(cap=caps[k],used=used[k],reserved=held[k],successor=future[k],
            matrix=MATRIX[k],final=FINAL.get(k,0),projected=total,shortfall=max(0,total-caps[k]))
    return dict(schema='V5_CD_EXISTING_CAPS_WHOLE_PACKAGE_CPU_RECOMPUTATION_V1',task=TASK,
        task_source=TASK_SOURCE,order=list(ORDER),dimensions=rows,
        shortfalls={k:v['shortfall'] for k,v in rows.items() if v['shortfall']},
        remaining_package=future,cap_changes={},native_admission=False,new_native_effects=0)
