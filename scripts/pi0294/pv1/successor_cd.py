"""V5 successor's cold whole-package calculation using existing PV1 dimensions.

Pure proposals only. No registration, new cap, submission, model or physics.
Old ledgers and their scopes are inputs, never rewritten by this module.
"""
from .future_budget import _vector, FINAL
from .evaluation_work import MATRIX
from .evaluation_accounting import output_upper
from .evaluation_core_accounting import MODEL_BYTES
from .evaluation_admission import adopted_caps as legacy_caps, resource_floor

TASK='V5_CD_AUTONOMOUS_SUCCESSOR_V1'
TASK_SOURCE='5b5de6f5d26ac90a6002c83ebcb60005acf06462'
ORDER=('TRAIN:C50:1024','EVAL:C50:N294101','TRAIN:D25:1024','EVAL:D25:N294101')
WINDOW_S=168*3600
TRAIN_STOP_MARGIN_S=24*3600
GIB=1024**3
FIELD='V5_CD_successor_scope'
GRANT_SHA='3bb1d98abcb083e9b1204c06135e52b008f796680c79271878f749a5b2e29c5a'
NEW_STORAGE=dict(Hopper_storage_bytes=1021*GIB,Core_storage_bytes=393*GIB)


def scope(t_cd):
    if type(t_cd) not in (int,float) or t_cd<=0:
        raise PermissionError('V5_ACTUAL_ONCE_REGISTRATION_EPOCH_REQUIRED')
    import math
    if not math.isfinite(t_cd):raise PermissionError('V5_FINITE_REGISTRATION_EPOCH')
    return dict(task=TASK,task_source=TASK_SOURCE,T_CD=t_cd,deadline_epoch=t_cd+WINDOW_S,
        stop_new_training_epoch=t_cd+WINDOW_S-TRAIN_STOP_MARGIN_S,order=list(ORDER),
        engineering_replacements_total=4,engineering_replacements_per_cell=1,
        same_root_cause_native_attempts_max=3,old_clocks_unchanged=True)


def execution_order(value):
    from .mix_vanda_first import validated
    marker=validated(value)
    return tuple(marker['physical_order']) if marker else ORDER


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
    from .mix_vanda_first import budget_LOCAL
    local_ledger=budget_LOCAL(global_ledger,local_ledger)
    used,held=resource_floor(global_ledger,local_ledger);caps=legacy_caps(global_ledger)
    if FIELD in global_ledger:caps.update(validated(global_ledger)['authorized_storage_caps'])
    for k,v in dict(Core_storage_bytes=Core_storage_upper_bytes,local_storage_bytes=local_storage_upper_bytes).items():
        if type(v) is not int or v<used[k]:raise PermissionError('V5_RETAIN_FULL_HISTORICAL_STORAGE_UPPER')
        used[k]=v
    if global_ledger['pending_updates'] or global_ledger['unknown_effects']:
        raise PermissionError('V5_NO_NEW_PACKAGE_WHILE_NATIVE_EFFECTS_UNKNOWN')
    future=whole_package();credit={k:0 for k in future}
    package=global_ledger.get('V5_CD_execution_package')
    if package is not None:
        if (package.get('schema')!='V5_CD_EXISTING_LEDGER_PACKAGE_ESCROW_V1' or package.get('scope_grant_sha256')!=GRANT_SHA):raise PermissionError('V5_REAL_EXISTING_PACKAGE_CREDIT_ONLY')
        future=_vector(package['remaining_resources'])
        for k in ('H200_s','optimizer_updates','neural_queries','Hopper_CPU_s','Hopper_storage_bytes','Core_CPU_s','local_CPU_s'):
            credit[k]=min(held[k],future[k])
    rows={}
    for k in caps:
        total=used[k]+held[k]+future[k]-credit[k]+MATRIX[k]+FINAL.get(k,0)
        rows[k]=dict(cap=caps[k],used=used[k],reserved=held[k],successor=future[k],
            exact_existing_package_credit=credit[k],matrix=MATRIX[k],final=FINAL.get(k,0),projected=total,shortfall=max(0,total-caps[k]))
    return dict(schema='V5_CD_EXISTING_CAPS_WHOLE_PACKAGE_CPU_RECOMPUTATION_V1',task=TASK,
        task_source=TASK_SOURCE,order=list(ORDER),dimensions=rows,
        shortfalls={k:v['shortfall'] for k,v in rows.items() if v['shortfall']},
        remaining_package=future,cap_changes={},native_admission=False,new_native_effects=0)


def grant(data):
    import hashlib,json
    if type(data) is not bytes or hashlib.sha256(data).hexdigest()!=GRANT_SHA:
        raise PermissionError('V5_EXACT_PUBLISHED_DIRECT_HUMAN_GRANT_BYTES')
    v=json.loads(data)
    if (v['direct_human_answer']!='允许启动，' or v['task']!=TASK or v['task_source']!=TASK_SOURCE
        or {k:r['after'] for k,r in v['storage_caps_bytes'].items()}!=NEW_STORAGE
        or v['other_caps_unchanged'] is not True):
        raise PermissionError('V5_ONLY_THE_EXACT_TWO_STORAGE_CORRECTIONS')
    return v


def marker(t_cd,before,grant_data):
    import re
    grant(grant_data)
    if set(before)!={'GLOBAL','LOCAL'} or any(not re.fullmatch('[0-9a-f]{64}',v) for v in before.values()):
        raise PermissionError('V5_ACTUAL_TWO_LEDGER_IDENTITIES')
    return dict(scope(t_cd),grant_sha256=GRANT_SHA,before_sha256=dict(before),
        authorized_storage_caps=dict(NEW_STORAGE),old_clock_and_history_preserved=True,
        scope_only_not_GPU_or_physics_admission=True)


def validated(value):
    import re
    m=value.get(FIELD)
    if type(m) is not dict:raise PermissionError('V5_SCOPE_NOT_ACTUALLY_ADOPTED')
    expected=dict(scope(m.get('T_CD')),grant_sha256=GRANT_SHA,before_sha256=m.get('before_sha256'),
        authorized_storage_caps=dict(NEW_STORAGE),old_clock_and_history_preserved=True,
        scope_only_not_GPU_or_physics_admission=True)
    b=m.get('before_sha256')
    if (m!=expected or type(b)is not dict or set(b)!={'GLOBAL','LOCAL'}
        or any(type(x)is not str or not re.fullmatch('[0-9a-f]{64}',x) for x in b.values())):
        raise PermissionError('V5_EXACT_ONCE_SCOPE_MARKER_AND_CLOCK')
    return m


def adopt(global_raw,local_raw,*,grant_data,now_epoch,Core_storage_upper_bytes,local_storage_upper_bytes):
    """Append the same immutable successor scope to the TWO EXISTING ledgers.

    Caller persists with existing onceIO and the original owner locks. This is
    control-plane adoption only; all actual GPU/model/physics gates still apply.
    """
    import hashlib,json
    from copy import deepcopy
    g,l=json.loads(global_raw),json.loads(local_raw)
    if FIELD in g or FIELD in l:raise FileExistsError('V5_ONCE_SCOPE_NEVER_RESET_OR_REPLAY')
    proof=recompute(g,l,Core_storage_upper_bytes=Core_storage_upper_bytes,local_storage_upper_bytes=local_storage_upper_bytes)
    m=marker(now_epoch,dict(GLOBAL=hashlib.sha256(global_raw).hexdigest(),LOCAL=hashlib.sha256(local_raw).hexdigest()),grant_data)
    proposed_g,proposed_l=deepcopy(g),deepcopy(l)
    proposed_g[FIELD]=deepcopy(m);proposed_l[FIELD]=deepcopy(m)
    # Recompute using the now-authorized caps, but before reserving any job.
    approved=recompute(proposed_g,proposed_l,Core_storage_upper_bytes=Core_storage_upper_bytes,local_storage_upper_bytes=local_storage_upper_bytes)
    if approved['shortfalls']:raise PermissionError('V5_WHOLE_PAIR_STILL_UNPAYABLE:'+str(approved['shortfalls']))
    if any(proposed_g[k]!=v for k,v in g.items()) or any(proposed_l[k]!=v for k,v in l.items()):
        raise AssertionError('V5_ADOPTION_CHANGED_ORIGINAL_LEDGER_MEMBERS')
    return dict(proposed_global=proposed_g,proposed_LOCAL=proposed_l,marker=m,
        before_cap_proof=proof,adopted_cap_proof=approved,native_dispatch_admitted=False)
