"""One shared night source allowance in the existing owner's run account.

Pure authority, forecast and reservation arithmetic. Native registry/lease,
node-local input, safety and independent scoring admission remain separate.
This module cannot launch a process, download assets or change any ledger.
"""
from copy import deepcopy
from datetime import datetime
import hashlib,json,math

APPROVAL_SHA='d7ba87c7711b484b71907fb22b7c6b334c5d67dceeef369ecb8ba12cdc8364e4'
SCOPE='PINNED_OPEN_SOURCE_NIGHT_20261030_V1'
TASKS=('ROBOCASA_OPEN_CABINET','LIBERO_BOOK_LEFT','LIBERO_BOOK_RIGHT')
STOP_NEW=datetime.fromisoformat('2026-10-30T08:00:00+08:00').timestamp()
DEADLINE=datetime.fromisoformat('2026-10-30T09:00:00+08:00').timestamp()
GiB=1024**3
CAPS=dict(attempts=12,physics_integrations=600000,RGB_pairs=24000,
    Core_CPU_s=41400,PC_CPU_s=1800,render_device_s=7200,
    Core_peak_bytes=20*GiB,PC_evidence_bytes=2*GiB,transfer_bytes=20*GiB)


def need(ok,why):
    if not ok:raise PermissionError('NIGHT_SOURCE_'+why)


def authority(body,*,now):
    need(type(body)is bytes and hashlib.sha256(body).hexdigest()==APPROVAL_SHA,'EXACT_DIRECT_HUMAN_AUTHORITY')
    value=json.loads(body)
    need(value['direct_answer']=='正式批准全文并执行'
        and value['aggregate_new_source_caps']==dict(physics_attempts=12,all_physics_integrations=600000,
            RGB_pairs=24000,CPU_core_seconds=43200,active_CPU_cores_per_node=4,
            render_device_occupation_seconds=7200,Core_peak_bytes=20*GiB,
            PC_evidence_bytes=2*GiB,cumulative_transfer_bytes=20*GiB)
        and type(now)in(int,float)and math.isfinite(now)and now<DEADLINE,
        'FINITE_APPROVED_WINDOW_AND_AGGREGATE_CAPS')
    return value


def add_to_original(original):
    """Keep every old obligation inside its old cap before adding this scope."""
    from scripts.pi0294.pv1.future_budget import _vector
    extra=_vector(dict(Core_CPU_s=CAPS['Core_CPU_s'],local_CPU_s=CAPS['PC_CPU_s'],
        episode_starts=12,physics_ticks=600000,engine_starts=12,
        Core_storage_bytes=20*GiB,local_storage_bytes=2*GiB,transfer_bytes=20*GiB))
    need(set(extra)==set(original['dimensions']),'COMPLETE_ORIGINAL_FOURTEEN_DIMENSIONS')
    need(original['cap_changes_required']=={},'NEW_ALLOWANCE_CANNOT_FILL_OLD_SHORTFALL')
    rows={}
    for key,amount in extra.items():
        before=original['dimensions'][key];cap=before['cap']
        need(cap is None and key=='PBS_submissions'and before.get('count_still_accounted')is True
            or type(cap)in(int,float)and before['projected']<=cap,'ALL_OLD_WORK_WITHIN_OLD_CAP')
        rows[key]=dict(original_cap=cap,original_all_work_projected=before['projected'],
            new_named_source_allowance=amount,combined_projected=before['projected']+amount,
            combined_named_ceiling=None if cap is None else cap+amount)
    return dict(schema='NIGHT_SOURCE_ORIGINAL_ALL14_PLUS_DISTINCT_NAMED_ALLOWANCE_V1',
        original_forecast=deepcopy(original),dimensions=rows,source_caps=dict(CAPS),
        render_device_s_separate_from_training_GPU=7200,
        original_shared_caps_changed=False,old_work_may_spend_new_allowance=False,
        new_GR00T_training_PBS_neural_queries=0,native_adopted=False,launch_allowed=False)


def forecast(approval,global_bytes,local_bytes,*,observation,now):
    authority(approval,now=now)
    for key,b in [('GLOBAL',global_bytes),('LOCAL',local_bytes)]:
        v=observation[key]
        need(v['sha256']==hashlib.sha256(b).hexdigest()and v['bytes']==len(b)
            and 0<=now-v['observed_epoch']<=120,'ACTUAL_FRESH_COMPLETE_'+key)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    return add_to_original(first512_forecast(global_bytes,local_bytes))


def validate_account(v):
    need(v['scope']==SCOPE and v['caps']==CAPS and v['approval_sha256']==APPROVAL_SHA
        and v['stop_new_epoch']==STOP_NEW and v['deadline_epoch']==DEADLINE
        and type(v['operations'])is list and type(v['native_adopted'])is bool,
        'ONE_SHARED_ACCOUNT_NOT_PER_TASK_REPLICATED')
    need(set(v['charged_upper'])==set(CAPS)
        and all(type(x)in(int,float)and math.isfinite(x)and 0<=x<=CAPS[k]
            for k,x in v['charged_upper'].items())
        and v['charged_upper']['Core_CPU_s']+v['charged_upper']['PC_CPU_s']<=43200,
        'AGGREGATE_WHOLE_NIGHT_RESOURCE_BOUNDS')
    need(type(v['revision'])is int and v['revision']>=0,'NONNEGATIVE_ACCOUNT_REVISION')
    return v


def reserve(v,*,operation,kind,upper,now,task=None,qualification=None):
    """Reserve full upper bounds before effects; UNKNOWN reservations stay paid."""
    validate_account(v)
    need(type(now)in(int,float)and math.isfinite(now)and now<DEADLINE,'HARD_FINISH')
    need(type(operation)is str and operation and all(r['operation']!=operation for r in v['operations']),
        'NO_REPLAY_OR_DUPLICATE_OPERATION')
    need(kind in('PREPARATION','ATTEMPT','CLOSEOUT')and set(upper)<=set(CAPS)
        and all(type(x)is int and 0<=x for x in upper.values()),'FINITE_EXPLICIT_EFFECT_UPPERS')
    if kind=='ATTEMPT':
        need(v['native_adopted']is True and now<STOP_NEW and task in TASKS
            and qualification==v.get('qualification_sha256')and type(qualification)is str
            and len(qualification)==64,'ACTUAL_ADOPTED_SAFE_FIXED_TASK_BEFORE08')
        need(upper.get('attempts')==1 and 0<upper.get('physics_integrations',0)<=50000
            and 0<upper.get('RGB_pairs',0)<=2000 and 0<upper.get('render_device_s',0)<=600,
            'WARMUP_RESET_SHADOW_FAIL_QUALIFICATION_ALL_IN_ONE_ATTEMPT')
    else:
        need(all(upper.get(k,0)==0 for k in('attempts','physics_integrations','RGB_pairs','render_device_s')),
            'PREPARATION_CANNOT_STEP_OR_RENDER')
    out=deepcopy(v)
    for key,amount in upper.items():out['charged_upper'][key]+=amount
    out['revision']+=1;out['operations'].append(dict(operation=operation,kind=kind,task=task,
        upper=dict(upper),status='RESERVED_UNKNOWN_UNTIL_EXACT_NATIVE_RESULT',started_epoch=now))
    validate_account(out);return out
