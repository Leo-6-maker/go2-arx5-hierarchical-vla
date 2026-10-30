"""One shared night source allowance in the existing owner's run account.

Pure authority, forecast and reservation arithmetic. Native registry/lease,
node-local input, safety and independent scoring admission remain separate.
This module cannot launch a process, download assets or change any ledger.
"""
from copy import deepcopy
from datetime import datetime
import base64,hashlib,json,math

APPROVAL_SHA='d7ba87c7711b484b71907fb22b7c6b334c5d67dceeef369ecb8ba12cdc8364e4'
SCOPE='PINNED_OPEN_SOURCE_NIGHT_20261030_V1'
TASKS=('ROBOCASA_OPEN_CABINET','LIBERO_BOOK_LEFT','LIBERO_BOOK_RIGHT')
STOP_NEW=datetime.fromisoformat('2026-10-30T08:00:00+08:00').timestamp()
DEADLINE=datetime.fromisoformat('2026-10-30T09:00:00+08:00').timestamp()
GiB=1024**3
CAPS=dict(attempts=12,physics_integrations=600000,RGB_pairs=24000,
    Core_CPU_s=41400,PC_CPU_s=1800,render_device_s=7200,
    Core_peak_bytes=20*GiB,PC_evidence_bytes=2*GiB,transfer_bytes=20*GiB)
SUCCESSOR_AUTH='49506e3a135ad6ad931cd672cb9f0b9f6214e47fb072fd160b3f29b5932f007b'
CLOSED_PARENT='fc92d9548b2e5bb134e4b88880df456f1ebc457d071406a71b5bfb05b3f0f468'
CLOSED_SUPPLEMENT='06e3560d5180ff6ce74a1015f40e8a410fa65c4fbb71ab29b43bfdf9d7f302bd'
SUCCESSOR_STOP=datetime.fromisoformat('2026-10-31T08:00:00+08:00').timestamp()
SUCCESSOR_DEADLINE=datetime.fromisoformat('2026-10-31T09:00:00+08:00').timestamp()
# Four retained <=512MiB child roots plus <=128MiB source/assets and
# <=128MiB owner/account metadata. A separate 768MiB margin makes3GiB.
# This is a phase storage forecast, not a reduction of the shared20GiB cap.
FOUR_TASK_CORE_STORAGE=3*GiB


def successor_authority(body):
    need(type(body)is bytes and hashlib.sha256(body).hexdigest()==SUCCESSOR_AUTH,
         'EXACT_SECTION5_DIRECT_APPROVAL')
    value=json.loads(body)
    need(value['direct_answer']=='正式批准第5节全文并执行'
         and value['source_pool']=='carry exact prior costs and remaining allowance; no fresh pool'
         and value['S1_closed_T2_quarantined']is True and value['infrastructure_changes']is False,
         'SECTION5_SAME_SOURCE_POOL_AND_SAFETY')
    return value


def need(ok,why):
    if not ok:raise PermissionError('NIGHT_SOURCE_'+why)


def authority(body,*,now,successor_body=None):
    need(type(body)is bytes and hashlib.sha256(body).hexdigest()==APPROVAL_SHA,'EXACT_DIRECT_HUMAN_AUTHORITY')
    value=json.loads(body)
    need(value['direct_answer']=='正式批准全文并执行'
        and value['aggregate_new_source_caps']==dict(physics_attempts=12,all_physics_integrations=600000,
            RGB_pairs=24000,CPU_core_seconds=43200,active_CPU_cores_per_node=4,
            render_device_occupation_seconds=7200,Core_peak_bytes=20*GiB,
            PC_evidence_bytes=2*GiB,cumulative_transfer_bytes=20*GiB)
        and type(now)in(int,float)and math.isfinite(now)
        and now<(SUCCESSOR_DEADLINE if successor_body is not None else DEADLINE),
        'FINITE_APPROVED_WINDOW_AND_AGGREGATE_CAPS')
    if successor_body is not None:successor_authority(successor_body)
    return value


def add_to_original(original,*,Core_storage_bytes=None):
    """Keep every old obligation inside its old cap before adding this scope."""
    from scripts.pi0294.pv1.future_budget import _vector
    planned_storage=CAPS['Core_peak_bytes']if Core_storage_bytes is None else Core_storage_bytes
    need(planned_storage in(CAPS['Core_peak_bytes'],FOUR_TASK_CORE_STORAGE),'FIXED_DECLARED_SOURCE_STORAGE_PHASE')
    extra=_vector(dict(Core_CPU_s=CAPS['Core_CPU_s'],local_CPU_s=CAPS['PC_CPU_s'],
        episode_starts=12,physics_ticks=600000,engine_starts=12,
        Core_storage_bytes=planned_storage,local_storage_bytes=2*GiB,transfer_bytes=20*GiB))
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


def forecast(approval,global_bytes,local_bytes,*,observation,now,successor_body=None):
    authority(approval,now=now,successor_body=successor_body)
    for key,b in [('GLOBAL',global_bytes),('LOCAL',local_bytes)]:
        v=observation[key]
        need(v['sha256']==hashlib.sha256(b).hexdigest()and v['bytes']==len(b)
            and 0<=now-v['observed_epoch']<=120,'ACTUAL_FRESH_COMPLETE_'+key)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    return add_to_original(first512_forecast(global_bytes,local_bytes),
        Core_storage_bytes=FOUR_TASK_CORE_STORAGE if successor_body is not None else None)


def account_window(v):
    extension=v.get('section5_successor')
    if extension is None:return STOP_NEW,DEADLINE
    need(extension['authority_sha256']==SUCCESSOR_AUTH,'ACCOUNT_SECTION5_AUTHORITY')
    pb=base64.b64decode(extension['closed_parent_base64'],validate=True)
    sb=base64.b64decode(extension['closed_PC_supplement_base64'],validate=True)
    need(hashlib.sha256(pb).hexdigest()==CLOSED_PARENT
         and hashlib.sha256(sb).hexdigest()==CLOSED_SUPPLEMENT,'EXACT_BOTH_CLOSED_PC_COST_RECORDS')
    parent,supp=map(json.loads,(pb,sb));validate_account(parent);validate_account(supp)
    need(parent['native_adopted']is supp['native_adopted']is False
         and supp['sealed_closed_parent_account_sha256']==CLOSED_PARENT
         and supp['operations'][:len(parent['operations'])]==parent['operations']
         and v['operations'][:len(supp['operations'])]==supp['operations']
         and all(v['charged_upper'][k]>=supp['charged_upper'][k]>=parent['charged_upper'][k]for k in CAPS),
         'CLOSED_AGGREGATE_CARRIED_ONCE_NO_REFUND_OR_DOUBLE_POOL')
    return SUCCESSOR_STOP,SUCCESSOR_DEADLINE


def successor_account(parent_bytes,supplement_bytes,authority_bytes,*,destination,now):
    """Pure new account proposal; the two historical accounts stay closed."""
    successor_authority(authority_bytes)
    need(type(destination)is str and destination.endswith('-night-source-four01')
         and type(now)in(int,float)and math.isfinite(now)and now<SUCCESSOR_STOP,
         'ONE_FOUR_CONDITION_SUCCESSOR_BEFORE_STOP')
    old=json.loads(supplement_bytes)
    out={k:deepcopy(old[k])for k in('scope','caps','approval_sha256','charged_upper','operations')}
    out.update(stop_new_epoch=SUCCESSOR_STOP,deadline_epoch=SUCCESSOR_DEADLINE,
        native_adopted=False,revision=old['revision']+1,
        custody='ROOT_SOURCE_SUCCESSOR_PREPARATION_NO_NATIVE_ADOPTION',native_destination=destination,
        section5_successor=dict(authority_sha256=SUCCESSOR_AUTH,
            closed_parent_base64=base64.b64encode(parent_bytes).decode(),
            closed_PC_supplement_base64=base64.b64encode(supplement_bytes).decode()))
    validate_account(out);return out


def for_root(root):
    """Explicit cold-root authority selection; historical roots keep old clocks."""
    from pathlib import Path
    from types import SimpleNamespace
    import sys
    path=Path(root)/'SOURCE_SUCCESSOR_AUTHORITY.json'
    if not path.exists():return sys.modules[__name__]
    body=path.read_bytes();successor_authority(body)
    return SimpleNamespace(SCOPE=SCOPE,CAPS=CAPS,APPROVAL_SHA=APPROVAL_SHA,
        STOP_NEW=SUCCESSOR_STOP,DEADLINE=SUCCESSOR_DEADLINE,
        authority=lambda original,*,now:authority(original,now=now,successor_body=body),
        validate_account=validate_account,reserve=reserve,
        forecast=lambda original,g,l,*,observation,now:forecast(original,g,l,observation=observation,now=now,successor_body=body))


def validate_account(v):
    stop,deadline=account_window(v)
    need(v['scope']==SCOPE and v['caps']==CAPS and v['approval_sha256']==APPROVAL_SHA
        and v['stop_new_epoch']==stop and v['deadline_epoch']==deadline
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
    stop,deadline=account_window(v)
    need(type(now)in(int,float)and math.isfinite(now)and now<deadline,'HARD_FINISH')
    need(type(operation)is str and operation and all(r['operation']!=operation for r in v['operations']),
        'NO_REPLAY_OR_DUPLICATE_OPERATION')
    need(kind in('PREPARATION','ATTEMPT','CLOSEOUT')and set(upper)<=set(CAPS)
        and all(type(x)is int and 0<=x for x in upper.values()),'FINITE_EXPLICIT_EFFECT_UPPERS')
    if kind=='ATTEMPT':
        need(v['native_adopted']is True and now<stop and task in TASKS
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
