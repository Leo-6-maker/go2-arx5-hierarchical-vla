"""Pure once-reservation proposal; no IO, registration or native admission.

Owner must first validate the complete source intake, then compare the actual
Core snapshot bytes under its mature once writer and reserve LOCAL PC parent
resources plus Core DEC1 exactly once. This module implements neither write.
A returned proposal is not permission to run a worker or spend resources.

"""
from copy import deepcopy
from pathlib import PurePosixPath
import hashlib
import json
import re
from . import source_cpu_intake as intake

ROLE=intake.ROLE
SCHEMA='DEC1_SOURCE_CPU_COMPONENT_NATIVE_RESERVATION_V1'
DEC1_SNAPSHOT='plans/SOURCE_CORE_DEC1_PRE_RESERVATION_SNAPSHOT.private.json'
PROJECTION='plans/SOURCE_CPU_NATIVE_RESOURCE_PROJECTION.json'


def _need(ok,reason):
    if not ok:raise PermissionError(reason)


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _hash(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _dims(value,name):return intake._mapping(value,name)


def _same(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def _prior_rows(current,request_id,root,campaign,history=None):
    rows=current.get('source_cpu_reservations',{})
    _need(type(rows) is dict and request_id not in rows,'SOURCE_RESERVE_REQUEST_NEVER_REUSED_EVEN_SETTLED')
    total=dict.fromkeys(intake.DIMS,0);slots=0;engineering=0
    for key,item in rows.items():
        _need(type(item) is dict and item.get('schema')==SCHEMA and item.get('role')==ROLE
            and item.get('request_id')==key and item.get('authority_release')==intake.AUTHORITY
            and item.get('campaign_run_uuid')==campaign and item.get('unknown_effects')==[]
            and type(item.get('child_run_uuid')) is str and bool(item['child_run_uuid'])
            and type(item.get('child_root')) is str and item['child_root'].startswith('/')
            and all(_hash(item.get(n)) for n in ('child_manifest_sha256','component_plan_sha256',
                'budget_sha256','parent_projection_sha256','pre_reservation_DEC1_sha256','native_projection_receipt_sha256'))
            and item.get('status') in ('RESERVED','RUNNING','SETTLED'),
            'SOURCE_RESERVE_PRIOR_ROWS_TYPED_KNOWN_NO_UNKNOWN')
        _need(item.get('child_run_uuid')!=root.name and item.get('child_root')!=str(root),
            'SOURCE_RESERVE_CHILD_NEVER_REGISTERED_TWICE')
        original=_dims(item.get('requested'),'old requested');left=_dims(item.get('remaining'),'old remaining')
        _need(all(left[k]<=original[k] for k in intake.DIMS),'SOURCE_RESERVE_PRIOR_REMAINING_NOT_EXPANDED')
        counts=intake.history.row_slots(item,current,history)
        slots+=counts['commissioning_count'];engineering+=counts['engineering_count']
        if item['status']=='SETTLED':_need(not any(left.values()),'SOURCE_RESERVE_SETTLED_REMAINDER_ZERO')
        else:
            for k in intake.DIMS:total[k]+=left[k]
    _need(not rows or 'source_cpu_reserved' in current,'SOURCE_RESERVE_PRIOR_AGGREGATE_REQUIRED')
    aggregate=_dims(current.get('source_cpu_reserved',dict.fromkeys(intake.DIMS,0)),'old aggregate')
    _need(_same(aggregate,total),'SOURCE_RESERVE_CURRENT_AGGREGATE_EXACT')
    _need(engineering<=1,'SOURCE_RESERVE_ONE_SOURCE_ENGINEERING_REPLACEMENT')
    return rows,aggregate,slots


def propose(*,snapshot_bytes,budget_bytes,manifest_bytes,stage_bytes,projection_bytes,
            source_seal_bytes,child_root,now_epoch,amendment_bytes=None,
            history_bytes=None,engineering_bytes=None,history_parent_bytes=None):
    """Return a copied ledger and request row; every input byte remains intact.

    Legacy Core may lack source CPU extension fields only before its FIRST row.
    Complete first classification comes from the separately sealed actual owner
    projection, never extrapolated from old scalar CPU or invented zero costs.
    """
    current=intake._object(snapshot_bytes,'immutable Core DEC1 snapshot')
    budget=intake._object(budget_bytes,'budget');m=intake._object(manifest_bytes,'final child manifest')
    stage=intake._object(stage_bytes,'stage');projection=intake._object(projection_bytes,'native projection')
    seal=intake._object(source_seal_bytes,'source seal')
    now=intake._number(now_epoch,'native now')
    caps=intake.effective_caps(current,amendment_bytes)
    amendments=[intake.AUTHORITY]+([intake.VAP1_AUTHORITY] if amendment_bytes is not None else [])
    _need(current.get('schema')=='DEC1_NONADDITIVE_RESOURCE_CLASSIFICATION_V1'
        and current.get('authority_release')==intake.AUTHORITY
        and current.get('unknown_effects')==[],'SOURCE_RESERVE_EXACT_KNOWN_CORE_SNAPSHOT')
    campaign=current['campaign_run_uuid'];t0=intake._number(current['original_T0_epoch'],'original T0')
    _need(current.get('hard_close_epoch')==intake.HARD_CLOSE and t0+168*3600==intake.HARD_CLOSE,
        'SOURCE_RESERVE_ORIGINAL_CLOCK')
    _need(m.get('schema')=='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1' and m.get('scope')==ROLE
        and m.get('owner')=='EXISTING_LOCAL_PC'
        and m.get('authority',{}).get('protocol_ref')==intake.PARENT_AUTHORITY
        and m['authority'].get('amendments')==amendments,
        'SOURCE_RESERVE_EXPLICIT_SOURCE_CPU_MANIFEST')
    _need(type(child_root) is str and child_root.startswith('/'),'SOURCE_RESERVE_ABSOLUTE_CHILD')
    root=PurePosixPath(child_root)
    _need(str(root)==child_root and '..' not in root.parts and root.parent.name=='children'
        and root.parent.parent.name==campaign and root.name==m.get('run_uuid')
        and root.name not in ('','.', '..') and '\\' not in root.name,
        'SOURCE_RESERVE_EXACT_CHILD_IDENTITY_NO_IO')
    _need(stage.get('schema')==intake.STAGE_SCHEMA and stage.get('role')==ROLE
        and stage.get('authority_release')==intake.AUTHORITY
        and stage.get('campaign_run_uuid')==campaign
        and stage.get('component_plan_owner_release')==intake.PLAN_OWNER_RELEASE
        and stage.get('runtime_source_release')==m['code']['vla_commit'],
        'SOURCE_RESERVE_STAGE_AND_SEPARATE_FIXED_PLAN_OWNER')
    runtime=m['code']['vla_commit']
    _need(type(runtime) is str and re.fullmatch('[0-9a-f]{40}',runtime) is not None
        and seal.get('source_sha')==runtime and _sha(source_seal_bytes)==m['code']['source_seal_sha256']
        and type(seal.get('files')) is dict and bool(seal['files'])
        and all(type(n) is str and n and not PurePosixPath(n).is_absolute()
            and '..' not in PurePosixPath(n).parts and _hash(h) for n,h in seal['files'].items()),
        'SOURCE_RESERVE_FINAL_MANIFEST_SOURCE_SEAL_BINDING')
    sealed=m.get('execution',{}).get('sealed_inputs',{})
    if history_bytes is not None:
        _need(type(history_parent_bytes) is bytes and _sha(history_parent_bytes)==budget['parent_snapshot_sha256'],
            'SOURCE_RESERVE_HISTORY_REQUIRES_EXACT_BOUND_PARENT_BYTES')
        parent=intake._object(history_parent_bytes,'history LOCAL parent')
    else:parent={}
    known=intake.closed_history(m,budget,parent,current,history_bytes=history_bytes,
        engineering_bytes=engineering_bytes,now_epoch=now,source_members=seal.get('files'))
    _need(type(sealed) is dict and ((amendment_bytes is None and intake.VAP1_PLAN not in sealed)
        or (amendment_bytes is not None and sealed.get(intake.VAP1_PLAN)==_sha(amendment_bytes))),
        'SOURCE_RESERVE_EXACT_MANIFEST_ADOPTED_VAP1_SEAL')
    _need(type(sealed) is dict and sealed.get(intake.PATHS['stage'])==_sha(stage_bytes)
        and sealed.get(intake.PATHS['budget'])==_sha(budget_bytes)
        and sealed.get(DEC1_SNAPSHOT)==_sha(snapshot_bytes)
        and sealed.get(PROJECTION)==_sha(projection_bytes)
        and m.get('data',{}).get('manifest_sha256')==intake.PLAN_SHA256
        and sealed.get(intake.PATHS['plan'])==intake.PLAN_SHA256,
        'SOURCE_RESERVE_FINAL_SEALED_REQUEST_SNAPSHOT_AND_PLAN')
    _need(budget.get('schema')==intake.BUDGET_SCHEMA and budget.get('native_complete') is True
        and budget.get('campaign_run_uuid')==campaign and budget.get('unknown_effects')==[]
        and budget.get('DEC1_snapshot_sha256')==_sha(snapshot_bytes)
        and budget.get('native_projection_receipt_sha256')==_sha(projection_bytes)
        and _hash(budget.get('parent_snapshot_sha256')) and _hash(budget.get('adoption_sha256')),
        'SOURCE_RESERVE_COMPLETE_ACTUAL_BUDGET_AND_SNAPSHOT')
    request_id=budget.get('request_id')
    _need(type(request_id) is str and bool(request_id),'SOURCE_RESERVE_ONCE_REQUEST_ID')
    observed=intake._number(budget['observed_epoch'],'observed')
    valid_until=intake._number(budget['valid_until_epoch'],'budget cutoff')
    deadline=intake._number(m['limits']['operation_deadline_epoch'],'child cutoff')
    _need(t0<=observed<=now<deadline<=valid_until<=t0+160*3600,
        'SOURCE_RESERVE_FRESH_NATIVE_INTERVAL_NO_LATE_RESERVATION')
    _need(m['limits']['PV1_T0_epoch']==t0 and m['limits']['PV1_hard_close_epoch']==intake.HARD_CLOSE,
        'SOURCE_RESERVE_MANIFEST_ORIGINAL_CLOCK')
    _need(projection.get('schema')=='DEC1_SOURCE_CPU_NATIVE_RESOURCE_PROJECTION_V1'
        and projection.get('authority_release')==intake.AUTHORITY
        and projection.get('campaign_run_uuid')==campaign
        and projection.get('parent_projection_origin')=='EXISTING_LOCAL_PC'
        and projection.get('unknown_effects')==[] and projection.get('native_safety_unknown_effects')==[],
        'SOURCE_RESERVE_ACTUAL_COMPLETE_OWNER_PROJECTION')
    for key in ('parent_snapshot_sha256','DEC1_snapshot_sha256','adoption_sha256','observed_epoch',
            'valid_until_epoch','DEC1_used','DEC1_reserved','parent_remaining_after_spent_reserved_and_final',
            'site_remaining','source_commissioning_used','source_commissioning_reserved','target_component_reserved'):
        _need(key in projection and key in budget and _same(projection[key],budget[key]),
            'SOURCE_RESERVE_PROJECTION_BUDGET_BINDING:'+key)
    _need(projection.get('continuation')==budget.get('continuation'),
        'SOURCE_RESERVE_CONTINUATION_PROJECTION_BINDING')
    components=budget.get('components')
    _need(type(components) is dict and set(components)==intake.COMPONENTS,
        'SOURCE_RESERVE_COMPLETE_COMPONENTS_NOT_JUST_MAIN')
    for name,row in components.items():_dims(row,'component.'+name)
    requested=_dims(budget.get('requested'),'requested')
    _need(requested=={k:sum(c[k] for c in components.values()) for k in intake.DIMS}
        and requested['episode_starts']==4,'SOURCE_RESERVE_EXACT_FOUR_REQUEST_TOTAL')
    _need(all(type(budget.get(k)) is int and budget[k]==0 for k in
        ('GR00T_queries','GPU_submissions','training_updates')),'SOURCE_RESERVE_ZERO_MODEL_GPU_UPDATES')
    used=_dims(budget.get('DEC1_used'),'used');reserved=_dims(budget.get('DEC1_reserved'),'reserved')
    for key,field in (('CoreLab_CPU_s','CPU_s'),('episode_starts','episode_starts'),
            ('physics_ticks','physics_ticks'),('constructors','constructors')):
        _need(intake._number(current[field],'old.'+field,key!='CoreLab_CPU_s')==used[key],
            'SOURCE_RESERVE_ALL_OLD_NATIVE_COSTS_RETAINED:'+field)
    if 'source_cpu_current_classification' in current:
        _need(_same(_dims(current['source_cpu_current_classification'],'old classification'),used),
            'SOURCE_RESERVE_COMPLETE_CURRENT_CLASSIFICATION_NOT_RESET')
    counts=intake.history.validate_continuation(budget,known)
    rows,aggregate,slots=_prior_rows(current,request_id,root,campaign,known)
    _need(slots+counts['commissioning_count']<=4
        and intake._number(budget['source_commissioning_used'],'old source starts',True)
            +intake._number(budget['source_commissioning_reserved'],'pending source starts',True)+counts['commissioning_count']<=4
        and type(budget.get('target_component_reserved')) is int and budget['target_component_reserved']==4,
        'SOURCE_RESERVE_ORIGINAL_FOUR_SOURCE_FOUR_TARGET')
    parent_left=_dims(budget.get('parent_remaining_after_spent_reserved_and_final'),'parent remaining')
    site_left=_dims(budget.get('site_remaining'),'site remaining')
    _need(all(reserved[k]>=aggregate[k] and used[k]+reserved[k]+requested[k]<=caps[k]
        and requested[k]<=min(parent_left[k],site_left[k]) for k in intake.DIMS),
        'SOURCE_RESERVE_NONADDITIVE_DEC1_PARENT_AND_SITE_CAPS')
    row=dict(schema=SCHEMA,request_id=request_id,authority_release=intake.AUTHORITY,role=ROLE,
        campaign_run_uuid=campaign,child_run_uuid=root.name,child_root=child_root,
        child_manifest_sha256=_sha(manifest_bytes),component_plan_sha256=intake.PLAN_SHA256,
        budget_sha256=_sha(budget_bytes),parent_projection_sha256=budget['parent_snapshot_sha256'],
        pre_reservation_DEC1_sha256=_sha(snapshot_bytes),native_projection_receipt_sha256=_sha(projection_bytes),
        status='RESERVED',unknown_effects=[],reserved_epoch=now,deadline_epoch=deadline,
        requested=deepcopy(requested),remaining=deepcopy(requested))
    if known is not None:row['continuation']=deepcopy(budget['continuation'])
    after=deepcopy(current)
    after.setdefault('source_cpu_reservations',{})[request_id]=row
    after['source_cpu_reserved']={k:aggregate[k]+requested[k] for k in intake.DIMS}
    after['source_cpu_current_classification']=deepcopy(used)
    return dict(status='PROPOSAL_ONLY_NOT_REGISTRATION_OR_AUTHORIZATION',before_snapshot_sha256=_sha(snapshot_bytes),
        child_manifest_sha256=_sha(manifest_bytes),source_seal_sha256=_sha(source_seal_bytes),
        reservation=deepcopy(row),proposed_ledger=after,
        original_parent_global_reservation_and_native_once_write_still_required=True,native_dispatch_admitted=False)


def propose_parent(*,parent_bytes,adoption_bytes,**reservation_inputs):
    """Copy LOCAL parent formal reservations; never debit spent counters or IO.

    Full intake must precede owner IO. Only caller-supplied exact local snapshot,
    complete projection/budget and adopted bytes are used; old output/transfer
    consumption is never inferred as zero. Core and local writes remain pending.
    """
    if reservation_inputs.get('history_bytes') is not None:
        _need(reservation_inputs.get('history_parent_bytes',parent_bytes)==parent_bytes,
            'SOURCE_PARENT_SAME_HISTORY_SNAPSHOT_BYTES')
        reservation_inputs=dict(reservation_inputs,history_parent_bytes=parent_bytes)
    core=propose(**reservation_inputs)
    parent=intake._object(parent_bytes,'immutable LOCAL parent')
    adoption=intake._object(adoption_bytes,'actual DEC1 adoption')
    current=intake._object(reservation_inputs['snapshot_bytes'],'Core snapshot')
    budget=intake._object(reservation_inputs['budget_bytes'],'budget')
    m=intake._object(reservation_inputs['manifest_bytes'],'manifest')
    row=core['reservation'];requested=row['requested'];root=PurePosixPath(row['child_root'])
    campaign=row['campaign_run_uuid'];t0=current['original_T0_epoch']
    sealed=m['execution']['sealed_inputs']
    _need(parent.get('run_uuid')==campaign and parent.get('T0_epoch')==t0
        and parent.get('hard_closeout_epoch')==intake.HARD_CLOSE and parent.get('unknown_effects')==[]
        and _sha(parent_bytes)==budget['parent_snapshot_sha256']
        and sealed.get('plans/SOURCE_PARENT_RESOURCE_SNAPSHOT.private.json')==_sha(parent_bytes),
        'SOURCE_PARENT_EXACT_LOCAL_SNAPSHOT_NO_UNKNOWN')
    _need(_sha(adoption_bytes)==budget['adoption_sha256']
        and sealed.get(intake.PATHS['adoption'])==_sha(adoption_bytes)
        and adoption.get('schema')=='DEC1_NATIVE_ADOPTION_AND_BUDGET_V1'
        and adoption.get('authority_release')==intake.AUTHORITY
        and adoption.get('original_authority_release')==intake.PARENT_AUTHORITY
        and adoption.get('campaign_run_uuid')==campaign and adoption.get('original_T0_epoch')==t0
        and adoption.get('original_hard_close_epoch')==intake.HARD_CLOSE
        and adoption.get('clock_reset') is False and adoption.get('additive_budget') is False,
        'SOURCE_PARENT_EXACT_ACTUAL_ADOPTION_BYTES')
    caps=intake.effective_caps(current,reservation_inputs.get('amendment_bytes'))
    # Reuse original parent spent/pending/final arithmetic, not a new allowance.
    intake.validate_budget(budget,parent,current,adoption,
        now_epoch=reservation_inputs['now_epoch'],caps=caps,history_bytes=reservation_inputs.get('history_bytes'))
    known=intake.history.validate_history(reservation_inputs.get('history_bytes'),parent,current,
        now_epoch=reservation_inputs['now_epoch'])
    counts=intake.history.validate_continuation(budget,known)
    rows,aggregate,slots=_prior_rows(parent,row['request_id'],root,campaign,known)
    _need(slots+counts['commissioning_count']<=4,'SOURCE_PARENT_ORIGINAL_FOUR_SOURCE_SLOTS')
    children=parent['children']
    _need(all(c.get('unknown_effects',[])==[] for c in children),
        'SOURCE_PARENT_CHILD_UNKNOWN_NO_NEW_RESERVATION')
    _need(not any(row['request_id'] in (c.get('source_budget_request_id'),c.get('request_id'))
        or root.name in (c.get('name'),c.get('run_uuid'),c.get('child_run_uuid'))
        or str(root) in (c.get('root'),c.get('path'),c.get('child_root')) for c in children),
        'SOURCE_PARENT_REQUEST_OR_CHILD_NEVER_REUSED')
    _need(aggregate['CoreLab_CPU_s']<=parent['CPU_reserved_s']['CoreLab']
        and aggregate['local_CPU_s']<=parent['CPU_reserved_s']['local_PC']
        and all(aggregate[key]<=sum(c.get(field,0) for c in children) for key,field in (
            ('episode_starts','episode_starts_reserved'),('physics_ticks','physics_ticks_reserved'),
            ('constructors','engine_starts_reserved'))),
        'SOURCE_PARENT_SOURCE_RESERVATIONS_ALREADY_INSIDE_FORMAL_TOTALS')
    after=deepcopy(parent)
    after['CPU_reserved_s']['CoreLab']+=requested['CoreLab_CPU_s']
    after['CPU_reserved_s']['local_PC']+=requested['local_CPU_s']
    child=dict(name=root.name,run_uuid=root.name,child_root=str(root),role=ROLE,
        status='RESERVED_FOR_SOURCE_CPU_REGISTRATION_NOT_STARTED',
        source_budget_request_id=row['request_id'],CPU_reserved_s=requested['CoreLab_CPU_s'],
        local_CPU_reserved_s=requested['local_CPU_s'],episode_starts_reserved=requested['episode_starts'],
        physics_ticks_reserved=requested['physics_ticks'],engine_starts_reserved=requested['constructors'],
        unknown_effects=[])
    after['children'].append(child)
    after.setdefault('source_cpu_reservations',{})[row['request_id']]=deepcopy(row)
    after['source_cpu_reserved']={k:aggregate[k]+requested[k] for k in intake.DIMS}
    return dict(status='PARENT_PROPOSAL_ONLY_NOT_REGISTRATION_OR_AUTHORIZATION',
        before_snapshot_sha256=_sha(parent_bytes),Core_before_snapshot_sha256=core['before_snapshot_sha256'],
        child_manifest_sha256=core['child_manifest_sha256'],reservation=deepcopy(row),
        proposed_ledger=after,Core_proposal=core,
        original_parent_global_reservation_and_native_once_write_still_required=True,native_dispatch_admitted=False)
