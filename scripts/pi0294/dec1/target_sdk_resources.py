"""Typed once-resource proposals for current zero-physics SDK qualification.

Pure: no registry/lease writer, build, subprocess, model or physics. The single
existing owner applies BOTH proposals with source_cpu_once_io, independently
reads them back, registers the standard child and only then dispatches it.
"""
from copy import deepcopy
import base64
import math
import re
import uuid
from . import target_sdk_budget as budget
from . import target_zero_counter_contract as common
from .target_sdk_client import SDK_ROLE, CAMPAIGN, T0, STOP_NEW, HARD_CLOSE

ROLE=SDK_ROLE
REQUEST_SCHEMA='DEC1_TARGET_SDK_CPU_REQUEST_V1'
PROJECTION='plans/SDK_COMPLETE_RESOURCE_PROJECTION.private.json'
FORECAST='plans/SDK_FULL_REMAINING_RESOURCE_INPUTS.private.json'
need=budget.need


def forecast_pack(inputs):
    """Seal every original forecast input, including each exact grant byte."""
    allowed={'snapshot','plan','now_epoch','progress_bytes','amendment_bytes','recovery_bytes',
        'segment_bytes','storage_bytes','heartbeat_bytes','zero_update_bytes','b_recovery_bytes',
        'retry_bytes','vanda2_bytes','precuda_bytes'}
    need(set(inputs)<=allowed and {'snapshot','plan','progress_bytes'}<=set(inputs),
         'SDK_COMPLETE_ORIGINAL_FORECAST_INPUT_NAMES')
    values={}
    for key,value in inputs.items():
        if key.endswith('_bytes'):
            need(type(value) is bytes or value is None,'SDK_EXACT_FORECAST_GRANT_BYTES')
            values[key]=None if value is None else base64.b64encode(value).decode('ascii')
        else:values[key]=deepcopy(value)
    return common.encode(dict(schema='DEC1_SDK_FULL_REMAINING_FORECAST_INPUTS_V1',inputs=values))


def unpack_forecast(raw):
    value=common.obj(raw)
    need(value['schema']=='DEC1_SDK_FULL_REMAINING_FORECAST_INPUTS_V1','SDK_FULL_FORECAST_PACK_SCHEMA')
    out={k:(base64.b64decode(v,validate=True) if k.endswith('_bytes') and v is not None else v)
         for k,v in value['inputs'].items()}
    need(common.same(common.obj(forecast_pack(out)),value),'SDK_EXACT_COMPLETE_FORECAST_PACK')
    return out


def _identity(p,c):
    need(p['run_uuid']==c['campaign_run_uuid']==CAMPAIGN
         and p['T0_epoch']==c['original_T0_epoch']==T0
         and p['hard_closeout_epoch']==c['hard_close_epoch']==HARD_CLOSE
         and c['authority_release']==common.AUTHORITY
         and c['VAP1_authority_release']==common.VAP1_AUTHORITY
         and p['unknown_effects']==p.get('Hopper_unknown_effects',[])==p['pending_updates']==c['unknown_effects']==[],
         'SDK_CURRENT_ORIGINAL_AUTHORITY_CLOCK_AND_KNOWN_WRITES')


def reserve(parent_bytes,Core_bytes,request_bytes,manifest_bytes,projection_bytes,VAP1_bytes,
            *,forecast_inputs,namespace_proof,now_epoch):
    p,c,q,m,x=map(common.obj,(parent_bytes,Core_bytes,request_bytes,manifest_bytes,projection_bytes))
    _identity(p,c)
    need(common.sha(VAP1_bytes)==common.VAP1_SHA,'SDK_EXACT_ADOPTED_NON_GPU_GRANT')
    prior,spent,held=budget.rows(p,c)
    need(not any(held.values()),'SDK_NO_SECOND_CURRENT_NATIVE_CPU_OWNER')
    requested=common.vector(q['requested'])
    for k in common.DIMS:
        need(requested[k]<=budget.REQUESTED[k],'SDK_REQUEST_WITHIN_PREDECLARED_COMPONENT_UPPER')
    need(600<requested['CoreLab_CPU_s'] and 30<requested['local_CPU_s']
         and requested['episode_starts']==requested['physics_ticks']==0
         and requested['constructors']==2
         and requested['CoreLab_output_bytes']==budget.REQUESTED['CoreLab_output_bytes']
         and requested['local_output_bytes']==budget.REQUESTED['local_output_bytes']
         and requested['transfer_bytes']==budget.REQUESTED['transfer_bytes'], 'SDK_BOUNDED_FINITE_BUILD_AND_MODEL_ONLY')
    need(q['schema']==REQUEST_SCHEMA and q['scope']==m['scope']==ROLE
         and q['campaign_run_uuid']==CAMPAIGN and q['original_T0_epoch']==T0
         and q['authority_release']==common.AUTHORITY
         and q['sdk_initialization_mode']==m['execution']['sdk_initialization_mode']=='ORIGINAL_SDK_INITIALIZE'
         and q['funded_work_id']==budget.WORK and q['component_id']==budget.COMPONENT,
         'SDK_EXPLICIT_CURRENT_ROLE_AND_SAME_MATRIX_WORK')
    need(all(type(q[k]) is int and q[k]==0 for k in
         ('GPU_submissions','training_updates','GR00T_queries','policy_episodes','physics_ticks')),
         'SDK_NO_GPU_NEURAL_TASK_OR_PHYSICS_PERMISSION')
    rid=q['request_id'];run=q['child_run_uuid']
    need(str(uuid.UUID(rid))==rid and rid not in prior and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',run)
         and m['run_uuid']==run,'SDK_NEW_ONCE_REQUEST_AND_CHILD')
    need(q['parent_before_sha256']==x['parent_before_sha256']==common.sha(parent_bytes)
         and q['Core_before_sha256']==x['Core_before_sha256']==common.sha(Core_bytes)
         and x['schema']=='DEC1_TARGET_SDK_COMPLETE_NATIVE_PROJECTION_V1'
         and x['observed_epoch']==q['observed_epoch']==forecast_inputs['snapshot']['observed_epoch']
         and 0<=now_epoch-q['observed_epoch']<120
         and x['origin']=='EXISTING_LOCAL_PC' and x['unknown_effects']==[],
         'SDK_FRESH_EXACT_NATIVE_PAIR_AND_SNAPSHOT')
    full_inputs=forecast_pack(forecast_inputs)
    need(q['full_forecast_inputs_sha256']==x['full_forecast_inputs_sha256']==common.sha(full_inputs)
         and m['execution']['sealed_inputs'][FORECAST]==common.sha(full_inputs),
         'SDK_EVERY_ORIGINAL_GRANT_PROGRESS_PLAN_AND_SNAPSHOT_SEALED')
    need(now_epoch<q['deadline_epoch']<=min(STOP_NEW,now_epoch+1800)
         and m['limits']['operation_deadline_epoch']==q['deadline_epoch']
         and m['limits']['PV1_T0_epoch']==T0 and m['limits']['PV1_hard_close_epoch']==HARD_CLOSE,
         'SDK_ORIGINAL_CLOCK_AND_FINITE_CURRENT_OPERATION')
    need(m['schema']=='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1'
         and m['authority']['protocol_ref']==common.PARENT_AUTHORITY
         and m['authority']['amendments']==[common.AUTHORITY,common.VAP1_AUTHORITY]
         and m['execution']['profile']=='4080-cpu-data'
         and x['site_permission']=='EXISTING_REGISTERED_CORELAB_4080_CPU_DATA'
         and type(x['concurrent_CoreLab_cores']) is int and 1<=x['concurrent_CoreLab_cores']<=2,
         'SDK_EXISTING_STANDARD_CPU_PROFILE_NOT_EXPIRED_PILOT')
    need(all(common.vector(x['site_remaining'])[k]>=requested[k] for k in common.DIMS),
         'SDK_ALL_EIGHT_NATIVE_SITE_DIMENSIONS')
    floor=p['final_reserve_inside_total']
    need(floor['H200_s']==72000 and floor['episode_starts']==100
         and floor['live_ticks']==floor['shadow_ticks']==1600000,
         'SDK_ORIGINAL_PARENT_FINAL_PROTECTION_UNCHANGED')
    need(re.fullmatch('[0-9a-f]{40}',m['code']['vla_commit']) is not None
         and m['limits']['engine_starts']==requested['constructors']
         and m['limits']['physics_ticks']==m['limits']['policy_episodes']==0
         and m['execution']['sealed_inputs']['REQUEST.json']==common.sha(request_bytes)
         and m['execution']['sealed_inputs'][PROJECTION]==common.sha(projection_bytes),
         'SDK_FINAL_MANIFEST_SEALS_ZERO_PHYSICS_REQUEST_AND_FULL_PROJECTION')
    need(common.sha(common.encode(namespace_proof))==x['namespace_proof_sha256']
         and q['LOCAL_namespace']==namespace_proof['SDK_namespace'],'SDK_SEALED_ALL_NAMESPACE_PROOF')
    # Local source/CPU costs use original actual parent counters. No current
    # GPU, paid failed update, old UNKNOWN source or fenced Hopper record moves.
    row=dict(schema=budget.ROW_SCHEMA,scope=ROLE,request_id=rid,child_run_uuid=run,
        funded_work_id=budget.WORK,component_id=budget.COMPONENT,source_sha=m['code']['vla_commit'],
        manifest_sha256=common.sha(manifest_bytes),request_sha256=common.sha(request_bytes),
        projection_sha256=common.sha(projection_bytes),requested=deepcopy(requested),
        remaining=deepcopy(requested),status='RESERVED',unknown_effects=[],deadline_epoch=q['deadline_epoch'])
    outputs=[]
    for ledger in (p,c):
        out=deepcopy(ledger);out.setdefault(budget.ROWS,{})[rid]=deepcopy(row)
        out['target_sdk_client_reserved']=deepcopy(requested);outputs.append(out)
    parent,core=outputs
    parent['CPU_reserved_s']['CoreLab']+=requested['CoreLab_CPU_s']
    parent['CPU_reserved_s']['local_PC']+=requested['local_CPU_s']
    parent['children'].append(dict(id=run,kind=ROLE,request_id=rid,status='RESERVED',
        CPU_reserved_s=requested['CoreLab_CPU_s'],local_CPU_reserved_s=requested['local_CPU_s'],
        engine_starts_reserved=2,episode_starts_reserved=0,physics_ticks_reserved=0,unknown_effects=[]))
    args=deepcopy(forecast_inputs)
    prepared_at=args.pop('now_epoch',q['observed_epoch'])
    need(prepared_at==q['observed_epoch'] and 0<=now_epoch-prepared_at<120,
         'SDK_PREPARED_INPUT_CLOCK_AND_FRESH_ACTUAL_NATIVE_RECOMPUTATION')
    result=budget.forecast(parent_bytes=common.encode(parent),Core_bytes=common.encode(core),
        namespace_proof=namespace_proof,now_epoch=now_epoch,**args)
    need(result['admissible_forecast'] is True,'SDK_ALL_REMAINING_DIMENSIONS_AND_ORIGINAL_CLOCK')
    return dict(status='PURE_SDK_DUAL_RESERVATION_PROPOSAL_NOT_NATIVE_ADMISSION',reservation=row,
        proposed_parent=parent,proposed_Core=core,forecast=result,
        old_GPU_source_CPU_source_counter_and_historical_costs_preserved=True,native_dispatch_admitted=False)


def settle(parent_bytes,Core_bytes,request_id,*,actor_proof_bytes,result_bytes,LOCAL_meter_bytes):
    """Only actual terminal actor proof permits removing eight holds.

    Missing/unknown worker or LOCAL cost data keeps that full requested upper.
    Known LOCAL measurements include all coordinator children plus30s tail;
    this does not refund any old prepayment or original controller envelope.
    """
    p,c=map(common.obj,(parent_bytes,Core_bytes));_identity(p,c)
    all_rows,spent,total=budget.rows(p,c);row=all_rows[request_id]
    need(row['status']=='RESERVED' and row['remaining']==row['requested'],'SDK_ONE_UNSETTLED_ORIGINAL_ROW')
    proof=common.obj(actor_proof_bytes)
    need(proof['schema']=='DEC1_TARGET_SDK_ACTUAL_TERMINAL_ACTOR_PROOF_V1'
         and proof['request_id']==request_id and proof['manifest_sha256']==row['manifest_sha256']
         and proof['all_registered_actors_terminal'] is True and proof['current_matching_actors']==0
         and type(proof['pid']) is int and proof['pid']>0 and str(proof['proc_start']).isdigit(),
         'SDK_ACTUAL_ACTOR_PID_START_EXIT_AND_NO_LIVE_BUILD_OR_CLIENT')
    amount=deepcopy(row['requested']);status='SDK_ENGINEERING_UNKNOWN_FULL_UPPER_CHARGED';result_hash=None
    if result_bytes is not None:
        result=common.obj(result_bytes);result_hash=common.sha(result_bytes)
        need(proof['result_present'] is True and proof['result_sha256']==result_hash
             and result['request_id']==request_id and result['manifest_sha256']==row['manifest_sha256']
             and result['source_sha']==row['source_sha']
             and result['scope']==ROLE and result['target_task_native_qualified'] is False
             and all(type(result[k]) is int and result[k]==0 for k in
                 ('physics_ticks','policy_episodes','GR00T_queries','MuJoCo_engine_constructors')),
             'SDK_KNOWN_CURRENT_ZERO_PHYSICS_RESULT_NOT_TARGET_TASK_PASS')
        cpu=result['actual_worker_and_children_CPU_s']
        need(type(cpu) in (int,float) and math.isfinite(cpu) and 0<=cpu<=row['requested']['CoreLab_CPU_s']-600,
             'SDK_ALL_CORE_COMPILER_CLIENT_AND_WORKER_CPU_BOUNDED')
        amount['CoreLab_CPU_s']=600+cpu
        need(type(result['SDK_interface_constructor_attempts']) is int
             and 0<=result['SDK_interface_constructor_attempts']<=1,'SDK_SINGLE_INTERFACE_CONSTRUCTION')
        status='SDK_NATIVE_COMPONENT_RESULT_NOT_TASK_QUALIFICATION'
    else:
        need(proof.get('result_present') is False and proof.get('result_sha256') is None,
             'SDK_NO_DROPPED_KNOWN_RESULT')
    if LOCAL_meter_bytes is not None:
        meter=common.obj(LOCAL_meter_bytes);cpu=meter['coordinator_and_children_CPU_s_before_settle']
        need(meter['schema']=='DEC1_TARGET_SDK_ACTUAL_LOCAL_COORDINATOR_METER_V1'
             and meter['request_id']==request_id and meter['manifest_sha256']==row['manifest_sha256']
             and meter['all_pre_settlement_control_children_waited'] is True
             and type(cpu) in (int,float) and math.isfinite(cpu) and 0<=cpu<=row['requested']['local_CPU_s']-30,
             'SDK_KNOWN_WHOLE_LOCAL_METER_PLUS_CLOSURE_TAIL')
        amount['local_CPU_s']=cpu+30
    record=dict(schema='DEC1_TARGET_SDK_CPU_SETTLEMENT_V1',scope=ROLE,request_id=request_id,
        settlement_id=request_id+'-sdk-terminal',manifest_sha256=row['manifest_sha256'],source_sha=row['source_sha'],
        actor_proof_sha256=common.sha(actor_proof_bytes),result_sha256=result_hash,
        LOCAL_meter_sha256=common.sha(LOCAL_meter_bytes) if LOCAL_meter_bytes is not None else None,
        scientific_verdict=status,financial_charged=amount,
        financial_constructor_upper_not_claimed_as_actual_MuJoCo_models=True,
        target_task_native_qualified=False,core64_cells=0)
    outputs=[]
    for ledger in (p,c):
        out=deepcopy(ledger);out[budget.ROWS][request_id].update(status='SETTLED',
            remaining=dict.fromkeys(common.DIMS,0),settlement_sha256=common.sha(common.encode(record)))
        out['target_sdk_client_reserved']={k:total[k]-row['requested'][k] for k in common.DIMS}
        out.setdefault(budget.SETTLEMENTS,{})[request_id]=deepcopy(record);outputs.append(out)
    parent,core=outputs
    for site,key in (('CoreLab','CoreLab_CPU_s'),('local_PC','local_CPU_s')):
        parent['CPU_reserved_s'][site]-=row['requested'][key];parent['CPU_s'][site]+=amount[key]
    need(min(parent['CPU_reserved_s'].values())>=0,'SDK_NO_RELEASE_OF_OTHER_HOLDS')
    parent['engine_starts']+=amount['constructors'];parent['network_transfer_accounted_bytes']+=amount['transfer_bytes']
    children=[x for x in parent['children'] if x.get('request_id')==request_id and x.get('kind')==ROLE]
    need(len(children)==1 and children[0]['status']=='RESERVED','SDK_EXACT_ORIGINAL_PARENT_CHILD')
    children[0].update(status='CLOSED_SDK_EFFECTS_SETTLED',CPU_reserved_s=0,local_CPU_reserved_s=0,
        episode_starts_reserved=0,physics_ticks_reserved=0,engine_starts_reserved=0)
    core['CPU_s']+=amount['CoreLab_CPU_s'];core['constructors']+=amount['constructors']
    return dict(status='PURE_SDK_TERMINAL_SETTLEMENT_PROPOSAL_NOT_NATIVE_CLOSURE',
        settlement=record,proposed_parent=parent,proposed_Core=core,
        actual_once_Core_LOCAL_lease_close_and_independent_readback_still_required=True,
        native_dispatch_admitted=False)
