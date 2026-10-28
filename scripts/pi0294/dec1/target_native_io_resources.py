"""Pure current-target IO reserve/settle proposals, never a writer or GO.

Existing once IO must commit each exact pair, standard registry must bind the
new child, and the current owner must independently re-read both reservations
before importing/dispatching native components. Old R4 is never reopened.
"""
from copy import deepcopy
import math
import re
import uuid

from . import target_native_io_budget as budget
from .target_paused_stop import FROZEN_CLOCK

common=budget.common
need=common.need
ROLE=budget.ROLE
REQUEST_SCHEMA=budget.REQUEST_SCHEMA
PROJECTION=budget.PROJECTION
FORECAST=budget.FORECAST


def reserve(parent_bytes,Core_bytes,request_bytes,manifest_bytes,projection_bytes,VAP1_bytes,
            *,forecast_inputs,DEC1_bytes,SDK_closures,namespace_proof,now_epoch,predecessor_bytes=None):
    p,c,q,m,x=map(common.obj,(parent_bytes,Core_bytes,request_bytes,manifest_bytes,projection_bytes))
    budget.identity(p,c);prior,spent,held=budget.rows(p,c)
    need(len(prior)<3 and not any(held.values()),'IO_NEW_COMPONENT_ONCE_NOT_CLOSED_REQUEST_REPLAY')
    need(common.sha(VAP1_bytes)==common.VAP1_SHA,'IO_EXACT_ADOPTED_NON_GPU_GRANT')
    need(common.vector(q['requested'])==budget.REQUESTED,'IO_FIXED_ALL_EIGHT_REQUESTED_DIMENSIONS')
    need(q['schema']==REQUEST_SCHEMA and q['scope']==m['scope']==ROLE
         and q['campaign_run_uuid']==budget.CAMPAIGN and q['original_T0_epoch']==budget.T0
         and q['authority_release']==common.AUTHORITY
         and q['funded_work_id']==budget.WORK and q['component_id']==budget.COMPONENT
         and q['purpose']=='CURRENT_NON_OBJECT_BRIDGE_ZERO_INTEGRATION'
         and q['LOCAL_namespace']==budget.LOCAL_NAMESPACE,
         'IO_CURRENT_PURPOSE_NOT_A_TASK_PILOT_OR_OLD_COUNTER')
    need(all(type(q[k]) is int and q[k]==0 for k in
         ('GPU_submissions','training_updates','GR00T_queries','policy_episodes','physics_ticks'))
         and q['sdk_initialization_mode']=='SEALED_CACHE_LOAD' and q['maximum_SDK_requests']==1
         and type(q['maximum_SDK_requests']) is int,'IO_ZERO_PHYSICS_AND_NO_NEW_SDK_CACHE_COMPILATION')
    rid=q['request_id'];run=q['child_run_uuid']
    need(str(uuid.UUID(rid))==rid and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',run)
         and m['run_uuid']==run and m['owner']=='EXISTING_LOCAL_PC','IO_EXACT_NEW_REQUEST_AND_OWNER')
    need(rid not in prior and all(row['child_run_uuid']!=run for row in prior.values()),'IO_NEVER_REPLAY_CLOSED_REQUEST_OR_CHILD')
    certificate=None
    if prior:
        old=budget.predecessor
        if len(prior)==2:
            old=budget.elf;old.adopted(p,c)
            need(set(prior)=={budget.predecessor.REQUEST,old.REQUEST},'IO_EXACT_TWO_CLOSED_PREDECESSORS')
            need(q['LOCAL_minute_grant_sha256']==old.GRANT_SHA
                 and m['authority']['supplemental_grants']==[old.GRANT_SHA]
                 and m['execution']['sealed_inputs'][old.GRANT]==old.GRANT_SHA,'IO_SEALED_ONE_MINUTE_GRANT')
        else:need(set(prior)=={old.REQUEST},'IO_ONE_EXACT_KNOWN_ZERO_PREDECESSOR_REQUIRED')
        need(type(predecessor_bytes) is bytes,'IO_EXACT_PREDECESSOR_BYTES_REQUIRED')
        certificate=old.verify(predecessor_bytes,prior_row=prior[old.REQUEST],prior_settlement=p[budget.SETTLEMENTS][old.REQUEST])
        need(q['closed_predecessor_packet_sha256']==common.sha(predecessor_bytes)
             and m['execution']['sealed_inputs'][old.PACK]==common.sha(predecessor_bytes),
             'IO_NEW_REQUEST_SEALS_ORIGINAL_CLOSED_PREDECESSOR_BYTES')
    else:
        need(predecessor_bytes is None and 'closed_predecessor_packet_sha256' not in q,
             'IO_NO_FAKE_PREDECESSOR_ON_FIRST_COMPONENT')
    need(q['parent_before_sha256']==x['parent_before_sha256']==common.sha(parent_bytes)
         and q['Core_before_sha256']==x['Core_before_sha256']==common.sha(Core_bytes)
         and q['observed_epoch']==x['observed_epoch']==forecast_inputs['snapshot']['observed_epoch']
         and 0<=now_epoch-q['observed_epoch']<120
         and x['schema']=='DEC1_TARGET_NATIVE_IO_COMPLETE_NATIVE_PROJECTION_V1'
         and x['origin']=='EXISTING_LOCAL_PC' and x['unknown_effects']==[],
         'IO_FRESH_ACTUAL_PAIR_AND_COMPLETE_PROJECTION')
    packed=budget.forecast_pack(forecast_inputs,DEC1_bytes=DEC1_bytes,SDK_closures=SDK_closures)
    need(q['full_forecast_inputs_sha256']==x['full_forecast_inputs_sha256']==common.sha(packed),
         'IO_ALL_GRANTS_PROGRESS_PLAN_QUEUE_COST_CONTEXT_AND_SDK_CLOSURES_SEALED')
    need(now_epoch<q['deadline_epoch']<=min(budget.STOP_NEW,now_epoch+1800)
         and m['limits']['operation_deadline_epoch']==q['deadline_epoch']
         and m['limits']['PV1_T0_epoch']==budget.T0 and m['limits']['PV1_hard_close_epoch']==budget.HARD_CLOSE,
         'IO_CURRENT_FINITE_OPERATION_AND_ORIGINAL_CUTOFFS')
    need(m['schema']=='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1'
         and m['authority']['protocol_ref']==common.PARENT_AUTHORITY
         and m['authority']['amendments']==[common.AUTHORITY,common.VAP1_AUTHORITY]
         and m['execution']['profile']=='4080-simulation'
         and m['execution']['sdk_initialization_mode']=='SEALED_CACHE_LOAD'
         and m['execution']['frozen_task_clock_sha256']==FROZEN_CLOCK
         and x['site_permission']=='EXISTING_REGISTERED_CORELAB_4080_SIMULATION'
         and type(x['concurrent_CoreLab_cores']) is int and 1<=x['concurrent_CoreLab_cores']<=2,
         'IO_ACTUAL_EXISTING_SIMULATION_PROFILE_AND_FROZEN_CLOCK')
    need(all(common.vector(x['site_remaining'])[k]>=budget.REQUESTED[k] for k in common.DIMS),
         'IO_ALL_EIGHT_NATIVE_SITE_DIMENSIONS_FUNDED')
    floor=p['final_reserve_inside_total']
    need(floor['H200_s']==72000 and floor['episode_starts']==100
         and floor['live_ticks']==floor['shadow_ticks']==1600000,
         'IO_ORIGINAL_GLOBAL_FINAL_PROTECTION')
    need(re.fullmatch('[0-9a-f]{40}',m['code']['vla_commit']) is not None
         and m['limits']['engine_starts']==budget.REQUESTED['constructors']
         and type(m['limits']['engine_starts']) is int
         and all(type(m['limits'][k]) is int and m['limits'][k]==0 for k in ('physics_ticks','policy_episodes')),
         'IO_PINNED_SOURCE_AND_ZERO_NATIVE_PHYSICS_MANIFEST')
    e=m['execution']['sealed_inputs']
    need(e['REQUEST.json']==common.sha(request_bytes) and e[PROJECTION]==common.sha(projection_bytes)
         and e[FORECAST]==common.sha(packed)
         and x['namespace_proof_sha256']==common.sha(common.encode(namespace_proof)),
         'IO_FINAL_MANIFEST_SEALS_ALL_ADMISSION_INPUTS')
    row=dict(schema=budget.ROW_SCHEMA,scope=ROLE,request_id=rid,child_run_uuid=run,
        funded_work_id=budget.WORK,component_id=budget.COMPONENT,source_sha=m['code']['vla_commit'],
        manifest_sha256=common.sha(manifest_bytes),request_sha256=common.sha(request_bytes),
        projection_sha256=common.sha(projection_bytes),requested=deepcopy(budget.REQUESTED),
        remaining=deepcopy(budget.REQUESTED),status='RESERVED',unknown_effects=[],deadline_epoch=q['deadline_epoch'])
    if certificate is not None:
        import base64
        row.update(predecessor_packet_base64=base64.b64encode(predecessor_bytes).decode(),predecessor_certificate=certificate)
    parent,core=deepcopy(p),deepcopy(c)
    for ledger in (parent,core):
        ledger.setdefault(budget.ROWS,{})[rid]=deepcopy(row)
        ledger[budget.AGGREGATE]=deepcopy(budget.REQUESTED)
    parent['CPU_reserved_s']['CoreLab']+=budget.REQUESTED['CoreLab_CPU_s']
    parent['CPU_reserved_s']['local_PC']+=budget.REQUESTED['local_CPU_s']
    parent['children'].append(dict(id=run,kind=ROLE,request_id=rid,status='RESERVED',
        CPU_reserved_s=budget.REQUESTED['CoreLab_CPU_s'],local_CPU_reserved_s=budget.REQUESTED['local_CPU_s'],
        engine_starts_reserved=budget.REQUESTED['constructors'],episode_starts_reserved=0,physics_ticks_reserved=0,
        unknown_effects=[]))
    args=deepcopy(forecast_inputs);prepared=args.pop('now_epoch',q['observed_epoch'])
    need(prepared==q['observed_epoch'],'IO_SEALED_PREPARATION_CLOCK_NOT_REWRITTEN')
    result=budget.forecast(parent_bytes=common.encode(parent),Core_bytes=common.encode(core),
        DEC1_bytes=DEC1_bytes,SDK_closures=SDK_closures,namespace_proof=namespace_proof,
        now_epoch=now_epoch,**args)
    need(result['admissible_forecast'] is True,'IO_COMPLETE_REMAINING_RESOURCES_AND_CLOCK')
    return dict(status='PURE_NATIVE_IO_DUAL_RESERVATION_PROPOSAL_NOT_NATIVE_ADMISSION',reservation=row,
        proposed_parent=parent,proposed_Core=core,forecast=result,native_dispatch_admitted=False)


def settle(parent_bytes,Core_bytes,request_id,*,actor_proof_bytes,result_bytes,LOCAL_meter_bytes):
    p,c=map(common.obj,(parent_bytes,Core_bytes));rows,spent,held=budget.rows(p,c);row=rows[request_id]
    need(row['status']=='RESERVED' and row['remaining']==row['requested'],'IO_ONCE_ORIGINAL_UNSETTLED_ROW')
    proof=common.obj(actor_proof_bytes)
    need(proof['schema']=='DEC1_TARGET_NATIVE_IO_ACTUAL_TERMINAL_ACTOR_PROOF_V1'
         and proof['request_id']==request_id and proof['manifest_sha256']==row['manifest_sha256']
         and proof['all_registered_actors_terminal'] is True
         and type(proof['current_matching_actors']) is int and proof['current_matching_actors']==0
         and type(proof['pid']) is int and proof['pid']>0 and str(proof['proc_start']).isdigit(),
         'IO_ACTUAL_ALL_WORKER_SIM_SOLVER_RECEIVER_SDK_ACTORS_TERMINAL')
    charged=deepcopy(row['requested']);result_hash=None;verdict='NATIVE_IO_UNKNOWN_RESULT_FULL_UPPER_CHARGED'
    if result_bytes is not None:
        result=common.obj(result_bytes);result_hash=common.sha(result_bytes)
        need(proof['result_present'] is True and proof['result_sha256']==result_hash
             and result['request_id']==request_id and result['manifest_sha256']==row['manifest_sha256']
             and result['source_sha']==row['source_sha'] and result['scope']==ROLE
             and result['target_task_native_qualified'] is False
             and result['native_prefix_safety_qualified'] is False
             and all(type(result[k]) is int and result[k]==0 for k in
                ('physics_ticks','policy_episodes','GR00T_queries','training_updates','GPU_submissions')),
             'IO_KNOWN_ZERO_INTEGRATION_RESULT_NEVER_TARGET_TASK_OR_PREFIX_PASS')
        cpu=result['actual_worker_and_children_CPU_s']
        need(type(cpu) in (int,float) and math.isfinite(cpu) and 0<=cpu<=charged['CoreLab_CPU_s']-600,
             'IO_ALL_COMPILER_NATIVE_GRAPH_SDK_AND_WORKER_CPU_BOUNDED')
        need(type(result['actual_constructor_attempts']) is int
             and 0<=result['actual_constructor_attempts']<=charged['constructors'],
             'IO_ALL_ACTUAL_CONSTRUCTOR_ATTEMPTS_WITHIN_RESERVED_UPPER')
        charged['CoreLab_CPU_s']=cpu+600
        verdict='NATIVE_ZERO_IO_COMPONENT_RESULT_NOT_TASK_QUALIFICATION'
    else:
        need(proof['result_present'] is False and proof['result_sha256'] is None,
             'IO_NO_DROPPED_KNOWN_RESULT')
    if LOCAL_meter_bytes is not None:
        meter=common.obj(LOCAL_meter_bytes);cpu=meter['coordinator_and_children_CPU_s_before_settle']
        need(meter['schema']=='DEC1_TARGET_NATIVE_IO_ACTUAL_LOCAL_COORDINATOR_METER_V1'
             and meter['request_id']==request_id and meter['manifest_sha256']==row['manifest_sha256']
             and meter['all_pre_settlement_control_children_waited'] is True
             and type(cpu) in (int,float) and math.isfinite(cpu) and 0<=cpu<=charged['local_CPU_s']-30,
             'IO_WHOLE_LOCAL_CONTROL_CPU_PLUS_FIXED_CLOSURE_TAIL')
        charged['local_CPU_s']=cpu+30
    record=dict(schema='DEC1_TARGET_NATIVE_IO_CPU_SETTLEMENT_V1',scope=ROLE,request_id=request_id,
        settlement_id=request_id+'-native-io-terminal',manifest_sha256=row['manifest_sha256'],source_sha=row['source_sha'],
        actor_proof_sha256=common.sha(actor_proof_bytes),result_sha256=result_hash,
        LOCAL_meter_sha256=common.sha(LOCAL_meter_bytes) if LOCAL_meter_bytes is not None else None,
        scientific_verdict=verdict,financial_charged=charged,
        financial_constructor_upper_not_claimed_as_actual_models=True,
        target_task_native_qualified=False,native_prefix_safety_qualified=False,core64_cells=0)
    parent,core=deepcopy(p),deepcopy(c)
    for ledger in (parent,core):
        ledger[budget.ROWS][request_id].update(status='SETTLED',remaining=dict.fromkeys(common.DIMS,0),
            settlement_sha256=common.sha(common.encode(record)))
        ledger[budget.AGGREGATE]={k:held[k]-row['requested'][k] for k in common.DIMS}
        ledger.setdefault(budget.SETTLEMENTS,{})[request_id]=deepcopy(record)
    for site,key in (('CoreLab','CoreLab_CPU_s'),('local_PC','local_CPU_s')):
        parent['CPU_reserved_s'][site]-=row['requested'][key];parent['CPU_s'][site]+=charged[key]
    need(min(parent['CPU_reserved_s'].values())>=0,'IO_DO_NOT_RELEASE_OTHER_HOLDS')
    parent['engine_starts']+=charged['constructors'];parent['network_transfer_accounted_bytes']+=charged['transfer_bytes']
    children=[x for x in parent['children'] if x.get('request_id')==request_id and x.get('kind')==ROLE]
    need(len(children)==1 and children[0]['status']=='RESERVED','IO_EXACT_ORIGINAL_REGISTERED_CHILD')
    children[0].update(status='CLOSED_NATIVE_IO_EFFECTS_SETTLED',CPU_reserved_s=0,local_CPU_reserved_s=0,
        episode_starts_reserved=0,physics_ticks_reserved=0,engine_starts_reserved=0)
    core['CPU_s']+=charged['CoreLab_CPU_s'];core['constructors']+=charged['constructors']
    return dict(status='PURE_NATIVE_IO_SETTLEMENT_PROPOSAL_NOT_NATIVE_CLOSURE',settlement=record,
        proposed_parent=parent,proposed_Core=core,native_dispatch_admitted=False,
        actual_once_Core_LOCAL_lease_close_and_independent_readback_still_required=True)
