"""Pure eight-dimension reserve/settle proposals; no native dispatch or writer."""
from copy import deepcopy
import pytest

from . import target_native_io_resources as unit
from .test_target_native_io_budget import context
from .test_target_sdk_budget import bundle,NOW
from .test_target_sdk_resources import packet
from .test_target_sdk_closed_costs import closed


@pytest.fixture
def proposal(context,monkeypatch):
    v=deepcopy(context);b=unit.budget;pb=unit.common.encode(v['p']);cb=unit.common.encode(v['c'])
    packed=b.forecast_pack(v['inputs'],DEC1_bytes=unit.common.encode(v['d']),SDK_closures=v['closures'])
    x=dict(schema='DEC1_TARGET_NATIVE_IO_COMPLETE_NATIVE_PROJECTION_V1',origin='EXISTING_LOCAL_PC',unknown_effects=[],
        parent_before_sha256=unit.common.sha(pb),Core_before_sha256=unit.common.sha(cb),observed_epoch=NOW,
        full_forecast_inputs_sha256=unit.common.sha(packed),site_permission='EXISTING_REGISTERED_CORELAB_4080_SIMULATION',
        concurrent_CoreLab_cores=2,site_remaining=deepcopy(b.REQUESTED),namespace_proof_sha256=unit.common.sha(unit.common.encode(v['space'])))
    q=dict(schema=unit.REQUEST_SCHEMA,scope=unit.ROLE,campaign_run_uuid=b.CAMPAIGN,original_T0_epoch=b.T0,
        authority_release=unit.common.AUTHORITY,funded_work_id=b.WORK,component_id=b.COMPONENT,
        purpose='CURRENT_NON_OBJECT_BRIDGE_ZERO_INTEGRATION',LOCAL_namespace=b.LOCAL_NAMESPACE,
        sdk_initialization_mode='SEALED_CACHE_LOAD',maximum_SDK_requests=1,
        request_id='54c59aa8-f6b6-4438-800f-ce2e50f01818',child_run_uuid='current-native-zero-IO-CPU-fixture',
        requested=deepcopy(b.REQUESTED),GPU_submissions=0,training_updates=0,GR00T_queries=0,policy_episodes=0,physics_ticks=0,
        parent_before_sha256=unit.common.sha(pb),Core_before_sha256=unit.common.sha(cb),observed_epoch=NOW,
        deadline_epoch=NOW+1800,full_forecast_inputs_sha256=unit.common.sha(packed))
    monkeypatch.setattr(unit.common,'VAP1_SHA',unit.common.sha(v['VAP1']))
    v.update(q=q,x=x);return v


def manifest(v):
    q=v['q'];b=unit.budget;qb=unit.common.encode(q);xb=unit.common.encode(v['x'])
    packed=b.forecast_pack(v['inputs'],DEC1_bytes=unit.common.encode(v['d']),SDK_closures=v['closures'])
    return dict(schema='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1',scope=unit.ROLE,run_uuid=q['child_run_uuid'],owner='EXISTING_LOCAL_PC',
        authority=dict(protocol_ref=unit.common.PARENT_AUTHORITY,amendments=[unit.common.AUTHORITY,unit.common.VAP1_AUTHORITY]),
        code=dict(vla_commit='a'*40),limits=dict(operation_deadline_epoch=q['deadline_epoch'],PV1_T0_epoch=b.T0,
            PV1_hard_close_epoch=b.HARD_CLOSE,engine_starts=8,physics_ticks=0,policy_episodes=0),
        execution=dict(profile='4080-simulation',sdk_initialization_mode='SEALED_CACHE_LOAD',frozen_task_clock_sha256=unit.FROZEN_CLOCK,
            sealed_inputs={'REQUEST.json':unit.common.sha(qb),unit.PROJECTION:unit.common.sha(xb),unit.FORECAST:unit.common.sha(packed)}))


def reserve(v,*,now=NOW,m=None):
    return unit.reserve(unit.common.encode(v['p']),unit.common.encode(v['c']),unit.common.encode(v['q']),
        unit.common.encode(m or manifest(v)),unit.common.encode(v['x']),v['VAP1'],forecast_inputs=v['inputs'],
        DEC1_bytes=unit.common.encode(v['d']),SDK_closures=v['closures'],namespace_proof=v['space'],now_epoch=now)


def terminal(r,*,known=True,meter=True):
    row=r['reservation'];rid=row['request_id']
    result=dict(request_id=rid,manifest_sha256=row['manifest_sha256'],source_sha=row['source_sha'],scope=unit.ROLE,
        target_task_native_qualified=False,native_prefix_safety_qualified=False,
        physics_ticks=0,policy_episodes=0,GR00T_queries=0,training_updates=0,GPU_submissions=0,
        actual_worker_and_children_CPU_s=120.25,actual_constructor_attempts=7)
    rb=unit.common.encode(result) if known else None
    proof=dict(schema='DEC1_TARGET_NATIVE_IO_ACTUAL_TERMINAL_ACTOR_PROOF_V1',request_id=rid,
        manifest_sha256=row['manifest_sha256'],all_registered_actors_terminal=True,current_matching_actors=0,
        pid=999,proc_start='12345',result_present=known,result_sha256=unit.common.sha(rb) if rb else None)
    local=dict(schema='DEC1_TARGET_NATIVE_IO_ACTUAL_LOCAL_COORDINATOR_METER_V1',request_id=rid,
        manifest_sha256=row['manifest_sha256'],all_pre_settlement_control_children_waited=True,
        coordinator_and_children_CPU_s_before_settle=2.5)
    return dict(actor_proof_bytes=unit.common.encode(proof),result_bytes=rb,
        LOCAL_meter_bytes=unit.common.encode(local) if meter else None)


def settle(r,args):
    return unit.settle(unit.common.encode(r['proposed_parent']),unit.common.encode(r['proposed_Core']),r['reservation']['request_id'],**args)


def test_reserve_known_terminal_preserves_all_history_and_releases_only_own_holds(proposal):
    before=deepcopy(proposal);r=reserve(proposal);assert proposal==before
    assert r['forecast']['admissible_forecast'] and r['native_dispatch_admitted'] is False
    assert r['proposed_parent']['CPU_reserved_s']['CoreLab']==3600
    assert r['proposed_parent']['CPU_reserved_s']['local_PC']==60
    final=settle(r,terminal(r));cost=final['settlement']['financial_charged']
    assert cost==dict(unit.budget.REQUESTED,CoreLab_CPU_s=720.25,local_CPU_s=32.5)
    assert final['proposed_parent']['CPU_reserved_s']==dict(CoreLab=0,local_PC=0,Hopper=0)
    for k,v in proposal['p'].items():
        if k not in ('CPU_s','CPU_reserved_s','engine_starts','network_transfer_accounted_bytes','children'):
            assert final['proposed_parent'][k]==v
    unit.budget.rows(final['proposed_parent'],final['proposed_Core'])
    assert final['actual_once_Core_LOCAL_lease_close_and_independent_readback_still_required']
    assert not final['settlement']['target_task_native_qualified'] and not final['settlement']['native_prefix_safety_qualified']
    with pytest.raises(PermissionError):
        unit.settle(unit.common.encode(final['proposed_parent']),unit.common.encode(final['proposed_Core']),r['reservation']['request_id'],**terminal(r))


def test_missing_result_or_meter_charges_full_unknown_upper_after_actual_actor_terminal(proposal):
    r=reserve(proposal);s=settle(r,terminal(r,known=False,meter=False))
    assert s['settlement']['financial_charged']==unit.budget.REQUESTED
    assert s['settlement']['scientific_verdict']=='NATIVE_IO_UNKNOWN_RESULT_FULL_UPPER_CHARGED'


def test_native_later_fresh_recomputation_keeps_sealed_clock_and_request(proposal):
    before=deepcopy(proposal);a=reserve(proposal);b=reserve(proposal,now=NOW+3)
    assert proposal==before and a['reservation']==b['reservation']
    assert b['forecast']['projected_finish_epoch']==a['forecast']['projected_finish_epoch']+3
    with pytest.raises(PermissionError):reserve(proposal,now=NOW+121)


@pytest.mark.parametrize('fault',['old_clock','expired','too_late','physics','queries','new_GPU','old_scope',
    'SDK_recompile','SDK_many_requests','SDK_bool_requests','wrong_namespace','budget_over',
    'wrong_native_before','unknown_projection','other_work','wrong_profile_permission','site_output',
    'namespace_omitted','unsealed_grant','unsealed_full_plan','unsealed_closure'])
def test_request_fails_before_reservation_if_current_complete_scope_missing(proposal,fault):
    v=deepcopy(proposal);q,x=v['q'],v['x']
    if fault=='old_clock':q['original_T0_epoch']+=1
    elif fault=='expired':q['deadline_epoch']=NOW
    elif fault=='too_late':q['deadline_epoch']=NOW+1801
    elif fault=='physics':q['physics_ticks']=1
    elif fault=='queries':q['GR00T_queries']=1
    elif fault=='new_GPU':q['GPU_submissions']=True
    elif fault=='old_scope':q['scope']='OLD_R4_PILOT'
    elif fault=='SDK_recompile':q['sdk_initialization_mode']='ORIGINAL_SDK_INITIALIZE'
    elif fault=='SDK_many_requests':q['maximum_SDK_requests']=2
    elif fault=='SDK_bool_requests':q['maximum_SDK_requests']=True
    elif fault=='wrong_namespace':q['LOCAL_namespace']='dropped-LOCAL-pool'
    elif fault=='budget_over':q['requested']['local_CPU_s']=61
    elif fault=='wrong_native_before':q['Core_before_sha256']='f'*64
    elif fault=='unknown_projection':x['unknown_effects']=['UNKNOWN']
    elif fault=='other_work':q['funded_work_id']='NEW_EXPERIMENT'
    elif fault=='wrong_profile_permission':x['site_permission']='EXPIRED_R4_SIMULATION'
    elif fault=='site_output':x['site_remaining']['CoreLab_output_bytes']=0
    elif fault=='namespace_omitted':v['space']['allocated_bytes_by_namespace'].pop('science-next')
    elif fault=='unsealed_grant':v['VAP1']+=b' '
    elif fault=='unsealed_full_plan':v['inputs']['plan']['queue_delay_upper_s_per_job']+=1
    elif fault=='unsealed_closure':next(iter(v['closures'].values()))['dual_bytes']+=b' '
    with pytest.raises((PermissionError,KeyError)):reserve(v)


@pytest.mark.parametrize('fault',['wrong_profile','wrong_clock_member','bad_source','nonzero_manifest_step','wrong_owner','missing_seal'])
def test_wrong_standard_entry_or_manifest_cannot_dispatch(proposal,fault):
    m=manifest(proposal)
    if fault=='wrong_profile':m['execution']['profile']='4080-cpu-data'
    elif fault=='wrong_clock_member':m['execution']['frozen_task_clock_sha256']='0'*64
    elif fault=='bad_source':m['code']['vla_commit']='main'
    elif fault=='nonzero_manifest_step':m['limits']['physics_ticks']=True
    elif fault=='wrong_owner':m['owner']='OLD_PILOT'
    elif fault=='missing_seal':del m['execution']['sealed_inputs'][unit.FORECAST]
    with pytest.raises((PermissionError,KeyError)):reserve(proposal,m=m)


@pytest.mark.parametrize('fault',['live_actor','bool_actor_count','wrong_request','dropped_result','wrong_source',
    'physics','fake_task_PASS','fake_prefix_PASS','constructor_over','Core_CPU_over','LOCAL_CPU_over','meter_not_waited'])
def test_all_actor_and_actual_cost_evidence_required_before_hold_release(proposal,fault):
    r=reserve(proposal);args=terminal(r)
    proof=unit.common.obj(args['actor_proof_bytes']);res=unit.common.obj(args['result_bytes']);meter=unit.common.obj(args['LOCAL_meter_bytes'])
    if fault=='live_actor':proof['current_matching_actors']=1
    elif fault=='bool_actor_count':proof['current_matching_actors']=False
    elif fault=='wrong_request':proof['request_id']='other'
    elif fault=='dropped_result':args['result_bytes']=None
    elif fault=='wrong_source':res['source_sha']='b'*40
    elif fault=='physics':res['physics_ticks']=1
    elif fault=='fake_task_PASS':res['target_task_native_qualified']=True
    elif fault=='fake_prefix_PASS':res['native_prefix_safety_qualified']=True
    elif fault=='constructor_over':res['actual_constructor_attempts']=9
    elif fault=='Core_CPU_over':res['actual_worker_and_children_CPU_s']=3001
    elif fault=='LOCAL_CPU_over':meter['coordinator_and_children_CPU_s_before_settle']=31
    elif fault=='meter_not_waited':meter['all_pre_settlement_control_children_waited']=False
    if args['result_bytes'] is not None:
        args['result_bytes']=unit.common.encode(res);proof['result_sha256']=unit.common.sha(args['result_bytes'])
    args['actor_proof_bytes']=unit.common.encode(proof);args['LOCAL_meter_bytes']=unit.common.encode(meter)
    with pytest.raises(PermissionError):settle(r,args)
