"""Original full arithmetic and paired proposals; never native admission."""
from copy import deepcopy
import pytest
from . import target_sdk_resources as unit
from .test_target_sdk_budget import bundle, NOW


@pytest.fixture
def packet(bundle,monkeypatch):
    s,progress,plan,p,c,space,a=deepcopy(bundle)
    p['children']=[]
    p['final_reserve_inside_total']=dict(H200_s=72000,episode_starts=100,live_ticks=1600000,shadow_ticks=1600000)
    monkeypatch.setattr(unit.common,'VAP1_SHA',unit.common.sha(a))
    args=dict(snapshot=s,progress_bytes=unit.common.encode(progress),plan=plan,amendment_bytes=a,now_epoch=NOW)
    digest=unit.common.sha(unit.forecast_pack(args));pb=unit.common.encode(p);cb=unit.common.encode(c)
    x=dict(schema='DEC1_TARGET_SDK_COMPLETE_NATIVE_PROJECTION_V1',origin='EXISTING_LOCAL_PC',unknown_effects=[],
        parent_before_sha256=unit.common.sha(pb),Core_before_sha256=unit.common.sha(cb),
        observed_epoch=NOW,full_forecast_inputs_sha256=digest,site_permission='EXISTING_REGISTERED_CORELAB_4080_CPU_DATA',
        concurrent_CoreLab_cores=2,site_remaining=deepcopy(unit.budget.REQUESTED),
        namespace_proof_sha256=unit.common.sha(unit.common.encode(space)))
    q=dict(schema=unit.REQUEST_SCHEMA,scope=unit.ROLE,campaign_run_uuid=unit.CAMPAIGN,
        original_T0_epoch=unit.T0,authority_release=unit.common.AUTHORITY,
        sdk_initialization_mode='ORIGINAL_SDK_INITIALIZE',funded_work_id=unit.budget.WORK,
        component_id=unit.budget.COMPONENT,request_id='f98bd87f-dbe8-4e68-974f-e6a02ece3b43',
        child_run_uuid='sdk-qualification-CPU-fixture',requested=deepcopy(unit.budget.REQUESTED),
        GPU_submissions=0,training_updates=0,GR00T_queries=0,policy_episodes=0,physics_ticks=0,
        parent_before_sha256=unit.common.sha(pb),Core_before_sha256=unit.common.sha(cb),
        observed_epoch=NOW,deadline_epoch=NOW+1800,full_forecast_inputs_sha256=digest,LOCAL_namespace=space['SDK_namespace'])
    return dict(parent=pb,Core=cb,q=q,x=x,args=args,space=space,VAP1=a)


def reserve(v,*,now_epoch=NOW):
    q,x=v['q'],v['x'];qb=unit.common.encode(q);xb=unit.common.encode(x)
    m=dict(schema='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1',scope=unit.ROLE,run_uuid=q['child_run_uuid'],
        authority=dict(protocol_ref=unit.common.PARENT_AUTHORITY,amendments=[unit.common.AUTHORITY,unit.common.VAP1_AUTHORITY]),
        code=dict(vla_commit='a'*40),limits=dict(operation_deadline_epoch=q['deadline_epoch'],PV1_T0_epoch=unit.T0,
            PV1_hard_close_epoch=unit.HARD_CLOSE,engine_starts=2,physics_ticks=0,policy_episodes=0),
        execution=dict(profile='4080-cpu-data',sdk_initialization_mode=q['sdk_initialization_mode'],sealed_inputs={
            'REQUEST.json':unit.common.sha(qb),unit.PROJECTION:unit.common.sha(xb),
            unit.FORECAST:unit.common.sha(unit.forecast_pack(v['args']))}))
    return unit.reserve(v['parent'],v['Core'],qb,unit.common.encode(m),xb,v['VAP1'],
        forecast_inputs=v['args'],namespace_proof=v['space'],now_epoch=now_epoch)


def terminal(r,*,result=True,meter=True):
    row=r['reservation'];rid=row['request_id']
    value=dict(scope=unit.ROLE,request_id=rid,manifest_sha256=row['manifest_sha256'],source_sha=row['source_sha'],
        target_task_native_qualified=False,physics_ticks=0,policy_episodes=0,GR00T_queries=0,MuJoCo_engine_constructors=0,
        actual_worker_and_children_CPU_s=111.5,SDK_interface_constructor_attempts=1)
    rb=unit.common.encode(value) if result else None
    proof=dict(schema='DEC1_TARGET_SDK_ACTUAL_TERMINAL_ACTOR_PROOF_V1',request_id=rid,
        manifest_sha256=row['manifest_sha256'],all_registered_actors_terminal=True,current_matching_actors=0,
        pid=123,proc_start='456',result_present=result,result_sha256=unit.common.sha(rb) if rb else None)
    local=dict(schema='DEC1_TARGET_SDK_ACTUAL_LOCAL_COORDINATOR_METER_V1',request_id=rid,
        manifest_sha256=row['manifest_sha256'],all_pre_settlement_control_children_waited=True,
        coordinator_and_children_CPU_s_before_settle=2.25)
    return dict(actor_proof_bytes=unit.common.encode(proof),result_bytes=rb,
        LOCAL_meter_bytes=unit.common.encode(local) if meter else None)


def finish(r,inputs):
    return unit.settle(unit.common.encode(r['proposed_parent']),unit.common.encode(r['proposed_Core']),
        r['reservation']['request_id'],**inputs)


def test_reserve_then_known_effects_keeps_history_and_all_eight_costs(packet):
    before=deepcopy(packet);r=reserve(packet)
    assert packet==before and r['native_dispatch_admitted'] is False
    assert r['forecast']['admissible_forecast'] is True
    p=r['proposed_parent'];assert p['CPU_reserved_s']['CoreLab']==3600 and p['CPU_reserved_s']['local_PC']==120
    closed=finish(r,terminal(r));amount=closed['settlement']['financial_charged']
    assert amount['CoreLab_CPU_s']==711.5 and amount['local_CPU_s']==32.25 and amount['constructors']==2
    assert amount['CoreLab_output_bytes']==512*1024**2 and amount['transfer_bytes']==64*1024**2
    assert closed['proposed_parent']['CPU_reserved_s']==dict(CoreLab=0,local_PC=0,Hopper=0)
    assert closed['proposed_parent']['children'][-1]['status']=='CLOSED_SDK_EFFECTS_SETTLED'
    assert closed['proposed_parent']['final_reserve_inside_total']==unit.common.obj(packet['parent'])['final_reserve_inside_total']
    unit.budget.rows(closed['proposed_parent'],closed['proposed_Core'])
    with pytest.raises(PermissionError):
        unit.settle(unit.common.encode(closed['proposed_parent']),unit.common.encode(closed['proposed_Core']),
            r['reservation']['request_id'],**terminal(r))


def test_unknown_result_and_meter_charge_full_upper_and_never_claim_closed_lease(packet):
    r=reserve(packet);v=finish(r,terminal(r,result=False,meter=False))
    assert v['settlement']['financial_charged']==unit.budget.REQUESTED
    assert v['settlement']['scientific_verdict']=='SDK_ENGINEERING_UNKNOWN_FULL_UPPER_CHARGED'
    assert v['actual_once_Core_LOCAL_lease_close_and_independent_readback_still_required'] is True
    assert v['settlement']['target_task_native_qualified'] is False


def test_native_recomputation_uses_actual_later_now_without_rewriting_sealed_inputs(packet):
    before=deepcopy(packet);first=reserve(packet);later=reserve(packet,now_epoch=NOW+3)
    assert packet==before and later['reservation']==first['reservation']
    assert later['forecast']['projected_finish_epoch']==first['forecast']['projected_finish_epoch']+3
    assert later['forecast']['admissible_forecast'] is True
    with pytest.raises(PermissionError):reserve(packet,now_epoch=NOW+121)


@pytest.mark.parametrize('fault',['original_clock','new_physics','too_late','expired','unsealed_grant','unsealed_plan',
    'another_component','native_unknown','site_CPU','site_output','missing_final','wrong_profile_permission',
    'stale_namespace','full_remaining_GPU','wrong_adoption','CPU_over_component'])
def test_reject_unsealed_stale_unknown_or_unfunded_reservation(packet,fault):
    q,x=packet['q'],packet['x']
    if fault=='original_clock':q['original_T0_epoch']+=1
    elif fault=='new_physics':q['physics_ticks']=1
    elif fault=='too_late':q['deadline_epoch']=NOW+1801
    elif fault=='expired':q['deadline_epoch']=NOW
    elif fault=='unsealed_grant':packet['args']['amendment_bytes']+=b' '
    elif fault=='unsealed_plan':packet['args']['plan']['accounting_pad_s']+=1
    elif fault=='another_component':q['component_id']='NEW_EXPERIMENT'
    elif fault=='native_unknown':x['unknown_effects']=['UNKNOWN']
    elif fault=='site_CPU':x['site_remaining']['CoreLab_CPU_s']=3599
    elif fault=='site_output':x['site_remaining']['local_output_bytes']=0
    elif fault=='missing_final':
        p=unit.common.obj(packet['parent']);p['final_reserve_inside_total']['H200_s']=0;packet['parent']=unit.common.encode(p)
        q['parent_before_sha256']=x['parent_before_sha256']=unit.common.sha(packet['parent'])
    elif fault=='wrong_profile_permission':x['site_permission']='OLD_R4_PILOT'
    elif fault=='stale_namespace':packet['space']['observed_epoch']-=121
    elif fault=='full_remaining_GPU':
        packet['args']['snapshot']['dec1_used']['H200_s']=200000
        q['full_forecast_inputs_sha256']=x['full_forecast_inputs_sha256']=unit.common.sha(unit.forecast_pack(packet['args']))
    elif fault=='wrong_adoption':packet['VAP1']+=b' '
    elif fault=='CPU_over_component':q['requested']['local_CPU_s']=121
    with pytest.raises((PermissionError,KeyError)):reserve(packet)


@pytest.mark.parametrize('fault',['live_actor','other_request','known_result_missing','wrong_result_source',
    'worker_CPU_over','local_meter_over','physics','constructor_replay','meter_not_waited'])
def test_terminal_requires_exact_actor_and_whole_cpu_measurements(packet,fault):
    r=reserve(packet);args=terminal(r);proof=unit.common.obj(args['actor_proof_bytes'])
    result=unit.common.obj(args['result_bytes']);meter=unit.common.obj(args['LOCAL_meter_bytes'])
    if fault=='live_actor':proof['current_matching_actors']=1
    elif fault=='other_request':proof['request_id']='different'
    elif fault=='known_result_missing':args['result_bytes']=None
    elif fault=='wrong_result_source':result['source_sha']='b'*40
    elif fault=='worker_CPU_over':result['actual_worker_and_children_CPU_s']=3001
    elif fault=='local_meter_over':meter['coordinator_and_children_CPU_s_before_settle']=91
    elif fault=='physics':result['physics_ticks']=1
    elif fault=='constructor_replay':result['SDK_interface_constructor_attempts']=2
    elif fault=='meter_not_waited':meter['all_pre_settlement_control_children_waited']=False
    if args['result_bytes'] is not None:
        args['result_bytes']=unit.common.encode(result);proof['result_sha256']=unit.common.sha(args['result_bytes'])
    args['actor_proof_bytes']=unit.common.encode(proof);args['LOCAL_meter_bytes']=unit.common.encode(meter)
    with pytest.raises(PermissionError):finish(r,args)
