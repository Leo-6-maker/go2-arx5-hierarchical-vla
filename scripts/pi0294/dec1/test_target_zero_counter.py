"""CPU resource fixtures only; no native constructors, entry or admission."""
from copy import deepcopy
import sys
import pytest
from . import target_zero_counter_contract as unit
from . import target_zero_counter as runtime


@pytest.fixture
def fixture(monkeypatch):
    adoption=b'{"CPU_fixture":true}\n';monkeypatch.setattr(unit,'VAP1_SHA',unit.sha(adoption))
    p=dict(run_uuid='20261018T182956Z-localpc-pv1',T0_epoch=1792348196,hard_closeout_epoch=unit.HARD_CLOSE,
        unknown_effects=[],Hopper_unknown_effects=[],pending_updates=[],CPU_s=dict(CoreLab=10000.,local_PC=2000.,Hopper=5000.),
        CPU_reserved_s=dict(CoreLab=0.,local_PC=0.,Hopper=100.),episode_starts=10,physics_ticks=dict(live=5000,shadow=6000,reconstruction=10),
        engine_starts=100,network_transfer_accounted_bytes=1000,children=[],
        final_reserve_inside_total=dict(H200_s=72000,episode_starts=100,live_ticks=1600000,shadow_ticks=1600000),
        GPU_allocated_s=61462,GPU_reserved_s=22560,PBS_submissions=['old-preserved'],optimizer_updates=5891,
        old_source_scientific_UNKNOWN='PRESERVED_WITH_FINANCIAL_CLOSURE')
    c=dict(authority_release=unit.AUTHORITY,campaign_run_uuid=p['run_uuid'],original_T0_epoch=p['T0_epoch'],hard_close_epoch=unit.HARD_CLOSE,
        VAP1_authority_release=unit.VAP1_AUTHORITY,unknown_effects=[],CPU_s=100.,episode_starts=5,physics_ticks=50,constructors=4,
        source_cpu_reserved=dict.fromkeys(unit.DIMS,0),historical_source_settlements={'fixture':'unchanged'})
    q=dict(schema=unit.REQUEST_SCHEMA,request_id='counter-fixture-01',scope=unit.ROLE,authority_release=unit.AUTHORITY,
        campaign_run_uuid=p['run_uuid'],original_T0_epoch=p['T0_epoch'],observed_epoch=1792720000.,deadline_epoch=1792721800.,
        requested=deepcopy(unit.REQUESTED),GPU_submissions=0,training_updates=0,GR00T_queries=0,policy_episodes=0,physics_ticks=0,
        child_run_uuid='target-counter-fixture')
    used=dict(zip(unit.DIMS,(100.,2000.,5,50,4,100000,100000,1000)))
    x=dict(schema='DEC1_TARGET_ZERO_COUNTER_ACTUAL_NATIVE_RESOURCE_PROJECTION_V1',origin='EXISTING_LOCAL_PC',
        observed_epoch=q['observed_epoch'],unknown_effects=[],DEC1_used=used,DEC1_reserved=dict.fromkeys(unit.DIMS,0),
        site_remaining=deepcopy(unit.CAPS),parent_remaining=deepcopy(unit.CAPS),
        site_permission='EXISTING_REGISTERED_CORELAB_4080_SIMULATION',concurrent_CoreLab_cores=2)
    m=dict(schema='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1',scope=unit.ROLE,run_uuid=q['child_run_uuid'],
        authority=dict(protocol_ref=unit.PARENT_AUTHORITY,amendments=[unit.AUTHORITY,unit.VAP1_AUTHORITY]),
        limits=dict(operation_deadline_epoch=q['deadline_epoch'],PV1_T0_epoch=p['T0_epoch'],PV1_hard_close_epoch=unit.HARD_CLOSE,
            engine_starts=2,policy_episodes=0,physics_ticks=0),execution=dict(sealed_inputs={}),code=dict(vla_commit='e'*40))
    future=dict(Core_CPU_s=200360,local_CPU_s=7440,episode_starts=88,physics_ticks=2816000,engine_starts=7134,Core_storage_bytes=17968398336,local_storage_bytes=0,transfer_bytes=10468982784)
    protected=dict(budget_proof=dict(full_remaining_work_admissible=True,all_old_spent_and_reserved_preserved=True,
        original_T0_epoch=p['T0_epoch'],hard_close_epoch=unit.HARD_CLOSE,zero_update_resume_release=unit.RECOVERY_RELEASE,
        zero_update_resume_grant_sha256=unit.RECOVERY_GRANT,forecast=dict(admissible_forecast=True,
            original_final_protection=dict(H200_s=72000,Hopper_storage_bytes=40*1024**3),future_required_additional=future,projected_finish_epoch=1792863000.,clock=dict(stop_new_jobs_epoch=1792866596))))
    return p,c,q,m,x,adoption,unit.encode(protected)


def call(fixture,now=1792720001.):
    p,c,q,m,x,a,protected=deepcopy(fixture);pr,cr=map(unit.encode,(p,c));q['parent_before_sha256']=x['parent_before_sha256']=unit.sha(pr)
    q['Core_before_sha256']=x['Core_before_sha256']=unit.sha(cr);qr,xr=map(unit.encode,(q,x))
    remaining=unit.obj(protected)['budget_proof']['forecast']['future_required_additional']
    mapping=dict(CoreLab_CPU_s='Core_CPU_s',local_CPU_s='local_CPU_s',episode_starts='episode_starts',physics_ticks='physics_ticks',constructors='engine_starts',CoreLab_output_bytes='Core_storage_bytes',local_output_bytes='local_storage_bytes',transfer_bytes='transfer_bytes')
    x['protected_remaining']={k:remaining[v] for k,v in mapping.items()};x['protected_budget_sha256']=unit.sha(protected);xr=unit.encode(x)
    m['execution']['sealed_inputs'].update({unit.PROJECTION:unit.sha(xr),'REQUEST.json':unit.sha(qr),unit.PROTECTED:unit.sha(protected)})
    return unit.reserve(pr,cr,qr,unit.encode(m),xr,a,protected_budget_bytes=protected,now_epoch=now)


def test_current_reservation_keeps_gpu_clocks_source_failures_and_final_unchanged(fixture):
    before=deepcopy(fixture);r=call(fixture)
    assert fixture==before and r['native_dispatch_admitted'] is False
    for k in ('GPU_allocated_s','GPU_reserved_s','PBS_submissions','optimizer_updates','final_reserve_inside_total','old_source_scientific_UNKNOWN'):
        assert r['proposed_parent'][k]==fixture[0][k]
    assert r['proposed_Core']['historical_source_settlements']==fixture[1]['historical_source_settlements']
    assert r['reservation']['requested']['physics_ticks']==r['reservation']['requested']['episode_starts']==0
    assert r['proposed_parent']['CPU_reserved_s']['Hopper']==100.


@pytest.mark.parametrize('fault',['policy','physical','constructors','GPU','old_scope','unknown_Core','unknown_LOCAL',
    'time','local_CPU','Core_CPU','missing_dimension','source_hold_omitted','site','final','bool_cost'])
def test_counter_cannot_borrow_science_authority_reset_history_or_drop_cost(fixture,fault):
    p,c,q,m,x,a,protected=fixture
    if fault=='policy':q['policy_episodes']=1
    elif fault=='physical':q['requested']['physics_ticks']=1
    elif fault=='constructors':q['requested']['constructors']=3
    elif fault=='GPU':q['GPU_submissions']=1
    elif fault=='old_scope':m['scope']='R4A_NATIVE_PILOT'
    elif fault=='unknown_Core':c['unknown_effects']=['lost-write']
    elif fault=='unknown_LOCAL':p['unknown_effects']=['lost-write']
    elif fault=='time':q['observed_epoch']-=121;x['observed_epoch']=q['observed_epoch']
    elif fault=='local_CPU':p['CPU_s']['local_PC']=unit.CAPS['local_CPU_s'];x['DEC1_used']['local_CPU_s']=p['CPU_s']['local_PC']
    elif fault=='Core_CPU':c['CPU_s']=unit.CAPS['CoreLab_CPU_s'];x['DEC1_used']['CoreLab_CPU_s']=c['CPU_s']
    elif fault=='missing_dimension':x['site_remaining'].pop('constructors')
    elif fault=='source_hold_omitted':c['source_cpu_reserved']['CoreLab_CPU_s']=500.
    elif fault=='site':x['concurrent_CoreLab_cores']=4
    elif fault=='final':p['final_reserve_inside_total']['H200_s']=0
    elif fault=='bool_cost':q['requested']['physics_ticks']=False
    with pytest.raises(PermissionError):call(fixture)


def terminal(fixture,*,missing=False):
    r=call(fixture);rid=r['reservation']['request_id'];row=r['reservation']
    result=dict(scope=unit.ROLE,request_id=rid,manifest_sha256=row['manifest_sha256'],constructor_attempts=2,
        actual_worker_CPU_s=1.25,status='TARGET_ZERO_COUNTER_ENGINEERING_FAILURE_PRESERVED',physics_ticks=0,
        policy_episodes=0,WBC_or_leg_policy_forwards=0,GR00T_queries=0,target_task_native_qualified=False,integration_attempts=[])
    raw=None if missing else unit.encode(result)
    proof=dict(schema='DEC1_TARGET_ZERO_COUNTER_ACTUAL_TERMINAL_ACTOR_PROOF_V1',request_id=rid,
        manifest_sha256=row['manifest_sha256'],actor_terminal=True,current_matching_actors=0,pid=123,
        proc_start='456',result_sha256=None if missing else unit.sha(raw),result_present=not missing)
    return r,rid,raw,proof


def test_known_failure_charges_actual_attempts_and_overhead_once_never_task_qualification(fixture):
    r,rid,raw,proof=terminal(fixture)
    s=unit.settle(unit.encode(r['proposed_parent']),unit.encode(r['proposed_Core']),rid,result_bytes=raw,actor_proof_bytes=unit.encode(proof))
    assert s['settlement']['financial_charged']['CoreLab_CPU_s']==601.25
    assert s['settlement']['scientific_verdict']=='KNOWN_ZERO_TIME_COUNTER_ENGINEERING_FAILURE'
    assert s['proposed_parent']['CPU_reserved_s']['CoreLab']==s['proposed_parent']['CPU_reserved_s']['local_PC']==0
    assert s['proposed_parent']['GPU_reserved_s']==22560 and not s['settlement']['target_task_native_qualified']
    with pytest.raises(PermissionError):unit.settle(unit.encode(s['proposed_parent']),unit.encode(s['proposed_Core']),rid,result_bytes=raw,actor_proof_bytes=unit.encode(proof))


def test_missing_result_retains_full_cost_and_unknown_without_faking_counter_pass(fixture):
    r,rid,raw,proof=terminal(fixture,missing=True)
    s=unit.settle(unit.encode(r['proposed_parent']),unit.encode(r['proposed_Core']),rid,result_bytes=raw,actor_proof_bytes=unit.encode(proof))
    assert s['settlement']['financial_charged']==unit.REQUESTED
    assert s['settlement']['scientific_verdict']=='COUNTER_ENGINEERING_INCOMPLETE_UNKNOWN_FULL_UPPER_CHARGED'


@pytest.mark.parametrize('fault',['live','wrong_start','known_result_dropped','false_PASS'])
def test_settlement_needs_actual_terminal_actor_and_complete_result(fixture,fault):
    r,rid,raw,proof=terminal(fixture)
    if fault=='live':proof['actor_terminal']=False
    elif fault=='wrong_start':proof['proc_start']='unknown'
    elif fault=='known_result_dropped':raw=None
    elif fault=='false_PASS':
        result=unit.obj(raw);result['status']='PASS_FROZEN_FULLBODY_TASK_ZERO_INTEGRATION_ASSET_COUNTER_ONLY';result['constructor_attempts']=0
        raw=unit.encode(result);proof['result_sha256']=unit.sha(raw)
    with pytest.raises((PermissionError,KeyError)):unit.settle(unit.encode(r['proposed_parent']),unit.encode(r['proposed_Core']),rid,result_bytes=raw,actor_proof_bytes=unit.encode(proof))


def test_no_admission_no_model_import_or_intent(tmp_path,monkeypatch):
    import builtins
    before=set(sys.modules);native=[];original=builtins.__import__
    def check(name,*a,**kw):
        if name.split('.')[0] in ('mujoco','torch','rclpy'):native.append(name);raise AssertionError('NATIVE_IMPORT_BEFORE_ADMISSION')
        return original(name,*a,**kw)
    with monkeypatch.context() as m:
        m.setattr(builtins,'__import__',check)
        with pytest.raises(FileNotFoundError):runtime.run(tmp_path)
    assert native==[] and not(tmp_path/'evidence/ZERO_COUNTER_INTENT.json').exists()


def test_original_remaining_work_cannot_be_reduced_or_final_clock_extended(fixture):
    p,c,q,m,x,a,protected=fixture
    value=unit.obj(protected);value['budget_proof']['forecast']['projected_finish_epoch']=1792866590.
    modified=(*fixture[:6],unit.encode(value))
    with pytest.raises(PermissionError,match='WALL_HEADROOM'):call(modified)
