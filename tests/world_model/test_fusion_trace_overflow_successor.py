"""Synthetic overflow-lineage tests; never a real qualification."""
import copy
import json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as c
from go2arx5_vla.world_model.storage import digest


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def failure(tmp_path,monkeypatch):
    native=tmp_path/'frozen-native';native.write_bytes(b'fixture-native')
    observer=tmp_path/'frozen-observer';observer.write_bytes(b'fixture-observer')
    scene=tmp_path/'scene';scene.write_bytes(b'fixture-scene')
    root=tmp_path/'failed-full';path=root/'native-01/RESULT.json'
    value=dict(status='NATIVE_STAGE_FAILED_PRESERVED',phase=c.SLOW_PHASE,
        actual_target_integrations=29753,reached_exact_pause=True,shutdown_clean=True,
        failure=None,physics_stops=[],original_joint_or_warning_samples=0,native_outer_guard_rejection_count=0,
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_FAIL',
            errors=['WRITER_EVENT_COUNT_DIFFERS','WRITER_REPORTED_DROPPED_EVENTS'],missing=[],
            writer=dict(events=266763,written=262144,dropped=4619,sealed_at_explicit_pause=True,stop_latched=False)),
        reference_qualification=dict(all_40_knots_consumed=True,complete_task_atoms=1090,
                                    solver_target_confirmed_atoms=1090,gripper_confirmed_atoms=1090),
        native_binary_sha256=digest(native),observer_sha256=digest(observer),scene_sha256=digest(scene),
        owned_process_exits=[dict(name=n,exit_code=0)for n in ['sim','controller','solver','planner']])
    put(path,value);put(root/'EXECUTION_SPEC.json',dict(max_ticks=29753))
    put(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=1,
                                     spec_sha256=digest(root/'EXECUTION_SPEC.json')))
    put(root/'WORKER_EXIT.private.json',dict(exit_code=1))
    spec=dict(trace_capacity_failed_result=str(path),trace_capacity_failed_result_sha256=digest(path),
              native_binary=str(native),observer_library=str(observer),model_path=str(scene))
    monkeypatch.setattr(c,'TRACE_CAPACITY_FAILED_RESULT',digest(path))
    return spec,value,path


def test_actual_overflow_shape_is_a_retained_failure_not_a_pass(failure):
    spec,value,_=failure
    assert c.validate_trace_capacity_failure(spec)==value
    assert value['status']=='NATIVE_STAGE_FAILED_PRESERVED'


@pytest.mark.parametrize('mutation',['pass','unsealed','no_drop','extra_error','missing','stopped','incomplete','child'])
def test_cannot_substitute_an_unrelated_or_incomplete_failure(failure,monkeypatch,mutation):
    spec,value,path=failure;value=copy.deepcopy(value)
    if mutation=='pass':value['status']='NATIVE_STAGE_PASSED'
    if mutation=='unsealed':value['trace_audit']['writer']['sealed_at_explicit_pause']=False
    if mutation=='no_drop':value['trace_audit']['writer']['dropped']=0
    if mutation=='extra_error':value['trace_audit']['errors'].append('NEW_ERROR')
    if mutation=='missing':value['trace_audit']['missing']=['missing']
    if mutation=='stopped':value['physics_stops']=[{'reason':'ILLEGAL_CONTACT'}]
    if mutation=='incomplete':value['reference_qualification']['gripper_confirmed_atoms']=1089
    if mutation=='child':value['owned_process_exits'][0]['exit_code']=1
    put(path,value);spec['trace_capacity_failed_result_sha256']=digest(path)
    monkeypatch.setattr(c,'TRACE_CAPACITY_FAILED_RESULT',digest(path))
    with pytest.raises(ValueError,match='ACTUAL_FAILED_TRACE_EVIDENCE'):c.validate_trace_capacity_failure(spec)


def test_changed_native_failure_bytes_are_rejected(failure):
    spec,value,path=failure;value['failure']='changed';put(path,value)
    with pytest.raises(ValueError,match='EXACT_RETAINED_FULL1'):c.validate_trace_capacity_failure(spec)


def test_new_task_scene_keeps_historical_overflow_failed(failure,monkeypatch):
    from go2arx5_vla.world_model import stable_task_scene_v1 as current
    from go2arx5_vla.world_model import stable_task_layout_static_v1 as layouts
    spec,value,path=failure
    historical=spec['model_path'];pin=digest(historical)
    previous=path.parent.parent/'EXECUTION_SPEC.json'
    put(previous,dict(max_ticks=29753,model_path=historical))
    lease=path.parent.parent/'control/LEASE.json'
    put(lease,dict(status='CLOSED',worker_exit_code=1,spec_sha256=digest(previous)))
    spec=dict(spec,phase=current.PHASE,model_path='new-scene-not-historical')
    seen=[]
    monkeypatch.setattr(current,'scene',lambda s:seen.append(s))
    monkeypatch.setattr(layouts,'BASE_SCENE',pin)
    assert c.validate_trace_capacity_failure(spec,historical_scene_sha256=pin)==value
    assert seen==[spec]
    assert value['status']=='NATIVE_STAGE_FAILED_PRESERVED'
    with pytest.raises(ValueError,match='EXACT_HISTORICAL_T06'):
        c.validate_trace_capacity_failure(spec,historical_scene_sha256='0'*64)


def test_unregistered_scene_cannot_use_new_task_route(failure):
    spec,_,_=failure
    with pytest.raises(ValueError,match='DISTINCT_DIAGNOSTIC_QUERY_ONLY_SCOPE'):
        c.validate_trace_capacity_failure(dict(spec,phase='unknown'),historical_scene_sha256='0'*64)
