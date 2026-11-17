import copy
import hashlib
import json
import time

import pytest

from go2arx5_vla.world_model import sim_only_scope_v51 as scope
from scripts.wm02.sim_only_campaign_v51 import interface_gate, fatal_control_violation, pilot_gate, panel_preflight


@pytest.fixture
def authorized(tmp_path,monkeypatch):
    binaries={};frozen={};digests={}
    for key in scope.BINARIES:
        p=tmp_path/key;p.write_bytes(('CPU_SIMULATION_FIXTURE_'+key).encode())
        binaries[key]=str(p);digests[key]=hashlib.sha256(p.read_bytes()).hexdigest();frozen[str(p)]=digests[key]
    monkeypatch.setattr(scope,'BINARIES',digests)
    now=time.time();authority=dict(schema='USER_PROJECT_AUTHORIZED_ISOLATED_MUJOCO_V51',
        scope=scope.SCOPE,source='CURRENT_USER_EXPLICIT_V51_INSTRUCTION',new_training_updates=0,
        hardware_allowed=False,source_sha='a'*40,deadline_epoch=min(now+1000,1795027883.855488))
    p=tmp_path/'USER_AUTH.json';p.write_text(json.dumps(authority))
    spec=dict(sim_only_project_authorization=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
        source_sha='a'*40,mobile_condition=dict(family='F1',mode='M_FREE',method='SHIFT7_SINGLE'),
        sim_only_role='STUDENT',mobile_experiment_version=scope.VERSION,plan_commit=scope.PLAN_COMMIT,
        execution_clock_profile='SIM_PAUSED_NATIVE',scope=scope.SCOPE,physics=True,
        native_backend='MUJOCO_ONLY',hardware_connections_allowed=False,
        max_ticks=scope.TOTAL_TICKS,physics_dt_s=.002,SHIFT7_package=dict(sha256=scope.PACKAGE),
        optimizer_updates=0,WBC_changes=False,deadline_epoch=authority['deadline_epoch']-60,
        frozen_files=frozen,ROS_DOMAIN_ID=198,native_environment=dict(ROS_LOCALHOST_ONLY='1',
            ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES=''),**binaries)
    return spec


def test_owner_authorized_simulation_needs_no_invented_independent_signature(authorized):
    result=scope.validate(authorized)
    assert result['independent_signature_required'] is False
    assert result['hardware_allowed'] is False
    assert result['total_ticks_max']==5703+30000
    assert 'independent_reviewer'not in authorized


@pytest.mark.parametrize('fault',['hardware','clock','ticks','method','model','nonlocal','binary','source','real_backend'])
def test_project_authorization_does_not_waive_actual_isolation_or_frozen_contract(authorized,fault):
    s=copy.deepcopy(authorized)
    if fault=='hardware':s['hardware_connections_allowed']=True
    elif fault=='clock':s['execution_clock_profile']='WALL_CLOCK_NATIVE'
    elif fault=='ticks':s['max_ticks']+=1
    elif fault=='method':s['mobile_condition']['method']='EXPERT'
    elif fault=='model':s['SHIFT7_package']['sha256']='b'*64
    elif fault=='nonlocal':s['native_environment']['ROS_LOCALHOST_ONLY']='0'
    elif fault=='binary':s['frozen_files'][s['controller_binary']]='b'*64
    elif fault=='source':s['source_sha']='b'*40
    else:s['native_backend']='REAL_ROBOT'
    with pytest.raises(PermissionError):scope.validate(s)


def test_interface_has_its_own_small_native_bound(authorized):
    s=copy.deepcopy(authorized);s['sim_only_role']='INTERFACE';s['mobile_condition']['method']='SIM_INTERFACE_HOLD'
    s['max_ticks']=scope.INTERFACE_TICKS
    assert scope.validate(s)['total_ticks_max']==5953
    s['max_ticks']=scope.TOTAL_TICKS
    with pytest.raises(PermissionError):scope.validate(s)


def test_one_simulated_collision_is_a_failed_case_without_waiving_STOP_integrity():
    row=dict(hard_STOP=True,integrations_after_STOP=0,trace_errors=[])
    assert not fatal_control_violation(row)
    assert fatal_control_violation(dict(row,integrations_after_STOP=1))
    assert fatal_control_violation(dict(row,trace_errors=['DECLARED_NATIVE_TICK_LIMIT_EXCEEDED']))


def test_stage_gates_preserve_first_round_and_do_not_call_zero_of_eight_success():
    interfaces=[dict(role='INTERFACE',completed=True,hard_STOP=False,interface_protocol_pass=True)for _ in range(2)]
    assert interface_gate(interfaces)
    assert not interface_gate(interfaces[:1])
    students=[dict(role='STUDENT',family='F1'if i<4 else'F2',hard_STOP=False,
        task_and_mobile_success=False)for i in range(8)]
    assert not pilot_gate(interfaces+students)
    students[0]['task_and_mobile_success']=True;students[4]['task_and_mobile_success']=True
    assert pilot_gate(interfaces+students)
    students[2]['hard_STOP']=True
    assert not pilot_gate(interfaces+students)


def test_panel_catches_different_historical_collector_lock_before_native_launch(tmp_path):
    paths=[]
    for i in range(18):
        p=tmp_path/('%02d.json'%i);p.write_text(json.dumps(dict(mobile_world_lock='ros-198.lock',
            ROS_DOMAIN_ID=198,mobile_group=str(tmp_path))));paths.append(str(p))
    parent=dict(cases=paths,mobile_world_lock='ros-198.lock',ROS_DOMAIN_ID=198,registered_root=str(tmp_path))
    assert panel_preflight(parent)==[]
    wrong=json.loads((tmp_path/'01.json').read_text());wrong['mobile_world_lock']='ros-199.lock'
    (tmp_path/'01.json').write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match='PARENT_DOMAIN_AND_LOCK'):panel_preflight(parent)
