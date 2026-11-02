"""Reset policy/fault tests; these are not native MuJoCo qualification."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0294.multitask import book_reset as reset
from scripts.pi0294.multitask.book_policy import Student, dev_card


def sealed():
    panel = reset.plan(dict(Phase4096='a'*64, D0='b'*64, D1='c'*64))
    qualification = dict(schema='BOOK_RESET_NATIVE_QUALIFICATION_V1',passed=True,
        host=os.uname().nodename,uid=os.getuid(),numpy_version=np.__version__,MuJoCo_version='3.2.7',
        checks=dict.fromkeys(reset.NATIVE_CHECKS,True),template_manifest_sha256='d'*64,
        source_files=reset.source_files())
    manifest = dict(kind=reset.KIND,run_uuid=reset.condition_run_name(panel,'D1',0),
        reset_model='D1',reset_condition_index=0,delta_checkpoint_manifest_sha256='c'*64,
        limits=dict(reset.LIMITS),source_template_manifest_sha256='d'*64,
        reset_source_files=qualification['source_files'],sealed_inputs={
            'BOOK_RESET_DEV10_PLAN.json':reset.digest(panel),
            'RESET_NATIVE_QUALIFICATION.json':reset.digest(qualification),
            'RESET_PREDECESSOR.json':'e'*64})
    return manifest,panel,qualification


def test_same_original_conditions_noise_and_one_finite_native_worker():
    from scripts.pi0294.multitask.seed_set_export import DEV_OFFSETS
    m,p,q=sealed()
    assert reset.validate_manifest(m,p,q)==dict(id='BOOK_DEV000',xy_offset_m=DEV_OFFSETS[0])
    for i,c in enumerate(p['conditions']):
        old=Student(None,c['id'],None,None)
        assert old.generation_seed(0)==Student(None,c['id'],None,None,seed_namespace=p['seed_namespace']).generation_seed(0)
        assert c['xy_offset_m']==DEV_OFFSETS[i]
    assert p['exposed_DEV_not_blind_TEST'] and p['original_results_and_STOP_preserved']
    assert reset.LIMITS['physics_integrations']==18500 and reset.LIMITS['source_constructors']==1


@pytest.mark.parametrize('field,value',[('reset_condition_index',True),('reset_condition_index',10),
    ('delta_checkpoint_manifest_sha256','f'*64),('kind','V7_BOOK_FROZEN_DEV10'),('run_uuid','retry-same-condition')])
def test_changed_model_condition_or_retry_identity_rejected(field,value):
    m,p,q=sealed();m[field]=value
    with pytest.raises(PermissionError):reset.validate_manifest(m,p,q)


def test_missing_native_checks_changed_implementation_and_extra_training_rejected():
    for fault in ('check','source','steps','predecessor','qualified_template','host','numpy'):
        m,p,q=sealed()
        if fault=='check':q['checks']['residual_writer_rejected']=False
        elif fault=='source':q['source_files'][reset.SOURCES[0]]='0'*64
        elif fault=='steps':m['limits']['optimizer_updates']=1
        elif fault=='predecessor':del m['sealed_inputs']['RESET_PREDECESSOR.json']
        elif fault=='qualified_template':q['template_manifest_sha256']='0'*64
        elif fault=='host':q['host']='not-the-qualified-runtime'
        else:q['numpy_version']='different'
        m['sealed_inputs']['RESET_NATIVE_QUALIFICATION.json']=reset.digest(q)
        with pytest.raises(PermissionError):reset.validate_manifest(m,p,q)


def closed(tmp_path):
    root=tmp_path/'previous';root.mkdir()
    m=dict(kind=reset.KIND,reset_model='D1',reset_condition_index=0,
        sealed_inputs={'BOOK_RESET_DEV10_PLAN.json':'p'})
    folder=root/'conditions/BOOK_DEV000';folder.mkdir(parents=True);witness={}
    for name,value in [('RESET_INITIAL.json',dict(controller_and_IK_clear=True)),
        ('RESET_CAMERA.json',dict(original_dual_camera_qualified=True)),
        ('RESET_CLOSE.json',dict(engine_close_returned=True,raw_writer_closed=True))]:
        body=json.dumps(value).encode();(folder/name).write_bytes(body);witness[name]=hashlib.sha256(body).hexdigest()
    result=dict(status='COMPLETE_BOOK_RESET_DEV_CONDITION',error=None,
        conditions=[dict(id='BOOK_DEV000',reset_protocol=reset.PROTOCOL,reset_witness_files=witness,
            valid_model_condition=True,hard_native_safety=True,success=False,reset_close_confirmed=True)])
    pins={}
    for name,v in [('RUN_MANIFEST.json',m),('RESULT.json',result)]:
        body=json.dumps(v).encode();(root/name).write_bytes(body);pins[name]=hashlib.sha256(body).hexdigest()
    (root/'control').mkdir();body=json.dumps(dict(status='CLOSED',run_uuid=root.name,
        manifest_sha256=pins['RUN_MANIFEST.json'],result_sha256=pins['RESULT.json'])).encode()
    (root/'control/LEASE.json').write_bytes(body);pins['control/LEASE.json']=hashlib.sha256(body).hexdigest()
    return dict(root=str(root),unit='previous.service',files=pins)


def test_hard_negative_remains_valid_but_next_condition_requires_actual_exit(tmp_path):
    pred=closed(tmp_path);opts=dict(current_root=tmp_path/'next',expected_previous=('D1',0,'p'),writers=lambda root:[])
    assert reset.verify_predecessor(pred,observe_unit=lambda n:dict(MainPID='0',SubState='dead'),**opts)['process_terminal']
    for state in (dict(MainPID='321',SubState='running'),dict(MainPID='0',SubState='unknown'),{}):
        with pytest.raises(PermissionError,match='PROCESS_STILL_LIVE'):
            reset.verify_predecessor(pred,observe_unit=lambda n:state,**opts)


def test_child_cgroup_and_residual_writer_stop_even_with_dead_main_pid(tmp_path):
    pred=closed(tmp_path);cg=tmp_path/'cgroup';(cg/'unit/child').mkdir(parents=True)
    (cg/'unit/cgroup.procs').write_text('');(cg/'unit/child/cgroup.procs').write_text('123\n')
    state=lambda n:dict(MainPID='0',SubState='dead',ControlGroup='/unit')
    with pytest.raises(PermissionError,match='CHILD_PROCESS'):
        reset.verify_predecessor(pred,current_root=tmp_path/'next',observe_unit=state,writers=lambda r:[],cgroup_root=cg)
    (cg/'unit/child/cgroup.procs').write_text('')
    with pytest.raises(PermissionError,match='RESIDUAL_WRITER'):
        reset.verify_predecessor(pred,current_root=tmp_path/'next',observe_unit=state,writers=lambda r:[{'PID':123}],cgroup_root=cg)


def test_unknown_close_or_wrong_prior_condition_never_advances(tmp_path):
    pred=closed(tmp_path);opts=dict(current_root=tmp_path/'next',observe_unit=lambda n:dict(MainPID='0',SubState='dead'),writers=lambda r:[])
    for expected in [('D1',1,'p'),('D0',0,'p'),('D1',0,'other')]:
        with pytest.raises(PermissionError):reset.verify_predecessor(pred,expected_previous=expected,**opts)
    (Path(pred['root'])/'control/LEASE.json').write_text('{}')
    with pytest.raises(PermissionError,match='CLOSURE_CHANGED'):reset.verify_predecessor(pred,**opts)


def test_changed_reset_witness_stops_next_condition(tmp_path):
    pred=closed(tmp_path)
    (Path(pred['root'])/'conditions/BOOK_DEV000/RESET_CLOSE.json').write_text('{}')
    with pytest.raises(PermissionError,match='RESET_WITNESS_CHANGED'):
        reset.verify_predecessor(pred,current_root=tmp_path/'next',expected_previous=('D1',0,'p'),
            observe_unit=lambda n:dict(MainPID='0',SubState='dead'),writers=lambda root:[])


def test_actual_writable_descriptor_is_detected_and_readonly_consumer_is_allowed(tmp_path):
    root=tmp_path/'raw';root.mkdir();file=root/'records';file.write_text('retained')
    # Use the live process's /proc entries, not a claimed PID or a stale marker.
    proc=tmp_path/'proc';proc.mkdir();(proc/str(os.getpid())).symlink_to(Path('/proc')/str(os.getpid()))
    with file.open('rb'):
        assert reset.live_writers(root,proc=proc)==[]
    with file.open('ab'):
        rows=reset.live_writers(root,proc=proc)
        assert len(rows)==1 and rows[0]['PID']==os.getpid() and rows[0]['relative_path']=='records'
    assert reset.live_writers(root,proc=proc)==[]


def fresh():
    card=dev_card(dict(id='BOOK_DEV000',xy_offset_m=[.006,0.]))
    d=SimpleNamespace(time=0.,qpos=np.array([0.,2.13,1.73,-1.17,0.,0.,.042,.042]),
        qvel=np.zeros(8),ctrl=np.zeros(8),act=np.zeros(0),qacc_warmstart=np.zeros(8),
        qfrc_applied=np.zeros(8),xfrc_applied=np.zeros((10,6)))
    original_origin=np.array(card['initial_book_native_world'])[:3,3].copy()
    engine=SimpleNamespace(data=d,ik_data=SimpleNamespace(time=0.,qvel=np.zeros(8)),steps=0,
        safety=[],renderer=None,_ik_target=None,_cached_ik_solution=None,_cached_ik_error=None,
        ik_solve_calls=0,ik_cache_hits=0,arm_q=np.arange(6),finger_q=np.array([6,7]),
        arm_targets=d.qpos[:6].copy(),finger_targets=np.full(2,.042),xml='<original-native-scene/>',
        pose=lambda name:np.array(card['initial_tcp']),snapshot=lambda:dict(safety_valid=True,native_grasp=False,
            native_book_body_origin=original_origin.copy()))
    return engine,Student(None,card['id'],None,None),card


@pytest.fixture
def fake_pristine(monkeypatch):
    # Actual MuJoCo reference-data reconstruction is a separate native check.
    monkeypatch.setattr(reset,'pristine_warmstart',lambda e:np.zeros_like(e.data.qacc_warmstart))


@pytest.mark.parametrize('fault',['time','qvel','ctrl','warmstart','force','IK','target','finger','student','seed','object'])
def test_dirty_initial_state_is_rejected_without_repair(fault,fake_pristine):
    e,s,c=fresh()
    if fault=='time':e.data.time=1.
    elif fault=='qvel':e.data.qvel[0]=.1
    elif fault=='ctrl':e.data.ctrl[0]=1.
    elif fault=='warmstart':e.data.qacc_warmstart[0]=1.
    elif fault=='force':e.data.xfrc_applied[0,0]=1.
    elif fault=='IK':e._cached_ik_solution=np.ones(6)
    elif fault=='target':e.arm_targets[0]=1.
    elif fault=='finger':e.finger_targets[0]=0.
    elif fault=='student':s.last_plan=0
    elif fault=='seed':s.seed_namespace='changed-noise'
    else:c['initial_book_native_world'][0][3]+=.01
    with pytest.raises(PermissionError):reset.initial_witness(e,s,c)
    if fault=='qvel':assert e.data.qvel[0]==.1


def test_fresh_witness_keeps_seed_and_actual_home_without_integrating(fake_pristine):
    e,s,c=fresh();w=reset.initial_witness(e,s,c)
    assert w['steps']==0 and w['first_generation_seed']==s.generation_seed(0)
    assert e.steps==0 and e.data.time==0 and s.queries==0


def test_nonzero_derived_acceleration_is_allowed_only_when_pristine_matches(monkeypatch):
    e,s,c=fresh();e.data.qacc_warmstart[:]=np.arange(8)
    monkeypatch.setattr(reset,'pristine_warmstart',lambda engine:np.arange(8))
    assert reset.initial_witness(e,s,c)['acceleration_cache_matches_pristine_native_forwards']
    e.data.qacc_warmstart[0]=1.
    with pytest.raises(PermissionError,match='PRISTINE_NATIVE_FORWARD'):reset.initial_witness(e,s,c)
