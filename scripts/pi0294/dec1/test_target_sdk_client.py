"""Real finite process/pipe tests with generated CPU-only SDK API stand-ins.

The actual C++ companion is compiled and executed. No vendor SDK, model,
physics, ROS, GPU, lease or ledger is involved. This is not native ABI proof.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from . import target_sdk_client as unit
from .test_target_sdk_constraint_wire import SDK_FIXTURE
from .target_mpc_prediction import BoundMPCPrediction


MOBILE = r'''
#pragma once
#include "cpu_sdk_fixture.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
namespace ocs2::mobile_manipulator {
struct FrameOverrides {std::string baseFrame,eeFrame,eeFrame1;};
struct ModelInfo {size_t stateDim=12,inputDim=12;int manipulatorModelType=3;};
class MobileManipulatorInterface {
  OptimalControlProblem problem_; ModelInfo info_;
 public:
  MobileManipulatorInterface(const std::string&,const std::string& cache,const std::string&,const FrameOverrides& frames) {
    std::cout<<"CPU_GENERATED_SDK_FIXTURE_LOG_NOT_MACHINE_REPLY\n";
    if(frames.baseFrame=="CPU_TEST_BAD_MODEL") info_.stateDim=13;
    if(frames.baseFrame=="CPU_TEST_NO_EXEC" && std::system("exit 0")==0)
      throw std::runtime_error("COMPILER_OR_SHELL_EXEC_WAS_NOT_DENIED");
    if(frames.baseFrame=="CPU_TEST_INITIALIZE") {
      if(std::system("exit 0")!=0)throw std::runtime_error("EXPLICIT_COLD_INIT_EXEC_DENIED");
      std::ofstream(cache+"/CPU_generated_cache.so")<<"CPU_STANDIN_NOT_NATIVE_AD";
    }
    problem_.softConstraintPtr->terms["jointLimits"]=std::make_unique<StateInputSoftBoxConstraint>();
    problem_.stateSoftConstraintPtr->terms["selfCollision"]=std::make_unique<StateSoftConstraint>();
    if(frames.baseFrame=="CPU_TEST_INITIALIZE")
      problem_.stateSoftConstraintPtr->get<StateSoftConstraint>("selfCollision").underlying.require_exec_denied=true;
  }
  const ModelInfo& getManipulatorModelInfo()const{return info_;}
  const OptimalControlProblem& getOptimalControlProblem()const{return problem_;}
};
}
'''


@pytest.fixture(scope='module')
def binary(tmp_path_factory):
    if not shutil.which('g++'): pytest.skip('CPU C++ fixture compiler unavailable; no native qualification')
    folder = tmp_path_factory.mktemp('SDK_client_generated_CPU_API')
    for name in ('ocs2_core/soft_constraint/StateSoftConstraint.h',
                 'ocs2_core/soft_constraint/StateInputSoftBoxConstraint.h',
                 'ocs2_oc/oc_problem/OptimalControlProblem.h'):
        p = folder/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('#include "cpu_sdk_fixture.hpp"\n')
    p = folder/'ocs2_mobile_manipulator/MobileManipulatorInterface.h';p.parent.mkdir()
    p.write_text(MOBILE); (folder/'cpu_sdk_fixture.hpp').write_text(SDK_FIXTURE)
    output = folder/'SDK_client_CPU_fixture'
    source = Path(__file__).with_name('target_sdk_client.cpp')
    compiled = subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread',
                    '-I'+str(folder), '-I'+str(source.parent), str(source), '-o', str(output)],
                   capture_output=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr.decode()
    return output


@pytest.fixture
def pins(binary, tmp_path, monkeypatch):
    # Explicit CPU fixture clock only. Do not replace process-wide time, and
    # never pass a historical now to a native entry/lease.
    monkeypatch.setattr(unit, 'time', SimpleNamespace(time=lambda:unit.T0+1000, monotonic=time.monotonic))
    files = {}
    for name in ('client_task', 'urdf', 'mobile_archive', 'core_archive'):
        p = tmp_path/name;p.write_bytes(('CPU_FIXTURE_'+name).encode());files[name] = p
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(unit, 'CLIENT_TASK_SHA256', digest(files['client_task']))
    monkeypatch.setattr(unit, 'MOBILE_SDK_SHA256', digest(files['mobile_archive']))
    monkeypatch.setattr(unit, 'CORE_SDK_SHA256', digest(files['core_archive']))
    cache = tmp_path/'private_cache';cache.mkdir();member=cache/'CPU_fixture_not_native_AD.so'
    member.write_bytes(b'CPU_FIXTURE_NOT_A_NATIVE_LIBRARY');member.chmod(0o444);cache.chmod(0o555)
    manifest=tmp_path/'RUN_MANIFEST.json';request=tmp_path/'REQUEST.json'
    request.write_text(json.dumps(dict(request_id='12345678-1234-4123-8123-123456789abc',
        child_run_uuid='CPU_FIXTURE_ONLY',scope=unit.SDK_ROLE,campaign_run_uuid=unit.CAMPAIGN,
        original_T0_epoch=unit.T0,deadline_epoch=unit.STOP_PHYSICS,sdk_initialization_mode='SEALED_CACHE_LOAD')))
    manifest.write_text(json.dumps(dict(schema='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1',scope=unit.SDK_ROLE,
        run_uuid='CPU_FIXTURE_ONLY',code=dict(vla_commit='c'*40),
        execution=dict(sealed_inputs={'REQUEST.json':digest(request)},sdk_initialization_mode='SEALED_CACHE_LOAD'),
        limits=dict(PV1_T0_epoch=unit.T0,PV1_hard_close_epoch=unit.HARD_CLOSE,
                    operation_deadline_epoch=unit.STOP_PHYSICS))))
    value = unit.SdkClientPins(str(binary), digest(binary), str(files['client_task']), str(files['urdf']),
        digest(files['urdf']), str(files['mobile_archive']), str(files['core_archive']), str(cache),
        ((member.name, digest(member)),), ('-', '-', '-'), str(manifest), digest(manifest),
        'CPU_FIXTURE_ONLY','c'*40,'12345678-1234-4123-8123-123456789abc','SEALED_CACHE_LOAD')
    yield value
    cache.chmod(0o755);member.chmod(0o644)


@pytest.fixture
def prediction():
    return BoundMPCPrediction('a'*64, 'b'*64, 'CPU_ONLY', 0., (0., .2, .4),
                              ((0.,)*12,)*3, ((0.,)*12,)*3)


def folder(tmp_path):
    p = tmp_path/'registered_CPU_fixture_output';p.mkdir();return p


def test_actual_cpp_process_persistent_exact_wire_and_no_exec(pins, prediction, tmp_path):
    guard_calls=[];root=folder(tmp_path)
    client=unit.SdkClient(root,replace(pins,frames=('CPU_TEST_NO_EXEC','-','-')),
                          guard=lambda **kw:guard_calls.append(kw),maximum_requests=2)
    pid=client.process.pid
    for _ in range(2):
        value=client.exchange(prediction)
        assert value.minimum_joint_limit_cost==-3.5 and value.minimum_self_collision_residual==-.002
        assert value.native_prefix_safety_qualified is value.physics_advance_authorized is False
        assert client.process.pid==pid
    client.close()
    assert client.process.returncode==0 and any(x['closeout'] for x in guard_calls)
    assert 'CPU_GENERATED_SDK_FIXTURE_LOG_NOT_MACHINE_REPLY' in (root/'SDK_CLIENT.log').read_text()
    assert len(list(root.glob('*_CONFIRMED.json')))==2
    result=json.loads((root/'SDK_CLIENT_EXIT.json').read_bytes())
    assert result['requests_confirmed']==2 and result['failed_or_unknown'] is False
    assert result['lease_closed'] is result['constructor_and_CPU_settled'] is False


def initialize_pins(pins,tmp_path,monkeypatch,scope=unit.SDK_ROLE):
    original=tmp_path/'explicit_original_task_CPU_fixture';original.write_bytes(b'EXPLICIT_ORIGINAL_SDK_TASK_CPU_FIXTURE')
    monkeypatch.setattr(unit,'FROZEN_TASK_SHA256',unit.digest(original.read_bytes()))
    cache=tmp_path/'new_registered_model_cache';cache.mkdir()
    p=replace(pins,client_task=str(original),private_readonly_cache=str(cache),cache_members=(),
              initialization_mode='ORIGINAL_SDK_INITIALIZE',frames=('CPU_TEST_INITIALIZE','-','-'))
    mpath=Path(p.owner_manifest);qpath=mpath.parent/'REQUEST.json'
    m=json.loads(mpath.read_bytes());q=json.loads(qpath.read_bytes())
    q.update(scope=scope,sdk_initialization_mode=p.initialization_mode,GR00T_queries=0,physics_ticks=0,policy_episodes=0)
    qpath.write_text(json.dumps(q));m['execution']['sealed_inputs']['REQUEST.json']=unit.digest(qpath.read_bytes())
    m['execution']['sdk_initialization_mode']=p.initialization_mode;m['scope']=scope
    mpath.write_text(json.dumps(m));return replace(p,owner_manifest_sha256=unit.digest(mpath.read_bytes()))


def test_explicit_original_init_then_no_exec_requests_and_post_exit_cache_seal(pins,prediction,tmp_path,monkeypatch):
    p=initialize_pins(pins,tmp_path,monkeypatch);root=folder(tmp_path)
    client=unit.SdkClient(root,p,guard=lambda **_:None,maximum_requests=1)
    assert not(root/'SDK_CLIENT_CACHE_SEALED.json').exists()
    client.exchange(prediction);client.close()
    proof=json.loads((root/'SDK_CLIENT_CACHE_SEALED.json').read_bytes())
    assert proof['source']=='EXPLICIT_ORIGINAL_SDK_INITIALIZATION_AFTER_NATURAL_EXIT'
    assert proof['original_task_sha256']==unit.FROZEN_TASK_SHA256 and len(proof['members'])==1
    assert proof['resource_settlement_or_lease_closure'] is proof['native_prefix_safety_qualified'] is False
    cache=Path(p.private_readonly_cache)
    try:assert not(cache.stat().st_mode & 0o222)
    finally:
        cache.chmod(0o755)
        for f in cache.iterdir():f.chmod(0o644)


def test_task_cannot_enable_cache_compilation_or_reuse_nonempty_init(pins,tmp_path,monkeypatch):
    p=initialize_pins(pins,tmp_path,monkeypatch,scope=unit.TASK_ROLE)
    monkeypatch.setattr(unit.subprocess,'Popen',lambda *a,**kw:pytest.fail('compiler mode in task role'))
    with pytest.raises(PermissionError,match='COLD_CPU_QUALIFICATION_ONLY'):
        unit.SdkClient(folder(tmp_path),p,guard=lambda **_:None,maximum_requests=1)


def test_sdk_cannot_inherit_simulation_interposers(pins,tmp_path,monkeypatch):
    monkeypatch.setenv('LD_PRELOAD','/CPU_FIXTURE_NONEXISTENT_SIM_INTERPOSER.so')
    monkeypatch.setenv('LD_AUDIT','/CPU_FIXTURE_NONEXISTENT_AUDITOR.so')
    root=folder(tmp_path);client=unit.SdkClient(root,pins,guard=lambda **_:None,maximum_requests=1)
    client.close()
    assert 'cannot be preloaded' not in (root/'SDK_CLIENT.log').read_text()
    assert 'cannot be loaded as audit interface' not in (root/'SDK_CLIENT.log').read_text()
    assert json.loads((root/'SDK_CLIENT_STARTED.json').read_bytes())['simulation_preloads_removed'] is True


def test_no_new_request_after_finite_cap_and_no_reopen(pins, prediction, tmp_path):
    root=folder(tmp_path);client=unit.SdkClient(root,pins,guard=lambda **_:None,maximum_requests=1)
    client.exchange(prediction)
    with pytest.raises(PermissionError):client.exchange(prediction)
    client.close()
    with pytest.raises(FileExistsError):unit.SdkClient(root,pins,guard=lambda **_:None,maximum_requests=1)
    assert len(list(root.glob('SDK_REQUEST_*_INTENT.json')))==1


@pytest.mark.parametrize('change',['task','archive','binary','writable_cache','extra_cache','escape','frame'])
def test_changed_inputs_reject_before_process_start(pins,tmp_path,monkeypatch,change):
    def forbidden(*a,**kw):raise AssertionError('Popen before exact pins')
    monkeypatch.setattr(unit.subprocess,'Popen',forbidden)
    p=pins
    if change=='task':Path(p.client_task).write_bytes(b'CHANGED')
    elif change=='archive':Path(p.core_archive).write_bytes(b'CHANGED')
    elif change=='binary':p=replace(p,executable_sha256='0'*64)
    elif change=='writable_cache':Path(p.private_readonly_cache).chmod(0o755)
    elif change=='extra_cache':
        cache=Path(p.private_readonly_cache);cache.chmod(0o755);(cache/'extra').write_bytes(b'X');cache.chmod(0o555)
    elif change=='escape':p=replace(p,cache_members=(('../escape','0'*64),))
    elif change=='frame':p=replace(p,frames=('bad\x00frame','-','-'))
    with pytest.raises(PermissionError):unit.SdkClient(folder(tmp_path),p,guard=lambda **_:None,maximum_requests=1)


def test_wrong_native_model_fails_and_closes_without_fake_settlement(pins,tmp_path):
    root=folder(tmp_path)
    with pytest.raises(PermissionError):
        unit.SdkClient(root,replace(pins,frames=('CPU_TEST_BAD_MODEL','-','-')),guard=lambda **_:None,maximum_requests=1)
    result=json.loads((root/'SDK_CLIENT_EXIT.json').read_bytes())
    assert result['returncode']==1 and result['failed_or_unknown'] is True
    assert result['constructor_and_CPU_settled'] is result['lease_closed'] is False
    assert not(root/'SDK_CLIENT_READY.json').exists()


@pytest.mark.parametrize('change',['old_scope','wrong_request','reset_clock'])
def test_even_sealed_old_or_mismatched_child_cannot_start_client(pins,tmp_path,monkeypatch,change):
    mpath=Path(pins.owner_manifest);qpath=mpath.parent/'REQUEST.json'
    m=json.loads(mpath.read_bytes());q=json.loads(qpath.read_bytes());p=pins
    if change=='old_scope':m['scope']=q['scope']='OLD_EXPIRED_R4_PILOT'
    elif change=='wrong_request':p=replace(p,owner_request_id='87654321-1234-4123-8123-123456789abc')
    elif change=='reset_clock':m['limits']['PV1_T0_epoch']=q['original_T0_epoch']=unit.T0+1
    qpath.write_text(json.dumps(q));m['execution']['sealed_inputs']['REQUEST.json']=unit.digest(qpath.read_bytes())
    mpath.write_text(json.dumps(m));p=replace(p,owner_manifest_sha256=unit.digest(mpath.read_bytes()))
    monkeypatch.setattr(unit.subprocess,'Popen',lambda *a,**kw:pytest.fail('Popen before current identity'))
    with pytest.raises(PermissionError):unit.SdkClient(folder(tmp_path),p,guard=lambda **_:None,maximum_requests=1)


def test_bad_policy_before_intent_does_not_touch_native_client(pins,prediction,tmp_path):
    root=folder(tmp_path);client=unit.SdkClient(root,pins,guard=lambda **_:None,maximum_requests=1)
    try:
        with pytest.raises(PermissionError):client.exchange(replace(prediction,time_trajectory=(False,.2,.4)))
        assert not list(root.glob('SDK_REQUEST_*_INTENT.json'))
        client.exchange(prediction)
    finally:client.close()


def test_cutoff_crossed_during_pin_read_cannot_start_constructor(pins,tmp_path,monkeypatch):
    clock={'now':unit.T0+1000};original=unit.validate_pins
    monkeypatch.setattr(unit,'time',SimpleNamespace(time=lambda:clock['now'],monotonic=time.monotonic))
    def validated(value):
        result=original(value);clock['now']=unit.STOP_NEW;return result
    monkeypatch.setattr(unit,'validate_pins',validated)
    monkeypatch.setattr(unit.subprocess,'Popen',lambda *a,**kw:pytest.fail('new constructor after cutoff'))
    root=folder(tmp_path)
    with pytest.raises(PermissionError,match='FRESH_NO_NEW_CONSTRUCTOR'):
        unit.SdkClient(root,pins,guard=lambda **_:None,maximum_requests=1)
    assert not(root/'SDK_CLIENT_STARTED.json').exists()


def transport_stub(tmp_path,pins,code):
    p=tmp_path/'explicit_CPU_transport_fixture'
    p.write_text('#!/usr/bin/env python3\nimport os,sys,struct,time\nfd=int(sys.argv[7])\n'+code)
    p.chmod(0o755)
    return replace(pins,executable=str(p),executable_sha256=unit.digest(p.read_bytes()))


@pytest.mark.parametrize('fault',['truncated','wrong_version','timeout','nan'])
def test_unknown_reply_permanently_latches_and_never_resends(pins,prediction,tmp_path,fault):
    code="os.write(fd,b'D1SDKI01'+struct.pack('<III',12,12,3))\nraw=os.read(0,100000)\n"
    if fault=='truncated':code+="os.write(fd,b'D1SDKR01'+struct.pack('<I',3))\n"
    elif fault=='wrong_version':code+="os.write(fd,b'WRONGTAG'+struct.pack('<I',3))\n"
    elif fault=='timeout':code+='time.sleep(.3)\n'
    elif fault=='nan':code+="os.write(fd,b'D1SDKR01'+struct.pack('<I',3)+b''.join(struct.pack('<ddId',t,float('nan'),1,.01) for t in (0.,.2,.4)))\n"
    root=folder(tmp_path);client=unit.SdkClient(root,transport_stub(tmp_path,pins,code),
                                               guard=lambda **_:None,maximum_requests=2,timeout_s=.08)
    try:
        with pytest.raises((PermissionError,ValueError)):client.exchange(prediction)
        with pytest.raises(PermissionError):client.exchange(prediction)
        assert len(list(root.glob('SDK_REQUEST_*_INTENT.json')))==1
        assert not list(root.glob('*_CONFIRMED.json'))
    finally:client.close()
    assert json.loads((root/'SDK_CLIENT_EXIT.json').read_bytes())['failed_or_unknown'] is True


def test_guard_denial_still_cleans_owned_child(pins,tmp_path):
    root=folder(tmp_path);state={'deny':False}
    def guard(**kw):
        if state['deny']:raise PermissionError('CPU_FIXTURE_CURRENT_ENTRY_REVOKED')
    client=unit.SdkClient(root,pins,guard=guard,maximum_requests=1)
    state['deny']=True
    with pytest.raises(PermissionError):client.close()
    assert client.closed and client.process.poll() is not None


def test_other_thread_cannot_exchange_or_kill_owner(pins,prediction,tmp_path):
    client=unit.SdkClient(folder(tmp_path),pins,guard=lambda **_:None,maximum_requests=1);errors=[]
    def other():
        for f in (lambda:client.exchange(prediction),client.close):
            try:f()
            except PermissionError:errors.append(True)
    thread=threading.Thread(target=other);thread.start();thread.join()
    try:assert len(errors)==2;client.exchange(prediction)
    finally:client.close()
