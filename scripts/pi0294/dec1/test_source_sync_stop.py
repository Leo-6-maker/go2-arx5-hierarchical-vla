"""Synthetic CPU fixtures, independently executed by the owner. No MuJoCo/model/native stop qualification.

Original numerical primitives are fakes; tests exercise ownership/IO/lifecycle.
No background integrator or native engine is created by these fixtures.
"""
from copy import deepcopy
import hashlib
import json
import pickle
from types import SimpleNamespace
import threading
import weakref
import pytest
from scripts.pi0292.durable import read,write
from scripts.pi0294.dec1 import source_sync_stop as unit
from go2arx5_vla.adapters.execution_progress_binding import NativeStopReceipt

HOOKS=weakref.WeakKeyDictionary()  # Never stored in the fake engine's dictionary.


class FakeEngine:
    def __init__(self):
        self.mode='normal';self.model=SimpleNamespace(opt=SimpleNamespace(timestep=.002))
        self.data=SimpleNamespace(time=0.);self.ik_data=SimpleNamespace(time=0.)
        self.steps=0;self.renderer=None;self.raw_calls=0;self.safety=[];self.failure=None
    def step(self,target,opening):
        self.raw_calls+=1
        if self in HOOKS:HOOKS[self]()
        if self.failure=='before':raise RuntimeError('FAKE_ERROR_BEFORE_RELIABLE_ADVANCE')
        self.data.time+=.002;self.steps+=1
        if self.failure=='after':
            self.safety.append('ORIGINAL_FAKE_S1');raise RuntimeError('ORIGINAL_FAKE_S1')
    def snapshot(self):return dict(time_s=self.data.time)
    def close(self):self.renderer=None  # Like real close, this proves no stop.


def install(engine,tmp_path,**kwargs):
    fingerprint=kwargs.pop('fingerprint',fake_fingerprint)
    return unit.SourceSyncStopBarrier(engine,episode_id='synthetic-episode',executor_id='synthetic-source',
        receipt_path=tmp_path/'STOP.json',source_step_sha256='a'*64,
        expected_step=FakeEngine.step,expected_snapshot=FakeEngine.snapshot,fingerprint=fingerprint,**kwargs)


def fake_fingerprint(engine):
    # Same Python-instance-field exclusion as original S1; fake native state is
    # included separately. Locks/closures in engine.__dict__ would fail pickle.
    fields={k:v for k,v in engine.__dict__.items() if k not in ('model','data','ik_data','renderer')}
    return hashlib.sha256(pickle.dumps(fields)+pickle.dumps(engine.data)+pickle.dumps(engine.ik_data)).hexdigest()


def fake_original_clone(engine):
    # Original S1 object.__new__(type(engine)) and per-field copy semantics.
    result=object.__new__(type(engine))
    for key,value in engine.__dict__.items():
        if key=='model':setattr(result,key,value)
        elif key=='renderer':setattr(result,key,None)
        else:setattr(result,key,deepcopy(value))
    return result


def test_live_forwards_exact_original_once_then_real_clock_counter_and_durable_stop_readback(tmp_path):
    engine=FakeEngine();before=fake_fingerprint(engine);keys=tuple(engine.__dict__)
    barrier=install(engine,tmp_path)
    assert tuple(engine.__dict__)==keys and fake_fingerprint(engine)==before
    target=object();engine.step(target,.5)
    assert engine.steps==1 and engine.raw_calls==1 and engine.data.time==.002
    ack=barrier.stop('CPU_FIXTURE_STOP')
    assert isinstance(ack,NativeStopReceipt) and ack.stopped is True
    document=read(tmp_path/'STOP.json')
    assert document['native_steps']==1 and document['sim_time_s']==.002
    assert document['last_attempt']['native_delta']==1
    assert document['stopped'] and document['native_frame_inflight'] is False
    assert ack.receipt_id.endswith(hashlib.sha256((tmp_path/'STOP.json').read_bytes()).hexdigest())
    with pytest.raises(PermissionError,match='STOP_LATCHED'):engine.step(target,.5)
    assert engine.steps==1 and engine.raw_calls==1
    with pytest.raises(PermissionError,match='ALREADY_ATTEMPTED'):barrier.stop('NO_REACK')
    assert barrier.state()['stop_acknowledged'] and not barrier.state()['unknown']


def test_clone_and_fingerprint_ABI_unchanged_no_lock_or_stop_state_copied(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);before=fake_fingerprint(engine)
    clone=fake_original_clone(engine)
    assert type(clone) is type(engine) and clone.model is engine.model and clone.data is not engine.data
    clone.step(object(),.3)
    assert clone.steps==1 and engine.steps==0 and fake_fingerprint(engine)==before
    assert barrier.state()['last_attempt'] is None  # Clone did not touch live barrier.
    raw_calls=clone.raw_calls
    barrier.stop('NO_AUTOMATIC_CLONE_INTEGRATIONS')
    assert clone.raw_calls==raw_calls and clone.steps==1 and engine.steps==0
    # Closing original renderer is independent of stop and adds no integrations.
    clone.close();assert engine.steps==0


def test_native_safety_error_after_actual_integration_preserves_original_error_and_known_tick(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);engine.failure='after'
    with pytest.raises(RuntimeError,match='ORIGINAL_FAKE_S1'):engine.step(object(),.5)
    assert engine.steps==1 and barrier.state()['last_attempt']['native_delta']==1
    assert barrier.state()['latched'] and not barrier.state()['unknown']
    ack=barrier.stop('ORIGINAL_NATIVE_SAFETY')
    assert ack.stopped and read(tmp_path/'STOP.json')['native_steps']==1
    assert read(tmp_path/'STOP.json')['last_attempt']['native_error'].startswith('RuntimeError:')


def test_error_without_reliable_native_advance_remains_unknown_and_cannot_ACK(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);engine.failure='before'
    with pytest.raises(PermissionError,match='EFFECT_UNKNOWN'):engine.step(object(),.5)
    assert engine.steps==0 and barrier.state()['unknown']
    assert barrier.state()['last_attempt']['native_delta']==0
    with pytest.raises(PermissionError,match='UNKNOWN_NO_ACK'):barrier.stop('UNKNOWN')
    assert not (tmp_path/'STOP.json').exists() and not barrier.state()['stop_acknowledged']


def test_same_owner_inflight_stop_never_returns_ack_or_loses_final_native_counter(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);errors=[]
    def during_step():
        try:barrier.stop('INFLIGHT_IS_NOT_STOPPED')
        except PermissionError as error:errors.append(str(error))
    HOOKS[engine]=during_step
    with pytest.raises(PermissionError):engine.step(object(),.5)
    assert engine.steps==1 and engine.raw_calls==1
    assert errors and barrier.state()['last_attempt']['tick_after']==1
    assert barrier.state()['unknown'] and not barrier.state()['stop_acknowledged']
    assert not (tmp_path/'STOP.json').exists()
    with pytest.raises(PermissionError):barrier.stop('NO_RETRY')


def test_same_thread_reentrant_owned_step_is_rejected_before_second_native_call(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);errors=[]
    def reenter():
        try:engine.step(object(),.5)
        except PermissionError as error:errors.append(str(error))
    HOOKS[engine]=reenter
    with pytest.raises(PermissionError):engine.step(object(),.5)
    assert engine.raw_calls==1 and engine.steps==1 and errors
    with pytest.raises(PermissionError):barrier.stop('REENTRANT_NATIVE_EFFECTS_NOT_ACKED')
    assert not barrier.state()['stop_acknowledged']


@pytest.mark.parametrize('operation',['step','stop'])
def test_foreign_thread_cannot_advance_or_forge_stop_ack(tmp_path,operation):
    engine=FakeEngine();barrier=install(engine,tmp_path);errors=[]
    def foreign():
        try:
            if operation=='step':engine.step(object(),.5)
            else:barrier.stop('FOREIGN')
        except PermissionError as error:errors.append(str(error))
    thread=threading.Thread(target=foreign);thread.start();thread.join(timeout=1.)
    assert not thread.is_alive() and errors and engine.raw_calls==0 and engine.steps==0
    assert barrier.state()['unknown'] and not barrier.state()['stop_acknowledged']
    with pytest.raises(PermissionError):engine.step(object(),.5)
    assert not (tmp_path/'STOP.json').exists()


def test_second_owner_conflict_rejected_without_altering_first_barrier_or_original_class(tmp_path):
    engine=FakeEngine();original=FakeEngine.step;barrier=install(engine,tmp_path)
    with pytest.raises(PermissionError):install(engine,tmp_path)
    assert FakeEngine.step is original
    engine.step(object(),.5);barrier.stop('FIRST_OWNER_ONLY')
    assert engine.steps==1


@pytest.mark.parametrize('failure',['write-before','write-after','noop-write','reader-error','reader-corrupt','bytes-corrupt'])
def test_unknown_write_or_readback_never_retries_or_returns_ack(tmp_path,failure):
    calls=[]
    def writer(path,document):
        calls.append('write')
        if failure=='write-before':raise OSError('LOST_WRITE_OUTCOME')
        if failure=='noop-write':return
        write(path,document)
        if failure=='write-after':raise OSError('LOST_AFTER_DURABLE_WRITE')
        if failure=='bytes-corrupt':path.write_text(json.dumps(document,sort_keys=True))
    def reader(path):
        calls.append('read')
        if failure=='reader-error':raise OSError('READ_OUTCOME_UNKNOWN')
        result=read(path)
        if failure=='reader-corrupt':result['native_steps']=99
        return result
    engine=FakeEngine();barrier=install(engine,tmp_path,writer=writer,reader=reader)
    with pytest.raises((PermissionError,OSError)):barrier.stop('READBACK_OR_WRITE_UNKNOWN')
    assert barrier.state()['latched'] and barrier.state()['unknown'] and not barrier.state()['stop_acknowledged']
    original_calls=list(calls)
    with pytest.raises(PermissionError,match='ALREADY_ATTEMPTED'):barrier.stop('NO_RETRY')
    assert calls==original_calls and calls.count('write')==1
    with pytest.raises(PermissionError):engine.step(object(),.5)
    assert engine.steps==0


def test_unowned_counter_change_between_ticks_rejects_stop_and_keeps_latch(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);engine.steps=1;engine.data.time=.002
    with pytest.raises(PermissionError,match='DIFFERS_FROM_LAST_OWNED'):barrier.stop('FOREIGN_ADVANCE')
    assert not barrier.state()['stop_acknowledged'] and barrier.state()['unknown']


def test_live_counter_change_during_durable_readback_has_no_stop_ack(tmp_path):
    engine=FakeEngine()
    def reader(path):
        result=read(path);engine.steps+=1;engine.data.time+=.002
        return result
    barrier=install(engine,tmp_path,reader=reader)
    with pytest.raises(PermissionError):barrier.stop('BACKGROUND_EFFECT_DISALLOWED')
    assert not barrier.state()['stop_acknowledged'] and barrier.state()['unknown']
    with pytest.raises(PermissionError):engine.step(object(),.5)


def test_renderer_close_is_never_used_as_stop_evidence(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);engine.close()
    assert not barrier.state()['latched'] and not barrier.state()['stop_acknowledged']
    engine.step(object(),.5)
    assert engine.steps==1 and not (tmp_path/'STOP.json').exists()


def test_original_step_conflict_after_install_is_detected_before_any_native_advance(tmp_path,monkeypatch):
    engine=FakeEngine();barrier=install(engine,tmp_path)
    monkeypatch.setattr(FakeEngine,'step',lambda *a:None)
    with pytest.raises(PermissionError,match='BINDING_CHANGED'):engine.step(object(),.5)
    assert engine.steps==0 and barrier.state()['unknown']


def test_bad_clock_before_dispatch_is_rejected_without_calling_original_step(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);engine.data.time=float('nan')
    with pytest.raises(PermissionError,match='CLOCK_CHANGE'):engine.step(object(),.5)
    assert engine.raw_calls==0 and engine.steps==0 and barrier.state()['unknown']


def test_pid_change_never_forges_owner_stop_ACK(tmp_path,monkeypatch):
    engine=FakeEngine();barrier=install(engine,tmp_path);pid=unit.os.getpid()
    monkeypatch.setattr(unit.os,'getpid',lambda:pid+1)
    with pytest.raises(PermissionError,match='FOREIGN_PROCESS'):barrier.stop('NOT_OWN_PROCESS')
    assert not (tmp_path/'STOP.json').exists() and not barrier.state()['stop_acknowledged']


def test_preexisting_receipt_and_nonfresh_engine_rejected_before_class_change(tmp_path):
    engine=FakeEngine();write(tmp_path/'STOP.json',{'prior':'PRESERVE'})
    with pytest.raises(PermissionError):install(engine,tmp_path)
    assert type(engine) is FakeEngine
    other=tmp_path/'other';other.mkdir();engine.steps=1;engine.data.time=.002
    with pytest.raises(PermissionError):install(engine,other)
    assert type(engine) is FakeEngine


def test_unchanged_clock_but_native_state_mutation_during_IO_is_no_stop_ack(tmp_path):
    engine=FakeEngine()
    def reader(path):
        result=read(path);engine.ik_data.time+=.001  # Invisible to live counter/clock.
        return result
    barrier=install(engine,tmp_path,reader=reader)
    with pytest.raises(PermissionError,match='STATE_OR_RNG'):barrier.stop('ORIGINAL_FINGERPRINT_MUST_STAY_FIXED')
    assert engine.steps==0 and barrier.state()['unknown'] and not barrier.state()['stop_acknowledged']


def test_installation_uses_original_fingerprint_on_both_sides_of_class_forwarding(tmp_path):
    engine=FakeEngine();seen=[]
    def fingerprint(actual):
        seen.append((type(actual),fake_fingerprint(actual)))
        return seen[-1][1]
    barrier=install(engine,tmp_path,fingerprint=fingerprint)
    assert len(seen)==2 and seen[0][0] is FakeEngine and seen[1][0] is type(engine)
    assert seen[0][1]==seen[1][1]
    assert engine.steps==0 and engine.raw_calls==0 and not barrier.state()['unknown']


def test_install_fingerprint_mismatch_restores_class_but_never_restores_native_or_rng_state(tmp_path):
    engine=FakeEngine();calls=[]
    def changing_fingerprint(actual):
        calls.append(type(actual))
        if len(calls)==2:actual.ik_data.time=.001
        return fake_fingerprint(actual)
    with pytest.raises(PermissionError,match='INSTALLATION_CHANGED_ORIGINAL'):
        install(engine,tmp_path,fingerprint=changing_fingerprint)
    assert type(engine) is FakeEngine and engine.steps==0 and engine.raw_calls==0
    assert engine.ik_data.time==.001  # No repair to manufacture a matching hash.
    assert not (tmp_path/'STOP.json').exists()


def test_stop_never_calls_mutating_original_snapshot_or_changes_integration_fingerprint(tmp_path,monkeypatch):
    calls=[]
    def native_style_snapshot(engine):
        calls.append(engine.steps)
        engine.ik_data.time+=.001  # Stand-in for mj_forward warm-start mutation.
        return dict(time_s=engine.data.time)
    monkeypatch.setattr(FakeEngine,'snapshot',native_style_snapshot)
    engine=FakeEngine();barrier=install(engine,tmp_path)
    engine.failure='after'
    with pytest.raises(RuntimeError,match='ORIGINAL_FAKE_S1'):engine.step(object(),.5)
    before=fake_fingerprint(engine)
    ack=barrier.stop('KNOWN_NATIVE_REJECTION_NO_EXTRA_FORWARD')
    assert ack.stopped and calls==[] and fake_fingerprint(engine)==before
    assert barrier.state()['stop_acknowledged'] and not barrier.state()['unknown']
    assert read(tmp_path/'STOP.json')['native_state_sha256']==before


def test_stop_direct_clock_must_still_match_owned_integration_count(tmp_path):
    engine=FakeEngine();barrier=install(engine,tmp_path);engine.data.time=.002
    with pytest.raises(PermissionError,match='COUNTER_OR_CLOCK_NOT_QUIESCENT'):
        barrier.stop('UNOWNED_CLOCK_NO_ACK')
    assert barrier.state()['unknown'] and not barrier.state()['stop_acknowledged']


def test_stop_still_rejects_changed_frozen_snapshot_function_without_calling_it(tmp_path,monkeypatch):
    engine=FakeEngine();barrier=install(engine,tmp_path)
    monkeypatch.setattr(FakeEngine,'snapshot',lambda engine:dict(time_s=engine.data.time))
    with pytest.raises(PermissionError,match='BINDING_CHANGED'):
        barrier.stop('FROZEN_FUNCTION_IDENTITY_MUST_REMAIN')
    assert barrier.state()['unknown'] and not barrier.state()['stop_acknowledged']
