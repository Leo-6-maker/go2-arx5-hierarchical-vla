"""Synthetic single-pose CPU fixtures; not native qualification.

The native clone/fingerprint/observed_state/bind_install primitives are mocked.
No MuJoCo instance, physics or neural model is constructed by these fixtures.
These tests run under a bounded CPU envelope without remote dispatch.
"""
from copy import deepcopy
import hashlib
from unittest.mock import patch

import numpy as np
import pytest
from scripts.pi0294.dec1 import source_single_pose_prefix as unit
from go2arx5_vla.adapters.execution_progress import ExecutionProgressAdapter,ProgressConfig
from go2arx5_vla.adapters.groot_to_wbc import ReferenceAdapterConfig
from go2arx5_vla.adapters.execution_progress_binding import NativeStopReceipt


def raw(stationary=False):
    values=np.zeros((40,10));values[:,0]=.4;values[:,2]=.4
    if not stationary:values[:,0]+=np.arange(40)*.002
    values[:,3]=1.;values[:,7]=1.;values[:,9]=.5
    if stationary:values[3,9]=.2;values[4:,9]=.8
    return values


class FakeEngine:
    def __init__(self):
        self.steps=0;self.tcp=np.eye(4);self.tcp[:3,3]=[.4,0.,.4]
        self.grip=.5;self.safety=[];self.role='live';self.closed=False
        self.reject_tick=None;self.raise_before_tick=None;self.live_drift_tick=None
        self.clock_shift_at=None
    def snapshot(self):
        now=self.steps*.002
        if self.clock_shift_at is not None and self.steps>=self.clock_shift_at:now+=.08
        return dict(time_s=now,tcp=self.tcp.copy(),gripper_fraction=self.grip,
            object_pose='FORBIDDEN_PHASE_INPUT',teacher_phase='FORBIDDEN_PHASE_INPUT',task='FORBIDDEN_PHASE_INPUT')
    def step(self,target,opening):
        assert np.asarray(target).shape==(4,4)  # Never accepts preview9.
        if self.role=='shadow' and self.steps+1==self.raise_before_tick:
            raise RuntimeError('NATIVE_STEP_RETURNED_NO_RELIABLE_COUNTER')
        self.tcp=np.asarray(target).copy();self.grip=float(opening);self.steps+=1
        if self.role=='shadow' and self.steps==self.reject_tick:
            self.safety.append('ORIGINAL_NATIVE_FAILURE')
            raise RuntimeError('ORIGINAL_NATIVE_FAILURE')  # Actual Engine.step raises after charging the tick.
        if self.role=='live' and self.steps==self.live_drift_tick:self.tcp[0,3]+=1e-5
    def close(self):self.closed=True


class Harness:
    def __init__(self,mode='I_PROGRESS'):
        self.engine=FakeEngine();self.clones=[];self.events=[];self.charges=[];self.finishes=[];self.reservations=[];self.stops=[]
        self.rng_identity=[0];self.change_rng_on_clone=False;self.fail_finish=False;self.fail_reserve=False;self.stop_ack=True
        self.gate_reject_time=None;self.gates=[]
        adapter=ExecutionProgressAdapter(ReferenceAdapterConfig(unit.FRAME,unit.FRAME,unit.TCP,stale_timeout_s=3.),
            ProgressConfig(),mode=mode,episode_id='CPU_SOURCE_EPISODE',
            validate_raw=lambda c:None,validate_mapped_full=lambda p:None,validate_prefix=lambda p:None)
        self.executor=unit.SinglePosePrefixExecutor(self.engine,adapter,executor_id='CPU_SINGLE_POSE',
            guard=lambda:None,reserve=self.reserve,charge=self.charge,finish=self.finish,
            record=self.events.append,stop=self.stop)
    def clone(self,engine):
        assert engine is self.engine
        copy=deepcopy(engine);copy.role='shadow';self.clones.append(copy)
        if self.change_rng_on_clone:self.rng_identity[0]+=1
        return copy
    def observed(self,engine):return np.r_[engine.tcp.ravel(),engine.grip,engine.steps]
    def fingerprint(self,engine):
        return hashlib.sha256(self.observed(engine).tobytes()+str(self.rng_identity[0]).encode()).hexdigest()
    def bind(self,engine,poses,opening,forecast,nonce):
        assert forecast['decision']=='PASS' and nonce==forecast['nonce']
        if forecast['state_sha256']!=self.fingerprint(engine):raise PermissionError('ORIGINAL_BIND_STATE_CHANGED')
        if forecast['action_sha256']!=hashlib.sha256(poses.tobytes()+opening.tobytes()).hexdigest():
            raise PermissionError('ORIGINAL_BIND_ACTION_CHANGED')
    def gate(self,action,state,time,identifier):
        assert np.asarray(action).shape==(40,10) and np.asarray(state).shape==(10,)
        self.gates.append((identifier,time,state.copy()))
        bad=(np.max(np.abs(action[:,:3]))>2 or self.gate_reject_time is not None and
            identifier.startswith('CPU_SOURCE_EPISODE:') and time>=self.gate_reject_time)
        return dict(decision='REJECT' if bad else 'ACCEPT')
    def reserve(self,kind,ticks,clones):
        self.reservations.append((kind,ticks,clones))
        if self.fail_reserve:raise RuntimeError('RESERVATION_REPLY_UNKNOWN')
        return kind+':'+str(len(self.reservations))
    def charge(self,kind,count,unknown,context):self.charges.append((kind,count,unknown,context))
    def finish(self,kind,reservation,count,unknown,status):
        self.finishes.append((kind,reservation,count,unknown,status))
        if self.fail_finish:raise RuntimeError('FINISH_WRITE_FAILED')
    def stop(self,reason):
        self.stops.append(reason)
        return NativeStopReceipt('CPU_SOURCE_EPISODE','CPU_SINGLE_POSE','actual-stop-ack',self.stop_ack)
    def patches(self):
        return [patch.object(unit.native,'clone',side_effect=self.clone),
            patch.object(unit.native,'fingerprint',side_effect=self.fingerprint),
            patch.object(unit.native,'observed_state',side_effect=self.observed),
            patch.object(unit.native,'bind_install',side_effect=self.bind),
            patch.object(unit.native,'predict',side_effect=AssertionError('NO_NESTED_PREFIX_PREDICTOR')),
            patch.object(unit,'gate_action',side_effect=self.gate)]


@pytest.fixture
def harness():
    objects=[]
    def make(mode='I_PROGRESS'):
        h=Harness(mode);started=h.patches()
        for p in started:p.start()
        objects.extend(started);return h
    yield make
    for p in reversed(objects):p.stop()


def prepare(h,values=None,chunk_id='raw-0'):
    return h.executor.prepare(raw() if values is None else values,chunk_id=chunk_id,
        observation_timestamp=h.engine.snapshot()['time_s'],
        point_timestamps=h.engine.snapshot()['time_s']+np.arange(40)*.05)


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
def test_one_clone_200_shadow_ticks_then_same_eight_actual_proposals_200_live(harness,mode):
    h=harness(mode);before=h.fingerprint(h.engine)
    result=prepare(h)
    assert result['shadow_ticks']==200 and result['clone_attempts']==1
    assert len(h.clones)==1 and h.clones[0].steps==200 and h.clones[0].closed
    assert h.engine.steps==0 and h.fingerprint(h.engine)==before
    assert len(h.executor.pending['records'])==8
    assert all(len(r['expected_states'])==25 for r in h.executor.pending['records'])
    actual=h.executor.execute_prefix(result['nonce'])
    assert actual['actual_ticks']==200 and h.engine.steps==200
    assert h.executor.state()['known_ticks']==dict(shadow=200,live=200)
    assert h.executor.state()['unknown']==dict(shadow=False,live=False)
    assert h.reservations==[('shadow',200,1),('live',200,0)]
    assert sum(v[1] for v in h.charges if v[0]=='shadow')==200
    assert sum(v[1] for v in h.charges if v[0]=='live')==200
    assert len([v for v in h.events if v['kind']=='SOURCE_REFERENCE_ACK'])==8
    assert len([v for v in h.events if v['kind']=='SOURCE_PREFIX200_ACK'])==1
    assert not h.stops


def test_stationary_gripper_events_keep_original_order_without_object_or_teacher_inputs(harness):
    h=harness('I_PROGRESS');result=prepare(h,raw(stationary=True))
    planned=[r['opening'] for r in h.executor.pending['records']]
    assert planned==[.5,.5,.5,.2,.8,.8,.8,.8]
    h.executor.execute_prefix(result['nonce'])
    assert all(state.shape==(10,) for _,_,state in h.gates)


def test_illegal_raw_tail_stops_before_clone_or_live(harness):
    h=harness();bad=raw();bad[-1,0]=9.
    with pytest.raises(PermissionError):prepare(h,bad)
    assert not h.clones and h.engine.steps==0 and len(h.stops)==1
    assert h.executor.state()['known_ticks']==dict(shadow=0,live=0)


def test_mapped_H40_rejection_mid_shadow_keeps_50_known_ticks_and_no_live(harness):
    h=harness();h.gate_reject_time=.1
    with pytest.raises(PermissionError):prepare(h)
    assert h.executor.state()['known_ticks']['shadow']==50 and h.engine.steps==0
    assert h.executor.state()['unknown']['shadow'] is False
    assert h.finishes[-1][2:]==(50,False,'REFERENCE_REJECT') and len(h.stops)==1


def test_known_native_shadow_reject_keeps_37_ticks_without_unknown_or_live(harness):
    h=harness();h.engine.reject_tick=37
    with pytest.raises(PermissionError):prepare(h)
    assert h.executor.state()['known_ticks']['shadow']==37
    assert not h.executor.state()['unknown']['shadow'] and h.engine.steps==0
    assert h.finishes[-1][2:]==(37,False,'S1_REJECT') and len(h.stops)==1
    with pytest.raises(PermissionError,match='LATCHED'):prepare(h)
    assert len(h.clones)==1 and len(h.stops)==1


def test_native_counter_uncertainty_retains_reservation_and_partial_lower_bound(harness):
    h=harness();h.engine.raise_before_tick=7
    with pytest.raises(PermissionError,match='ACCOUNTING_UNCERTAIN'):prepare(h)
    state=h.executor.state()
    assert state['known_ticks']['shadow']==6 and state['unknown']['shadow'] is True
    assert any(v[1]==0 and v[2] is True for v in h.charges)
    assert h.engine.steps==0 and len(h.stops)==1


def test_live_parity_difference_stops_after_exactly11_charged_ticks(harness):
    h=harness();h.engine.live_drift_tick=11
    result=prepare(h)
    with pytest.raises(PermissionError,match='PARITY_MISMATCH'):h.executor.execute_prefix(result['nonce'])
    assert h.executor.state()['known_ticks']==dict(shadow=200,live=11)
    assert h.finishes[-1][2:]==(11,False,'NATIVE_SAFETY_ERROR')
    assert len(h.stops)==1
    with pytest.raises(PermissionError,match='LATCHED'):h.executor.execute_prefix(result['nonce'])
    assert h.engine.steps==11 and len(h.stops)==1


def test_live_proposal_digest_mismatch_rejects_before_first_live_tick(harness):
    h=harness();result=prepare(h)
    h.executor.pending['records'][0]['digest']='foreign-proposal'
    with pytest.raises(PermissionError,match='PROPOSAL_DIFFERS'):h.executor.execute_prefix(result['nonce'])
    assert h.engine.steps==0 and h.executor.state()['known_ticks']['live']==0 and len(h.stops)==1


def test_shadow_clock_gap_rejects_first_reference_ACK_after25_known_ticks(harness):
    h=harness();h.engine.clock_shift_at=25
    with pytest.raises(PermissionError):prepare(h)
    assert h.executor.state()['known_ticks']['shadow']==25 and h.engine.steps==0
    assert len(h.stops)==1


def test_clone_changing_live_rng_fingerprint_is_not_repaired_or_hidden(harness):
    h=harness();h.change_rng_on_clone=True
    with pytest.raises(PermissionError,match='LIVE_OR_RNG'):prepare(h)
    assert h.executor.state()['unknown']['live'] and h.engine.steps==0
    assert h.clones[0].closed and len(h.stops)==1


def test_actual_stop_ACK_missing_stays_latched_not_a_successful_stop(harness):
    h=harness();h.engine.reject_tick=37;h.stop_ack=False
    with pytest.raises(PermissionError,match='STOP_ACK_MISSING'):prepare(h)
    assert h.executor.state()['stopped'] and not h.executor.state()['stop_confirmed']
    assert h.executor.state()['known_ticks']['shadow']==37
    with pytest.raises(PermissionError,match='LATCHED'):prepare(h)
    assert len(h.stops)==1


def test_reservation_reply_unknown_never_enters_clone_or_live(harness):
    h=harness();h.fail_reserve=True
    with pytest.raises(RuntimeError):prepare(h)
    assert not h.clones and h.executor.state()['unknown']['shadow']
    assert h.engine.steps==0 and len(h.stops)==1


def test_finish_write_failure_latches_and_preserves_unknown_accounting(harness):
    h=harness();h.fail_finish=True
    with pytest.raises(RuntimeError,match='FINISH_WRITE_FAILED'):prepare(h)
    assert h.executor.state()['unknown']['shadow'] and h.executor.state()['stopped']
    assert h.executor.state()['known_ticks']['shadow']==200 and h.engine.steps==0
    assert len(h.stops)==1 and h.clones[0].closed


def test_wrong_nonce_does_not_replay_completed_shadow(harness):
    h=harness();result=prepare(h)
    with pytest.raises(PermissionError,match='EXACT_PENDING'):h.executor.execute_prefix(result['nonce']+'wrong')
    assert len(h.clones)==1 and h.engine.steps==0 and len(h.stops)==1


def test_invalid_native_observed_state_retains_charged_ticks_and_unknown_observation(harness):
    h=harness();original=h.observed
    def invalid(engine):
        return np.array([np.nan]) if engine.role=='shadow' and engine.steps==3 else original(engine)
    with patch.object(unit.native,'observed_state',side_effect=invalid):
        with pytest.raises(PermissionError,match='OBSERVED_STATE'):prepare(h)
    assert h.executor.state()['known_ticks']['shadow']==3
    assert h.executor.state()['unknown']['shadow'] and h.engine.steps==0
    assert len(h.stops)==1 and h.clones[0].closed


def test_authority_guard_error_stops_before_shadow_or_live(harness):
    h=harness()
    def refused():raise PermissionError('ORIGINAL_LEASE_OR_BUDGET_GUARD')
    h.executor.guard=refused
    with pytest.raises(PermissionError,match='LEASE_OR_BUDGET'):prepare(h)
    assert not h.clones and h.engine.steps==0 and h.executor.state()['stopped']
    assert h.executor.state()['stop_confirmed'] and len(h.stops)==1


@pytest.mark.parametrize('changed',['lambda','offsets'])
def test_frozen_shared_reference_parameters_cannot_change(changed):
    from dataclasses import replace
    h=Harness();reference=h.executor.adapter.reference
    kwargs={'lambda_m_per_rad':.2} if changed=='lambda' else {'preview_offsets_m':np.linspace(0,.5,9)}
    h.executor.adapter.reference=replace(reference,**kwargs)
    with pytest.raises(PermissionError,match='FIXED_CONFIG'):
        unit.SinglePosePrefixExecutor(h.engine,h.executor.adapter,executor_id='CPU_SINGLE_POSE',
            guard=lambda:None,reserve=h.reserve,charge=h.charge,finish=h.finish,record=h.events.append,stop=h.stop)
    assert not h.clones and h.engine.steps==0
