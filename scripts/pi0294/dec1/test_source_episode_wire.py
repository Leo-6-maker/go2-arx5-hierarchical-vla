"""Synthetic CPU fixtures; never evidence of native execution qualification.

Reuse the existing fake primitives only. No model, renderer or native physics.
For integration place source_episode_wire.py next to the existing DEC1 module.
"""
import numpy as np
import pytest
from scripts.pi0294.dec1 import source_episode_wire as wire
from scripts.pi0294.dec1.test_source_single_pose_prefix import harness,raw


def outer(harness,mode='I_PROGRESS'):
    h=harness(mode)
    h.executor=wire.OuterLoopSinglePosePrefixExecutor(h.engine,h.executor.adapter,
        executor_id='CPU_SINGLE_POSE',guard=lambda:None,reserve=h.reserve,charge=h.charge,
        finish=h.finish,record=h.events.append,stop=h.stop)
    return h


def start(h):
    proof=h.executor.prepare(raw(),chunk_id='raw-0',observation_timestamp=0.,
        point_timestamps=np.arange(40)*.05)
    h.executor.begin_prefix(proof['nonce'])
    return proof


def advance(h,target,opening,record_error=None):
    h.executor.before_native_tick(h.engine,target,opening)
    error=None
    try:h.engine.step(target,opening)  # Sole native advance, like original episode.
    except Exception as caught:error=caught
    h.executor.after_native_tick(h.engine,error,record_error)
    if error is not None:raise error


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
def test_hooks_never_advance_live_and_reuse_one200_shadow_for_all8_refs(harness,mode):
    h=outer(harness,mode);proof=start(h);rows=[]
    assert len(h.clones)==1 and h.clones[0].steps==200 and h.engine.steps==0
    for reference in range(8):
        before=h.engine.steps
        target,opening=h.executor.reference_action(reference)
        assert h.engine.steps==before and len(h.clones)==1
        for tick in range(25):
            advance(h,target,opening);rows.append(h.observed(h.engine))
    result=h.executor.complete_prefix(h.engine,rows)
    assert h.engine.steps==200 and result['actual_ticks']==200
    assert result['nonce']==proof['nonce'] and len(result['proposal_digests'])==8
    assert len([e for e in h.events if e['kind']=='SOURCE_REFERENCE_ACK'])==8
    assert h.executor.state()['known_ticks']==dict(shadow=200,live=200)
    assert h.finishes==[('shadow','shadow:1',200,False,'PASS'),('live','live:2',200,False,'PASS')]
    assert h.reservations==[('shadow',200,1),('live',200,0)] and not h.stops


def test_mapped_dispatch_mismatch_before_native_step_stops_with_known_zero_live(harness):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    target[0,3]+=1e-4
    with pytest.raises(PermissionError,match='DISPATCH_ACTION_DIFFERS'):
        h.executor.before_native_tick(h.engine,target,opening)
    assert h.engine.steps==0 and h.executor.state()['known_ticks']['live']==0
    assert h.finishes[-1][2:]==(0,False,'ENGINEERING_MISSING') and len(h.stops)==1


def test_actual_feedback_reanchors_ref1_instead_of_reusing_precomputed_target(harness):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    for tick in range(25):advance(h,target,opening)
    # Bring feedback inside the local projection window. The first spatial
    # preview is already .01m forward; further forward perturbations legitimately
    # keep the same upper-clamped phase and need not change its proposal.
    h.engine.tcp[0,3]=.4
    with pytest.raises(PermissionError,match='PROPOSAL_DIFFERS'):
        h.executor.reference_action(1)
    assert h.engine.steps==25 and h.executor.state()['known_ticks']['live']==25
    assert len(h.clones)==1 and len(h.stops)==1


def test_after_hook_preserves_native_runtime_error_safety37_as_known_effect(harness):
    h=outer(harness);start(h)
    for reference in range(2):
        target,opening=h.executor.reference_action(reference)
        for tick in range(25):
            if h.engine.steps==36:
                h.executor.before_native_tick(h.engine,target,opening)
                h.engine.step(target,opening);h.engine.safety.append('ORIGINAL_S1')
                with pytest.raises(PermissionError,match='ORIGINAL_NATIVE_S1_REJECT'):
                    h.executor.after_native_tick(h.engine,RuntimeError('NATIVE_SAFETY'))
                assert h.finishes[-1][2:]==(37,False,'NATIVE_SAFETY_ERROR')
                assert h.engine.steps==37 and len(h.stops)==1
                return
            advance(h,target,opening)
    raise AssertionError('fixture did not reach tick37')


def test_record_tick_failure_is_charged_known_step_and_retains_audit_uncertainty(harness):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    for tick in range(11):advance(h,target,opening)
    with pytest.raises(RuntimeError,match='RAW_RECORDER_FAILED'):
        advance(h,target,opening,RuntimeError('RAW_RECORDER_FAILED'))
    assert h.engine.steps==12 and h.executor.state()['known_ticks']['live']==12
    assert h.finishes[-1][2:]==(12,True,'ENGINEERING_MISSING') and len(h.stops)==1


def test_missing_after_hook_cannot_be_silently_retried_or_known_zero(harness):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    h.executor.before_native_tick(h.engine,target,opening);h.engine.step(target,opening)
    with pytest.raises(PermissionError,match='DUPLICATE_SKIPPED'):
        h.executor.before_native_tick(h.engine,target,opening)
    assert h.engine.steps==1 and h.executor.state()['unknown']['live'] is True
    # Last hook-confirmed lower bound is0; liability remains unknown, NOT knownzero.
    assert h.finishes[-1][2:]==(0,True,'ENGINEERING_MISSING') and len(h.stops)==1


def test_counter_uncertainty_after_external_engine_error_keeps6_known_ticks(harness):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    for tick in range(6):advance(h,target,opening)
    h.executor.before_native_tick(h.engine,target,opening)
    with pytest.raises(PermissionError,match='ACCOUNTING_UNCERTAIN'):
        h.executor.after_native_tick(h.engine,RuntimeError('STEP_FAILED_BEFORE_RELIABLE_ADVANCE'))
    assert h.engine.steps==6 and h.finishes[-1][2:]==(6,True,'ENGINEERING_MISSING')
    assert len(h.stops)==1


def test_live_parity_failure_at11_stops_before_next_reference_or_more_clones(harness):
    h=outer(harness);h.engine.live_drift_tick=11;start(h)
    target,opening=h.executor.reference_action(0)
    for tick in range(10):advance(h,target,opening)
    with pytest.raises(PermissionError,match='PARITY_MISMATCH'):advance(h,target,opening)
    assert h.finishes[-1][2:]==(11,False,'NATIVE_SAFETY_ERROR')
    assert h.engine.steps==11 and len(h.clones)==1 and len(h.stops)==1
    with pytest.raises(PermissionError,match='LATCHED'):h.executor.reference_action(1)
    assert h.engine.steps==11 and len(h.stops)==1


def test_internal200_tick_convenience_method_is_forbidden_for_episode_subclass(harness):
    h=outer(harness);prepared=h.executor.prepare(raw(),chunk_id='raw-0',
        observation_timestamp=0.,point_timestamps=np.arange(40)*.05)
    with pytest.raises(PermissionError,match='OUTER_EPISODE_OWNS'):
        h.executor.execute_prefix(prepared['nonce'])
    assert h.engine.steps==0 and len(h.clones)==1 and len(h.stops)==1


def test_old_raw_prefix_ACK_is_not_selected_by_default(harness):
    h=outer(harness)
    with pytest.raises(TypeError):wire.SourceInterfaceTeacher(object(),h.executor,persist=lambda *a:None)
    # Explicit protocol-aware owner callback remains a real integration dependency.


class FakeTeacher:
    def __init__(self):self.queries=0;self.old_acks=0;self.finished=[]
    def plan(self,state):
        raise AssertionError('FIXTURE_REQUIRES_EXPLICIT_RAW_PLAN')
    def acknowledge(self,*args):self.old_acks+=1;raise AssertionError('OLD_RAW_ACK_MUST_NOT_RUN')
    def finish(self,*args):self.finished.append(args)


@pytest.mark.parametrize('bad',['none','raw_hash','ticks','episode',None])
def test_teacher_emits_one_raw_plan_and_requires_exact_actual_mapped_ACK(harness,bad):
    from dataclasses import replace
    from scripts.pi0292.eval_contract import matrix10
    import hashlib
    h=outer(harness);base=FakeTeacher();persisted=[];acks=[];values=raw()
    plan=dict(poses=matrix10(values),opening=values[:,9],offset_s=np.arange(40)*.05,
        reference_uuid='wire-raw-0',stage='AUDIT_ONLY')
    def model_plan(state):base.queries+=1;return plan
    base.plan=model_plan
    def acknowledge(engine,rows,proof):
        acks.append(proof)
        ack=wire.NativeMappedPrefixAck(h.executor.episode_id,h.executor.executor_id,
            proof['nonce'],proof['raw_h40_sha256'],proof['action_sha256'],
            proof['tick_before'],proof['tick_after'],'CPU_FIXTURE_ACK',
            hashlib.sha256(np.asarray(rows).tobytes()).hexdigest(),hashlib.sha256(b'CPU_FIXTURE_NOT_NATIVE').hexdigest())
        if bad=='none':return None
        if bad=='raw_hash':return replace(ack,raw_h40_sha256='f'*64)
        if bad=='ticks':return replace(ack,tick_after=199)
        if bad=='episode':return replace(ack,episode_id='OTHER_EPISODE')
        return ack
    teacher=wire.SourceInterfaceTeacher(base,h.executor,acknowledge_source=acknowledge,
        persist=lambda *v:persisted.append(v))
    returned=teacher.plan(h.engine.snapshot());assert returned is plan and base.queries==1
    assert np.array_equal(returned['poses'],matrix10(values))
    rows=[]
    for ref in range(8):
        target,opening=teacher.reference_action(ref)
        for tick in range(25):
            teacher.before_native_tick(h.engine,target,opening);h.engine.step(target,opening)
            teacher.after_native_tick(h.engine);rows.append(h.observed(h.engine))
    if bad is None:
        teacher.acknowledge(h.engine,rows);teacher.finish('VALID_COMPLETE',None)
        assert not teacher.protocol_ack_unknown and persisted[-1][0]=='VALID_COMPLETE'
        assert not h.stops
    else:
        with pytest.raises(PermissionError,match='PROTOCOL_ACK_BINDING'):teacher.acknowledge(h.engine,rows)
        teacher.finish('VALID_COMPLETE',None)  # Cannot hide an unresolved ACK behind a success label.
        assert teacher.protocol_ack_unknown and persisted[-1][0]=='FAILED_PRESERVED'
        assert persisted[-1][2]['protocol_ack_unknown'] and base.finished[-1][0]=='FAILED_PRESERVED'
        assert h.executor.stopped and len(h.stops)==1
    assert base.old_acks==0 and len(acks)==1 and base.queries==1 and h.engine.steps==200
    assert len(h.clones)==1
