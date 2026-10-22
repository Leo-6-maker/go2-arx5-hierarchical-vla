"""Synthetic CPU interrupts, never native commissioning evidence.

The owner independently executed these beside the reviewed source modules.
Independent audit rejection is installed BEFORE any project imports.
"""
import sys

_AUDIT={'active':True,'blocked':[]}
def _deny_external(event,args):
    if _AUDIT['active'] and (event.startswith(('subprocess.','socket.')) or event in (
            'os.system','os.fork','os.forkpty','os.posix_spawn','os.spawn','pty.spawn')):
        _AUDIT['blocked'].append(event)
        raise PermissionError('CPU_INTERRUPT_FIXTURE_FORBIDS_EXTERNAL_EFFECT:'+event)
sys.addaudithook(_deny_external)

from unittest.mock import patch
import pytest
from scripts.pi0294.dec1 import source_single_pose_prefix as unit
from scripts.pi0294.dec1 import source_episode_wire as wire
from scripts.pi0294.dec1.test_source_single_pose_prefix import harness,prepare,FakeEngine
from scripts.pi0294.dec1.test_source_episode_wire import outer,start,advance,FakeTeacher

assert not _AUDIT['blocked'],'PROJECT_IMPORT_ATTEMPTED_EXTERNAL_EFFECT'
_AUDIT['active']=False
@pytest.fixture(autouse=True)
def _isolated_cpu_boundary():
    before=len(_AUDIT['blocked']);_AUDIT['active']=True
    try:yield
    finally:
        _AUDIT['active']=False
        assert len(_AUDIT['blocked'])==before,'FORBIDDEN_EXTERNAL_ATTEMPT_WAS_NOT_A_MOCK'


def assert_latched(h):
    assert h.executor.stopped and h.executor.stop_calls==1
    with pytest.raises(PermissionError,match='LATCHED'):h.executor._open()
    assert h.executor.stop_calls==1


@pytest.mark.parametrize('interrupt',[KeyboardInterrupt,SystemExit])
@pytest.mark.parametrize('integrated',[False,True])
def test_shadow_native_interrupt_accounts_actual_counter_before_unwinding(harness,interrupt,integrated):
    h=harness();original=FakeEngine.step
    def interrupted(engine,target,opening):
        if engine.role=='shadow':
            if integrated:original(engine,target,opening)
            raise interrupt('NATIVE_STEP_INTERRUPTED')
        return original(engine,target,opening)
    with patch.object(FakeEngine,'step',interrupted):
        with pytest.raises(interrupt):prepare(h)
    n=int(integrated)
    assert h.clones[0].steps==n and h.clones[0].closed and h.engine.steps==0
    assert h.executor.counts['shadow']==n and h.executor.unknown['shadow'] is (not integrated)
    assert h.finishes[-1][2:]==(n,not integrated,'ENGINEERING_MISSING')
    assert sum(c[1] for c in h.charges)==n
    if not integrated:assert h.charges[-1][2] is True  # NOT zero-effect proof.
    assert_latched(h)


@pytest.mark.parametrize('interrupt',[KeyboardInterrupt,SystemExit])
def test_internal_live_native_interrupt_retains_one_real_integration(harness,interrupt):
    h=harness();proof=prepare(h);original=h.engine.step
    def interrupted(target,opening):original(target,opening);raise interrupt('LIVE_STEP')
    h.engine.step=interrupted
    with pytest.raises(interrupt):h.executor.execute_prefix(proof['nonce'])
    assert h.engine.steps==1 and h.executor.counts==dict(shadow=200,live=1)
    assert h.finishes[-1][2:]==(1,False,'ENGINEERING_MISSING')
    assert_latched(h)


def test_reserve_persisted_then_interrupt_is_unknown_without_clone_or_retry(harness):
    h=harness()
    def interrupted(*args):h.reserve(*args);raise KeyboardInterrupt('RESERVE_REPLY_UNKNOWN')
    h.executor.reserve=interrupted
    with pytest.raises(KeyboardInterrupt):prepare(h)
    assert len(h.reservations)==1 and not h.clones and not h.finishes
    assert h.executor.unknown['shadow'] and h.executor.counts['shadow']==0
    assert_latched(h)


@pytest.mark.parametrize('domain',['shadow','live'])
def test_durable_charge_then_interrupt_preserves_count_unknown_and_single_finish(harness,domain):
    h=harness();proof=prepare(h) if domain=='live' else None
    def interrupted(kind,n,unknown,context):
        h.charge(kind,n,unknown,context)
        if kind==domain:raise KeyboardInterrupt('CHARGE_REPLY_UNKNOWN')
    h.executor.charge=interrupted
    with pytest.raises(KeyboardInterrupt):
        h.executor.execute_prefix(proof['nonce']) if proof else prepare(h)
    assert h.executor.counts[domain]==1 and h.executor.unknown[domain]
    assert h.finishes[-1][0]==domain and h.finishes[-1][2:]==(1,True,'ENGINEERING_MISSING')
    assert len([c for c in h.charges if c[0]==domain])==1
    assert_latched(h)


def test_post_observation_interrupt_retains_charged_tick_and_unknown(harness):
    h=harness();original=h.observed
    def interrupted(engine):
        if engine.role=='shadow' and engine.steps==1:raise KeyboardInterrupt('OBSERVATION_UNKNOWN')
        return original(engine)
    with patch.object(unit.native,'observed_state',side_effect=interrupted):
        with pytest.raises(KeyboardInterrupt):prepare(h)
    assert h.finishes[-1][2:]==(1,True,'ENGINEERING_MISSING')
    assert h.clones[0].closed and h.executor.counts['shadow']==1
    assert_latched(h)


def test_shadow_cleanup_interrupt_still_finishes_unknown_once(harness):
    h=harness();original=FakeEngine.close
    def interrupted(engine):original(engine);raise KeyboardInterrupt('CLOSE_REPLY_UNKNOWN')
    with patch.object(FakeEngine,'close',interrupted):
        with pytest.raises(KeyboardInterrupt):prepare(h)
    assert h.clones[0].closed and h.finishes[-1][2:]==(200,True,'ENGINEERING_MISSING')
    assert len(h.finishes)==1 and h.executor.unknown['shadow']
    assert_latched(h)


def test_shadow_completion_evidence_interrupt_cannot_finish_pass(harness):
    h=harness()
    def interrupted(event):
        h.events.append(event)
        if event['kind']=='SOURCE_PREFIX_SHADOW_COMPLETE':raise KeyboardInterrupt('RECORD_REPLY_UNKNOWN')
    h.executor.record=interrupted
    with pytest.raises(KeyboardInterrupt):prepare(h)
    assert h.finishes[-1][2:]==(200,False,'ENGINEERING_MISSING')
    assert_latched(h)


def test_durable_shadow_finish_interrupt_is_unknown_and_never_repeated(harness):
    h=harness()
    def interrupted(*args):h.finish(*args);raise KeyboardInterrupt('FINISH_REPLY_UNKNOWN')
    h.executor.finish=interrupted
    with pytest.raises(KeyboardInterrupt):prepare(h)
    assert len(h.finishes)==1 and h.finishes[0][2]==200
    assert h.executor.unknown['shadow'] and h.executor.counts['shadow']==200
    assert_latched(h)
    assert len(h.finishes)==1  # A persisted PASS request is not a confirmed outcome.


@pytest.mark.parametrize('integrated',[False,True])
def test_outer_afterhook_native_interrupt_closes_lower_bound_without_success(harness,integrated):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    h.executor.before_native_tick(h.engine,target,opening)
    if integrated:h.engine.step(target,opening)
    with pytest.raises(KeyboardInterrupt):
        h.executor.after_native_tick(h.engine,KeyboardInterrupt('OUTER_STEP_INTERRUPTED'))
    n=int(integrated)
    assert h.finishes[-1][2:]==(n,not integrated,'ENGINEERING_MISSING')
    assert h.executor.state()['live_reservation_closed'] is True
    assert h.executor.counts['live']==n
    assert_latched(h)


def test_missing_afterhook_finally_keeps_unknown_zero_lower_bound_not_refund(harness):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    h.executor.before_native_tick(h.engine,target,opening)
    h.engine.step(target,opening)  # Original episode interrupted before after-hook.
    saved=[];base=FakeTeacher()
    teacher=wire.SourceInterfaceTeacher(base,h.executor,acknowledge_source=lambda *a:None,
        persist=lambda *a:saved.append(a))
    teacher.finish('VALID_COMPLETE',None)
    assert h.engine.steps==1 and h.executor.counts['live']==0
    assert h.finishes[-1][2:]==(0,True,'ENGINEERING_MISSING')
    assert saved[-1][0]=='FAILED_PRESERVED' and saved[-1][2]['unknown']['live']
    assert_latched(h)


def test_outer_charge_interrupt_after_persist_marks_unknown_and_finishes_once(harness):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    def interrupted(*args):h.charge(*args);raise KeyboardInterrupt('LIVE_CHARGE_UNKNOWN')
    h.executor.charge=interrupted
    h.executor.before_native_tick(h.engine,target,opening);h.engine.step(target,opening)
    with pytest.raises(KeyboardInterrupt):h.executor.after_native_tick(h.engine)
    assert h.finishes[-1][2:]==(1,True,'ENGINEERING_MISSING')
    assert h.executor.counts['live']==1 and len([f for f in h.finishes if f[0]=='live'])==1
    assert_latched(h)


def test_outer_finish_interrupt_is_attempted_not_confirmed_and_not_retried(harness):
    h=outer(harness);start(h);rows=[]
    for reference in range(8):
        target,opening=h.executor.reference_action(reference)
        for tick in range(25):advance(h,target,opening);rows.append(h.observed(h.engine))
    def interrupted(*args):h.finish(*args);raise KeyboardInterrupt('LIVE_FINISH_UNKNOWN')
    h.executor.finish=interrupted
    with pytest.raises(KeyboardInterrupt):h.executor.complete_prefix(h.engine,rows)
    state=h.executor.state()
    assert state['known_ticks']['live']==200 and state['unknown']['live']
    assert state['live_finish_attempted'] is True and state['live_reservation_closed'] is False
    assert len([f for f in h.finishes if f[0]=='live'])==1
    with pytest.raises(RuntimeError):h.executor.abort_outer(RuntimeError('NO_FINISH_RETRY'))
    assert len([f for f in h.finishes if f[0]=='live'])==1
    assert_latched(h)


def test_stop_interrupt_latches_without_false_stop_ack_and_still_finishes_shadow(harness):
    h=harness();h.engine.reject_tick=1
    def interrupted(reason):h.stops.append(reason);raise KeyboardInterrupt('STOP_OUTCOME_UNKNOWN')
    h.executor.stop=interrupted
    with pytest.raises(KeyboardInterrupt):prepare(h)
    assert h.executor.stopped and not h.executor.stop_confirmed and h.executor.unknown['live']
    assert h.finishes[-1][2]==1 and h.clones[0].closed
    assert_latched(h)


def test_owner_stop_ack_record_interrupt_is_unknown_not_confirmed_or_retried(harness):
    h=harness();h.engine.reject_tick=1
    def interrupted(event):
        h.events.append(event)
        if event['kind']=='SOURCE_STOP_ACK':raise KeyboardInterrupt('OWNER_STOP_RECORD_UNKNOWN')
    h.executor.record=interrupted
    with pytest.raises(KeyboardInterrupt):prepare(h)
    assert h.executor.stopped and not h.executor.stop_confirmed and h.executor.unknown['live']
    assert len(h.stops)==1 and h.clones[0].closed and h.executor.counts['shadow']==1
    assert_latched(h)


def test_teacher_cleanup_stop_interrupt_preserves_evidence_then_propagates(harness):
    h=outer(harness);start(h);target,opening=h.executor.reference_action(0)
    h.executor.before_native_tick(h.engine,target,opening);h.engine.step(target,opening)
    def interrupted(reason):h.stops.append(reason);raise KeyboardInterrupt('STOP_UNKNOWN')
    h.executor.stop=interrupted;saved=[];base=FakeTeacher()
    teacher=wire.SourceInterfaceTeacher(base,h.executor,acknowledge_source=lambda *a:None,
        persist=lambda *a:saved.append(a))
    with pytest.raises(KeyboardInterrupt):teacher.finish('VALID_COMPLETE',None)
    assert saved[-1][0]=='FAILED_PRESERVED' and base.finished[-1][0]=='FAILED_PRESERVED'
    assert h.finishes[-1][2:]==(0,True,'ENGINEERING_MISSING')
    assert h.executor.unknown['live'] and not h.executor.stop_confirmed
    assert_latched(h)


@pytest.mark.parametrize('callback',['persist','teacher_finish'])
def test_teacher_finish_durable_callback_interrupt_is_latched_not_retried(harness,callback):
    h=outer(harness);base=FakeTeacher();saved=[]
    def persist(*args):
        saved.append(args)
        if callback=='persist':raise KeyboardInterrupt('RESULT_PERSIST_UNKNOWN')
    def finish(*args):base.finished.append(args);raise KeyboardInterrupt('TEACHER_FINISH_UNKNOWN')
    if callback=='teacher_finish':base.finish=finish
    teacher=wire.SourceInterfaceTeacher(base,h.executor,acknowledge_source=lambda *a:None,persist=persist)
    with pytest.raises(KeyboardInterrupt):teacher.finish('FAILED_PRESERVED','ORIGINAL_ERROR')
    assert teacher.finish_unknown and h.executor.unknown['live']
    assert len(saved)==1 and len(base.finished)==1 and base.finished[-1][0]=='FAILED_PRESERVED'
    with pytest.raises(PermissionError,match='NO_RETRY'):teacher.finish('FAILED_PRESERVED','DO_NOT_RETRY')
    assert len(saved)==1 and len(base.finished)==1
    assert_latched(h)
