"""Targeted CPU integration fixtures; text prepared, NOT run by helper.

No MuJoCo/model/scheduler/admission. Real original Ledger, component Meters,
Evidence and component ACK constructors are used. The numerical engine/STOP
are explicit fakes: these fixtures qualify orchestration, not native physics.
The closing helper below is an illustrative owner-finally boundary, not a new
runner or proof that the future pv1.core registered branch already exists.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
from scripts.pi0292.durable import read, write
from scripts.pi0294.test_pv1_core_ledger import fixture as real_ledger
from scripts.pi0294.dec1.test_source_commissioning_plan import population
from scripts.pi0294.dec1 import source_component_factory as unit
from scripts.pi0294.dec1 import source_component_ack as ack_unit
from scripts.pi0294.dec1 import source_commissioning_plan, source_single_pose_prefix
from scripts.pi0294 import s1_preflight
from go2arx5_vla.adapters.execution_progress import ExecutionProgressAdapter, ProgressConfig
from go2arx5_vla.adapters.groot_to_wbc import ReferenceAdapterConfig


class FakeEngine:
    def __init__(self, card):
        self.card = deepcopy(card); self.steps = 0; self.safety = []
        self.data = SimpleNamespace(time=0.)
        self.model = SimpleNamespace(nq=2, nv=2, nu=1, opt=SimpleNamespace(timestep=.002))
        self.native_calls = self.renderer_closes = 0

    def step(self, target, opening):
        self.native_calls += 1
        raise AssertionError('FACTORY_OR_PERSIST_MUST_NOT_INTEGRATE')

    def snapshot(self):
        return dict(time_s=self.data.time, tcp=np.eye(4), gripper_fraction=.5)

    def close(self):
        self.renderer_closes += 1  # Explicitly no STOP or physics effect.


class FakeStopBarrier:
    """Synthetic typed-stop producer with real CPU receipt file/readback.

    Actual SourceSyncStopBarrier numerical/fingerprint behavior is covered by
    its separate targeted fixtures; this fake does not claim that qualification.
    """
    def __init__(self, engine, *, episode_id, executor_id, receipt_path, **pins):
        self.engine = engine; self.episode = episode_id; self.executor = executor_id
        self.path = receipt_path; self.calls = 0; self.attempted = False
        self.acknowledged = False; self.unknown = False; self.failure = None

    def stop(self, reason):
        self.calls += 1
        if self.attempted:
            raise AssertionError('NATIVE_STOP_MUST_NOT_BE_RETRIED')
        self.attempted = True
        write(self.path, dict(schema='SYNTHETIC_COMPONENT_STOP', reason=reason,
            steps=self.engine.steps, clock=self.engine.data.time, stopped=True))
        if self.failure == 'interrupt':
            self.unknown = True
            raise KeyboardInterrupt('ORIGINAL_STOP_INTERRUPT')
        if self.failure == 'ordinary':
            self.unknown = True
            raise OSError('SECONDARY_STOP_CLEANUP_ERROR')
        if self.failure == 'noop':
            return None
        self.acknowledged = True
        return unit.NativeStopReceipt(self.episode, self.executor,
            self.path.name + '#sha256=' + hashlib.sha256(self.path.read_bytes()).hexdigest(), True)

    def state(self):
        return dict(stop_attempted=self.attempted, stop_acknowledged=self.acknowledged,
            unknown=self.unknown, latched=self.attempted)


class OriginalTeacherSpy:
    def __init__(self):
        self.plan_calls = 0

    def plan(self, state):
        self.plan_calls += 1
        pose = np.eye(4); pose[:3, 3] = [.4, 0., .4]
        return dict(poses=np.repeat(pose[None], 40, axis=0), opening=np.full(40, .5),
            offset_s=np.arange(40) / 20, stage='SYNTHETIC_ORIGINAL_TEACHER')


def forbidden(*args, **kwargs):
    raise AssertionError('NO_NATIVE_CLONE_PREDICT_OR_UNBOUND_SAFETY_CALL_IN_FACTORY_FIXTURES')


@pytest.fixture
def case(population, tmp_path, monkeypatch):
    data, encode = population
    population_bytes = encode(data)
    plan = source_commissioning_plan.build(population_bytes, owner_release='a' * 40)
    plan_bytes = (json.dumps(plan, sort_keys=True) + '\n').encode()
    monkeypatch.setattr(unit, 'SourceSyncStopBarrier', FakeStopBarrier)
    monkeypatch.setattr(ack_unit, 'PickPlaceTeacher', OriginalTeacherSpy)
    monkeypatch.setattr(ack_unit, 'PushTeacher', OriginalTeacherSpy)
    monkeypatch.setattr(s1_preflight, 'clone', forbidden)
    monkeypatch.setattr(s1_preflight, 'predict', forbidden)
    counter = [0]

    @contextmanager
    def open_case(condition_index=0):
        counter[0] += 1
        base = tmp_path / ('case-' + str(counter[0])); base.mkdir()
        root, ledger = real_ledger(base)
        ledger.charge('main_starts', 1); ledger.charge('engine_starts', 1)
        condition = plan['conditions'][condition_index]
        folder = root / 'episodes' / condition['condition_id']; folder.mkdir(parents=True)
        engine = FakeEngine(condition['card']); teacher = OriginalTeacherSpy(); calls = []
        adapter = ExecutionProgressAdapter(
            ReferenceAdapterConfig(ack_unit.FRAME, ack_unit.FRAME, ack_unit.TCP, stale_timeout_s=3.),
            ProgressConfig(), mode=condition['interface'], episode_id='CPU_COMPONENT',
            validate_raw=forbidden, validate_mapped_full=forbidden, validate_prefix=forbidden)
        def original(card, initial, actual):
            calls.append((deepcopy(card), np.asarray(initial).copy(), actual))
            return teacher
        with ledger.steps('live_and_recovery_integrations', 16000, 'original-live-owner') as live:
            factory = unit.SourceComponentFactory(folder=folder, ledger=ledger,
                outer_live_meter=live, original_make_teacher=original, adapter=adapter,
                guard=ledger.guard, plan_bytes=plan_bytes,
                expected_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                population_bytes=population_bytes, plan_owner_release='a' * 40,
                runtime_release='b' * 40, condition_id=condition['condition_id'],
                episode_id='CPU_COMPONENT', executor_id='CPU_SOURCE', source_step_sha256='c' * 64,
                expected_step=FakeEngine.step, expected_snapshot=FakeEngine.snapshot,
                fingerprint=lambda e:'d' * 64, event_maximum_bytes=16 * 1024**2)
            h = SimpleNamespace(factory=factory, engine=engine, teacher=teacher, calls=calls,
                root=root, ledger=ledger, live=live, folder=folder, condition=condition, adapter=adapter)
            try:
                yield h
            finally:
                # Test-resource cleanup only, after all relevant assertions.
                # Real close-failure injection closes the fd BEFORE raising.
                evidence = factory.evidence
                if evidence is not None and not evidence.closed:
                    evidence.close()
    return open_case


def install(h):
    return h.factory(h.condition['card'], np.eye(4), h.engine)


def owner_finally(h, operation, status='FAILED_PRESERVED'):
    """Illustrative real-ledger closeout, preserving first interrupt priority.

    Root must apply equivalent primary-exception preservation in its actual
    owner branch; blindly nesting finally can mask KI with reconcile errors.
    """
    errors = []
    try:
        operation()
    except BaseException as error:
        errors.append(error)
    for close in (
        lambda:h.factory.persist_if_unfinished(status, None if not errors else str(errors[0])),
        h.factory.close_evidence,
        h.factory.reconcile_outer,
    ):
        try:
            close()
        except BaseException as error:
            errors.append(error)
    if errors:
        primary = next((e for e in errors if not isinstance(e, Exception)), errors[0])
        secondary = next((e for e in errors if e is not primary), None)
        if primary.__cause__ is None and secondary is not None:
            raise primary from secondary
        raise primary


def add_complete_prefix(h, sequence):
    """Synthetic completed records; no native step, source-proposal or model."""
    f = h.factory
    shadow = f.meters.reserve('shadow', 200, 1)
    f.meters.charge('shadow', 200, False, dict(fixture=True))
    f.meters.finish('shadow', shadow, 200, False, 'PASS')
    live = f.meters.reserve('live', 200, 0)
    f.meters.charge('live', 200, False, dict(fixture=True))
    f.meters.finish('live', live, 200, False, 'PASS')
    f.executor.clone_attempts += 1
    f.executor.counts['shadow'] += 200; f.executor.counts['live'] += 200
    refs = []
    initial = None if sequence == 0 else f'fixture-{sequence - 1}-7'
    prior = initial
    for i in range(8):
        command = f'fixture-{sequence}-{i}'
        target = np.eye(4); target[:3, 3] = [.4 + i * .001, 0., .4]
        refs.append(dict(reference=i, command_id=command, ack_id=command,
            predecessor_ack_id=prior, proposal_sha256=hashlib.sha256(command.encode()).hexdigest(),
            phase=float(i), tick_before=sequence * 200 + i * 25,
            tick_after=sequence * 200 + (i + 1) * 25,
            execution_end_timestamp=(sequence * 200 + (i + 1) * 25) * .002,
            target=target.tolist(), opening=.5))
        prior = command
    for row in refs:
        f.record(dict(kind='SOURCE_ACTUAL_REFERENCE_ACK', episode_id=f.episode_id,
            executor_id=f.executor_id, **row))
    action = ack_unit._mapped_hash(refs); nonce = str(uuid.uuid4())
    f.record(dict(kind='SOURCE_PREFIX200_ACK', episode_id=f.episode_id, executor_id=f.executor_id,
        fixture=True, actual_ticks=200, tick_before=sequence * 200, tick_after=(sequence + 1) * 200,
        raw_chunk_id=f'CPU_COMPONENT:ref{sequence}', raw_h40_sha256='a' * 64,
        action_sha256=action, nonce=nonce, initial_adapter_ack_id=initial,
        proposal_digests=[r['proposal_sha256'] for r in refs], phases=[r['phase'] for r in refs],
        actual_reference_acks=refs, maximum_live_shadow_difference=0.,
        ABI='SINGLE_4x4_POSE_PLUS_GRIPPER_NOT_PREVIEW9',
        s1=dict(version=ack_unit.S1_VERSION, decision='PASS', nonce=nonce,
            state_sha256='b' * 64, action_sha256=action, shadow_integrations=200)))
    h.engine.steps = (sequence + 1) * 200; h.engine.data.time = h.engine.steps * .002
    f.executor.maximum_parity = 0.


def ledger_value(h):
    return read(h.ledger.path)


def chain_has(error, text):
    todo = [error]; seen = set()
    while todo:
        current = todo.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if text in str(current):
            return True
        todo.extend((current.__cause__, current.__context__))
    return False


def test_factory_calls_original_teacher_once_and_prebinds_without_old_protected_or_clone(case):
    with case() as h:
        interface = install(h)
        assert isinstance(interface, unit.SourceInterfaceTeacher)
        assert isinstance(interface.executor, unit.OuterLoopSinglePosePrefixExecutor)
        assert len(h.calls) == 1 and h.calls[0][2] is h.engine
        # Invoke only the raw-prebinding wrapper; actual source preflight is
        # already separately tested and is deliberately not run in this test.
        interface.teacher.plan(h.engine.snapshot())
        assert h.teacher.plan_calls == 1 and h.engine.native_calls == 0
        assert (h.folder / 'source_component_ack/plans/000.json').exists()
        assert h.factory.executor.clone_attempts == 0
        owner_finally(h, lambda:None)
        assert h.factory.barrier.calls == 1 and h.engine.renderer_closes == 0
    assert ledger_value(h)['engine_starts'] == ledger_value(h)['main_starts'] == 1
    assert ledger_value(h)['live_and_recovery_integrations'] == 0


def test_normal32s_stops_actual_barrier_returns_and_persists_owner_ack_before_renderer_close(case):
    with case() as h:
        interface = install(h)
        for sequence in range(80): add_complete_prefix(h, sequence)
        owner_finally(h, lambda:interface.finish('VALID_COMPLETE', None), status='VALID_COMPLETE')
        summary = read(h.folder / 'S1_RESULT.json')
        assert h.factory.barrier.calls == 1 and h.factory.actual_stop.stopped is True
        assert summary['actual_stop_event']['kind'] == 'SOURCE_NORMAL_STOP_ACK'
        assert summary['actual_stop_event']['receipt_id'] == h.factory.actual_stop.receipt_id
        assert summary['status'] == 'VALID_COMPLETE' and not summary['engineering_unknown']
        assert summary['shadow_integrations'] == summary['live_integrations_lower_bound'] == 16000
        assert len(summary['actual_completed_prefixes']) == 80 and len(summary['actual_25tick_ACKs']) == 640
        proof = summary['actual_completed_prefixes'][0]
        assert proof['s1']['nonce'] == proof['nonce'] and proof['s1']['action_sha256'] == proof['action_sha256']
        assert proof['actual_reference_acks'][0]['ack_id'] == 'fixture-0-0'
        assert summary['maximum_live_shadow_difference'] == 0. and not h.factory.executor.stopped
        h.engine.close()
        assert h.engine.renderer_closes == 1 and h.factory.barrier.calls == 1
    v = ledger_value(h)
    assert v['shadow_integrations'] == v['live_and_recovery_integrations'] == 16000
    assert not v['reservations'] and not v['unknown_effects']
    assert v['engine_starts'] == 1  # Fixture never calls a native clone; factory adds no debit.


def test_known_partial_shadow_s1_rejection_settles_actual37_and_never_stops_twice(case):
    with case() as h:
        interface = install(h); f = h.factory
        token = f.meters.reserve('shadow', 200, 1)
        f.meters.charge('shadow', 37, False, dict(fixture=True))
        f.executor.counts['shadow'] = 37; f.executor.clone_attempts = 1
        f.executor._halt(source_single_pose_prefix.PrefixFailure('SOURCE_ORIGINAL_NATIVE_S1_REJECT'))
        f.meters.finish('shadow', token, 37, False, 'S1_REJECT')
        assert f.barrier.calls == 1
        owner_finally(h, lambda:interface.finish('FAILED_PRESERVED', 'known S1 reject'))
        s1 = read(h.folder / 'S1_RESULT.json')
        assert f.barrier.calls == 1 and s1['actual_stop_event']['kind'] == 'SOURCE_STOP_ACK'
        assert s1['pre_execution_rejections'] == s1['known_partial_S1_rejections'] == 1
        assert s1['shadow_integrations'] == 37 and s1['native_live_integrations'] == 0
        assert not s1['engineering_unknown'] and not (h.folder / 'MODEL_REJECT.json').exists()
    v = ledger_value(h)
    assert v['shadow_integrations'] == 37 and v['live_and_recovery_integrations'] == 0
    assert not v['reservations'] and not v['unknown_effects']


@pytest.mark.parametrize('raw', [True, False])
def test_raw_and_mapped_full_h40_rejections_are_distinct_from_s1_and_unknown(case, raw):
    with case() as h:
        interface = install(h); f = h.factory
        reference = 'CPU_COMPONENT:ref0' if raw else 'CPU-mapped-command-0'
        f.record(dict(kind='SOURCE_FULL_H40_GATE', episode_id=f.episode_id, executor_id=f.executor_id,
            reference_id=reference, decision=dict(decision='REJECT', fixture=True)))
        f.executor._halt(source_single_pose_prefix.PrefixFailure('SOURCE_ORIGINAL_FULL_H40_REJECT'))
        owner_finally(h, lambda:interface.finish('FAILED_PRESERVED', 'known original gate reject'))
        s1 = read(h.folder / 'S1_RESULT.json'); reject = read(h.folder / 'MODEL_REJECT.json')
        assert s1['full_raw_gate_rejections'] == int(raw)
        assert s1['mapped_full_gate_rejections'] == int(not raw)
        assert s1['pre_execution_rejections'] == 0 and not s1['engineering_unknown']
        assert reject['source_gate_scope'] == ('FULL_RAW_H40' if raw else 'MAPPED_FULL_H40')
        assert reject['original_gate']['decision'] == 'REJECT' and f.barrier.calls == 1
    assert ledger_value(h)['live_and_recovery_integrations'] == ledger_value(h)['shadow_integrations'] == 0


def test_unknown_partial_preserves_known_lower_bounds_and_original_both_reservations(case):
    with pytest.raises(unit.SourceComponentUnknown):
        with case() as h:
            interface = install(h); f = h.factory
            token = f.meters.reserve('shadow', 200, 1)
            f.meters.charge('shadow', 7, True, dict(fixture=True))
            f.executor.counts['shadow'] = 7; f.executor.unknown['shadow'] = True
            f.meters.finish('shadow', token, 7, True, 'ENGINEERING_MISSING')
            h.engine.steps = 3; h.engine.data.time = .006
            f.executor._halt(RuntimeError('SYNTHETIC_UNKNOWN_EFFECT'))
            owner_finally(h, lambda:interface.finish('FAILED_PRESERVED', 'unknown partial'))
    s1 = read(h.folder / 'S1_RESULT.json'); v = ledger_value(h)
    assert s1['engineering_unknown'] and s1['status'] == 'FAILED_PRESERVED'
    assert s1['shadow_integrations'] == 7 and s1['native_live_integrations'] == 3
    assert s1['pre_execution_rejections'] == 0 and h.live['actual'] == 3
    assert {r['maximum'] for r in v['reservations'].values()} == {200, 16000}
    assert len(v['unknown_effects']) == 2 and v['live_and_recovery_integrations'] == 0


@pytest.mark.parametrize('boundary', ['stop', 'write', 'close'])
def test_interrupt_keeps_whole_live_counter_lower_bound_and_is_transparent_after_finally(case, monkeypatch, boundary):
    with pytest.raises(KeyboardInterrupt) as caught:
        with case() as h:
            interface = install(h); f = h.factory
            h.engine.steps = 13; h.engine.data.time = .026
            if boundary == 'stop': f.barrier.failure = 'interrupt'
            if boundary == 'write':
                original_write = unit.write
                def write_then_interrupt(path, body):
                    original_write(path, body)
                    if path.name == 'S1_RESULT.json': raise KeyboardInterrupt('ORIGINAL_S1_WRITE_INTERRUPT')
                monkeypatch.setattr(unit, 'write', write_then_interrupt)
            if boundary == 'close':
                original_close = f.evidence.close
                def close_then_interrupt():
                    original_close(); raise KeyboardInterrupt('ORIGINAL_EVIDENCE_CLOSE_INTERRUPT')
                monkeypatch.setattr(f.evidence, 'close', close_then_interrupt)
            owner_finally(h, lambda:interface.finish('FAILED_PRESERVED', 'fixture terminal'))
    assert 'ORIGINAL_' in str(caught.value) and h.live['actual'] == 13
    assert h.factory.unknown and h.factory.barrier.calls == 1
    v = ledger_value(h)
    assert v['reservations']['original-live-owner']['maximum'] == 16000 and v['unknown_effects']
    if boundary == 'stop':
        saved = read(h.folder / 'S1_RESULT.json')
        assert saved['engineering_unknown'] and saved['actual_stop_event'] is None
    if boundary == 'write':
        assert (h.folder / 'S1_RESULT.json').exists() and h.factory.persist_attempted
    if boundary == 'close':
        # First S1 file predates later close failure; never rewrite it as a
        # substitute for the authoritative unknown owner/ledger outcome.
        saved_bytes = (h.folder / 'S1_RESULT.json').read_bytes()
        h.factory.persist_if_unfinished('FAILED_PRESERVED', 'must not retry')
        assert (h.folder / 'S1_RESULT.json').read_bytes() == saved_bytes


def test_factory_original_interrupt_survives_secondary_stop_cleanup_exception(case):
    with pytest.raises(KeyboardInterrupt) as caught:
        with case() as h:
            f = h.factory
            def interrupted_original(*args):
                f.barrier.failure = 'ordinary'
                raise KeyboardInterrupt('ORIGINAL_TEACHER_CONSTRUCTOR_INTERRUPT')
            f.original = interrupted_original
            owner_finally(h, lambda:install(h))
    assert 'ORIGINAL_TEACHER_CONSTRUCTOR_INTERRUPT' in str(caught.value)
    assert chain_has(caught.value, 'SECONDARY_STOP_CLEANUP_ERROR')
    assert h.factory.barrier.calls == 1 and h.live['actual'] == 0
    assert ledger_value(h)['reservations']['original-live-owner']['maximum'] == 16000


@pytest.mark.parametrize('origin', ['stop', 'event-read'])
def test_persist_primary_interrupt_survives_secondary_s1_write_failure(case, monkeypatch, origin):
    with pytest.raises(KeyboardInterrupt) as caught:
        with case() as h:
            interface = install(h); f = h.factory
            h.engine.steps = 19; h.engine.data.time = .038
            if origin == 'stop': f.barrier.failure = 'interrupt'
            else:
                def interrupted_events(): raise KeyboardInterrupt('ORIGINAL_EVENTS_READ_INTERRUPT')
                monkeypatch.setattr(f.evidence, 'read_events', interrupted_events)
            original_write = unit.write
            attempts = []
            def secondary_write_error(path, body):
                attempts.append(path.name)
                if path.name == 'S1_RESULT.json': raise OSError('SECONDARY_S1_PERSIST_ERROR')
                return original_write(path, body)
            monkeypatch.setattr(unit, 'write', secondary_write_error)
            owner_finally(h, lambda:interface.finish('FAILED_PRESERVED', 'compound interruption'))
    assert 'ORIGINAL_' in str(caught.value) and chain_has(caught.value, 'SECONDARY_S1_PERSIST_ERROR')
    assert attempts.count('S1_RESULT.json') == 1 and h.factory.persist_attempted
    assert h.factory.barrier.calls == 1 and h.live['actual'] == 19
    assert ledger_value(h)['reservations']['original-live-owner']['maximum'] == 16000


def test_stop_noop_return_cannot_become_durable_normal_or_terminal_ack(case):
    with pytest.raises(unit.SourceComponentUnknown):
        with case() as h:
            interface = install(h); h.factory.barrier.failure = 'noop'
            owner_finally(h, lambda:interface.finish('FAILED_PRESERVED', 'no returned actual receipt'))
    saved = read(h.folder / 'S1_RESULT.json')
    assert saved['engineering_unknown'] and saved['actual_stop_event'] is None
    assert h.factory.actual_stop is None and h.factory.barrier.calls == 1
    assert ledger_value(h)['reservations']


def test_record_closeout_guard_preserves_real_effect_after_new_physics_cutoff(case):
    with case() as h:
        interface = install(h)
        (h.root / 'SOURCE_PHYSICS_STOP.json').write_text('{}')
        with pytest.raises(PermissionError): h.factory.guard()
        h.factory.record(dict(kind='ACTUAL_EFFECT_CLOSING_ONLY', episode_id='CPU_COMPONENT', executor_id='CPU_SOURCE'))
        owner_finally(h, lambda:interface.finish('FAILED_PRESERVED', 'source start gate closed'))
        assert any(e['kind'] == 'ACTUAL_EFFECT_CLOSING_ONLY' for e in h.factory.evidence.read_events())
    assert not ledger_value(h)['reservations']


def test_missing_factory_or_native_counter_never_reconciles_as_known_zero(case):
    with pytest.raises(unit.SourceComponentUnknown):
        with case() as h:
            # Simulates original engine construction failing before make_teacher.
            owner_finally(h, lambda:None)
    saved = read(h.folder / 'S1_RESULT.json')
    assert saved['engineering_unknown'] and saved['native_live_integrations'] is None
    assert saved['actual_stop_event'] is None and h.factory.unknown
    assert ledger_value(h)['reservations']['original-live-owner']['maximum'] == 16000
