"""Synthetic CPU fixtures independently executed by the integrating owner.

No native physics/model/lease/ledger. Population fixture deliberately uses the
existing synthetic sealed-plan fixture. A local JSON event-list reader stands
in for the owner's independent persisted-byte/hash-chain-verified read_events.
It does NOT qualify the actual SourceComponentEvidence implementation.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
from scripts.pi0292.durable import read, write
from scripts.pi0294.dec1 import source_component_ack as unit
from scripts.pi0294.dec1.test_source_commissioning_plan import population


class SpyTeacher:
    def __init__(self):
        self.calls = 0
        self.fail = False
        pose = np.eye(4); pose[:3, 3] = [.4, 0., .4]
        self.original = dict(poses=np.repeat(pose[None], 40, axis=0),
            opening=np.full(40, .5), offset_s=np.arange(40) / 20, stage='synthetic')

    def plan(self, state):
        self.calls += 1
        if self.fail:
            raise RuntimeError('SYNTHETIC_TEACHER_INTERRUPTED')
        return deepcopy(self.original)


class FakeAdapter:
    def __init__(self, mode):
        self.mode = mode; self.episode_id = 'CPU_COMPONENT_EPISODE'
        self.prior = None; self.pending = None

    def state(self):
        return dict(predecessor_ack_id=self.prior, pending_command_id=self.pending)


@pytest.fixture
def setup(population, tmp_path, monkeypatch):
    data, encode = population
    population_bytes = encode(data)
    plan = unit.source_commissioning_plan.build(population_bytes, owner_release='a' * 40)
    raw = (json.dumps(plan, sort_keys=True) + '\n').encode()
    monkeypatch.setattr(unit, 'PickPlaceTeacher', SpyTeacher)
    monkeypatch.setattr(unit, 'PushTeacher', SpyTeacher)

    def make(index=0, **kwargs):
        condition = plan['conditions'][index]
        directory = tmp_path / ('component-' + str(len(list(tmp_path.iterdir()))))
        directory.mkdir()
        event_path = tmp_path / (directory.name + '-events.json')
        engine = SimpleNamespace(card=deepcopy(condition['card']), steps=0, safety=[],
            data=SimpleNamespace(time=0.), model=SimpleNamespace(nq=2, nv=2, nu=1))
        adapter = FakeAdapter(condition['interface'])
        options = dict(plan_bytes=raw, expected_plan_sha256=hashlib.sha256(raw).hexdigest(),
            population_bytes=population_bytes, owner_release='a' * 40,
            runtime_release='b' * 40, adapter=adapter, condition_id=condition['condition_id'],
            episode_id='CPU_COMPONENT_EPISODE', executor_id='CPU_COMPONENT_EXECUTOR',
            read_events=lambda:read(event_path), guard=lambda:None)
        options.update(kwargs)
        owner = unit.SourceComponentAcknowledger(engine, directory, **options)
        teacher = SpyTeacher(); wrapped = owner.wrap_teacher(teacher)
        state = dict(time_s=0., tcp=np.eye(4), gripper_fraction=.5)
        return SimpleNamespace(owner=owner, teacher=teacher, wrapped=wrapped, engine=engine, adapter=adapter,
            state=state, directory=directory, events=event_path, options=options,
            condition=condition, raw_plan=raw)
    return make


def completed(h, *, tick=None, prior=None):
    """Fabricated completed evidence ONLY; no integration is performed."""
    if tick is None:
        tick = h.owner.state()['sequence'] * 200
    binding = read(h.directory / 'plans' / f'{tick // 200:03d}.json')
    nonce = str(uuid.uuid4()); refs = []
    for i in range(8):
        target = np.eye(4); target[:3, 3] = [.4 + .001 * i, 0., .4]
        command = f'CPU-command-{tick}-{i}'
        refs.append(dict(reference=i, command_id=command,
            proposal_sha256=hashlib.sha256(command.encode()).hexdigest(), phase=float(i),
            predecessor_ack_id=prior, ack_id=command,
            tick_before=tick + 25 * i, tick_after=tick + 25 * (i + 1),
            execution_end_timestamp=(tick + 25 * (i + 1)) * .002,
            target=target.tolist(), opening=.5))
        prior = command
    action_hash = unit._mapped_hash(refs)
    proof = dict(nonce=nonce, raw_chunk_id=binding['raw_chunk_id'],
        episode_id=h.owner.episode_id, executor_id=h.owner.executor_id,
        raw_h40_sha256=binding['raw_h40_sha256'], action_sha256=action_hash,
        proposal_digests=[r['proposal_sha256'] for r in refs], phases=[r['phase'] for r in refs],
        actual_ticks=200, tick_before=tick, tick_after=tick + 200,
        initial_adapter_ack_id=binding['initial_adapter_ack_id'],
        s1=dict(version=unit.S1_VERSION, decision='PASS', nonce=nonce,
            state_sha256='c' * 64, action_sha256=action_hash, shadow_integrations=200),
        actual_reference_acks=refs, maximum_live_shadow_difference=0.,
        ABI='SINGLE_4x4_POSE_PLUS_GRIPPER_NOT_PREVIEW9')
    identity = dict(episode_id=h.owner.episode_id, executor_id=h.owner.executor_id)
    events = [dict(kind='SOURCE_PREFIX_SHADOW_COMPLETE', **identity, nonce=nonce,
        shadow_ticks=200, clone_attempts=1, proposal_digests=proof['proposal_digests'], phases=proof['phases']),
        dict(kind='SOURCE_MAPPED_PREFIX_INSTALL_INTENT', **identity, nonce=nonce,
            raw_h40_sha256=proof['raw_h40_sha256'], action_sha256=action_hash,
            proposal_digests=proof['proposal_digests'], phases=proof['phases'],
            mapped_poses=[r['target'] for r in refs], mapped_grippers=[r['opening'] for r in refs],
            native_step_before=tick),
        dict(kind='SOURCE_MAPPED_PREFIX_BIND_CONFIRMED', **identity, nonce=nonce)]
    events.extend(dict(kind='SOURCE_ACTUAL_REFERENCE_ACK', **identity, nonce=nonce, **r) for r in refs)
    events.append(dict(kind='SOURCE_PREFIX200_ACK', **proof))
    previous = read(h.events) if h.events.exists() else []
    write(h.events, previous + events)
    rows = np.zeros((200, 1 + 2 + 2 + 1 + 6 + 1))
    rows[:, 0] = np.arange(tick + 1, tick + 201) * .002
    h.engine.steps = tick + 200; h.engine.data.time = (tick + 200) * .002
    h.adapter.prior = refs[-1]['ack_id']
    return rows, proof


@pytest.mark.parametrize('condition', range(4))
def test_four_fixed_source_conditions_prebind_raw_before_execution_and_ack_actual_npz(setup, condition):
    h = setup(condition)
    result = h.wrapped.plan(h.state)
    assert h.teacher.calls == 1 and h.engine.steps == 0
    binding = read(h.directory / 'plans' / '000.json')
    assert binding['raw_chunk_id'] == 'CPU_COMPONENT_EPISODE:ref0'
    assert binding['component_plan_sha256'] == h.options['expected_plan_sha256']
    assert binding['interface_mode'] == h.condition['interface']
    assert np.array_equal(result['poses'], h.teacher.original['poses'])
    rows, proof = completed(h)
    receipt = h.owner.acknowledge(h.engine, rows, proof)
    assert isinstance(receipt, unit.NativeMappedPrefixAck)
    physics = h.directory / 'physics' / '000.npz'
    with np.load(physics, allow_pickle=False) as saved:
        assert np.array_equal(saved['rows'], rows)
        assert int(saved['tick_before']) == 0 and int(saved['tick_after']) == 200
    ack_path = h.directory / receipt.receipt_id
    ack = read(ack_path)
    assert receipt.physics_receipt_sha256 == hashlib.sha256(physics.read_bytes()).hexdigest()
    assert receipt.acknowledgement_sha256 == hashlib.sha256(ack_path.read_bytes()).hexdigest()
    assert ack['plan_owner_release'] == 'a' * 40 and ack['runtime_release'] == 'b' * 40
    assert ack['policy'] == unit.POLICY and ack['GR00T_queries'] == 0
    assert 'identity' not in ack and 'query_uuid' not in ack and 'command_sha256' not in ack
    assert not ack['student_neural_ACK_qualification']
    assert not h.owner.state()['unknown'] and h.owner.state()['sequence'] == 1


def test_successful_next_original_teacher_plan_binds_actual_previous_adapter_ack(setup):
    h = setup(); h.wrapped.plan(h.state); rows, proof = completed(h)
    h.owner.acknowledge(h.engine, rows, proof)
    h.state['time_s'] = .4
    h.wrapped.plan(h.state)
    binding = read(h.directory / 'plans' / '001.json')
    assert h.teacher.calls == 2 and binding['tick_before'] == 200
    assert binding['initial_adapter_ack_id'] == proof['actual_reference_acks'][-1]['ack_id']
    assert binding['raw_chunk_id'] == 'CPU_COMPONENT_EPISODE:ref1'
    rows2, proof2 = completed(h, prior=binding['initial_adapter_ack_id'])
    ack2 = h.owner.acknowledge(h.engine, rows2, proof2)
    assert ack2.tick_before == 200 and ack2.tick_after == 400


@pytest.mark.parametrize('change', ['unbound', 'raw', 'chunk', 'nonce', 's1-state', 's1-action',
    's1-ticks', 'shadow-digest', 'phase', 'ack', 'predecessor', 'clock', 'action',
    'parity', 'missing-ack', 'duplicate-ack', 'unsafe', 'physics-clock', 'engine-clock',
    'adapter-mode', 'adapter-actual-ack', 'adapter-pending'])
def test_changed_or_missing_actual_identity_and_safety_chain_reject_once(setup, change):
    h = setup(); h.wrapped.plan(h.state); rows, proof = completed(h)
    if change == 'unbound': h.owner._pending = None
    if change == 'raw': proof['raw_h40_sha256'] = 'd' * 64
    if change == 'chunk': proof['raw_chunk_id'] = 'arbitrary-raw-proof'
    if change == 'nonce': proof['nonce'] = str(uuid.uuid4())
    if change == 's1-state': proof['s1']['state_sha256'] = 'not-a-sha'
    if change == 's1-action': proof['s1']['action_sha256'] = 'd' * 64
    if change == 's1-ticks': proof['s1']['shadow_integrations'] = True
    if change == 'shadow-digest': proof['proposal_digests'][0] = 'e' * 64
    if change == 'phase': proof['phases'][0] = 99.
    if change == 'ack': proof['actual_reference_acks'][0]['ack_id'] = 'not-actual-ack'
    if change == 'predecessor': proof['actual_reference_acks'][1]['predecessor_ack_id'] = None
    if change == 'clock': proof['actual_reference_acks'][0]['tick_after'] = 26
    if change == 'action': proof['actual_reference_acks'][0]['target'][0][3] += .001
    if change == 'parity': proof['maximum_live_shadow_difference'] = 1.1e-10
    if change == 'missing-ack': proof['actual_reference_acks'].pop()
    if change == 'duplicate-ack': proof['actual_reference_acks'][1] = deepcopy(proof['actual_reference_acks'][0])
    if change == 'unsafe': rows[-1, -1] = 1
    if change == 'physics-clock': rows[90, 0] += .001
    if change == 'engine-clock': h.engine.data.time = float('nan')
    if change == 'adapter-mode': h.adapter.mode = 'I_PROGRESS' if h.adapter.mode == 'I_TIME' else 'I_TIME'
    if change == 'adapter-actual-ack': h.adapter.prior = 'not-actually-acknowledged'
    if change == 'adapter-pending': h.adapter.pending = 'still-executing'
    with pytest.raises((ValueError, FileExistsError)):
        h.owner.acknowledge(h.engine, rows, proof)
    assert h.owner.state()['unknown']
    with pytest.raises(ValueError, match='UNKNOWN_NO_RETRY'):
        h.owner.acknowledge(h.engine, rows, proof)
    assert not list((h.directory / 'acks').iterdir())


@pytest.mark.parametrize('change', ['missing-complete', 'missing-live-ack', 'duplicate', 'forecast-only', 'bad-json', 'reordered'])
def test_proof_without_actual_readback_owner_events_is_not_native_ack(setup, change):
    h = setup(); h.wrapped.plan(h.state); rows, proof = completed(h)
    events = read(h.events)
    if change == 'missing-complete': events = [e for e in events if e['kind'] != 'SOURCE_PREFIX200_ACK']
    if change == 'missing-live-ack': events = [e for e in events if e.get('reference') != 3]
    if change == 'duplicate': events.append(deepcopy(events[-1]))
    if change == 'forecast-only': events = [e for e in events if e['kind'] != 'SOURCE_ACTUAL_REFERENCE_ACK']
    if change == 'reordered': events = [events[-1]] + events[:-1]
    write(h.events, events)
    if change == 'bad-json': h.events.write_text('{')
    with pytest.raises(ValueError): h.owner.acknowledge(h.engine, rows, proof)
    assert h.owner.state()['unknown'] and not list((h.directory / 'physics').iterdir())


def test_mutated_teacher_prebind_file_rejects_even_self_consistent_new_proof(setup):
    h = setup(); h.wrapped.plan(h.state); rows, proof = completed(h)
    path = h.directory / 'plans' / '000.json'
    binding = read(path); binding['raw_H40_eef10'][0][0] += .001; write(path, binding)
    with pytest.raises(ValueError, match='PREEXECUTION_TEACHER_PLAN_BYTES_CHANGED'):
        h.owner.acknowledge(h.engine, rows, proof)
    assert not list((h.directory / 'acks').iterdir())


def test_teacher_failure_and_missing_previous_ack_never_make_replacement_query(setup):
    h = setup(); h.teacher.fail = True
    with pytest.raises(RuntimeError, match='TEACHER_INTERRUPTED'): h.wrapped.plan(h.state)
    with pytest.raises(ValueError, match='UNKNOWN_NO_RETRY'): h.wrapped.plan(h.state)
    assert h.teacher.calls == 1
    other = setup(); other.wrapped.plan(other.state)
    with pytest.raises(ValueError, match='FRESH_TEACHER_PLAN'): other.wrapped.plan(other.state)
    assert other.teacher.calls == 1


@pytest.mark.parametrize('boundary', ['plan-write', 'ack-write', 'ack-read', 'event-read', 'event-final'])
def test_unknown_persistence_is_nonretryable_and_preserves_partial_artifacts(setup, boundary):
    h = setup(); writes = []; reads = []; event_calls = []
    original_write, original_read, original_events = h.owner.writer, h.owner.reader, h.owner.read_events
    def writer(path, body):
        writes.append(str(path)); original_write(path, body)
        if (boundary == 'plan-write' and path.parent.name == 'plans'
                or boundary == 'ack-write' and path.parent.name == 'acks'):
            raise KeyboardInterrupt('SYNTHETIC_PERSISTED_THEN_INTERRUPTED')
    def reader(path):
        reads.append(str(path))
        if boundary == 'ack-read' and path.parent.name == 'acks':
            raise OSError('SYNTHETIC_ACK_READ_UNKNOWN')
        return original_read(path)
    def read_events():
        event_calls.append(1)
        if boundary == 'event-read' or boundary == 'event-final' and len(event_calls) == 2:
            raise OSError('SYNTHETIC_PERSISTED_EVENT_READ_UNKNOWN')
        return original_events()
    h.owner.writer, h.owner.reader, h.owner.read_events = writer, reader, read_events
    if boundary == 'plan-write':
        with pytest.raises(KeyboardInterrupt): h.wrapped.plan(h.state)
        assert (h.directory / 'plans' / '000.json').exists()
        with pytest.raises(ValueError, match='UNKNOWN_NO_RETRY'): h.wrapped.plan(h.state)
        assert h.teacher.calls == 1 and len(writes) == 1
    else:
        h.wrapped.plan(h.state); rows, proof = completed(h)
        with pytest.raises((KeyboardInterrupt, OSError)): h.owner.acknowledge(h.engine, rows, proof)
        assert h.owner.state()['unknown']
        before = list(writes)
        with pytest.raises(ValueError, match='UNKNOWN_NO_RETRY'): h.owner.acknowledge(h.engine, rows, proof)
        assert writes == before
        if boundary != 'event-read': assert (h.directory / 'physics' / '000.npz').exists()


def test_existing_ack_or_physics_is_preserved_without_overwrite(setup):
    h = setup(); h.wrapped.plan(h.state); rows, proof = completed(h)
    path = h.directory / 'physics' / '000.npz'; path.write_bytes(b'FIRST_UNKNOWN_EFFECT')
    with pytest.raises(FileExistsError): h.owner.acknowledge(h.engine, rows, proof)
    assert path.read_bytes() == b'FIRST_UNKNOWN_EFFECT' and h.owner.state()['unknown']


def test_physics_readback_checks_full_rows_not_only_saved_file_hash(setup, monkeypatch):
    h = setup(); h.wrapped.plan(h.state); rows, proof = completed(h)
    wrong = rows.copy(); wrong[100, 2] = .01
    class WrongSavedRows:
        files = ['rows', 'tick_before', 'tick_after']
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __getitem__(self, key):
            return dict(rows=wrong, tick_before=np.array(0), tick_after=np.array(200))[key]
    monkeypatch.setattr(unit.np, 'load', lambda *args, **kwargs:WrongSavedRows())
    with pytest.raises(ValueError, match='COMPLETE_ACTUAL_PHYSICS_NPZ_READBACK'):
        h.owner.acknowledge(h.engine, rows, proof)
    assert (h.directory / 'physics' / '000.npz').exists()
    assert not list((h.directory / 'acks').iterdir()) and h.owner.state()['unknown']


def test_noop_ack_writer_cannot_produce_typed_native_ack(setup):
    h = setup(); h.wrapped.plan(h.state); rows, proof = completed(h)
    h.owner.writer = lambda path, body:None
    with pytest.raises(FileNotFoundError): h.owner.acknowledge(h.engine, rows, proof)
    assert h.owner.state()['unknown'] and not list((h.directory / 'acks').iterdir())


def test_clock_change_after_ack_write_is_unknown_without_return_or_retry(setup):
    h = setup(); h.wrapped.plan(h.state); rows, proof = completed(h)
    original = h.owner.writer
    def writer(path, body):
        original(path, body)
        if path.parent.name == 'acks': h.engine.data.time = float('nan')
    h.owner.writer = writer
    with pytest.raises(ValueError, match='FINAL_READBACK_CHANGED'):
        h.owner.acknowledge(h.engine, rows, proof)
    assert (h.directory / 'acks' / '000.json').exists()
    assert h.owner.state()['unknown'] and h.owner.state()['sequence'] == 0
    with pytest.raises(ValueError, match='UNKNOWN_NO_RETRY'):
        h.owner.acknowledge(h.engine, rows, proof)


@pytest.mark.parametrize('change', ['sha', 'condition', 'plan-owner', 'runtime'])
def test_owner_pins_cannot_be_selected_or_changed_by_ack_proof(setup, change):
    kwargs = {}
    if change == 'sha': kwargs['expected_plan_sha256'] = 'f' * 64
    if change == 'condition': kwargs['condition_id'] = 'UNREGISTERED-REQUEST-FLAG'
    if change == 'plan-owner': kwargs['owner_release'] = 'c' * 40
    if change == 'runtime': kwargs['runtime_release'] = 'not-a-runtime-pin'
    with pytest.raises((ValueError, PermissionError)): setup(**kwargs)


def test_real_source_event_chain_reader_integrates_with_component_ACK(setup,tmp_path):
    from scripts.pi0294.dec1.source_component_evidence import SourceComponentEvidence
    h=setup();h.wrapped.plan(h.state);rows,proof=completed(h)
    e=SourceComponentEvidence(tmp_path/'real-owner-events.jsonl',episode_id=h.owner.episode_id,
        executor_id=h.owner.executor_id,maximum_bytes=1024**2)
    for event in read(h.events):e.record(event)
    h.owner.read_events=e.read_events
    ack=h.owner.acknowledge(h.engine,rows,proof)
    assert isinstance(ack,unit.NativeMappedPrefixAck) and h.owner.state()['sequence']==1
    saved=read(h.directory/'acks/000.json')
    assert saved['verified_source_event_list_sha256']==hashlib.sha256(unit._bytes(e.read_events())).hexdigest()
    assert saved['student_neural_ACK_qualification'] is False
    e.close()


def test_changed_real_owner_event_chain_refuses_ACK_and_preserves_unknown(setup,tmp_path):
    from scripts.pi0294.dec1.source_component_evidence import SourceComponentEvidence
    h=setup();h.wrapped.plan(h.state);rows,proof=completed(h)
    e=SourceComponentEvidence(tmp_path/'changed-owner-events.jsonl',episode_id=h.owner.episode_id,
        executor_id=h.owner.executor_id,maximum_bytes=1024**2)
    for event in read(h.events):e.record(event)
    h.owner.read_events=e.read_events
    with e.path.open('ab') as f:f.write(b'{"partial":')
    with pytest.raises(ValueError):h.owner.acknowledge(h.engine,rows,proof)
    assert h.owner.state()['unknown'] and not (h.directory/'acks/000.json').exists()
    with pytest.raises(ValueError,match='UNKNOWN_NO_RETRY'):h.owner.acknowledge(h.engine,rows,proof)
    e.close()
