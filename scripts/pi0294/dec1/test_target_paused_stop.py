"""Temp-file native ABI fixtures, not actual target execution or qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import threading

import numpy as np
import pytest
from scripts.pi0292.durable import read, write
from . import target_paused_stop as unit

CLOCK = Path(__file__).with_name('frozen_assets') / 'task_clock_82004c31.py'


def native_packet(step=2500, q=0.):
    values = np.zeros(86, dtype='<f8'); values[0] = q
    return (struct.pack('<8sQdIII', b'GRTSTATE', step, step / 500, 34, 32, 20)
            + values.tobytes() + struct.pack('<I', 0))


def fixture(tmp_path, fault=None):
    calls = []
    def guard(*, closeout):
        calls.append(closeout)
        if fault == 'guard_stop' and closeout:
            raise PermissionError('CPU_CLOSEOUT_FAILURE')
    def spin(seconds, predicate):
        assert seconds == 2
        a = read(tmp_path / 'pause_ack.json')
        if fault != 'stale':
            a['ack_sequence'] += 1
        if fault == 'history': a['leg_history_steps'] += 1
        if fault == 'running': a['running'] = True
        if fault == 'clock': a['time_s'] += .002
        if fault == 'state': (tmp_path / 'state.bin').write_bytes(native_packet(q=1.))
        if fault == 'quota': (tmp_path / 'step_quota').write_text('2525\n')
        write(tmp_path / 'pause_ack.json', a)
        return predicate()
    quota = unit.OwnedTargetQuota(tmp_path, clock_path=CLOCK,
        episode_id='CPU-target-episode', executor_id='CPU-frozen-target',
        run_uuid='CPU-new-admitted-owner', manifest_sha256='1' * 64,
        guard=guard, spin=spin, maximum_native_step=18500)
    quota.quota(2500)
    (tmp_path / 'state.bin').write_bytes(native_packet())
    ack = dict(step=2500, time_s=5., motiontime=3900, leg_history_steps=120,
               gait_indices=.3, running=False, clock_paused=True, ack_sequence=1)
    write(tmp_path / 'pause_ack.json', ack)
    return quota, calls


def test_byte_exact_original_clock_overlay_and_no_HEAD_substitution(tmp_path):
    assert hashlib.sha256(CLOCK.read_bytes()).hexdigest() == unit.FROZEN_CLOCK
    actual = unit.frozen_clock(CLOCK)
    assert callable(actual['require_pause']) and callable(actual['finish_pause'])
    from scripts.pi0294 import task_clock
    with pytest.raises(unit.TargetStopError, match='FROZEN_CLOCK_BYTES_CHANGED'):
        unit.OwnedTargetQuota(tmp_path, clock_path=task_clock.__file__,
            episode_id='e', executor_id='x', run_uuid='r', manifest_sha256='1' * 64,
            guard=lambda **_: None, spin=lambda *_: True, maximum_native_step=18500)
    assert list(tmp_path.iterdir()) == []


def test_real_ABI_files_quota_latch_fresh_clock_and_durable_stop(tmp_path):
    q, calls = fixture(tmp_path)
    before = (tmp_path / 'state.bin').read_bytes()
    r = q.stop('CPU_BOUNDARY_REJECT')
    assert r.stopped and r.episode_id == q.episode_id and r.executor_id == q.executor_id
    raw = (tmp_path / 'TARGET_STOP_ACK.json').read_bytes()
    assert r.receipt_id.endswith(hashlib.sha256(raw).hexdigest())
    a = json.loads(raw)
    assert a['native_step'] == 2500 and a['pause_before']['ack_sequence'] == 1
    assert a['pause_after']['ack_sequence'] == 2 and a['task_clock_sha256'] == unit.FROZEN_CLOCK
    assert not a['target_task_native_qualified'] and not a['native_prefix_safety_qualified']
    assert (tmp_path / 'state.bin').read_bytes() == before
    assert (tmp_path / 'step_quota').read_bytes() == b'2500\n'
    assert q.state()['stop_acknowledged'] and not q.state()['unknown']
    assert calls == [False, False, True]
    with pytest.raises(unit.TargetStopError, match='LATCHED'): q.quota(2525)
    with pytest.raises(unit.TargetStopError, match='STOP_ALREADY'): q.stop('again')
    q.close()
    with pytest.raises(unit.TargetStopError, match='NEW_UNESCAPED'):
        unit.OwnedTargetQuota(tmp_path, clock_path=CLOCK, episode_id='e', executor_id='x',
            run_uuid='r', manifest_sha256='2' * 64, guard=lambda **_: None,
            spin=lambda *_: True, maximum_native_step=18500)


@pytest.mark.parametrize('fault', ['stale', 'history', 'running', 'clock', 'state', 'quota', 'guard_stop'])
def test_unknown_or_moving_stop_never_ACKs_or_retries(tmp_path, fault):
    q, _ = fixture(tmp_path, fault)
    with pytest.raises((ValueError, PermissionError)): q.stop('CPU_INCOMPLETE')
    assert q.state()['latched'] and q.state()['unknown'] and not q.state()['stop_acknowledged']
    assert not (tmp_path / 'TARGET_STOP_ACK.json').exists()
    with pytest.raises(unit.TargetStopError): q.quota(2525)
    with pytest.raises(unit.TargetStopError, match='STOP_ALREADY'): q.stop('again')
    q.close()


@pytest.mark.parametrize('fault', ['bool', 'nan', 'duplicate', 'symlink', 'clock_running', 'unreached_quota'])
def test_malformed_stale_or_unreached_native_boundary_cannot_stop_ACK(tmp_path, fault):
    q, _ = fixture(tmp_path)
    a = read(tmp_path / 'pause_ack.json')
    if fault == 'bool': a['step'] = True
    if fault == 'nan': a['gait_indices'] = float('nan')
    if fault == 'clock_running': a['running'] = True
    if fault == 'unreached_quota': q.quota(2525)
    if fault == 'symlink':
        other = tmp_path / 'other.bin'; (tmp_path / 'state.bin').rename(other)
        (tmp_path / 'state.bin').symlink_to(other)
    raw = json.dumps(a)
    if fault == 'duplicate': raw = raw[:-1] + ',"step":2500}'
    (tmp_path / 'pause_ack.json').write_text(raw)
    with pytest.raises((ValueError, PermissionError)): q.stop('CPU_BOUNDARY_BAD')
    assert q.state()['unknown'] and not q.state()['stop_acknowledged']
    q.close()


def test_lost_quota_replace_return_is_unknown_and_never_reissued(tmp_path, monkeypatch):
    q, _ = fixture(tmp_path)
    original = unit.os.replace; replacements = []
    def lose(source, target):
        original(source, target)
        if Path(target).name == 'step_quota':
            replacements.append(target)
            raise ConnectionError('CPU_LOST_QUOTA_WRITE_RETURN')
    monkeypatch.setattr(unit.os, 'replace', lose)
    with pytest.raises(ConnectionError): q.quota(2525)
    assert (tmp_path / 'step_quota').read_bytes() == b'2525\n'
    assert q.state()['last_native_quota'] == 2500 and q.state()['unknown']
    with pytest.raises(unit.TargetStopError): q.quota(2525)
    with pytest.raises(unit.TargetStopError, match='PRIOR_EFFECT_UNKNOWN'): q.stop('unknown')
    assert len(replacements) == 1
    q.close()


def test_lost_stop_write_return_or_post_write_motion_never_returns_receipt(tmp_path, monkeypatch):
    q, _ = fixture(tmp_path)
    original = unit.write
    def move_after_write(path, value):
        original(path, value)
        if Path(path).name == 'TARGET_STOP_ACK.json':
            (tmp_path / 'state.bin').write_bytes(native_packet(q=2.))
    monkeypatch.setattr(unit, 'write', move_after_write)
    with pytest.raises(unit.TargetStopError, match='DURABLE_STOP_READBACK'):
        q.stop('CPU_RACED_AFTER_DURABLE_WRITE')
    # An ACK-shaped file is NOT a returned acknowledgement or native evidence.
    assert (tmp_path / 'TARGET_STOP_ACK.json').exists()
    assert q.state()['unknown'] and not q.state()['stop_acknowledged']
    with pytest.raises(unit.TargetStopError): q.stop('again')
    q.close()


def test_foreign_thread_cannot_release_quota(tmp_path):
    q, _ = fixture(tmp_path); failures = []
    def foreign():
        try: q.quota(2525)
        except BaseException as error: failures.append(error)
    t = threading.Thread(target=foreign); t.start(); t.join()
    assert len(failures) == 1 and isinstance(failures[0], unit.TargetStopError)
    assert q.state()['unknown'] and (tmp_path / 'step_quota').read_bytes() == b'2500\n'
    q.close()


def test_stop_does_not_invent_task_or_prefix_safety_gate(tmp_path):
    q, _ = fixture(tmp_path)
    # A finite physically unsafe value is intentionally not edited/repaired by
    # stop. Stopped is distinct from safe/task-success; no mj_forward is called.
    (tmp_path / 'state.bin').write_bytes(native_packet(q=100.))
    assert q.stop('CPU_UNSAFE_BUT_QUIESCENT').stopped
    assert read(tmp_path / 'TARGET_STOP_ACK.json')['native_prefix_safety_qualified'] is False
    q.close()


@pytest.mark.parametrize('existing', ['state.bin', 'pause_ack.json', 'TARGET_QUOTA_OWNER.lock'])
def test_cannot_attach_to_escaped_worker_or_reopen_lost_constructor(tmp_path, existing):
    (tmp_path / existing).write_text('CPU_EXISTING_NOT_REUSABLE')
    with pytest.raises(unit.TargetStopError, match='NEW_UNESCAPED'):
        unit.OwnedTargetQuota(tmp_path, clock_path=CLOCK, episode_id='e', executor_id='x',
            run_uuid='r', manifest_sha256='1' * 64, guard=lambda **_: None,
            spin=lambda *_: True, maximum_native_step=18500)
    assert not (tmp_path / 'TARGET_QUOTA_OWNER.json').exists()


def test_original_frozen_clock_directly_binds_existing_paused_prediction_seam():
    from .test_target_paused_prediction import args
    from .target_time_transaction import prepare_for_paused_prediction
    raw, proposal, kw = args()
    kw['require_frozen_pause'] = unit.frozen_clock(CLOCK)['require_pause']
    candidate = prepare_for_paused_prediction(raw, proposal, **kw)
    assert candidate.native_prefix_validation_pending and not candidate.physics_advance_authorized
    kw['pause_after']['motiontime'] += 1
    with pytest.raises(ValueError, match='PHYSICS_FSM_OR_LEG_HISTORY'):
        prepare_for_paused_prediction(raw, proposal, **kw)


def test_existing_progress_executor_consumes_real_IO_stop_type_after_gate_failure(tmp_path):
    from go2arx5_vla.adapters.execution_progress_binding import BoundProgressExecutor
    from go2arx5_vla.adapters.groot_to_wbc import ReferenceAdapterConfig
    from go2arx5_vla.adapters.execution_progress import ProgressConfig
    from scripts.pi0292.eval_contract import FRAME, TCP
    q, _ = fixture(tmp_path)
    def reject(_): raise PermissionError('CPU_ORIGINAL_RAW_GATE_REJECT')
    def no_dispatch(*_): raise AssertionError('NO_NATIVE_DISPATCH_IN_CPU_FIXTURE')
    events = []
    executor = BoundProgressExecutor(ReferenceAdapterConfig(FRAME, FRAME, TCP,
        stale_timeout_s=ProgressConfig().new_chunk_max_age_s), mode='I_TIME',
        episode_id=q.episode_id, executor_id=q.executor_id, ticks_per_reference=25,
        validate_raw_eef10=reject, validate_mapped_full=no_dispatch,
        prefix_shadow=no_dispatch, dispatch_once=no_dispatch, actual_feedback=no_dispatch,
        stop=q.stop, record=events.append)
    raw = np.zeros((40, 10)); raw[:, 3] = raw[:, 7] = 1.; raw[:, 9] = .5
    with pytest.raises(PermissionError, match='RAW_GATE_REJECT'):
        executor.load_raw(raw_eef10=raw, chunk_id='CPU-raw', observation_timestamp=0.,
            point_timestamps=np.arange(40) * .05, source_frame=FRAME,
            tcp_semantic_id=TCP, controller_timestamp=0.)
    assert executor.state()['stop_confirmed'] and executor.state()['stop_calls'] == 1
    assert q.state()['stop_acknowledged'] and events[-1]['kind'] == 'STOP_ACK'
    assert executor.state()['live_ticks'] == executor.state()['shadow_ticks'] == 0
    q.close()
