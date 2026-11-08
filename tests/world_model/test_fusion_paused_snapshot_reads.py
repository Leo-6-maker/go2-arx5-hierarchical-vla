"""Synthetic live-file replacement faults; no renderer, model or native ticks."""
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from go2arx5_vla.world_model import fusion_paused_observation as p
from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
from tests.world_model.test_fusion_paused_observation import capture, capture_fixture


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=0., sleeps=[])
    def sleep(seconds):
        value.sleeps.append(seconds)
        value.now += seconds
    monkeypatch.setattr(p.time, 'monotonic', lambda: value.now)
    monkeypatch.setattr(p.time, 'sleep', sleep)
    return value


def reader_adapter(tmp_path):
    calls = []
    return SimpleNamespace(output=tmp_path, guard=lambda: calls.append('guard')), calls


@pytest.mark.parametrize('name,decode,failures,expected', [
    ('state.bin', False, [FileNotFoundError(), FileNotFoundError()], b'CURRENT_BINARY'),
    ('pause_ack.json', True, [FileNotFoundError(), b'{"ack_sequence":'], {'ack_sequence': 42}),
])
def test_transient_missing_or_partial_publication_reads_current_bytes(
        tmp_path, monkeypatch, clock, name, decode, failures, expected):
    adapter, guards = reader_adapter(tmp_path)
    target = tmp_path / name
    target.write_bytes(b'CURRENT_BINARY' if not decode else json.dumps(expected).encode())
    actual, reads = Path.read_bytes, []
    def fault(path):
        if path == target:
            reads.append(path)
            if failures:
                value = failures.pop(0)
                if isinstance(value, BaseException): raise value
                return value
        return actual(path)
    monkeypatch.setattr(Path, 'read_bytes', fault)
    assert p._read_live_snapshot(adapter, name, decode_json=decode) == expected
    assert len(reads) == len(guards) == 3
    assert clock.sleeps == [.002, .002]


@pytest.mark.parametrize('fault,exception', [('missing', FileNotFoundError), ('partial', json.JSONDecodeError)])
def test_persistent_publication_failure_is_monotonically_bounded(
        tmp_path, monkeypatch, clock, fault, exception):
    adapter, guards = reader_adapter(tmp_path)
    target = tmp_path / 'pause_ack.json'
    target.write_text('{')
    actual = Path.read_bytes
    def unreadable(path):
        if path == target and fault == 'missing': raise FileNotFoundError('SYNTHETIC_REPLACEMENT_GAP')
        return actual(path)
    monkeypatch.setattr(Path, 'read_bytes', unreadable)
    with pytest.raises(exception): p._read_live_snapshot(adapter, target.name, decode_json=True)
    assert .25 <= clock.now < .2521
    assert len(guards) == len(clock.sleeps) + 1
    assert all(seconds == .002 for seconds in clock.sleeps)


@pytest.mark.parametrize('error', [PermissionError('SYNTHETIC_EACCES'), OSError('SYNTHETIC_EIO'),
                                 UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'SYNTHETIC_ENCODING')])
def test_nonpublication_errors_are_not_retried(tmp_path, monkeypatch, clock, error):
    adapter, guards = reader_adapter(tmp_path)
    def fail(path): raise error
    monkeypatch.setattr(Path, 'read_bytes', fail)
    with pytest.raises(type(error)): p._read_live_snapshot(adapter, 'pause_ack.json', decode_json=True)
    assert guards == ['guard'] and clock.sleeps == []


def test_raw_binary_is_not_decoded_or_retried_as_partial_JSON(tmp_path, clock):
    adapter, guards = reader_adapter(tmp_path)
    raw = b'{"incomplete_json_is_a_valid_raw_snapshot":'
    (tmp_path / 'state.bin').write_bytes(raw)
    assert p._read_live_snapshot(adapter, 'state.bin') == raw
    assert guards == ['guard'] and not clock.sleeps


def test_original_guard_STOP_interrupts_retry_before_another_read(tmp_path, monkeypatch, clock):
    adapter, guards = reader_adapter(tmp_path)
    def guard():
        guards.append('guard')
        if len(guards) == 2: raise RuntimeError('SYNTHETIC_ORIGINAL_STOP')
    adapter.guard = guard
    reads = []
    def missing(path):
        reads.append(path)
        raise FileNotFoundError()
    monkeypatch.setattr(Path, 'read_bytes', missing)
    with pytest.raises(RuntimeError, match='ORIGINAL_STOP'):
        p._read_live_snapshot(adapter, 'pause_ack.json', decode_json=True)
    assert len(guards) == 2 and len(reads) == 1 and clock.sleeps == [.002]


def test_no_cached_ACK_substitution_after_a_success(tmp_path, monkeypatch, clock):
    adapter, _ = reader_adapter(tmp_path)
    target = tmp_path / 'pause_ack.json'
    target.write_text('{"ack_sequence": 5}')
    assert p._read_live_snapshot(adapter, target.name, decode_json=True) == {'ack_sequence': 5}
    target.write_text('{"ack_sequence": 6}')
    assert p._read_live_snapshot(adapter, target.name, decode_json=True) == {'ack_sequence': 6}
    target.unlink()
    with pytest.raises(FileNotFoundError): p._read_live_snapshot(adapter, target.name, decode_json=True)
    assert clock.now >= .25


@pytest.mark.parametrize('fault', ['running', 'not_paused', 'step', 'time', 'sequence', 'missing_frozen', 'quota'])
def test_valid_but_false_actual_pause_is_rejected_without_retry(tmp_path, monkeypatch, clock, fault):
    adapter, _, _, _ = capture_fixture(tmp_path, monkeypatch)
    path = adapter.output / 'pause_ack.json'
    ack = json.loads(path.read_text())
    if fault == 'running': ack['running'] = True
    elif fault == 'not_paused': ack['clock_paused'] = False
    elif fault == 'step': ack['step'] += 25
    elif fault == 'time': ack['time_s'] += .05
    elif fault == 'sequence': ack['ack_sequence'] = -1
    elif fault == 'missing_frozen': del ack['leg_history_steps']
    else: (adapter.output / 'step_quota').write_text('3728\n')
    path.write_text(json.dumps(ack))
    with pytest.raises(ValueError, match='ORIGINAL_NATIVE_PAUSE_ACK'):
        p._pause(adapter)
    assert clock.sleeps == []


@pytest.mark.parametrize('field', ['motiontime', 'leg_history_steps', 'gait_indices'])
def test_new_ACK_cannot_hide_changed_frozen_history(tmp_path, monkeypatch, clock, field):
    adapter, _, _, events = capture_fixture(tmp_path, monkeypatch)
    first = p._pause(adapter)
    path = adapter.output / 'pause_ack.json'
    ack = json.loads(path.read_text()); ack[field] += 1
    path.write_text(json.dumps(ack))
    with pytest.raises(ValueError, match='PHYSICS_FSM_OR_LEG_HISTORY'):
        p._after_pause(adapter, first)
    assert events.saves == [] and clock.sleeps == []


def test_missing_fresh_ACK_does_not_accept_previous_successful_read(tmp_path, monkeypatch, clock):
    adapter, _, _, events = capture_fixture(tmp_path, monkeypatch)
    first = p._pause(adapter)
    events.fresh_ack = False
    with pytest.raises(ValueError, match='FRESH_ACTUAL_NATIVE_PAUSE_ACK'):
        p._after_pause(adapter, first)
    assert events.saves == [] and clock.sleeps == []


@pytest.mark.parametrize('outcome', ['same_packet', 'changed_packet', 'corrupt_packet'])
def test_binary_gap_recovery_requires_fresh_exact_actual_state(tmp_path, monkeypatch, clock, outcome):
    adapter, _, _, _ = capture_fixture(tmp_path, monkeypatch)
    target = adapter.output / 'state.bin'
    raw = target.read_bytes()
    observed = adapter.observe()
    adapter.observe = lambda: observed
    current = bytearray(raw)
    if outcome == 'changed_packet': struct.pack_into('<d', current, 36, observed['qpos'][0] + .001)
    if outcome == 'corrupt_packet': current = b'INCOMPLETE_PACKET'
    target.write_bytes(current)
    actual, reads = Path.read_bytes, []
    def gap(path):
        if path == target:
            reads.append(path)
            if len(reads) == 1: raise FileNotFoundError('SYNTHETIC_STATE_PUBLICATION_GAP')
        return actual(path)
    monkeypatch.setattr(Path, 'read_bytes', gap)
    if outcome == 'same_packet':
        assert p._pause(adapter)['raw'] == raw
    else:
        reason = 'ADAPTER_OBSERVE_AND_CURRENT_PACKET' if outcome == 'changed_packet' else 'INCOMPLETE_ACTUAL_NATIVE_STATE'
        with pytest.raises(ValueError, match=reason): p._pause(adapter)
    assert len(reads) == 2 and clock.sleeps == [.002]


def test_waiting_bridge_survives_ACK_publication_gap_without_capture_or_ticks(tmp_path, monkeypatch, clock):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    record = capture(adapter, binding, task)
    before = dict(raw=(adapter.output / 'state.bin').read_bytes(),
                  quota=(adapter.output / 'step_quota').read_bytes(), saves=list(events.saves),
                  identities=events.load_calls)
    actual, reads = Path.read_bytes, []
    target = adapter.output / 'pause_ack.json'
    def gap(path):
        if path == target:
            reads.append(path)
            if len(reads) < 4: raise FileNotFoundError('SYNTHETIC_ACTUAL_FAILURE_SEAM')
        return actual(path)
    def forbidden(*args, **kwargs): pytest.fail('read retry attempted a native effect')
    adapter.advance_to = adapter.resume_clock_once = adapter.install_reference = forbidden
    monkeypatch.setattr(Path, 'read_bytes', gap)
    bridge._waiting_pause(adapter, record)
    assert clock.sleeps == [.002] * 3
    assert (adapter.output / 'state.bin').read_bytes() == before['raw']
    assert (adapter.output / 'step_quota').read_bytes() == before['quota']
    assert events.saves == before['saves'] and events.load_calls == before['identities']
    assert len(reads) >= 6  # first pause, fresh ACK and exact post-fresh pause.
