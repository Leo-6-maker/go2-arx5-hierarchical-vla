"""CPU ABI fixtures only: no simulator, model, ROS or native qualification."""
import hashlib

import pytest

from scripts.pi0292.durable import read, write
from . import target_zero_quota as unit
from . import target_paused_stop as stop
from .test_target_paused_stop import CLOCK, native_packet


def inputs(tmp_path, change=None, *, fault=None):
    root = tmp_path / 'output'; root.mkdir()
    path = tmp_path / 'REQUEST.json'
    q = dict(schema=unit.SCHEMA, scope=unit.ROLE, child_run_uuid='CPU-current-io',
        GPU_submissions=0, training_updates=0, GR00T_queries=0,
        policy_episodes=0, physics_ticks=0, requested=dict.fromkeys(unit.DIMS, 0))
    q['requested'].update(CoreLab_CPU_s=3600, local_CPU_s=60, constructors=8,
        CoreLab_output_bytes=512*1024**2, local_output_bytes=64*1024**2,
        transfer_bytes=128*1024**2)
    if change: change(q)
    write(path, q)
    calls = []
    def guard(*, closeout): calls.append(closeout)
    def spin(seconds, predicate):
        assert seconds == 2
        a = read(root / 'pause_ack.json')
        if fault != 'stale': a['ack_sequence'] += 1
        if fault == 'history': a['leg_history_steps'] += 1
        if fault == 'running': a['running'] = True
        if fault == 'state': (root / 'state.bin').write_bytes(native_packet(0, q=1.))
        write(root / 'pause_ack.json', a)
        return predicate()
    kw = dict(request_path=path, request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        clock_path=CLOCK, episode_id='CPU-zero', executor_id='CPU-target',
        run_uuid='CPU-current-io', manifest_sha256='2'*64, guard=guard, spin=spin)
    return root, path, kw, calls


def boundary(root):
    (root / 'state.bin').write_bytes(native_packet(0))
    write(root / 'pause_ack.json', dict(step=0, time_s=0., motiontime=0,
        leg_history_steps=0, gait_indices=0., running=False, clock_paused=True, ack_sequence=1))


def test_zero_arm_real_state_ABI_and_original_STOP_fresh_ACK(tmp_path):
    root, _, kw, calls = inputs(tmp_path)
    owner = unit.OwnedTargetZeroQuota(root, **kw)
    assert read(root/'TARGET_QUOTA_OWNER.json')['maximum_native_step'] == 0
    owner.arm(); boundary(root)
    before = (root/'state.bin').read_bytes()
    receipt = owner.stop('CPU_ZERO_NATIVE_IO_FINISHED')
    assert receipt.stopped and owner.state()['last_native_quota'] == 0
    assert (root/'step_quota').read_bytes() == b'0\n'
    assert (root/'state.bin').read_bytes() == before and calls == [False, False, True]
    ack = read(root/'TARGET_STOP_ACK.json')
    assert ack['native_step'] == ack['sim_time_s'] == 0
    assert ack['pause_after']['ack_sequence'] == 2
    assert ack['physics_advance_authorized'] is False
    assert ack['target_task_native_qualified'] is ack['native_prefix_safety_qualified'] is False
    with pytest.raises(stop.TargetStopError, match='LATCHED'): owner.arm()
    owner.close()


@pytest.mark.parametrize('step', [0, 1, 25, 18500, True, .002])
def test_no_step_input_can_turn_IO_component_into_physics_permission(tmp_path, step):
    root, _, kw, _ = inputs(tmp_path)
    owner = unit.OwnedTargetZeroQuota(root, **kw); owner.arm()
    with pytest.raises(stop.TargetStopError, match='ZERO_NEVER_ADVANCE'): owner.quota(step)
    assert (root/'step_quota').read_bytes() == b'0\n'
    with pytest.raises(stop.TargetStopError, match='LATCHED'): owner.arm()
    owner.close()


@pytest.mark.parametrize('field,value', [
    ('schema', 'PI0294_R4_PILOT'), ('scope', 'DEC1_TARGET_TASK_NATIVE_V1'),
    ('child_run_uuid', 'CPU-other'), ('physics_ticks', 1), ('GR00T_queries', 1),
    ('GPU_submissions', True), ('policy_episodes', False)])
def test_wrong_or_nonzero_role_request_rejected_before_owner_output(tmp_path, field, value):
    root, _, kw, _ = inputs(tmp_path, lambda q: q.update({field:value}))
    with pytest.raises(stop.TargetStopError): unit.OwnedTargetZeroQuota(root, **kw)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize('fault', ['missing', 'physics', 'episode', 'bool', 'negative', 'fractional_tick'])
def test_eight_dimension_binding_cannot_hide_physics_or_malformed_holds(tmp_path, fault):
    def change(q):
        v = q['requested']
        if fault == 'missing': del v['transfer_bytes']
        if fault == 'physics': v['physics_ticks'] = 1
        if fault == 'episode': v['episode_starts'] = 1
        if fault == 'bool': v['constructors'] = True
        if fault == 'negative': v['CoreLab_CPU_s'] = -1
        if fault == 'fractional_tick': v['physics_ticks'] = 0.
    root, _, kw, _ = inputs(tmp_path, change)
    with pytest.raises(stop.TargetStopError): unit.OwnedTargetZeroQuota(root, **kw)
    assert list(root.iterdir()) == []


def test_sealed_request_changed_after_construction_never_arms(tmp_path):
    root, path, kw, _ = inputs(tmp_path)
    owner = unit.OwnedTargetZeroQuota(root, **kw)
    q = read(path); q['physics_ticks'] = 25; write(path, q)
    with pytest.raises(stop.TargetStopError, match='ZERO_REQUEST_BYTES_CHANGED'): owner.arm()
    assert owner.state()['unknown'] and not (root/'step_quota').exists()
    owner.close()


def test_zero_arm_lost_return_retains_effect_and_no_retry(tmp_path, monkeypatch):
    root, _, kw, _ = inputs(tmp_path)
    owner = unit.OwnedTargetZeroQuota(root, **kw)
    original = stop.os.replace
    def lost(src, dst):
        original(src, dst)
        if dst.name == 'step_quota': raise ConnectionError('CPU_LOST_ZERO_RETURN')
    monkeypatch.setattr(stop.os, 'replace', lost)
    with pytest.raises(ConnectionError): owner.arm()
    assert (root/'step_quota').read_bytes() == b'0\n'
    assert owner.state()['unknown'] and owner.state()['last_native_quota'] is None
    with pytest.raises(stop.TargetStopError, match='LATCHED'): owner.arm()
    with pytest.raises(stop.TargetStopError, match='PRIOR_EFFECT_UNKNOWN'): owner.stop('UNKNOWN')
    owner.close()


@pytest.mark.parametrize('fault', ['stale', 'history', 'running', 'state'])
def test_zero_quota_is_not_native_pause_or_full_state_STOP_proof(tmp_path, fault):
    root, _, kw, _ = inputs(tmp_path, fault=fault)
    owner = unit.OwnedTargetZeroQuota(root, **kw); owner.arm(); boundary(root)
    with pytest.raises((ValueError, PermissionError)): owner.stop('CPU_BAD_PAUSE')
    assert owner.state()['unknown'] and not (root/'TARGET_STOP_ACK.json').exists()
    owner.close()


def test_rearm_before_stop_and_original_positive_class_zero_are_both_rejected(tmp_path):
    root, _, kw, _ = inputs(tmp_path)
    owner = unit.OwnedTargetZeroQuota(root, **kw); owner.arm()
    with pytest.raises(stop.TargetStopError, match='ZERO_ARM_ONCE'): owner.arm()
    assert owner.state()['unknown']; owner.close()
    fresh = tmp_path/'positive'; fresh.mkdir()
    kw = {k:v for k,v in kw.items() if not k.startswith('request_')}
    with pytest.raises(stop.TargetStopError, match='ORIGINAL_QUOTA_CAP'):
        stop.OwnedTargetQuota(fresh, maximum_native_step=0, **kw)
    assert list(fresh.iterdir()) == []


@pytest.mark.parametrize('old', ['step_quota.pending', 'TARGET_QUOTA_000000_INTENT.json'])
def test_partial_or_unknown_previous_publication_cannot_get_new_owner(tmp_path, old):
    root, _, kw, _ = inputs(tmp_path)
    (root/old).write_text('CPU_OLD_UNKNOWN_INTENT')
    with pytest.raises(stop.TargetStopError, match='NEW_UNESCAPED'):
        unit.OwnedTargetZeroQuota(root, **kw)
    assert not (root/'TARGET_QUOTA_OWNER.json').exists()
