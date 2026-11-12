import json
from pathlib import Path

import pytest

from scripts.wm02 import sla_queue_native_v1 as queue


def test_wait_retries_only_readonly_capacity_then_dispatch_permission():
    clock = [0]
    calls = []
    states = []
    def probe():
        calls.append(clock[0])
        if len(calls) == 1:
            raise ValueError(queue.CAPACITY_REASON)
        return {'free_GPU_bytes': 16 * 1024**3}
    assert queue.wait_capacity(probe, 120, states.append, now=lambda: clock[0],
                               sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert calls == [0, 60]
    assert states[0]['status'] == 'WAIT_ORIGINAL_GPU_HOST_CAPACITY'
    assert states[-1]['status'] == 'ORIGINAL_CAPACITY_READY'


def test_expired_wait_starts_nothing():
    calls = []
    assert not queue.wait_capacity(lambda: calls.append(1), 1, lambda value: None, now=lambda: 2)
    assert calls == []


@pytest.mark.parametrize('reason', ['CHANGED_MODEL', 'WRONG_OWNER'])
def test_other_admission_failures_do_not_loop(reason):
    def probe():
        raise PermissionError(reason)
    with pytest.raises(PermissionError, match=reason):
        queue.wait_capacity(probe, 10, lambda value: None, now=lambda: 0,
                            sleep=lambda seconds: pytest.fail('must not retry'))


def test_stop_prevents_probe():
    with pytest.raises(PermissionError, match='QUEUED_STOP_RETAINED'):
        queue.wait_capacity(lambda: pytest.fail('must not probe'), 10, lambda value: None,
                            stopped=lambda: True, now=lambda: 0)


def test_actual_original_capacity_guard_is_waitable(monkeypatch):
    from types import SimpleNamespace
    from go2arx5_vla.world_model import task_learning_student_batch_v1 as original
    gpu = 'GPU-845348f1-1511-dc19-0877-cc97658e3edd'
    monkeypatch.setattr(original.os, 'sched_getaffinity', lambda pid: set(range(12)))
    monkeypatch.setattr(original.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout=gpu + ', NVIDIA GeForce RTX 4080 SUPER, 3167'))
    spec = dict(GPU_uuid=gpu, checkpoint_parameter_bytes=7 * 1024**3,
                allocator_limit_bytes=13 * 1024**3)
    clock = [0]
    states = []
    assert not queue.wait_capacity(lambda: original.resources(spec), 60, states.append,
                                   now=lambda: clock[0],
                                   sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert [v['status'] for v in states] == ['WAIT_ORIGINAL_GPU_HOST_CAPACITY',
                                           'FINITE_QUEUE_WAIT_EXPIRED_NO_DISPATCH']


@pytest.mark.parametrize('intent', ['LAUNCH_INTENT.private.json', 'SUPERVISOR.private.json',
                                   'DISPATCH.private.json', 'journal'])
def test_any_launch_effect_prevents_capacity_retry(tmp_path, monkeypatch, intent):
    monkeypatch.setattr(queue, 'unit_state', lambda root: dict(LoadState='not-found', MainPID='0'))
    class Journal:
        def path(self, key):
            return tmp_path / 'journal'
    if intent:
        (tmp_path / intent).write_text('{}')
    assert not queue.never_dispatched(tmp_path, Journal())


def test_new_technical_failure_stops_chain_but_retained_prefix_does_not(tmp_path):
    (tmp_path / 'control').mkdir()
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    write('control/LEASE.json', dict(status='CLOSED', worker_exit_code=0))
    write('RESULT.private.json', dict(exit_code=0, unresolved_pair=None,
                                     summary=dict(complete=True, requested=4)))
    rows = [dict(technical_valid=value, hard_safety_stop=False) for value in (True, False, True, True)]
    write('OUTCOMES.private.json', rows)
    assert queue.terminal(tmp_path, 2)[0]
    rows[3]['technical_valid'] = False
    write('OUTCOMES.private.json', rows)
    assert not queue.terminal(tmp_path, 2)[0]
