"""Finite background wait around the existing native registration/dispatch path.

This wrapper allocates no GPU and implements no model, physics, or resource
admission. The sealed native capsule retains its original owner and Journal.
"""
import argparse
from contextlib import ExitStack, closing
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CAPACITY_REASON = 'TASK_LEARNING_GRASP_FRESH_SHARED_GPU_AND_HOST_HEADROOM_BEFORE_DEV_PAIR'


def pinned(value):
    path = Path(value['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise PermissionError('QUEUED_INPUT_CHANGED')
    return json.loads(raw)


def unit_state(root):
    reply = subprocess.run(
        ['systemctl', '--user', 'show', 'wm02-' + Path(root).name,
         '--property=LoadState,MainPID,ActiveState,Result'],
        capture_output=True, text=True, timeout=10, check=True)
    return dict(line.split('=', 1) for line in reply.stdout.splitlines() if '=' in line)


def wait_capacity(probe, deadline, save, *, stopped=lambda: False,
                  now=time.time, sleep=time.sleep):
    """Only a known read-only capacity refusal or busy owner is waitable."""
    while now() < deadline:
        if stopped():
            raise PermissionError('QUEUED_STOP_RETAINED')
        try:
            actual = probe()
        except BlockingIOError:
            save(dict(status='WAIT_EXISTING_OWNER', epoch=now()))
        except ValueError as exc:
            if str(exc) != CAPACITY_REASON:
                raise
            save(dict(status='WAIT_ORIGINAL_GPU_HOST_CAPACITY', epoch=now()))
        else:
            save(dict(status='ORIGINAL_CAPACITY_READY', epoch=now(), actual=actual))
            return True
        sleep(min(60, max(0, deadline - now())))
    save(dict(status='FINITE_QUEUE_WAIT_EXPIRED_NO_DISPATCH', epoch=now()))
    return False


def never_dispatched(root, journal):
    root = Path(root)
    state = unit_state(root)
    return (not any((root / name).exists() for name in
                    ('LAUNCH_INTENT.private.json', 'SUPERVISOR.private.json',
                     'DISPATCH.private.json'))
            and not journal.path(root.name.upper() + '-LAUNCH').exists()
            and state.get('LoadState') == 'not-found' and state.get('MainPID') == '0')


def terminal(root, start):
    """A policy failure can be terminal; any new technical failure stops chaining."""
    root = Path(root)
    lease = json.loads((root / 'control/LEASE.json').read_bytes())
    result = json.loads((root / 'RESULT.private.json').read_bytes())
    rows = json.loads((root / 'OUTCOMES.private.json').read_bytes())
    if (lease.get('status') != 'CLOSED'
            or lease.get('worker_exit_code') != result.get('exit_code')
            or result.get('unresolved_pair') is not None):
        raise RuntimeError('QUEUED_NATIVE_TERMINAL_UNKNOWN_NEVER_RESTART')
    good = (result['exit_code'] == 0 and result['summary']['complete']
            and len(rows) == result['summary']['requested']
            and not any(row['hard_safety_stop'] for row in rows)
            and all(row['technical_valid'] for row in rows[start:]))
    return good, dict(result=result, lease=lease,
                     result_sha256=hashlib.sha256((root / 'RESULT.private.json').read_bytes()).hexdigest())


def run(config, queue):
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from go2arx5_vla.world_model import task_learning_student_batch_v1 as batch
    from go2arx5_vla.world_model.task_learning_student_registration_v2 import register_group, dispatch_group

    campaign = Path(config['campaign_root'])
    journal = Journal(campaign / 'control/journal', campaign.name)
    gpu = pinned(config['gpu_spec'])
    stopped = lambda: (queue / 'STOP.json').exists() or (campaign / 'STOP.json').exists()

    def save(stage, value):
        write(queue / 'STATE.private.json', dict(value, stage=stage, queue_pid=os.getpid()))

    def probe():
        # Never retain another worker's owner while waiting for resources.
        with ExitStack() as locks:
            for path in config['original_owner_locks']:
                locks.enter_context(closing(OwnerLock(path)))
            return batch.resources(gpu)

    for index, stage in enumerate(config['stages']):
        request = stage['request']
        root = Path(request['root'])
        if root.exists():
            raise RuntimeError('QUEUED_GROUP_ALREADY_EXISTS_RECONCILE_NO_REPLAY')
        deadline = min(stage['latest_admission_epoch'], config['deadline_epoch'])
        if not wait_capacity(probe, deadline, lambda value: save(index, value), stopped=stopped):
            return dict(status='QUEUE_EXPIRED', completed_stages=index, exit_code=75)
        registered = register_group(request)  # Original once-only registration.
        write(queue / f'STAGE_{index}_REGISTERED.private.json', registered)
        # A capacity race before LAUNCH_INTENT is a read-only wait, not replay.
        while True:
            if stopped():
                raise PermissionError('QUEUED_STOP_RETAINED')
            try:
                dispatched = dispatch_group(registered['input']['path'])
                break
            except ValueError as exc:
                if str(exc) != CAPACITY_REASON or not never_dispatched(root, journal):
                    raise
                inp = pinned(registered['input'])
                until = min(deadline, inp['deadline_epoch'] - 1801)
                if not wait_capacity(probe, until, lambda value: save(index, value), stopped=stopped):
                    return dict(status='REGISTERED_UNDISPATCHED_QUEUE_EXPIRED',
                                completed_stages=index, exit_code=75)
        write(queue / f'STAGE_{index}_DISPATCH.private.json', dispatched)
        inp = pinned(registered['input'])
        while True:
            state = unit_state(root)
            if state.get('LoadState') != 'loaded':
                raise RuntimeError('QUEUED_NATIVE_UNIT_UNKNOWN_NEVER_RESTART')
            if state.get('MainPID') == '0':
                break
            if time.time() > inp['deadline_epoch'] + 180:
                raise RuntimeError('QUEUED_NATIVE_CLOSEOUT_UNKNOWN_NEVER_RESTART')
            save(index, dict(status='NATIVE_GROUP_LIVE', epoch=time.time(), unit=state,
                             group=str(root)))
            time.sleep(30)
        good, actual = terminal(root, request['assignment_slice'][0])
        write(queue / f'STAGE_{index}_TERMINAL.private.json', actual)
        if not good:
            return dict(status='NATIVE_TECHNICAL_OR_SAFETY_STOP_RETAINED',
                        completed_stages=index + 1, exit_code=1)
    if stopped():
        raise PermissionError('QUEUED_STOP_RETAINED')
    action = config['delivery_seal']
    raw = Path(action['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != action['sha256']:
        raise PermissionError('QUEUED_DELIVERY_SEAL_CHANGED')
    result = subprocess.run([config['python'], '-B', action['path']],
                            capture_output=True, text=True, timeout=600)
    write(queue / 'DELIVERY_SEAL_ACTUAL.private.json',
          dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
    if result.returncode:
        return dict(status='NATIVE_GROUPS_CLOSED_DELIVERY_SEAL_FAILED',
                    completed_stages=len(config['stages']), exit_code=1)
    return dict(status='S1_COMPLETE_NATIVE_DENOMINATORS_SEALED_NOT_TASK_SUCCESS',
                completed_stages=len(config['stages']), exit_code=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    config = pinned(dict(path=args.config, sha256=args.sha256))
    if (config['schema'] != 'SLA_FINITE_EXISTING_NATIVE_BACKGROUND_WAIT_V1'
            or os.getuid() != 2009 or os.uname().nodename != 'corelab-4080s'
            or time.time() >= config['deadline_epoch']):
        raise PermissionError('FINITE_EXISTING_CORE_QUEUE_ONLY')
    sys.path[:0] = [config['runtime_source_root'] + '/src', config['runtime_source_root']]
    from scripts.pi0292.durable import write
    from scripts.pi0294.e1.journal import Journal
    queue = Path(args.config).parent
    campaign = Path(config['campaign_root'])
    journal = Journal(campaign / 'control/journal', campaign.name)

    def execute():
        try:
            result = run(config, queue)
        except Exception as exc:
            result = dict(status='QUEUE_STOPPED_RECONCILE_NO_AUTOMATIC_RETRY',
                          error=type(exc).__name__ + ': ' + str(exc), exit_code=1)
        write(queue / 'RESULT.private.json', dict(result, epoch=time.time()))
        return result
    result = journal.once(queue.name.upper() + '-RUN', dict(config_sha256=args.sha256), execute)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
