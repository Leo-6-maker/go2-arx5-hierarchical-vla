"""Release finite SCALE successors from completed local or PBS owners."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import time

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_checkpoint import read_complete
from go2arx5_vla.sla_adapter.scale20k_contract import cursors
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.wm02.multisource_d4_vanda import pin


def verify_seal(train_result):
    result = read_pin(train_result)
    train = Path(train_result['path']).parent
    spec = read(train/'SPEC.private.json')
    arm, step = result['arm'], result['global_step']
    previous = spec.get('resume_from_closed_segment')
    start = read_pin(previous)['global_step'] if previous else 0
    if ((result['status'], result['exit_code']) not in
            (('FIXED_SEGMENT_SAVED', 0), ('BOUNDED_SAVED_PREFIX', 75))
            or not start < step <= spec['global_stop_step']
            or (result['exit_code'] == 0 and step != spec['global_stop_step'])
            or result.get('failure') or result['new_updates'] != step-start
            or result['new_native_requests'] or result['new_native_ticks']):
        raise ValueError('A_COMPLETE_FORWARD_TRAINING_TERMINAL_IS_REQUIRED')
    checkpoint, manifest = read_complete(result['checkpoint'], arm, read(train/'MODEL_AT_START.private.json')['binding'])
    if (checkpoint != train/f'step{step:05d}' or manifest['step'] != step
            or len(manifest['arrays']) != 140 or result['global_cursors'] != manifest['cursors']):
        raise ValueError('A_COMPLETE_140_ARRAY_CHECKPOINT_AND_EXACT_CURSORS_ARE_REQUIRED')
    for index, row in enumerate(manifest['arrays']):
        member = checkpoint/f'{index:06d}.npy'
        if member.is_symlink() or row['file'] != member.name or pin(member)['sha256'] != row['sha256']:
            raise ValueError(f'CHECKPOINT_ARRAY_CHANGED:{index}')
    rows = (train/'UPDATES.private.jsonl').read_bytes()
    ledger = rows.splitlines()
    if not rows.endswith(b'\n') or len(ledger) != step-start:
        raise ValueError('THE_COMPLETE_DURABLE_UPDATE_LEDGER_IS_REQUIRED')
    for expected, line in enumerate(ledger, start+1):
        row = json.loads(line)
        if row['step'] != expected or row['cursors'] != cursors(arm, expected):
            raise ValueError(f'UPDATE_LEDGER_CURSOR_CHANGED:{expected}')
    return dict(step=step, result=train_result, checkpoint=result['checkpoint'], arrays=140,
                ledger=pin(train/'UPDATES.private.jsonl'), cursors=manifest['cursors'])


def wait_owner(spec, ready):
    owner = unit_state(spec['predecessor_unit'])
    if (owner['pid'] != spec['predecessor_pid'] or owner.get('proc_start') != spec['predecessor_proc_start']
            or owner['ControlGroup'] != spec['predecessor_cgroup']):
        raise PermissionError('LIVE_PREDECESSOR_IDENTITY_CHANGED')
    fd = os.pidfd_open(owner['pid'])
    try:
        if unit_state(spec['predecessor_unit']).get('proc_start') != spec['predecessor_proc_start']:
            raise PermissionError('LIVE_PREDECESSOR_IDENTITY_CHANGED_AFTER_PIDFD_OPEN')
        ready()
        wait = select.poll()
        wait.register(fd, select.POLLIN)
        if not wait.poll(max(0, int(1000*(spec['watch_until_epoch']-time.time())))):
            raise TimeoutError('PREDECESSOR_DID_NOT_EXIT_WITHIN_REGISTERED_WINDOW')
    finally:
        os.close(fd)
    for _ in range(40):
        owner = unit_state(spec['predecessor_unit'])
        if not owner['pid'] and not owner['cgroup_pids'] and owner['ActiveState'] in ('inactive', 'failed'):
            return owner
        time.sleep(.25)
    raise PermissionError('PREDECESSOR_CGROUP_DID_NOT_CLEAR')


def core_handoff(spec):
    from scripts.sla.scale20k_core_after500_watch import headroom
    from scripts.sla.scale20k_core_prefix import closed_predecessor

    root, prev, next_root = (Path(spec[key]).resolve(strict=True) for key in ('root','previous','next_root'))
    if (os.uname().nodename != spec['host'] or os.getuid() != spec['uid']
            or unit_state(spec['unit'])['pid'] != os.getpid() or len({root, prev, next_root}) != 3
            or prev.parent != root.parent or next_root.parent != root.parent):
        raise PermissionError('FRESH_CORE_HANDOFF_OWNER_REQUIRED')
    lock = OwnerLock(root/'OWNER.lock')
    result = dict(status='BLOCKED', optimizer_updates=0, model_generations=0, native_requests=0)
    try:
        draft = read_pin(spec['next_draft'])
        previous = read_pin(spec['previous_schedule'])
        schedule = next_root/'SCHEDULE.private.json'
        if (previous['unit']+'.service' != spec['predecessor_unit'] or previous['source_sha'] != draft['source_sha']
                or previous['root'] != str(prev) or draft['root'] != str(next_root)
                or previous['deadline_epoch'] != spec['predecessor_deadline_epoch']):
            raise ValueError('CORE_HANDOFF_SCOPE_OR_ACTIVE_RUNTIME_CHANGED')
        def armed():
            write(root/'READY.private.json', dict(status='ARMED_WAITING_FOR_PREDECESSOR_PIDFD',
                unit=spec['unit'], pid=os.getpid(), predecessor=spec['predecessor_unit'],
                next_unit=draft['unit']+'.service', epoch=time.time()))

        result['predecessor_terminal'] = wait_owner(spec, armed)
        stops = (root/'STOP.json', prev/'STOP.json', next_root/'STOP.json')
        if any(path.exists() for path in stops):
            result['status'] = 'STOPPED_WITHOUT_SUCCESSOR'
            return 0
        terminal = read(prev/'RESULT.private.json')
        prior, _, step = closed_predecessor(dict(draft, previous_train_result=terminal['stages']['train']), next_root)
        result['seal'] = verify_seal(prior)
        if step >= draft['global_stop_step']:
            result['status'] = 'REGISTERED_MILESTONE_ALREADY_SEALED'
            return 0
        if (time.time() >= spec['watch_until_epoch'] or schedule.exists()
                or unit_state(draft['unit']+'.service')['pid'] or not headroom(draft, spec['protected_units'])):
            raise PermissionError('CORE_SUCCESSOR_DEADLINE_OWNER_OR_CAPACITY_NOT_AVAILABLE')
        if any(path.exists() for path in stops):
            result['status'] = 'STOPPED_WITHOUT_SUCCESSOR'
            return 0

        def launch():
            deadline = min(time.time()+14340, spec['latest_successor_deadline_epoch'])
            if deadline-time.time() <= 1800:
                raise PermissionError('CORE_SUCCESSOR_ALLOCATION_TOO_SHORT')
            write(schedule, dict(draft, deadline_epoch=deadline, previous_train_result=prior))
            cmd = ['systemd-run','--user','--unit='+draft['unit'],'--property=Restart=no',
                   '--property=UMask=0077','--property=KillMode=control-group',
                   '--property=CPUAffinity=12 13 14 15','--property=MemoryMax=24G',
                   '--property=StandardInput=null','--property=TimeoutStopSec=90',
                   '--property=RuntimeMaxSec='+str(int(deadline-time.time())+30),
                   '--property=WorkingDirectory='+spec['source'],
                   '--property=StandardOutput=append:'+str(next_root/'logs/queue.log'),
                   '--property=StandardError=append:'+str(next_root/'logs/queue.log')]
            cmd.extend('--setenv='+key+'='+value for key, value in spec['environment'].items())
            cmd.extend([spec['python'],'-B','-m','scripts.sla.scale20k_core_prefix',
                        str(schedule),pin(schedule)['sha256']])
            subprocess.run(cmd, check=True, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return dict(unit=unit_state(draft['unit']+'.service'), schedule=pin(schedule), deadline_epoch=deadline)

        result['successor'] = Journal(root/'handoff', root.name).once(
            'RELEASE_ONE_VERIFIED_CORE_SEGMENT', result['seal'], launch)
        result['status'] = 'SUCCESSOR_SUBMITTED'
        return 0
    except Exception as error:
        result['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write(root/'RESULT.private.json', result)
        lock.close()


def pbs_handoff(spec):
    from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
    from scripts.sla.scale20k_vanda_prefix import completed_prefix

    root, previous = (Path(spec[key]).resolve(strict=True) for key in ('root','previous'))
    if (os.environ.get('PBS_JOBID') != spec['job_id'] or root.stat().st_uid != os.getuid()
            or root == previous or root.parent != previous.parent or (root/'ACTIVATED.private.json').exists()
            or (root/'STOP.json').exists() or (previous/'STOP.json').exists()
            or (Path(spec['chain_root'])/'STOP.json').exists()):
        raise PermissionError('PBS_HANDOFF_STOPPED_OR_EXECUTION_IDENTITY_CHANGED')
    base = read_pin(spec['base_config'])
    continuation = dict(job_id=spec['previous_job'], train_result=pin(previous/'train/RESULT.private.json'),
        campaign_result=pin(previous/'RESULT.private.json'), training_spec=pin(previous/'train/SPEC.private.json'),
        model_at_start=pin(previous/'train/MODEL_AT_START.private.json'),
        worker_root=spec['worker_root'], worker_files=spec['worker_files'])
    config = dict(base, root=str(root), source_sha=spec['source_sha'], job_id=spec['job_id'],
                  deadline_epoch=spec['deadline_epoch'], continuation=continuation)
    prior, _ = read_pbs_with_auth_retry(completed_prefix, config, root)
    seal = verify_seal(prior)
    if spec['role'] == 'train':
        limit = spec['step']
        if seal['step'] >= limit:
            raise ValueError('PBS_HANDOFF_STEP_LIMIT_MUST_ADVANCE')
        config.update(global_stop_step=limit, max_walltime_s=21600)
        from scripts.sla.scale20k_vanda_prefix import run
    else:
        if seal['step'] != spec['step'] or read_pin(prior)['exit_code'] != 0:
            raise ValueError('THE_ORIGINAL_FORMAL_EVAL_ENDPOINT_IS_NOT_SEALED')
        config['step'] = spec['step']
        from scripts.sla.scale20k_vanda_offline import run
    write(root/'ACTIVATED.private.json', dict(seal=seal, execution=config))
    return run(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('core','pbs'))
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    spec = read_pin(dict(path=args.config, sha256=args.sha256))
    if args.role == 'core':
        raise SystemExit(core_handoff(spec))
    try:
        raise SystemExit(pbs_handoff(spec))
    except Exception as error:
        root = Path(spec['root'])
        if os.environ.get('PBS_JOBID') == spec['job_id'] and not (root/'ACTIVATED.private.json').exists():
            write(root/'BLOCKED.private.json', dict(status='PREDECESSOR_GATE_BLOCKED', job=spec['job_id'],
                predecessor=spec['previous_job'], failure=f'{type(error).__name__}: {error}',
                optimizer_updates=0, model_generations=0, native_requests=0))
        raise
