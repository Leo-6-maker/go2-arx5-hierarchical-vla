"""Run the remaining fixed H200 candidates under a single real PBS allocation."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.hopper_engineering_results import CANDIDATES, FM_WIDTH, pin, select, successful
from scripts.sla.hopper_parent_custody import PLAN, PARENT, write_once
from scripts.sla.hopper_site import pinned, site_epoch, wall_seconds


def observed_rejection(folder, run, role, source, action):
    interruption = folder/action/'INTERRUPTION.private.json'
    if not interruption.is_file():
        return None
    record = read(interruption)
    error = record['error']
    if action == 'train' and error == 'Candidate first-step numerical or input gate failed before Adam':
        status = 'NUMERICAL_REJECT'
    elif 'RESOURCE_EXHAUSTED' in error or 'out of memory' in error.lower():
        status = 'GPU_MEMORY_EXHAUSTED'
    else:
        return None
    train = folder/'train/TRAIN.private.json'
    t = read(train) if train.exists() else None
    first = folder/'train/first-step/COMPLETE.json'
    ordinary = folder/'ordinary/RESULT.private.json'
    return dict(plan=PLAN, parent=PARENT, source=source, role=role, qualified=False, status=status,
                job=os.environ['PBS_JOBID'], failed_action=action, error=error, interruption=pin(interruption),
                first_step=pin(first) if first.is_file() else None,
                ordinary_record=pin(ordinary) if ordinary.is_file() else None,
                completed_updates=t['optimizer_updates'] if t else record.get('observed_engine_steps'),
                visits=t['visits'] if t else record.get('completed_fm_windows'),
                training_generations_attempted=t['attempted_aux'] if t else record.get('attempted_aux'),
                panel_attempts=sum(1 for stage in ('ordinary','restore') for _ in (folder/stage).glob('*.intent.json')))


def batch(run):
    mount = read(run/'MOUNT.private.json')
    if mount['job'] != os.environ.get('PBS_JOBID') or mount['plan'] != PLAN:
        raise PermissionError('Unregistered engineering job')
    job = mount['job']
    receipt = pinned(mount['gpu_reservation'])
    grant = pinned(mount['site_grant'])
    import json
    q = json.loads(subprocess.check_output([mount['qstat'],'-f','-F','json',job],text=True,timeout=45))['Jobs'][job]
    wall = wall_seconds(q['Resource_List']['walltime'])
    if (q['job_state'] != 'R' or q['project'] != grant['project'] or q['Job_Owner'].split('@')[0] != grant['account']
            or receipt['job'] != job or receipt['reserved_seconds'] != wall or receipt['group'] != 'engineering'):
        raise PermissionError('Unmatched engineering allocation')
    deadline = min(site_epoch(q['stime']) + wall, mount['deadline_epoch'], grant['expiry_epoch'])
    root = Path(mount['engineering_root'])
    root.mkdir(parents=True,exist_ok=True)
    source = mount['source_revision']
    stop = False
    def signal_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM,signal_stop)
    signal.signal(signal.SIGINT,signal_stop)
    def expired():
        return stop or time.time() >= deadline-900 or (run/'STOP.json').exists() or (Path(mount['custody'])/'STOP.json').exists()
    with closing(OwnerLock(root/'batch.lock')):
        write_once(run/'START.private.json',dict(plan=PLAN,source=source,job=job,pid=os.getpid(),epoch=time.time(),deadline=deadline))
        for i, role in enumerate(CANDIDATES):
            outcome = root/(role+'.outcome.json')
            folder = root/role
            if outcome.is_file():
                prior = read(outcome)
                if prior['role'] != role or prior['source'] != source or prior['parent'] != PARENT:
                    raise ValueError('Candidate outcome identity changed')
                continue
            if expired():
                write_once(run/'RESULT.private.json', dict(status='SAVED_BEFORE_NEXT_CANDIDATE',job=job,next_role=role))
                return
            if folder.exists():
                raise RuntimeError('Prior candidate has an unresolved effect; reconcile before continuing')
            e1 = read(root/'E1.outcome.json') if i else None
            if i and (not e1.get('first_step') or not pinned(e1['first_step'])['passed']):
                raise RuntimeError('No complete original E1 backward reference for another candidate')
            group_run = run/role
            group_run.mkdir()
            folder.mkdir()
            scientific = dict(plan=PLAN,parent=PARENT,source=source,job=job,candidate=role,root=str(folder),
                              max_updates=16,max_panel_per_pass=48,first_step_reference=e1['first_step'] if i else None,
                              fm_outcomes={r:pin(root/(r+'.outcome.json')) for r in FM_WIDTH} if role in ('E5','E6') else {})
            group_file=group_run/'GROUP.private.json'
            write_once(group_file,scientific)
            group_mount = group_run/'MOUNT.private.json'
            write_once(group_mount,dict(mount,engineering_group=pin(group_file)))
            measurement = None
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            with (group_run/'gpu-profile.csv').open('x') as monitor:
                if visible:
                    measurement=subprocess.Popen(['nvidia-smi','-i',visible,
                        '--query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits','-l','5'],stdout=monitor,stderr=subprocess.STDOUT)
                try:
                    for action in ('train','ordinary','restore'):
                        argv=[sys.executable,'-B','-m','scripts.sla.hopper_engineering_worker',
                              str(group_run),pin(group_mount)['sha256'],role,action]
                        start=time.time()
                        with (group_run/(action+'.log')).open('xb') as log:
                            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
                            write_once(group_run/(action+'.START.json'),dict(job=job,pid=child.pid,epoch=start))
                            while child.poll() is None:
                                if expired():
                                    # The worker owns its seal margin. Forward an operator STOP to its normal guard.
                                    if not (group_run/'STOP.json').exists():
                                        write_once(group_run/'STOP.json',dict(reason='ALLOCATION_MARGIN_OR_STOP'))
                                time.sleep(1)
                        write_once(group_run/(action+'.EXIT.json'),dict(status=child.returncode,wall_s=time.time()-start))
                        if child.returncode:
                            rejected=observed_rejection(folder,run,role,source,action)
                            if not rejected:
                                raise RuntimeError(f'{role}/{action} interrupted; inspect the preserved counts and receipt')
                            write_once(outcome,rejected)
                            break
                    else:
                        write_once(outcome,successful(folder,role))
                finally:
                    if measurement is not None:
                        measurement.terminate()
                        measurement.wait(timeout=10)
        selected=select(root,source)
        write_once(run/'RESULT.private.json',dict(status='SELECTION_SEALED',job=job,
                   winner=selected['winner'],selection=pin(root/'SELECTION.json')))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('run',type=Path)
    args=cli.parse_args()
    os.umask(0o077)
    try:
        batch(args.run)
    except Exception as exc:
        write_once(args.run/'INTERRUPTION.private.json', dict(job=os.environ.get('PBS_JOBID'),error=str(exc),epoch=time.time()))
        raise
