"""Publish bounded, CPU-only daily receipts using VANDA's private endpoint copies."""
import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_formal_metrics import summarize
from scripts.sla.hopper_parent_custody import PARENT, PLAN, digest, write_once
from scripts.sla.hopper_site import DEADLINE, T0, pinned, wall_seconds
from scripts.sla.hopper_vanda_dispatch import PBS, PROJECT, job_row

SGT = timezone(timedelta(hours=8))
WALL = 600


def publication(root, cfg):
    formal = pinned(cfg['formal'])
    backup = Path(formal['backup'])
    control = Path(formal['control'])
    if formal['plan']!=PLAN or formal['project']!=PROJECT or cfg['formal']['path']!=str(control/'FORMAL.json'):
        raise ValueError('Formal scheduler identity changed')
    panel = Path(cfg['panel']['path'])
    pinned(cfg['panel'])
    endpoints, latest = {}, {}
    for file in sorted((backup/'receipts').glob('*-phase?????.json')):
        row = read(file)
        role, phase = row['role'],row['phase']
        key = f'{role}-phase{phase:05d}'
        if (role not in ('MAIN','CONTROL2000') or file.stem!=key or row['plan']!=PLAN
                or row['parent']!=PARENT or row['arrays']!=140):
            raise ValueError('Unrecognized checkpoint custody')
        if role not in latest or phase>latest[role]['phase']:
            latest[role] = row
        if not row['eval_due']:
            continue
        run = control/'jobs'/key
        terminal = run/'RESULT.json'
        entry = dict(checkpoint=row['checkpoint'],status='OFFLINE_EVAL_PENDING')
        if (run/'JOB.json').exists():
            entry['job'] = read(run/'JOB.json')['job']
        if terminal.exists():
            status = read(terminal)
            entry.update(status=status['status'],terminal_sha256=digest(terminal))
            if status['status']=='FORMAL_424_SEALED':
                if status['checkpoint']!=row['checkpoint'] or status['exact']!=48:
                    raise ValueError('Formal terminal and custody disagree')
                dst = root/'endpoints'/(key+'.json')
                if not dst.is_file():
                    summary = summarize(run,panel)
                    if (summary['checkpoint_sha256']!=row['checkpoint'] or summary['phase']!=phase
                            or summary['restore']['exact']!=48):
                        raise ValueError('Formal output differs from the registered endpoint')
                    write_once(dst,dict(summary,terminal_sha256=entry['terminal_sha256']))
                report = read(dst)
                if (report['terminal_sha256']!=entry['terminal_sha256'] or report['checkpoint_sha256']!=row['checkpoint']
                        or digest(run/'formal/RESULT.json')!=report['ordinary_receipt_sha256']
                        or digest(run/'formal-restore/RESULT.json')!=report['restore']['result_sha256']):
                    raise ValueError('Previously summarized endpoint changed')
                entry.update(metrics_sha256=digest(dst),source_MAE=report['source_normalized_active_MAE'])
        elif (run/'PBS_TERMINAL.json').is_file():
            entry.update(read(run/'PBS_TERMINAL.json'))
        if 'metrics_sha256' not in entry:
            entry['generation_receipts'] = {mode:dict(attempted=len(list((run/mode).glob('*.intent.json'))),
                    generated=len(list((run/mode).glob('*.generated.json')))) for mode in ('formal','formal-restore')}
        endpoints[key] = entry
    if len(endpoints)>7:
        raise ValueError('Formal endpoint limit exceeded')
    prefix = {}
    for role,row in latest.items():
        path = backup/role/f"phase{row['phase']:05d}"/'COMPLETE.json'
        if Path(row['custody'])!=path.parent:
            raise ValueError('Unexpected custody path')
        header = pinned(dict(path=str(path),sha256=row['checkpoint']))
        b,clock = header['binding'],header['progress']
        if (b['parent']!=PARENT or b['source']!=formal['training_source'] or b['role']!=role
                or clock['phase_step']!=row['phase'] or len(header['arrays'])!=140):
            raise ValueError('Complete-state identity disagrees with the copy receipt')
        prefix[role] = dict(phase=row['phase'],checkpoint=row['checkpoint'],bytes=row['bytes'],
                            source=b['source'],role_updates=clock['role_updates'],role_visits=clock['role_visits'],
                            training_H40=clock['role_training_full_generations'])
    reference = endpoints.get('MAIN-phase00000',{}).get('source_MAE')
    if reference is not None:
        for entry in endpoints.values():
            if 'source_MAE' in entry:
                entry['source_MAE_change_from_phase0'] = {d:v-reference[d] for d,v in entry['source_MAE'].items()}
    return dict(plan=PLAN,scope='VANDA_VERIFIED_CUSTODY_AND_FORMAL_EVAL',verified_prefix=prefix,endpoints=endpoints,
                new_optimizer_updates=0,new_generations=0,native_requests=0)


def due(n):
    return DEADLINE-1200 if n==14 else T0+n*86400+5


def submit(root, cfg, config_sha, n, predecessor=None):
    if not 0<=n<=14 or time.time()>=min(DEADLINE,cfg['grant_end'])-180 or (root/'STOP').exists():
        raise PermissionError('Daily report registration window closed')
    slot = root/f'{n:02d}'
    slot.mkdir(exist_ok=True)
    name = f'schv1sum{n:02d}'
    script = slot/'batch.pbs'
    body = '\n'.join(['#!/bin/bash','#PBS -N '+name,'#PBS -P '+PROJECT,'#PBS -q auto',
        '#PBS -l select=1:ncpus=1:mem=1GB','#PBS -l walltime=00:10:00','#PBS -m n','#PBS -r n',
        '#PBS -o '+str(slot/'stdout.log'),'#PBS -e '+str(slot/'stderr.log'),'set -eu','umask 077',
        'cd '+shlex.quote(str(root)),'export PYTHONPATH='+shlex.quote(str(root/'source')),
        'export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
        'exec '+shlex.join([cfg['python'],'-B','-m','scripts.sla.hopper_daily_custody',str(root),config_sha,'run',str(n)])])+'\n'
    if not script.exists():
        with script.open('x') as f:
            f.write(body)
    if script.read_text()!=body:
        raise ValueError('Registered report batch changed')
    payload = dict(name=name,config=config_sha,pbs=digest(script),predecessor=predecessor,
                   scheduled_epoch=due(n) if n else None)
    def enqueue():
        old = subprocess.check_output([str(PBS/'qselect'),'-u',os.environ['USER'],'-N',name],text=True,timeout=25)
        if old.strip():
            raise RuntimeError('Reconcile an already registered report job')
        opts = ['-W','depend=afterok:'+predecessor,'-a',datetime.fromtimestamp(due(n),SGT).strftime('%Y%m%d%H%M.%S')] if predecessor else []
        job = subprocess.check_output([str(PBS/'qsub'),*opts,str(script)],text=True,timeout=60,
                                      env=dict(os.environ,TZ='Asia/Singapore')).strip()
        if not re.fullmatch(r'\d+\.stdct-mgmt-02',job):
            raise RuntimeError('Ambiguous report JobID')
        write_once(slot/'JOB.json',dict(payload,job=job))
        return job
    return Journal(slot/'journal',name).once('submit',payload,enqueue)


def run(root,cfg,sha,n):
    slot = root/f'{n:02d}'
    for _ in range(20):
        if (slot/'JOB.json').is_file():
            break
        time.sleep(1)
    job = os.environ['PBS_JOBID']
    q = job_row(job)
    registration = read(slot/'JOB.json')
    if (job!=registration['job'] or q['Job_Name']!=registration['name'] or q['job_state']!='R'
            or q['project']!=PROJECT or q['Job_Owner'].split('@')[0]!=os.environ['USER']
            or int(q['Resource_List']['ncpus'])!=1 or int(q['Resource_List'].get('ngpus',0))!=0
            or wall_seconds(q['Resource_List']['walltime'])!=WALL or (n and time.time()<due(n))):
        raise PermissionError('Wrong report allocation or scheduled time')
    end = min(DEADLINE,cfg['grant_end'])
    if time.time()>=end or (root/'STOP').exists():
        write_once(slot/'CLOSED.json',dict(job=job,status='STOP_OR_DEADLINE'))
        return
    with closing(OwnerLock(root/'owner.lock')):
        start=time.time()
        actual=publication(root,cfg)
        write_once(slot/'ACTUAL.json',dict(actual,job=job,pid=os.getpid(),read_start=start,read_end=time.time(),
                                          source=cfg['source'],day_slot=n))
        after = max(n+1,math.floor((time.time()-T0)/86400)+1)
        if after<=14 and time.time()<due(after)<end and not (root/'STOP').exists():
            child=submit(root,cfg,sha,after,job)
            write_once(slot/'SUCCESSOR.json',dict(job=child,day_slot=after,afterok=job,scheduled_epoch=due(after)))


if __name__=='__main__':
    c=argparse.ArgumentParser()
    c.add_argument('root',type=Path)
    c.add_argument('config_sha')
    c.add_argument('mode',choices=('first','run'))
    c.add_argument('slot',type=int)
    a=c.parse_args()
    os.umask(0o077)
    cfg=pinned(dict(path=str(a.root/'CONFIG.json'),sha256=a.config_sha))
    if (cfg['plan']!=PLAN or cfg['T0']!=T0 or cfg['deadline']!=DEADLINE or cfg['project']!=PROJECT
            or cfg['uid']!=os.getuid() or any(digest(a.root/'source'/k)!=v for k,v in cfg['members'].items())):
        raise PermissionError('Unregistered VANDA report identity')
    if a.mode=='first':
        if a.slot!=0:
            raise ValueError('Only the initial slot is submitted interactively')
        print(submit(a.root,cfg,a.config_sha,0))
    else:
        run(a.root,cfg,a.config_sha,a.slot)
