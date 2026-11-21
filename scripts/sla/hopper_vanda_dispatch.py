"""Run a small, bounded PBS scheduler on VANDA for a single Hopper lineage."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_parent_custody import PLAN, SSH, HOST, write_once
from scripts.sla.hopper_site import DEADLINE, site_epoch, wall_seconds

PBS=Path('/opt/pbs/bin')
PROJECT='CFP04-CF-007'
MAX_SLOTS=60
WALL=21600


def job_row(job):
    return json.loads(subprocess.check_output([str(PBS/'qstat'),'-xf','-F','json',job],text=True,timeout=45))['Jobs'][job]


def submit(root,n,previous=None):
    slot=root/f'{n:02d}'
    slot.mkdir(exist_ok=True)
    cfg=read(root/'CONFIG.json')
    if n>MAX_SLOTS or cfg['plan']!=PLAN or time.time()>=min(DEADLINE,cfg['grant_end'])-240 or (root/'STOP').exists():
        raise PermissionError('Dispatcher limit reached')
    prefix='slachv1ev' if cfg.get('local_formal') else ('slachv1bk' if cfg.get('local_backup') else 'slachv1ctl')
    if cfg.get('local_exchange'):
        prefix='slachv1hf'
    if cfg.get('local_train_ingress'):
        prefix='schv1data'
    if cfg.get('reconciled_dispatch'):
        prefix='slachv1rc'
    if cfg.get('local_relay'):
        prefix='slachv1tr'
        if n==1:
            previous=cfg['local_relay']['job']
    if cfg.get('cold_predecessor'):
        prefix='slachv1ef' if cfg.get('local_formal') else 'slachv1rf'
        if n==1:
            previous=cfg['cold_predecessor']['job']
    name=prefix+f'{n:02d}'
    jobfile=slot/'JOB.json'
    journal=Journal(slot/'journal',slot.name)
    body='\n'.join(['#!/bin/bash','#PBS -N '+name,'#PBS -P '+PROJECT,'#PBS -q auto',
        '#PBS -l select=1:ncpus=1:mem=1GB','#PBS -l walltime=06:00:00','#PBS -m n','#PBS -r n',
        '#PBS -o '+str(slot/'stdout.log'),'#PBS -e '+str(slot/'stderr.log'),'set -eu','umask 077',
        'cd '+shlex.quote(str(root)),'export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1',
        'export PYTHONPATH='+shlex.quote(str(root/'source')),
        'exec /usr/bin/python3 -B driver.py run '+shlex.quote(str(root))+' '+str(n)])+'\n'
    script=slot/'dispatcher.pbs'
    if not script.exists():
        with script.open('x') as f:
            f.write(body)
    if script.read_text()!=body:
        raise ValueError('VANDA dispatcher batch file changed')
    deps=['-W','depend=afterok:'+previous] if previous else []
    request=dict(plan=PLAN,number=n,previous=previous,script_sha256=hashlib.sha256(body.encode()).hexdigest())
    def enqueue():
        if not previous or n==1:
            owned=subprocess.check_output([str(PBS/'qselect'),'-N',name,'-u',os.environ['USER']],text=True,timeout=30)
            if owned.strip():
                raise RuntimeError('Existing scheduler identity requires reconciliation')
        job=subprocess.check_output([str(PBS/'qsub'),*deps,str(script)],text=True,timeout=90).strip()
        if not re.fullmatch(r'\d+\.stdct-mgmt-02',job):
            raise RuntimeError('Indeterminate VANDA submission')
        write_once(jobfile,dict(job=job,**request))
        return job
    if jobfile.exists():
        receipt=read(jobfile)
        if any(receipt[k]!=v for k,v in request.items()):
            raise ValueError('Changed dispatcher dependency')
        return receipt['job']
    return journal.once('submit',request,enqueue)


def run(root,n):
    slot=root/f'{n:02d}'
    cfg=read(root/'CONFIG.json')
    job=os.environ['PBS_JOBID']
    for _ in range(45):
        if (slot/'JOB.json').is_file():
            break
        time.sleep(2)
    if (read(slot/'JOB.json')['job']!=job or cfg['plan']!=PLAN
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=cfg['driver_sha256']):
        raise PermissionError('Unbound VANDA dispatcher')
    row=job_row(job)
    if (row['job_state']!='R' or row['project']!=PROJECT or row['Job_Owner'].split('@')[0]!=os.environ['USER']
            or int(row['Resource_List']['ncpus'])!=1 or int(row['Resource_List'].get('ngpus',0))!=0
            or wall_seconds(row['Resource_List']['walltime'])!=WALL):
        raise PermissionError('Unrecognized VANDA CPU allocation')
    if n==1 and cfg.get('cold_predecessor'):
        prior=cfg['cold_predecessor']
        old=Path(prior['folder'])
        if (read(old/'01/JOB.json')['job']!=prior['job'] or (old/'01/SUCCESSOR.json').exists()
                or read(old/'STOP').get('successor')!=job or job_row(prior['job'])['job_state']!='F'
                or read(old/'01/RESULT.json')['status']!='STOP_LIMIT_OR_DEADLINE'):
            raise PermissionError('Prior scheduler has not completed the registered cold handoff')
    if n==1 and cfg.get('reconciled_dispatch'):
        prior=cfg['reconciled_dispatch']
        slot0=Path(prior['folder'])/'01'
        evidence=slot0/'INTERRUPTION.json'
        if (read(slot0/'JOB.json')['job']!=prior['job'] or job_row(prior['job'])['job_state']!='F'
                or (slot0/'SUCCESSOR.json').exists() or read(evidence)['job']!=prior['job']
                or hashlib.sha256(evidence.read_bytes()).hexdigest()!=prior['evidence_sha256']):
            raise PermissionError('Unreconciled dispatcher interruption')
    if n==1 and cfg.get('local_relay'):
        old=cfg['local_relay']
        target=Path(old['mount']['path']).parent
        prior=job_row(old['job'])
        terminal=read(target/'RESULT.private.json')
        if (prior['job_state']!='F' or int(prior['Exit_status'])!=0 or prior['project']!=PROJECT
                or read(target/'RELEASE.private.json')['job']!=old['job']
                or terminal['job']!=old['job']):
            raise PermissionError('First snapshot relay did not finish its natural allocation')
        if terminal['stop'] or (target/'STOP.json').exists():
            write_once(slot/'RESULT.json',dict(status='STOP_LIMIT_OR_DEADLINE'))
            return
    grant_limit=min(DEADLINE,cfg['grant_end'])
    end=site_epoch(row['stime'])+WALL
    def stopped():
        relay=cfg.get('local_relay')
        return ((root/'STOP').exists() or time.time()>=grant_limit-240 or
                bool(relay and (Path(relay['mount']['path']).parent/'STOP.json').exists()))
    with closing(OwnerLock(root/'scheduler.lock')):
        write_once(slot/'START.json',dict(job=job,pid=os.getpid(),node=os.uname().nodename,epoch=time.time(),
                                           site_end=end,grant_end=grant_limit))
        attempts=0
        window=3600 if cfg.get('local_exchange') or cfg.get('local_relay') else 540
        if cfg.get('local_train_ingress'):
            window=900
        while not stopped() and time.time()<end-window:
            # Arguments and identity are fixed at registration. Only scheduling and status cross this SSH connection.
            if cfg.get('local_train_ingress'):
                command=[cfg['hub_python'],'-B','-m','scripts.sla.core_hopper_hf_train',str(root),'receive',
                         cfg['local_train_ingress']]
            elif cfg.get('local_relay'):
                command=[sys.executable,'-B',str(root/'relay_tick.py'),str(root),str(n)]
            elif cfg.get('local_exchange'):
                command=[cfg['hub_python'],'-B',str(root/'exchange.py'),'tick',str(root)]
            elif cfg.get('local_formal'):
                command=[sys.executable,'-B',str(root/'formal_scheduler.py'),'tick',str(root)]
            elif cfg.get('local_backup'):
                command=[sys.executable,'-B',str(root/'backup.py'),'tick',str(root)]
            else:
                if not re.fullmatch(r'\d+\.stdct-mgmt-02',job):
                    raise PermissionError('Missing VANDA scheduler identity')
                command=[*SSH,HOST,'export SLA_VANDA_CPU_DISPATCH='+shlex.quote(job)+' && '+cfg['hopper_command']]
            proc=subprocess.run(command,text=True,capture_output=True,timeout=window-60)
            if proc.returncode:
                write_once(slot/'INTERRUPTION.json',dict(job=job,returncode=proc.returncode,epoch=time.time(),
                           reason=proc.stderr[-1800:]))
                return
            state=json.loads(proc.stdout)
            attempts+=1
            write(slot/'HEARTBEAT.json',dict(job=job,epoch=time.time(),polls=attempts,hopper=state))
            if state['status'] not in ('WAIT_PBS','SUBMITTED','SELECTION_RECONCILED','COPIED','WAIT_SEAL'):
                write_once(slot/'RESULT.json',state)
                return
            time.sleep(min(1 if state['status']=='COPIED' else 150,max(0,end-window-time.time())))
        if not stopped() and n<MAX_SLOTS:
            successor=submit(root,n+1,job)
            write_once(slot/'SUCCESSOR.json',dict(job=successor,afterok=job))
        else:
            write_once(slot/'RESULT.json',dict(status='STOP_LIMIT_OR_DEADLINE'))


if __name__=='__main__':
    c=argparse.ArgumentParser()
    c.add_argument('action',choices=['submit','run'])
    c.add_argument('root',type=Path)
    c.add_argument('number',type=int)
    a=c.parse_args()
    os.umask(0o077)
    if a.action=='submit':
        if a.number!=1:
            raise ValueError('Only the first dispatcher can be submitted manually')
        with closing(OwnerLock(a.root/'first-submit.lock')):
            print(submit(a.root,1))
    else:
        run(a.root,a.number)
