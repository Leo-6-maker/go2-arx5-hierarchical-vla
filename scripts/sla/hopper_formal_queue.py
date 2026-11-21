"""Release at most seven formal A40 jobs from the complete VANDA custody events."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_parent_custody import PLAN, PARENT, digest, write_once
from scripts.sla.hopper_site import DEADLINE, wall_seconds

PBS=Path('/opt/pbs/bin')
MAIN={0,2000,5000,10000,20000}
MATCHED={4000,7000,12000,17000}


def pin(p):
    return dict(path=str(p.resolve(strict=True)),sha256=digest(p))


def endpoint(root, cfg, path):
    b=Path(cfg['backup'])
    record=read(path)
    role,phase=record['role'],record['phase']
    key=f'{role}-phase{phase:05d}'
    common=read(b/'receipts'/(key+'.json'))
    if (path!=b/'formal-pending'/(key+'.json') or record!=dict(common,purpose='formal',status='PENDING')
            or common['plan']!=PLAN or common['parent']!=PARENT or common['eval_due'] is not True or common['arrays']!=140
            or Path(common['custody'])!=b/role/f'phase{phase:05d}'
            or (role=='MAIN' and phase not in MAIN|MATCHED) or (role=='CONTROL2000' and phase not in MATCHED)
            or role not in ('MAIN','CONTROL2000')):
        raise ValueError('Unrecognized formal endpoint or population')
    seal=Path(common['custody'])/'COMPLETE.json'
    if digest(seal)!=common['checkpoint']:
        raise ValueError('Local complete checkpoint changed')
    binding=read(seal)['binding']
    if (binding['source']!=cfg['training_source'] or binding['role']!=role or binding['parent']!=PARENT
            or (role=='CONTROL2000' and binding['role_start']+2000!=phase)):
        raise ValueError('Backed-up phase does not belong to the registered arm')
    return key


def register(root,cfg,path,key):
    run=root/'jobs'/key
    run.mkdir(exist_ok=True)
    jobfile=run/'JOB.json'
    config=root/'FORMAL.json'
    recovery=cfg.get('recovery')
    if recovery and key==recovery['key']:
        failure=read(Path(recovery['evidence']['path']))
        if (digest(Path(recovery['evidence']['path']))!=recovery['evidence']['sha256']
                or failure['checkpoint']!=read(path)['checkpoint'] or failure['status']!='FAILED'
                or failure['attempted'] or failure['generated'] or failure['rows']
                or 'cannot import name \'Queue\' from \'queue\'' not in failure['error']
                or not 0<recovery['paid_seconds']<cfg['max_job_seconds']-900):
            raise ValueError('Unproven pre-generation import interruption')
        config=run/'RECOVERED_CONFIG.json'
        write_once(config,dict(cfg,max_job_seconds=cfg['max_job_seconds']-recovery['paid_seconds']))
    spec=dict(config=pin(config),endpoint=pin(path))
    write_once(run/'SPEC.json',spec)
    jobname='schvf'+('m' if key.startswith('MAIN-') else 'c')+key[-5:]
    wall=read(config)['max_job_seconds']
    body='\n'.join(['#!/bin/bash -l','#PBS -N '+jobname,'#PBS -P '+cfg['project'],'#PBS -q auto',
        '#PBS -l select=1:ncpus=8:mem=112GB:ngpus=1:gpu_model=A40',
        '#PBS -l walltime='+f'{wall//3600:02d}:{wall//60%60:02d}:{wall%60:02d}',
        '#PBS -m n','#PBS -r n','#PBS -o '+str(run/'stdout.log'),'#PBS -e '+str(run/'stderr.log'),
        'set -eu','umask 077','cd '+shlex.quote(str(root)),'sha256sum --status -c SOURCE.sha256',
        'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'export PYTHONPATH='+shlex.quote(cfg['pythonpath']),'cd '+shlex.quote(str(run)),'exec '+shlex.join([cfg['python'],'-B','-m',
        'scripts.sla.hopper_formal_outputs',str(run),'batch'])])+'\n'
    script=run/'job.pbs'
    if not script.exists():
        with script.open('x') as f:
            f.write(body)
    if script.read_text()!=body:
        raise ValueError('Evaluation program changed')
    journal=Journal(run/'journal',key)
    def enqueue():
        old=subprocess.check_output([str(PBS/'qselect'),'-N',jobname,'-u',os.environ['USER']],text=True,timeout=30)
        if old.strip():
            raise RuntimeError('Existing formal job requires custody reconciliation')
        job=subprocess.check_output([str(PBS/'qsub'),'-h',str(script)],text=True,timeout=75).strip()
        if not re.fullmatch(r'\d+\.stdct-mgmt-02',job):
            raise RuntimeError('Ambiguous formal evaluation registration')
        write_once(jobfile,dict(job=job,spec=pin(run/'SPEC.json'),source=cfg['eval_source']))
        return job
    job=read(jobfile)['job'] if jobfile.exists() else journal.once('qsub',dict(spec=spec,pbs=pin(script)),enqueue)
    if not (run/'RELEASE.json').is_file():
        q=json.loads(subprocess.check_output([str(PBS/'qstat'),'-xf','-F','json',job],text=True,timeout=30))['Jobs'][job]
        if (q['job_state']!='H' or q['Job_Owner'].split('@')[0]!=os.environ['USER'] or q['project']!=cfg['project']
                or q['Job_Name']!=jobname or 'gpu_model=A40' not in q['Resource_List']['select']
                or int(q['Resource_List']['ngpus'])!=1 or wall_seconds(q['Resource_List']['walltime'])!=wall):
            raise ValueError('Halted at unmatched held job')
        def release():
            subprocess.run([str(PBS/'qrls'),job],check=True,timeout=45)
            return dict(job=job,epoch=time.time())
        write_once(run/'RELEASE.json',journal.once('release',dict(job=job,spec=spec),release))
    return job


def tick(root):
    cfg=read(root/'FORMAL.json')
    if (cfg['plan']!=PLAN or not re.fullmatch(r'\d+\.stdct-mgmt-02',os.environ.get('PBS_JOBID',''))
            or cfg['control']!=str(root) or cfg['max_job_seconds']!=3600):
        raise PermissionError('Expected a registered VANDA scheduler and one-hour evaluation limit')
    if (root/'STOP').exists() or time.time()>=min(cfg['grant_end'],DEADLINE)-900:
        return dict(status='STOP')
    b=Path(cfg['backup'])
    jobs=root/'jobs'
    jobs.mkdir(exist_ok=True)
    with closing(OwnerLock(root/'queue.lock')):
        pending=sorted((b/'formal-pending').glob('*.json'))
        if len(pending)>7:
            raise ValueError('Formal endpoint allowance exceeded')
        matched={p.stem[-5:] for p in pending if int(p.stem[-5:]) in MATCHED}
        if len(matched)>1:
            raise ValueError('More than one paired endpoint selected')
        complete=[]
        for path in pending:
            # A producer publishes its queue ticket just before the matching custody receipt.
            if not (b/'receipts'/path.name).exists():
                continue
            key=endpoint(root,cfg,path)
            run=jobs/key
            if not (run/'RELEASE.json').is_file():
                return dict(status='SUBMITTED',job=register(root,cfg,path,key),endpoint=key)
            if (run/'RESULT.json').is_file():
                complete.append((key,read(run/'RESULT.json')['status']))
            else:
                job=read(run/'JOB.json')['job']
                state=json.loads(subprocess.check_output([str(PBS/'qstat'),'-xf','-F','json',job],text=True,timeout=30))['Jobs'][job]
                if state['job_state']=='F':
                    missing=dict(status='TERMINAL_WITHOUT_OUTPUT_SEAL',job=job,exit=state['Exit_status'],
                                 pbs_wall=state.get('resources_used',{}).get('walltime'))
                    write_once(run/'PBS_TERMINAL.json',missing)
                    complete.append((key,missing['status']))
        ended=any((read(p).get('status')=='BACKUP_COMPLETE') for p in b.glob('[0-9][0-9]/RESULT.json'))
        if ended and len(complete)==len(pending):
            return dict(status='FORMAL_QUEUE_CLOSED',outcomes=complete)
        return dict(status='WAIT_SEAL',endpoints=len(pending),terminal=complete)


if __name__=='__main__':
    c=argparse.ArgumentParser()
    c.add_argument('mode',choices=['tick'])
    c.add_argument('root',type=Path)
    a=c.parse_args()
    print(json.dumps(tick(a.root)))
