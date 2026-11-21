"""Register a finite science capsule on Hopper with the real project and cumulative PBS cost."""
import argparse
import ast
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_parent_custody import PLAN, PARENT, digest, write_once
from scripts.sla.hopper_site import T0, DEADLINE, site_epoch, wall_seconds

PBS = Path('/cm/shared/apps/pbspro/current/bin')
PROJECT = 'CFP04-CF-007'
SGT = timezone(timedelta(hours=8))


def pin(path):
    return dict(path=str(path.resolve(strict=True)),sha256=digest(path))


def project_grant(seconds):
    account = os.environ['USER']
    amgr = '/opt/am/python/bin/amgr'
    records=json.loads(subprocess.check_output([amgr,'ls','project','--name',PROJECT,'--json-info'],text=True,timeout=30))
    balance=ast.literal_eval(subprocess.check_output([amgr,'checkbalance','project','--name',PROJECT],text=True,timeout=30))
    tariff=json.loads(subprocess.check_output([amgr,'ls','cluster','--name','hopper-m-02','--formula','--json-info'],
                                              text=True,timeout=30))
    constants=tariff['constants']
    expressions=tariff['formulas']
    if (constants['ngpus']!='job.ngpus' or constants['queue_charge_rate']!='job.queue_charge_rate'
            or constants['walltime']!='job.walltime' or float(constants['CONST_zero'])!=0
            or float(constants['CONST_seconds'])!=3600 or float(constants['CONST_ncpus_rate'])!=1
            or ''.join(expressions['gpu_hrs'].split())!='(ngpus*walltime/CONST_seconds)*(CONST_ngpus_rate*queue_charge_rate)'
            or ''.join(expressions['su'].split())!='((ngpus==CONST_zero)and((ncpus*walltime/CONST_seconds)*(CONST_ncpus_rate*queue_charge_rate))orCONST_zero)+(ngpus*walltime/CONST_seconds)*CONST_ngpus_rate*queue_charge_rate'):
        raise PermissionError('Unrecognized Hopper project tariff')
    rate=float(constants['CONST_ngpus_rate'])
    gpu,su=float(balance['gpu_hrs']),float(balance['su'])
    if not all(math.isfinite(v) and v>0 for v in (rate,gpu,su)):
        raise PermissionError('Invalid Hopper rate or project balance')
    required=seconds*rate/3600
    if len(records)!=1:
        raise PermissionError('Missing unambiguous current project grant')
    p=records[0]
    start=datetime.combine(date.fromisoformat(p['start_date']),datetime.min.time(),SGT).timestamp()
    expiry=datetime.combine(date.fromisoformat(p['end_date'])+timedelta(days=1),datetime.min.time(),SGT).timestamp()
    now=time.time()
    if (p['name']!=PROJECT or p['active'] is not True or account not in p['users'] or 'hopper-m-02' not in p['cluster']
            or not start<=now<expiry or min(gpu,su)<required):
        raise PermissionError('Insufficient current project grant on Hopper')
    return dict(project=PROJECT, account=account,valid_from_epoch=start,expiry_epoch=expiry,observed_epoch=now,
                balance_gpu_units=gpu,balance_su=su,units_per_gpu_hour_at_queue_rate_one=rate,
                required_units_at_queue_rate_one=required,source='HOPPER_AMGR_MEMBERSHIP_BALANCE_AND_CLUSTER_FORMULA')


def native(job):
    server=os.environ['PBS_SERVER']
    return json.loads(subprocess.check_output([str(PBS/'qstat'),'-x','-f','-F','json',job+'@'+server],
                      text=True,timeout=45))['Jobs'][job]


def charged(row, account, reservation):
    if (row['job_state']!='F' or row['Job_Owner'].split('@')[0]!=account
            or row['project']!=PROJECT or int(row['Resource_List']['ngpus'])!=1
            or wall_seconds(row['Resource_List']['walltime'])!=reservation['reserved_seconds']):
        raise PermissionError('Prior allocation is still live or its identity changed')
    if not row.get('stime'):
        if row.get('exec_host') or row.get('run_count',0):
            raise RuntimeError('Indeterminate allocation start')
        return 0
    elapsed=wall_seconds(row['resources_used']['walltime'])
    if row.get('obittime'):
        elapsed=max(elapsed,math.ceil(site_epoch(row['obittime'])-site_epoch(row['stime'])))
    return elapsed


def previous_cost(root,current):
    total=engineering=0
    for folder in sorted((root/'allocations').iterdir()):
        if folder==current or not folder.is_dir():
            continue
        if not (folder/'INTENT.json').exists():
            continue
        if not (folder/'RESERVATION.json').exists():
            raise RuntimeError('Unresolved prior submission or reservation')
        reservation=read(folder/'RESERVATION.json')
        proof=folder/'PBS_FINISHED.json'
        if not proof.exists():
            row=native(reservation['job'])
            used=charged(row,os.environ['USER'],reservation)
            write_once(proof,dict(job=reservation['job'],exit=row['Exit_status'],GPU_seconds=used,
                                 pbs_used_wall=row.get('resources_used',{}).get('walltime'),group=reservation['group']))
        settled=read(proof)
        if settled['job']!=reservation['job'] or settled['group']!=reservation['group']:
            raise ValueError('Incorrect prior settlement')
        total+=settled['GPU_seconds']
        if reservation['group']=='engineering':
            engineering+=settled['GPU_seconds']
    return total,engineering


def batch_text(run,source,runtime,name,group,wall):
    module={'engineering':'scripts.sla.hopper_engineering_batch','training':'scripts.sla.hopper_training',
            'reference':'scripts.sla.hopper_reference'}[group]
    python=runtime/'interpreter/bin/python3.12'
    paths=':'.join(str(p) for p in (source,source/'src',runtime/'data_dependencies',runtime/'upstream',runtime/'packages'))
    command=shlex.join([str(python),'-B','-m',module,str(run)])
    if group in ('training','reference'):
        command+=' "$(cut -d\' \' -f1 MOUNT.sha256)"'
    directives=['#!/bin/bash -l','#PBS -N '+name,'#PBS -P '+PROJECT,'#PBS -q auto',
        '#PBS -l select=1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB','#PBS -l walltime='+wall,
        '#PBS -m n','#PBS -r n','#PBS -o '+str(run/'stdout.log'),'#PBS -e '+str(run/'stderr.log')]
    env=['set -euo pipefail','umask 077','cd '+shlex.quote(str(run)),
         'export PBS_SERVER='+shlex.quote(os.environ['PBS_SERVER']),
         'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2',
         'export JAX_PLATFORMS=cuda PYTHONPATH='+shlex.quote(paths),'sha256sum --status -c MOUNT.sha256',
         'exec '+command]
    return '\n'.join(directives+env)+'\n'


def mixed_reader(segment, original, candidate, custody):
    from scripts.sla.hopper_site import pinned

    record = read(segment)
    if record['course'] is None:
        return None
    course = pinned(record['course'])
    return certified_reader(Path(course['folder']),course['seal_sha256'],course['manifest_sha256'],
                            original,candidate,custody)


def certified_reader(folder, seal_sha, manifest_sha, original, candidate, custody):
    from hopper_reader_compat import compare
    from scripts.sla.hopper_site import pinned

    if folder.parent.resolve() != custody.joinpath('train-ready').resolve() or digest(folder/'SEALED.private.json')!=seal_sha:
        raise ValueError('Course does not point to the sealed Hopper TRAIN intake')
    seal = read(folder/'SEALED.private.json')
    proof = pinned(dict(path=str(folder/'ACTUAL.private.json'), sha256=seal['admission_sha256']))
    if 'schema' not in proof:
        return None
    # Existing mixed courses keep the reader which certified their original intake.
    candidates = candidate if isinstance(candidate,(tuple,list)) else ([candidate] if candidate else [])
    name = seal_sha+'.json'
    matches = [c for c in candidates if (c/'certificates'/name).is_file()]
    if len(matches)!=1:
        raise ValueError('A mixed course must have one registered cold reader certificate')
    candidate = matches[0]
    verdict = candidate/'certificates'/name
    cert = read(verdict)
    sources = compare(original,candidate/'source')
    if (cert['plan']!=PLAN or cert['status']!='HOPPER_MIXED_TRAIN_APPROVED'
            or cert['seal']!=seal_sha or cert['manifest']!=manifest_sha
            or cert['windows']!=seal['windows'] or cert['episodes']!=seal['count']
            or cert['sources']!=sources or proof.get('producer_spec') is not None):
        raise ValueError('Course receipt or verified input runtime does not match its certificate')
    return dict(**sources, certificate=pin(verdict), source_directory=str(candidate/'source'))


def register(root,custody,source_folder,number,group,seconds,segment=None,recovery=None,reader_source=None):
    if number<1 or number>999 or not 900<seconds<=21600 or time.time()>=DEADLINE-900 or (custody/'STOP.json').exists():
        raise PermissionError('Campaign stop or invalid capsule allowance')
    if bool(segment)!=(group in ('training','reference')):
        raise ValueError('Training requires a frozen segment request')
    host=os.uname().nodename.split('.')[0]
    carried=os.environ.get('PBS_JOBID','')
    external=os.environ.get('SLA_VANDA_CPU_DISPATCH','')
    if (host!='hopper-l-01' or (carried and
            (not re.fullmatch(r'\d+\.stdct-mgmt-02',external) or external!=carried))):
        raise PermissionError(f'Expected the verified Hopper login node and submitting context: {host} {carried!r} {external!r}')
    root.mkdir(parents=True,exist_ok=True)
    allocations=root/'allocations'
    allocations.mkdir(exist_ok=True)
    run=allocations/f'{number:03d}'
    run.mkdir(exist_ok=True)
    with closing(OwnerLock(root/'submit.lock')):
        if [d for d in allocations.iterdir() if d.name.isdigit() and int(d.name)>number]:
            raise ValueError('Allocation order changed')
        if (run/'RELEASE.json').is_file():
            return read(run/'RELEASE.json')['job']
        login=run/'SUBMIT_HOST.json'
        if not login.is_file():
            write_once(login,dict(host=host,cpu_dispatch=external or None,inherited_job=carried or None))
        core=read(source_folder/'SOURCE_READY.json')
        source=source_folder/'source'
        manifest=read(source/'SOURCE_BINDING.private.json')
        if (core['source']!=manifest['new_revision'] or core['archive_sha256']!=digest(source_folder/'source.tar.gz')
                or not all(digest(source/n)==v['sha256'] for n,v in manifest['modules'].items())):
            raise ValueError('Scientific source registration changed')
        reader = mixed_reader(segment,source,reader_source,custody) if group=='training' else None
        execution_source = Path(reader['source_directory']) if reader else source
        runtime=custody/'sla-core-hopper-v1-runtime-20261121-01/runtime.verified'
        if digest(custody/'parent.verified/COMPLETE.json')!=PARENT:
            raise ValueError('Original parent checkpoint changed')
        tag={'engineering':'e','training':'t','reference':'r'}[group]
        name=f'slachv1{tag}{number:02d}'
        total,eng=previous_cost(root,run)
        engineering_root = root/'engineering'
        if recovery:
            rec = read(recovery)
            first = read(allocations/'001'/'PBS_FINISHED.json')
            if (recovery != custody/'TECHNICAL_RECOVERY.json' or rec['custody'] != str(custody)
                    or rec['source'] != core['source'] or rec['plan'] != PLAN or rec['job'] != first['job']
                    or first['group'] != 'engineering' or first['exit'] == 0 or number < 2
                    or rec['technical_reserve_groups'] != 1
                    or any(rec[k] != 0 for k in ('updates','optimization_visits','generations','native'))
                    or rec['engineering_root'] != str(root/'engineering-recovery-01')):
                raise ValueError('Technical reserve requires a reconciled, charged, pre-update interruption')
            engineering_root = Path(rec['engineering_root'])
        if total+seconds>320*3600 or (group=='engineering' and eng+seconds>8*3600):
            raise PermissionError('Remaining actual GPU budget exhausted')
        grantfile=run/'GRANT.json'
        if not grantfile.exists():
            write_once(grantfile,project_grant(seconds))
        grant=read(grantfile)
        if time.time()>=grant['expiry_epoch']:
            raise PermissionError('Site grant ended')
        intent=run/'INTENT.json'
        jobfile=run/'JOB.json'
        wall=f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'
        script=run/'job.pbs'
        body=batch_text(run,execution_source,runtime,name,group,wall)
        if not script.exists():
            with script.open('x') as f:
                f.write(body);f.flush();os.fsync(f.fileno())
        if script.read_text()!=body:
            raise ValueError('Registered batch program changed')
        request=dict(plan=PLAN,project=PROJECT,name=name,group=group,wall_seconds=seconds,source=core['source'],
                     pbs=pin(script),grant=pin(grantfile),science_request=pin(segment) if segment else None)
        if reader:
            request['input_reader'] = reader
        if recovery:
            request['technical_recovery']=pin(recovery)
        if not intent.exists():
            ids=subprocess.check_output([str(PBS/'qselect'),'-u',grant['account'],'-N',name,
                    '-q','@'+os.environ['PBS_SERVER']],text=True,timeout=45).split()
            if ids:
                raise RuntimeError('Same-name scheduler job predates this registration; reconcile')
        write_once(intent,request)
        journal=Journal(run/'journal',run.name)
        def submit():
            job=subprocess.check_output([str(PBS/'qsub'),'-h',str(script)],text=True,timeout=75).strip()
            if not re.fullmatch(r'\d+\.hopper-m-02',job):
                raise RuntimeError('Unrecognized scheduler reply; retain intent')
            write_once(jobfile,dict(job=job,name=name))
            return job
        job=read(jobfile)['job'] if jobfile.exists() else journal.once('qsub',request,submit)
        row=native(job)
        if (row['job_state']!='H' or row['Job_Owner'].split('@')[0]!=grant['account'] or row['project']!=PROJECT
                or row['Job_Name']!=name or row['Resource_List']['select']!='1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB'
                or wall_seconds(row['Resource_List']['walltime'])!=seconds):
            raise PermissionError('Wrong held job or prior release effect requires reconciliation')
        if 'required_units_at_queue_rate_one' in grant:
            queue_rate=float(row['Resource_List']['queue_charge_rate'])
            charge=grant['required_units_at_queue_rate_one']*queue_rate
            if not math.isfinite(queue_rate) or queue_rate<=0 or charge>min(grant['balance_gpu_units'],grant['balance_su']):
                raise PermissionError('Held job charge exceeds the verified project balance')
            write_once(run/'SITE_CHARGE_ESTIMATE.json',dict(job=job,project=PROJECT,reserved_seconds=seconds,
                       queue_rate=queue_rate,gpu_units=charge,su=charge,grant=pin(grantfile)))
        reservation=dict(plan=PLAN,job=job,group=group,reserved_gpu_count=1,reserved_seconds=seconds,
                         all_prior_seconds=total,engineering_prior_seconds=eng)
        write_once(run/'RESERVATION.json',reservation)
        m=dict(plan=PLAN,T0_epoch=T0,deadline_epoch=DEADLINE,uid=os.getuid(),project=PROJECT,job=job,
               qstat=str(PBS/'qstat'),source_revision=core['source'],custody=str(custody),lineage=str(root),
               engineering_root=str(engineering_root),compilation_cache_dir=str(root/'compilation-cache'),
               eval_assets=str(custody/'sla-core-hopper-v1-eval-inputs-20261121-01/inputs.verified'),
               gpu_reservation=pin(run/'RESERVATION.json'),site_grant=pin(grantfile))
        if segment:
            record=read(segment)
            if record['source']!=core['source'] or record['plan']!=PLAN or record['parent']!=PARENT:
                raise ValueError('Invalid proposed scientific segment')
            if reader:
                m['qualified_training_revision'] = core['source']
                m['actual_runtime_revision'] = reader['actual_reader_runtime_revision']
                m['input_reader'] = reader
                record = dict(record, input_reader=reader)
            write_once(run/'SEGMENT.json',dict(record,job=job))
            key='reference_request' if group=='reference' else 'training_segment'
            m[key]=pin(run/'SEGMENT.json')
        write_once(run/'MOUNT.private.json',m)
        check=run/'MOUNT.sha256'
        expected=digest(run/'MOUNT.private.json')+'  MOUNT.private.json\n'
        if not check.exists():
            with check.open('x') as f:
                f.write(expected);f.flush();os.fsync(f.fileno())
        if check.read_text()!=expected:
            raise ValueError('Changed mount checksum')
        def release():
            subprocess.run([str(PBS/'qrls'),job],check=True,timeout=60)
            return dict(job=job,epoch=time.time(),pbs_name=name)
        result=journal.once('release',dict(job=job,mount=pin(run/'MOUNT.private.json')),release)
        write_once(run/'RELEASE.json',result)
        return job


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path);p.add_argument('custody',type=Path);p.add_argument('source_folder',type=Path)
    p.add_argument('number',type=int);p.add_argument('group',choices=['engineering','training','reference'])
    p.add_argument('seconds',type=int);p.add_argument('--segment',type=Path)
    p.add_argument('--technical-recovery',type=Path)
    args=p.parse_args()
    os.umask(0o077)
    print(register(args.root,args.custody,args.source_folder,args.number,args.group,args.seconds,args.segment,args.technical_recovery),flush=True)
