"""Register only the two matched H200 arms after the original native block is settled."""
from argparse import ArgumentParser
from contextlib import closing
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from scripts.pi0292.durable import OwnerLock
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_pbs import PBS, PROJECT, BalanceLoginExpired, native, pin, project_grant
from scripts.sla.hopper_project_admission import SELECT, allocation, project_period
from scripts.sla.hopper_site import DEADLINE, T0, pinned, wall_seconds
from scripts.sla.main3800_auxiliary_training import ARMS, PARENT, SCHEMA, START, STOP, plan_rows


def run_script(root, cfg):
    runtime = Path(cfg['runtime'])
    source = Path(cfg['source'])
    python = runtime/'interpreter/bin/python3.12'
    paths = (source,source/'src',runtime/'upstream',runtime/'packages',runtime/'data_dependencies')
    bootstrap = shlex.join([str(python),'-B','-m','scripts.sla.main3800_auxiliary_training',str(root)])
    verifier = shlex.join([str(python),'-B','-m','scripts.sla.hopper_project_admission','running',str(root)])
    return '\n'.join(('#!/bin/bash -l',f'#PBS -P {PROJECT}','#PBS -q auto',f'#PBS -l select={SELECT}',
        f'#PBS -o {root}/stdout.log',f'#PBS -e {root}/stderr.log','#PBS -r n','#PBS -m n',
        'set -euo pipefail','umask 077','cd '+shlex.quote(str(root)),
        'export PBS_SERVER='+shlex.quote(os.environ['PBS_SERVER']),
        'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2',
        'export JAX_PLATFORMS=cuda PYTHONPATH='+shlex.quote(os.pathsep.join(map(str,paths))),
        'sha256sum --status -c MOUNT.sha256',verifier,'exec '+bootstrap+''' "$(cut -d' ' -f1 MOUNT.sha256)"''',''))


def submit(root, config):
    cfg = pinned(config)
    lineup = Path(cfg['lineage']).resolve(strict=True)
    custody = Path(cfg['custody']).resolve(strict=True)
    source = Path(cfg['source']).resolve(strict=True)
    seconds = cfg['wall_seconds_per_arm']
    gate = pinned(cfg['native_gate'])
    main = cfg['original_main_segment']
    if (os.uname().nodename.split('.')[0]!='hopper-l-01' or os.environ.get('PBS_JOBID')
            or cfg['schema']!=SCHEMA or gate['native_requests']!=6 or gate['parent']!=PARENT
            or gate['selected_intervention']!='AUXILIARY_SLOT_SELECTION' or main['previous']['sha256']!=PARENT
            or digest(Path(main['previous']['path'])/'COMPLETE.json')!=PARENT
            or cfg['deadline']>DEADLINE or time.time()+2*seconds>=cfg['deadline']
            or not 3600<=seconds<=21600 or cfg['prior_gpu_seconds_upper']+2*seconds>320*3600
            or shutil.disk_usage(root).free<45*1024**3 or any((p/'STOP.json').exists() for p in (root,custody,lineup))
            or source!=Path(__file__).resolve().parents[2]):
        raise PermissionError('Missing complete native decision, original state or usable Hopper allocation')
    plan_rows(cfg)
    for name,sha in pinned(cfg['source_members']).items():
        if digest(source/name)!=sha:
            raise ValueError('Registered exact source overlay changed')
    period = project_period(cfg['project_basis'],lineup,seconds)
    try:
        grant = project_grant(2*seconds)
    except BalanceLoginExpired:
        grant = period
    if time.time()+2*seconds>=grant['expiry_epoch']:
        raise PermissionError('Matched allocations exceed the actual project period')
    jobs, staged = {}, []
    with closing(OwnerLock(root/'submit.lock')), closing(OwnerLock(lineup/'MAIN/OWNER.lock')):
        for index,arm in enumerate(ARMS):
            folder = root/arm
            folder.mkdir(exist_ok=True)
            name = f'slam38aux{arm.lower()}01'
            script = folder/'job.pbs'
            body = run_script(folder,cfg)
            if not script.exists():
                script.write_text(body)
            if script.read_text()!=body:
                raise ValueError('Matched job program changed')
            grant_path = folder/'GRANT.json'
            if not grant_path.exists():
                write_once(grant_path,grant)
            grant_ref = pin(grant_path)
            request = dict(name=name,arm=arm,config=config,script=pin(script),grant=grant_ref,control_job=jobs.get('A'))
            journal = Journal(folder/'journal',SCHEMA+'-'+arm)
            wall = f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'

            def hold():
                command = [str(PBS/'qsub'),'-h','-N',name,'-l','walltime='+wall]
                if arm=='B':
                    command += ['-W','depend=afterok:'+jobs['A']]
                job = subprocess.check_output([*command,str(script)],text=True,timeout=75).strip()
                if not re.fullmatch(r'\d+\.hopper-m-02',job):
                    raise ValueError('Scheduler returned an unknown submission identity')
                return job

            job = journal.once('qsub-held',request,hold)
            jobs[arm] = job
            if (folder/'RELEASE.json').exists():
                continue
            row = native(job)
            rate = allocation(row,grant['account'],seconds,'H')
            if row['Job_Name']!=name or (arm=='B' and ('afterok:'+jobs['A']) not in row.get('depend','')):
                raise PermissionError('Held PBS identity or A to B dependency differs')
            if 'balance_su' in grant:
                required = grant['required_units_at_queue_rate_one']*rate
                if required>min(grant['balance_su'],grant['balance_gpu_units']):
                    raise PermissionError('Current balance does not cover the matched pair')
            spec = folder/'EXPERIMENT.json'
            write_once(spec,dict(cfg,arm=arm,start=START,stop=STOP,updates_per_arm=500,physics=False,
                                hardware_connections_allowed=False))
            reservation = folder/'RESERVATION.json'
            write_once(reservation,dict(plan=main['plan'],job=job,group='training',reserved_gpu_count=1,
                       reserved_seconds=seconds,all_prior_seconds=cfg['prior_gpu_seconds_upper']+index*seconds,
                       engineering_prior_seconds=cfg['engineering_seconds_upper']))
            path = folder/'MOUNT.private.json'
            write_once(path,dict(plan=main['plan'],T0_epoch=T0,deadline_epoch=cfg['deadline'],uid=os.getuid(),
                       project=PROJECT,job=job,qstat=str(PBS/'qstat'),source_revision=cfg['source_revision'],
                       custody=str(custody),lineage=str(lineup),comparison_root=str(root),control_job=request['control_job'],
                       compilation_cache_dir=cfg['compilation_cache_dir'],gpu_reservation=pin(reservation),
                       site_grant=grant_ref,pbs_project_basis=cfg['project_basis'],auxiliary_experiment=pin(spec)))
            checksum = folder/'MOUNT.sha256'
            expected = digest(path)+'  MOUNT.private.json\n'
            if not checksum.exists():
                checksum.write_text(expected)
            if checksum.read_text()!=expected:
                raise ValueError('Existing held job mount checksum changed')
            staged.append((folder,journal,job,path))

        for folder,journal,job,path in staged:
            def release():
                subprocess.run([str(PBS/'qrls'),job],check=True,timeout=45)
                actual = native(job)
                if (actual['job_state'] not in ('H','Q','R') or actual['project']!=PROJECT
                        or wall_seconds(actual['Resource_List']['walltime'])!=seconds):
                    raise ValueError('Released job requires independent scheduler reconciliation')
                return dict(job=job,state=actual['job_state'],dependency=actual.get('depend'),mount=pin(path))

            recorded = journal.once('release',dict(job=job,mount=pin(path)),release)
            write_once(folder/'RELEASE.json',recorded)
    return jobs


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('config_sha')
    args = cli.parse_args()
    os.umask(0o077)
    print(json.dumps(submit(args.root,pin(args.root/'CONFIG.json')|{'sha256':args.config_sha}),sort_keys=True))
