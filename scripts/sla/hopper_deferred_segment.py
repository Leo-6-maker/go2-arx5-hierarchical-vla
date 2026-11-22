"""Queue the next GPU training segment behind the actual predecessor's PBS exit."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_site import DEADLINE, pinned, site_epoch

from hopper_advance import segment
from hopper_pbs import PBS, PROJECT, BalanceLoginExpired, native, pin, previous_cost, project_grant
from hopper_project_admission import SELECT, allocation, project_period, running


def old_state(line, ordinal):
    previous = line/'allocations'/f'{ordinal:03d}'
    mount = read(previous/'MOUNT.private.json')
    release = read(previous/'RELEASE.json')
    request = pinned(mount['training_segment'])
    reservation = pinned(mount['gpu_reservation'])
    if (mount['plan']!=PLAN or mount['lineage']!=str(line) or mount['deadline_epoch']!=DEADLINE
            or request['job']!=release['job'] or mount['job']!=release['job']
            or reservation['job']!=release['job'] or reservation['group']!='training'
            or request['role']!='MAIN' or not request.get('course') or not mount.get('input_reader')):
        raise ValueError('Expected the current original MAIN and its exact data reader')
    pinned(mount['input_reader']['certificate'])
    pinned(request['course'])
    return previous, mount, request, reservation


def program(folder, mount, source, python, seconds, name, deadline):
    reader = Path(mount['input_reader']['source_directory'])
    runtime = python.parents[2]
    paths = os.pathsep.join(map(str,(reader,reader/'src',source.parent,runtime/'upstream',
                                    runtime/'packages',runtime/'data_dependencies')))
    clock = f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'
    command = shlex.join([str(python),'-B',str(source),'run',str(folder)])
    verify = 'printf "%s\\n" '+shlex.quote(digest(source)+'  '+str(source))+' | sha256sum --status -c -'
    return '\n'.join(('#!/bin/bash -l',f'#PBS -N {name}',f'#PBS -P {PROJECT}','#PBS -q auto',
        f'#PBS -l select={SELECT}',f'#PBS -l walltime={clock}','#PBS -m n','#PBS -r n',
        f'#PBS -o {folder}/stdout.log',f'#PBS -e {folder}/stderr.log','set -euo pipefail','umask 077',
        'cd '+shlex.quote(str(folder)),'test "$(date +%s)" -lt '+str(deadline),
        'export PBS_SERVER='+shlex.quote(os.environ['PBS_SERVER']),
        'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2',
        'export JAX_PLATFORMS=cuda PYTHONPATH='+shlex.quote(paths),verify,'exec '+command,''))


def register(line, ordinal, seconds, inspect_only):
    if os.uname().nodename.split('.')[0]!='hopper-l-01' or os.environ.get('PBS_JOBID'):
        raise PermissionError('Register from the original Hopper submit owner')
    if not 900<seconds<=21600 or not time.time()+seconds+900<DEADLINE:
        raise ValueError('The dependent allocation must fit the original absolute deadline')
    line = line.resolve(strict=True)
    previous, mount, request, reserve = old_state(line,ordinal)
    old_job = mount['job']
    row = native(old_job)
    allocation(row,os.environ['USER'],reserve['reserved_seconds'],'R')
    if any((p/'STOP.json').exists() for p in (line,Path(mount['custody']),previous)):
        raise PermissionError('Predecessor or lineage has been stopped')
    daily_end = min(DEADLINE,pinned(mount['site_grant'])['expiry_epoch'])
    projected_start = site_epoch(row['stime'])+reserve['reserved_seconds']
    if projected_start+seconds>=daily_end:
        raise PermissionError('The expected dependent job would cross the site period')
    folder = line/'allocations'/f'{ordinal+1:03d}'
    name = f'slachv1t{ordinal+1:02d}'
    source = Path(__file__).resolve()
    python = Path(mount['custody'])/'sla-core-hopper-v1-runtime-20261121-01/runtime.verified/interpreter/bin/python3.12'
    modules = [pin(source.with_name(n)) for n in ('hopper_advance.py','hopper_pbs.py','hopper_project_admission.py')]
    with closing(OwnerLock(line/'advance.lock')), closing(OwnerLock(line/'submit.lock')):
        if folder.exists() or max(int(p.name) for p in folder.parent.iterdir() if p.name.isdigit())!=ordinal:
            raise ValueError('A dependent allocation already has a registration or later owner')
        cost, engineering = previous_cost(line,previous)
        upper = cost+reserve['reserved_seconds']
        if upper+seconds>320*3600:
            raise PermissionError('Dependent reservation exceeds the original cumulative GPU cap')
        basis = mount['pbs_project_basis']
        period = project_period(basis,line,seconds)
        try:
            grant = project_grant(seconds)
        except BalanceLoginExpired:
            grant = period
        script = program(folder,mount,source,python,seconds,name,int(daily_end)-seconds)
        if inspect_only:
            subprocess.run(['bash','-n'],input=script,text=True,check=True,timeout=15)
            return dict(status='ACTUAL_PBS_AND_BUDGET_PREFLIGHT',predecessor=old_job,
                        planned_previous_stop=request['stop'],old_live_state=row['job_state'],
                        conservative_prior_GPU_seconds=upper,reserved_next_seconds=seconds,
                        project=PROJECT,new_PBS_jobs=0)
        ids = subprocess.check_output([str(PBS/'qselect'),'-u',grant['account'],'-N',name,
            '-q','@'+os.environ['PBS_SERVER']],text=True,timeout=45).split()
        if ids:
            raise ValueError('A same-name PBS job needs reconciliation')
        folder.mkdir()
        write_once(folder/'GRANT.json',grant)
        with (folder/'job.pbs').open('x') as f:
            f.write(script)
            f.flush()
            os.fsync(f.fileno())
        intent = dict(plan=PLAN,project=PROJECT,group='training',name=name,wall_seconds=seconds,
                      pbs=pin(folder/'job.pbs'),predecessor=old_job,previous_mount=pin(previous/'MOUNT.private.json'),
                      source=mount['source_revision'],deferred_worker=pin(source),deadline=DEADLINE,
                      grant=pin(folder/'GRANT.json'),project_basis=basis,ordinal=ordinal,modules=modules,
                      conservative_prior_GPU_seconds=upper)
        write_once(folder/'INTENT.json',intent)
        journal = Journal(folder/'journal',folder.name)

        def submit():
            job = subprocess.check_output([str(PBS/'qsub'),'-h','-W','depend=afterok:'+old_job,
                                           str(folder/'job.pbs')],text=True,timeout=75).strip()
            if not re.fullmatch(r'\d+\.hopper-m-02',job):
                raise ValueError('Unknown PBS reply, preserve the registration for reconciliation')
            write_once(folder/'JOB.json',dict(job=job,name=name))
            return job

        job = journal.once('qsub',intent,submit)
        held = native(job)
        rate = allocation(held,grant['account'],seconds,'H')
        if (held['Job_Name']!=name or not held.get('depend','').startswith('afterok:'+old_job)
                or ('balance_gpu_units' in grant and rate*grant['required_units_at_queue_rate_one']>
                    min(grant['balance_gpu_units'],grant['balance_su']))):
            raise PermissionError('Held job dependency or project reservation differs')
        write_once(folder/'RESERVATION.json',dict(plan=PLAN,job=job,group='training',
                   reserved_gpu_count=1,reserved_seconds=seconds,all_prior_seconds=upper,
                   engineering_prior_seconds=engineering,accounting='CONSERVATIVE_ACTIVE_UPPER_BOUND'))
        release = dict(job=job,epoch=time.time(),pbs_name=name,dependency='afterok:'+old_job)
        def expose():
            subprocess.run([str(PBS/'qrls'),job],check=True,timeout=60)
            return release
        got = journal.once('release',dict(job=job,intent=pin(folder/'INTENT.json')),expose)
        write_once(folder/'RELEASE.json',got)
        observed = native(job)
        return dict(status='PBS_DEPENDENT_TRAINING_REGISTERED',**got,observed_state=observed['job_state'],
                    observed_dependency=observed.get('depend'),reserved_seconds=seconds)


def run(folder):
    i = read(folder/'INTENT.json')
    r = read(folder/'RESERVATION.json')
    job = os.environ['PBS_JOBID']
    if (i['deferred_worker']!=pin(Path(__file__)) or i['pbs']!=pin(folder/'job.pbs') or r['job']!=job
            or any(pin(Path(ref['path']))!=ref for ref in i['modules'])):
        raise PermissionError('Dependent execution differs from the registered PBS owner')
    line = folder.parents[1]
    previous, old, request, _ = old_state(line,i['ordinal'])
    if old['job']!=i['predecessor'] or i['previous_mount']!=pin(previous/'MOUNT.private.json'):
        raise ValueError('PBS predecessor changed')
    last = native(i['predecessor'])
    result = read(previous/'RESULT.private.json')
    checkpoint = read(line/'MAIN'/'sealed'/f"phase{result['phase']:05d}.json")
    if (last['job_state']!='F' or int(last['Exit_status'])!=0 or not request['start']<result['phase']<=request['stop']
            or result['status'] not in ('SEGMENT_COMPLETE','SAVED_AT_ALLOCATION_OR_STOP')
            or (previous/'PARTIAL.private.json').exists() or checkpoint['checkpoint']!=result['last_checkpoint']
            or any((p/'STOP.json').exists() for p in (folder,previous,line,Path(old['custody'])))):
        raise ValueError('The predecessor has no clean sealed continuation')
    live = native(job)
    allocation(live,os.environ['USER'],i['wall_seconds'],'R')
    total, engineering = previous_cost(line,folder)
    if total>r['all_prior_seconds'] or engineering!=r['engineering_prior_seconds']:
        raise ValueError('Actual prior use exceeds the pending reservation')
    config = segment(line,request['selection'],Path(request['population_file']),Path(old['custody']),
                     old['source_revision'],i['wall_seconds'])
    if (config is None or config['start']!=result['phase'] or config['previous']!=result['last_checkpoint']
            or config['course']!=request['course'] or config['role']!='MAIN'):
        raise ValueError('This pending segment cannot change the original data course')
    write_once(folder/'SEGMENT.json',dict(config,job=job,input_reader=old['input_reader']))
    mount = dict(old,job=job,training_segment=pin(folder/'SEGMENT.json'),
                 gpu_reservation=pin(folder/'RESERVATION.json'),site_grant=i['grant'])
    target = folder/'MOUNT.private.json'
    write_once(target,mount)
    write_once(folder/'DEPENDENCY_ACTUAL.json',dict(job=job,predecessor=i['predecessor'],
        predecessor_terminal=pin(previous/'RESULT.private.json'),start=config['start'],stop=config['stop'],
        checkpoint=config['previous'],actual_prior_GPU_seconds=total))
    running(folder)
    os.execv(sys.executable,[sys.executable,'-B','-m','scripts.sla.hopper_training',str(folder),digest(target)])


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest='operation',required=True)
    command = sub.add_parser('register')
    command.add_argument('line',type=Path)
    command.add_argument('ordinal',type=int)
    command.add_argument('seconds',type=int)
    command.add_argument('--inspect',action='store_true')
    sub.add_parser('run').add_argument('folder',type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    if args.operation=='run':
        run(args.folder)
    else:
        import json
        print(json.dumps(register(args.line,args.ordinal,args.seconds,args.inspect)))
