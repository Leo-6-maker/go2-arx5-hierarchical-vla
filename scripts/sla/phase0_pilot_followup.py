"""One CPU-only project job compares sealed ordinary2 arrays and reconciles the H200 cost."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from compare_phase0_numeric import compare_one
from hopper_numeric_probe import CHECKPOINT, PACKAGE, PANEL, pin, record, sha


ORDS = (0, 7)
FILES = ['CONFIG.json','ordinary2/ENV.json','ordinary2/RESULT.json']
for ordinal in ORDS:
    FILES.extend('ordinary2/%03d.%s' % (ordinal,ext) for ext in ('intent.json','json','private.npz'))


def receive(command, dest):
    dest.mkdir(mode=0o700)
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        seen = set()
        with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
            for member in archive:
                if (not member.isfile() or member.name not in FILES or member.name in seen
                        or member.size > 2*1024**2):
                    raise ValueError('Unexpected private measurement archive')
                f = dest/member.name
                f.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                body = archive.extractfile(member).read()
                if len(body) != member.size:
                    raise EOFError('Incomplete private member')
                with f.open('xb') as out:
                    out.write(body)
                seen.add(member.name)
        if proc.wait(30) or seen != set(FILES):
            raise RuntimeError('Missing or incomplete private ordinary2 results')


def compare(root,cfg):
    spec = pin(root/'PREREGISTRATION.json',cfg['spec_sha'])
    if (spec['checkpoint'],spec['package'],spec['panel']) != (CHECKPOINT,PACKAGE,PANEL):
        raise ValueError('Unexpected phase or panel')
    rows = []
    for site in ('core','hopper'):
        folder = root/site
        bound = pin(folder/'CONFIG.json',cfg[site+'_config'])
        if bound['site'] != site or bound['preregistration'] != cfg['spec_sha']:
            raise ValueError('Wrong diagnostic capsule')
        if json.loads((folder/'ordinary2/RESULT.json').read_bytes())['completed'] != 2:
            raise ValueError('Expected exactly two complete production predictions')
    for ordinal in ORDS:
        row = compare_one(root/'hopper/ordinary2',root/'core/ordinary2',ordinal,
                          spec['equivalence']['per_point_absolute_limits'])
        rows.append(row)
    return dict(rows=rows,scope='TWO_ORDINARY_PRODUCTION_ROWS',checkpoint=CHECKPOINT,
                calibration_total=16,independent_total=12,native_admission=False)


def run(root,cfg):
    job = os.environ['PBS_JOBID']
    ticket = json.loads((root/'CPU_JOB.json').read_bytes())
    live = json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',job],
                      text=True,timeout=35))['Jobs'][job]
    if (job!=ticket['job'] or ticket['spec']!=sha(root/'CONFIG.json') or live['job_state']!='R'
            or live['Job_Owner'].split('@')[0]!=os.environ['USER'] or live['project']!=cfg['project']
            or int(live['Resource_List']['ncpus'])!=1 or int(live['Resource_List'].get('ngpus',0))!=0
            or live['Resource_List']['walltime']!='02:00:00' or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise PermissionError('Unexpected CPU allocation')
    ssh=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
         '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','hopper.nus.edu.sg']
    end=cfg['deadline']+600
    errors=0
    while time.time()<end and not (root/'STOP').exists():
        try:
            response=subprocess.check_output([*ssh,shlex.join(cfg['status_command'])],text=True,timeout=90)
            status=json.loads(response)
        except (subprocess.SubprocessError,ValueError) as error:
            errors+=1
            if errors>=3:
                raise RuntimeError('Three consecutive site receipt failures') from error
            time.sleep(45)
            continue
        errors=0
        if status['job']!=cfg['hopper_job']:
            raise ValueError('Unexpected H200 receipt')
        if status['ordinary'] and not (root/'COMPARISON.json').exists():
            receive([*ssh,shlex.join(['tar','-C',cfg['hopper_root'],'-cf','-','--',*FILES])],root/'hopper')
            record(root/'COMPARISON.json',compare(root,cfg))
        if status['settled']:
            record(root/'RESULT.json',dict(status='PBS_SETTLED',hopper=status,
                paired=2 if (root/'COMPARISON.json').exists() else 0,updates=0,native_requests=0))
            return
        time.sleep(120)
    record(root/'RESULT.json',dict(status='STOP_OR_RECEIPT_DEADLINE',
           paired=2 if (root/'COMPARISON.json').exists() else 0,updates=0,native_requests=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('spec_sha')
    args=parser.parse_args()
    os.umask(0o077)
    config=pin(args.root/'CONFIG.json',args.spec_sha)
    for name,digest in config['sources'].items():
        if sha(args.root/name)!=digest:
            raise ValueError('CPU follow-up source changed')
    try:
        run(args.root,config)
    except Exception as exc:
        record(args.root/'INTERRUPTED.json',dict(reason=type(exc).__name__+': '+str(exc),updates=0,native_requests=0))
        raise
