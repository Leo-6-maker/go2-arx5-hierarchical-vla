"""Copy sealed reader evidence to the scheduler, independently of H200 progress and Eval."""
from argparse import ArgumentParser
from contextlib import closing
import os
from pathlib import Path
import subprocess
import time

from hopper_mixed_custody import qualify_one
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.hopper_parent_custody import HOST, SSH, PLAN, digest, write_once
from scripts.sla.hopper_site import DEADLINE

# Receipts are small. Publish the original bytes with an atomic, no-replace link.
PUT = """
import hashlib,os,sys,tempfile
from pathlib import Path
root=Path(sys.argv[1]);name=sys.argv[2];sha=sys.argv[3]
body=sys.stdin.buffer.read(4*1024*1024+1)
assert len(body)<=4*1024*1024 and hashlib.sha256(body).hexdigest()==sha
os.umask(0o077);root.mkdir(exist_ok=True)
with tempfile.NamedTemporaryFile(dir=root,delete=False) as f:
    f.write(body);f.flush();os.fsync(f.fileno())
try:
    try:os.link(f.name,root/name)
    except FileExistsError:pass
    assert (root/name).read_bytes()==body
    fd=os.open(root,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
finally:os.unlink(f.name)
print(sha)
"""


def forward(file, destination):
    import shlex
    value=read(file)
    n=value['actual']['episodes']
    if not 16<n<=240 or value['plan']!=PLAN or value['status']!='MIXED_TRAIN_VANDA_READ_AND_HOPPER_STAGED':
        raise ValueError('Expected the qualified VANDA mixed-data receipt')
    sha=digest(file)
    command=shlex.join(['python3','-B','-c',PUT,destination,f'TRAIN-{n:03d}.json',sha])
    with file.open('rb') as f:
        copied=subprocess.check_output([*SSH,HOST,command],stdin=f,timeout=45).decode().strip()
    if copied!=sha:
        raise ValueError('Hopper readback differs from the local qualification receipt')
    return sha


def run(root, slot):
    cfg=read(root/'CONFIG.json')['mixed_dispatch']
    first=Path(cfg['first_root'])
    data_cfg=read(first/'CONFIG.json')
    if (digest(first/'CONFIG.json')!=cfg['first_config_sha256'] or data_cfg['plan']!=PLAN
            or read(first/'JOB.json')['job']!=cfg['first_job'] or read(slot/'JOB.json')['job']!=os.environ['PBS_JOBID']
            or digest(Path(__file__))!=cfg['loop_sha256'] or Path(cfg['destination']).name!='verified-inputs'):
        raise PermissionError('Unexpected mixed staging role or first allocation identity')
    end=min(read(slot/'START.json')['site_end'],DEADLINE)
    sent=root/'mixed-sent';sent.mkdir(exist_ok=True)
    checks=root/'mixed-checked';checks.mkdir(exist_ok=True)
    with closing(OwnerLock(root/'mixed-staging.lock')):
        while time.time()<end-600 and not (root/'STOP').exists():
            stage='WAIT_FIRST_SITE_QUALIFICATION'
            if (first/'CLOSED.json').exists() and not (first/'RESULT.json').exists():
                raise RuntimeError('First allocation ended before the mixed qualification sealed')
            if (first/'RESULT.json').is_file():
                first_result=read(first/'RESULT.json')
                if first_result['job']!=cfg['first_job']:
                    raise ValueError('First qualification has different PBS custody')
                prefix=read(first/'LEGACY.json')
                for entry in sorted(Path(data_cfg['intake_receipts']).glob('TRAIN-???.json')):
                    if time.time()>=end-600 or (root/'STOP').exists():
                        break
                    meta=read(entry)
                    if meta['status']!='MIXED_COLD_READER_REQUIRED':
                        continue
                    if (sent/entry.name).is_file():
                        if read(sent/entry.name)['seal']!=meta['seal']:
                            raise ValueError('Previously sent mixed snapshot changed')
                        continue
                    result=checks/entry.name
                    if meta['count']==first_result['actual']['episodes']:
                        result=first/'RESULT.json'
                    elif not result.exists():
                        write_once(result,qualify_one(first,data_cfg,entry,prefix,min(end-300,time.time()+3600)))
                    actual=read(result)
                    if actual['actual']['seal']!=meta['seal'] or actual['destination']['count']!=meta['count']:
                        raise ValueError('Checked source and destination differ from the original receipt')
                    sha=forward(result,cfg['destination'])
                    write_once(sent/entry.name,dict(seal=meta['seal'],report_sha256=sha,job=os.environ['PBS_JOBID']))
                stage='WAIT_NEXT_MIXED_INTAKE'
            write(slot/'MIXED_HEARTBEAT.json',dict(stage=stage,job=os.environ['PBS_JOBID'],epoch=time.time(),pid=os.getpid()))
            time.sleep(min(60,max(0,end-600-time.time())))
    write_once(slot/'MIXED_CLOSED.json',dict(status='STOP_OR_ALLOCATION',epoch=time.time()))


if __name__=='__main__':
    cli=ArgumentParser()
    cli.add_argument('root',type=Path)
    cli.add_argument('slot',type=Path)
    args=cli.parse_args()
    try:
        run(args.root,args.slot)
    except Exception as err:
        hold=dict(error=type(err).__name__,reason=str(err),epoch=time.time(),job=os.environ['PBS_JOBID'])
        write_once(args.slot/'MIXED_INTERRUPTION.json',hold)
        write_once(args.root/'MIXED_HOLD.json',hold)
        raise
