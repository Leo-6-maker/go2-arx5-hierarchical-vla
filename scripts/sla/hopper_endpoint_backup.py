"""Pull only sealed Hopper checkpoint members into private VANDA custody and enqueue independent endpoint work."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.hopper_parent_custody import PLAN, PARENT, HOST, SSH, digest, sync_dir, write_once
from scripts.sla.hopper_site import DEADLINE

SCHEMA='HOPPER_ADAPT20K_COMPLETE_STATE_V1'
ROLES=('MAIN','CONTROL2000')


def list_seals(line):
    result=[]
    for role in ROLES:
        for p in sorted((line/role/'sealed').glob('phase[0-9][0-9][0-9][0-9][0-9].json')):
            result.append(dict(role=role,phase=int(p.stem[5:]),event_sha256=digest(p)))
    trigger=line/'curriculum/FIRST_CONTROL_TRIGGER.json'
    final_controls=not trigger.exists() or (line/'CONTROL2000/sealed'/f"phase{int(read(trigger)['matched_endpoint']):05d}.json").is_file()
    complete=(line/'MAIN/sealed/phase20000.json').is_file() and final_controls
    return dict(events=result,training_complete=complete,stop=(line/'STOP.json').exists() or (line.parent/'STOP.json').exists())


def checkpoint(line,role,phase,seal):
    if role not in ROLES or not 0<=phase<=20000:
        raise ValueError('Unknown endpoint')
    event_file=line/role/'sealed'/f'phase{phase:05d}.json'
    if digest(event_file)!=seal:
        raise ValueError('Published event changed')
    event=read(event_file)
    cp=line/role/'checkpoints'/f'phase{phase:05d}'
    if (cp.is_symlink() or event['plan']!=PLAN or event['role']!=role or event['phase']!=phase
            or Path(event['checkpoint']['path'])!=cp or event['binding']['parent']!=PARENT
            or event['binding']['role']!=role or event['cursors']['phase_step']!=phase
            or event['backup_due'] is not True):
        raise ValueError('Unexpected sealed endpoint identity')
    if digest(cp/'COMPLETE.json')!=event['checkpoint']['sha256']:
        raise ValueError('Endpoint seal does not match the registered checkpoint')
    header=read(cp/'COMPLETE.json')
    if (header['schema']!=SCHEMA or header['binding']!=event['binding'] or header['progress']!=event['cursors']
            or header['predecessor']!=event['predecessor'] or len(header['arrays'])!=140):
        raise ValueError('Endpoint differs from its training event')
    members={'COMPLETE.json':event['checkpoint']['sha256']}
    for n,row in enumerate(header['arrays']):
        name=f'{n:06d}.npy'
        if row['file']!=name or not re.fullmatch(r'[0-9a-f]{64}',row['sha256']):
            raise ValueError('Unexpected checkpoint member')
        members[name]=row['sha256']
    if set(p.name for p in cp.iterdir())!=set(members) or any((cp/n).is_symlink() for n in members):
        raise ValueError('Checkpoint package membership changed')
    return dict(folder=str(cp),event=event,members=members,sizes={n:(cp/n).stat().st_size for n in members})


def remote(spec,*args):
    cmd=shlex.join(['python3','-B',spec['remote_program'],*map(str,args)])
    shell='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(spec['remote_source'])+' && exec '+cmd
    return json.loads(subprocess.check_output([*SSH,HOST,shell],text=True,timeout=90))


def copy_one(root,spec,item):
    role,phase,seal=(item[k] for k in ('role','phase','event_sha256'))
    info=remote(spec,'inspect',spec['line'],role,phase,seal)
    event=info['event']
    dest=root/role/f'phase{phase:05d}'
    dest.parent.mkdir(exist_ok=True)
    cp=dest if dest.is_dir() else dest.with_suffix('.incoming')
    cp.mkdir(exist_ok=True)
    sizes=info['sizes']
    if cp!=dest:
        remaining=sum(n for name,n in sizes.items() if not (cp/name).exists())
        if shutil.disk_usage(root).free<remaining+1024**3:
            raise OSError('Insufficient checkpoint staging space')
        command=['rsync','-rt','--partial','--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=','--files-from=-','-e',shlex.join(SSH),
                 HOST+':'+info['folder']+'/',str(cp)+'/']
        subprocess.run(command,input='\n'.join(info['members'])+'\n',text=True,check=True,timeout=300)
    for name,sha in info['members'].items():
        if (cp/name).stat().st_size!=sizes[name] or digest(cp/name,flush=True)!=sha:
            raise ValueError('Backup member size or SHA mismatch: '+name)
    if set(p.name for p in cp.iterdir())!=set(sizes):
        raise ValueError('Unexpected destination checkpoint entries')
    sync_dir(cp)
    if cp!=dest:
        cp.rename(dest)
        sync_dir(dest.parent)
    key=f'{role}-phase{phase:05d}'
    record=dict(plan=PLAN,role=role,phase=phase,event_sha256=seal,checkpoint=event['checkpoint']['sha256'],
                custody=str(dest),arrays=140,bytes=sum(sizes.values()),eval_due=event['eval_due'],
                deployment_due=event['deployment_due'],parent=PARENT)
    if record['eval_due'] or record['deployment_due']:
        for purpose,needed in [('formal',record['eval_due']),('reference',record['deployment_due'])]:
            if needed:
                queue=root/(purpose+'-pending')
                queue.mkdir(exist_ok=True)
                write_once(queue/(key+'.json'),dict(record,purpose=purpose,status='PENDING'))
    write_once(root/'receipts'/(key+'.json'),record)
    return record


def tick(root):
    job=os.environ.get('PBS_JOBID','')
    if not re.fullmatch(r'\d+\.stdct-mgmt-02',job):
        raise PermissionError('VANDA checkpoint custody is a scheduled CPU operation')
    root=root.resolve(strict=True)
    spec=read(root/'BACKUP.json')
    if (spec['plan']!=PLAN or time.time()>=DEADLINE-600 or (root/'STOP').exists()
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=spec['program_sha256']):
        return dict(status='STOP')
    receipts=root/'receipts'
    receipts.mkdir(exist_ok=True)
    with closing(OwnerLock(root/'custody.lock')):
        status=remote(spec,'list',spec['line'])
        events=status['events']
        if status['stop']:
            return dict(status='STOP')
        for item in events:
            r=receipts/f"{item['role']}-phase{item['phase']:05d}.json"
            if r.exists():
                if read(r)['event_sha256']!=item['event_sha256']:
                    raise ValueError('Previously backed-up source event changed')
                continue
            rec=copy_one(root,spec,item)
            return dict(status='COPIED',checkpoint=rec['checkpoint'],phase=rec['phase'],role=rec['role'])
        return dict(status='BACKUP_COMPLETE' if status['training_complete'] else 'WAIT_SEAL',copied=len(events))


if __name__=='__main__':
    c=argparse.ArgumentParser()
    c.add_argument('mode',choices=('list','inspect','tick'))
    c.add_argument('root',type=Path)
    c.add_argument('details',nargs='*')
    a=c.parse_args()
    os.umask(0o077)
    if a.mode=='inspect':
        role,phase,seal=a.details
        result=checkpoint(a.root,role,int(phase),seal)
    else:
        result=list_seals(a.root) if a.mode=='list' else tick(a.root)
    print(json.dumps(result))
