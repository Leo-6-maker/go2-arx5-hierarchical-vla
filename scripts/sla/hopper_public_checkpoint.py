"""Transport only goal-bound inference weights and the hashed RGB16 receipt via the authorized public repository."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_parent_custody import PLAN, PARENT, SSH, HOST, digest, write_once
from scripts.sla.hopper_site import DEADLINE

REPO='leoyuer/sla-goal-v2-checkpoints'
PREFIX='HOPPER-ADAPT20K-V1/Go2'
SCHEMA='SLA_CORE_HOPPER_GO2_RGB16_CHECKPOINT_V1'
ORDINALS=tuple(6*i+i%6 for i in range(16))
PANEL='3136404c06f890a9dd830b2cf5f40e487ef436ede16e805e14ffc0cb06b5eefd'
ACTION='f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e'
SHA_FIELDS=('checkpoint_sha256','frozen_model','foundation_sha256','partition_sha256','state_sha256','tokenizer_sha256')
REF_HASHES=('input_sha256','normalized_sha256','decoded_sha256','original_guard_sha256')


def is_sha(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def allowed(manifest):
    m=manifest
    keys={'schema','plan','parent','role','phase','training_source','panel','action_sha256','reference',
          'arrays','physics','hardware_allowed','native_admission',*SHA_FIELDS}
    if (set(m)!=keys or m['schema']!=SCHEMA or m['plan']!=PLAN or m['parent']!=PARENT or m['panel']!=PANEL
            or m['action_sha256']!=ACTION or not all(is_sha(m[k]) for k in SHA_FIELDS)
            or not re.fullmatch('[0-9a-f]{40}',m['training_source'])
            or any(m[k] is not False for k in ('physics','hardware_allowed','native_admission'))):
        raise ValueError('Unexpected public model manifest or non-checkpoint metadata')
    phases={4000,7000,12000,17000}
    if m['role']=='MAIN':
        phases|={0,2000,5000,10000,20000}
    if m['role'] not in ('MAIN','CONTROL2000') or type(m['phase']) is not int or m['phase'] not in phases:
        raise ValueError('Unexpected deployment phase')
    if len(m['reference'])!=16 or len(m['arrays'])!=29:
        raise ValueError('Incomplete reference proof or inference parameter set')
    for ordinal,row in zip(ORDINALS,m['reference']):
        if (set(row)!={'ordinal','index','noise_id',*REF_HASHES} or row['ordinal']!=ordinal
                or type(row['index']) is not int or row['index']<0 or type(row['noise_id']) is not int
                or not 0<=row['noise_id']<2**32 or not all(is_sha(row[k]) for k in REF_HASHES)):
            raise ValueError('Only ordinal/seed and input/output digests may accompany the weights')
    members={}
    keys=set()
    for row in m['arrays']:
        name=row['file']
        if (set(row)!={'key','file','shape','dtype','sha256'} or not re.fullmatch(r'parameters/\d{6}\.npy',name)
                or name in members or row['key'] in keys or not row['key'].startswith(
                    ("['parameters']['shared']","['parameters']['target']")) or not is_sha(row['sha256'])
                or row['dtype']!='float32' or not all(type(n)is int and n>0 for n in row['shape'])):
            raise ValueError('Only Go2 checkpoint parameter arrays are eligible')
        members[name]=row['sha256']
        keys.add(row['key'])
    return members


def list_ready(line):
    records=[]
    for path in sorted((line/'references').glob('*-phase[0-9][0-9][0-9][0-9][0-9]/RESULT.json')):
        result=read(path)
        if result['status']!='HOPPER_REF16_SEALED':
            continue
        folder=path.parent/'package'
        m=read(folder/'CHECKPOINT.json')
        allowed(m)
        role,phase=m['role'],m['phase']
        event=read(line/role/'sealed'/f'phase{phase:05d}.json')
        sha=digest(folder/'CHECKPOINT.json')
        if (event['plan']!=PLAN or event['deployment_due'] is not True or event['checkpoint']['sha256']!=m['checkpoint_sha256']
                or result['checkpoint']!=m['checkpoint_sha256'] or result['package']!=dict(path=str(folder),sha256=sha)
                or path.parent.name!=f'{role}-phase{phase:05d}' or result['complete']!=16):
            raise ValueError('Unbound inference checkpoint')
        records.append(dict(key=path.parent.name,manifest=sha,checkpoint=m['checkpoint_sha256'],folder=str(folder)))
    trigger=line/'curriculum/FIRST_CONTROL_TRIGGER.json'
    control_done=not trigger.is_file() or (line/'CONTROL2000/sealed'/f"phase{int(read(trigger)['matched_endpoint']):05d}.json").is_file()
    complete=(line/'MAIN/sealed/phase20000.json').is_file() and control_done
    if complete:
        for role in ('MAIN','CONTROL2000'):
            for f in (line/role/'sealed').glob('phase*.json'):
                event=read(f)
                if event['deployment_due'] and not (line/'references'/f"{role}-phase{event['phase']:05d}"/'RESULT.json').is_file():
                    complete=False
    return dict(rows=records,complete=complete,stop=(line/'STOP.json').exists() or (line.parent/'STOP.json').exists())


def copy_names(folder,to,names):
    subprocess.run(['rsync','-rt','--partial','--files-from=-','-e',shlex.join(SSH),HOST+':'+folder+'/',str(to)+'/'],
                   input='\n'.join(names)+'\n',text=True,check=True,timeout=300)


def verified_remote(api,prefix,members,folder,revision):
    paths=[prefix+'/'+n for n in members]
    rows=api.get_paths_info(REPO,paths,repo_type='dataset',revision=revision)
    if not rows:
        return False
    if {r.path for r in rows}!=set(paths):
        raise ValueError('Partial public package at the immutable endpoint path')
    for r in rows:
        name=r.path[len(prefix)+1:]
        f=folder/name
        if r.size!=f.stat().st_size:
            raise ValueError('Public checkpoint length differs')
        if r.lfs is not None:
            correct=r.lfs.sha256==members[name]
        else:
            h=hashlib.sha1(b'blob '+str(r.size).encode()+b'\x00')
            with f.open('rb') as data:
                for block in iter(lambda:data.read(2**20),b''):
                    h.update(block)
            correct=r.blob_id==h.hexdigest()
        if not correct:
            raise ValueError('Public checkpoint member digest differs')
    return True


def publish(root,entry):
    from huggingface_hub import HfApi, CommitOperationAdd

    if not re.fullmatch(r'(MAIN|CONTROL2000)-phase\d{5}',entry['key']) or not is_sha(entry['manifest']):
        raise ValueError('Unrecognized source')
    slot=root/'endpoints'/entry['key']
    slot.mkdir(exist_ok=True)
    folder=slot/'payload'
    folder.mkdir(exist_ok=True)
    copy_names(entry['folder'],folder,['CHECKPOINT.json'])
    manifest=folder/'CHECKPOINT.json'
    if digest(manifest)!=entry['manifest']:
        raise ValueError('Changed incoming manifest')
    m=read(manifest)
    members=allowed(m)
    if entry['key']!=f"{m['role']}-phase{m['phase']:05d}" or m['checkpoint_sha256']!=entry['checkpoint']:
        raise ValueError('Source role or checkpoint changed')
    copy_names(entry['folder'],folder,sorted(members))
    members['CHECKPOINT.json']=entry['manifest']
    if ({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}!=set(members)
            or any(digest(folder/n)!=v for n,v in members.items())):
        raise ValueError('Incomplete direct checkpoint transfer')
    api=HfApi()
    repo=api.repo_info(REPO,repo_type='dataset')
    if repo.private or api.whoami()['name']!='leoyuer':
        raise PermissionError('Incorrect authorized checkpoint repository or uploader')
    prefix=f"{PREFIX}/{entry['key']}-{entry['manifest'][:12]}"
    journal=Journal(slot/'journal',entry['key'])
    if verified_remote(api,prefix,members,folder,repo.sha):
        revision=repo.sha
    else:
        def upload():
            ops=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(folder/n)) for n in sorted(members)]
            return api.create_commit(REPO,repo_type='dataset',operations=ops,
                        commit_message='Seal Go2 checkpoint '+entry['key'],parent_commit=repo.sha).oid
        revision=journal.once('publish',dict(repository=REPO,prefix=prefix,manifest=entry['manifest']),upload)
    if not verified_remote(api,prefix,members,folder,revision):
        raise ValueError('Missing committed checkpoint')
    receipt=dict(plan=PLAN,repository=REPO,revision=revision,prefix=prefix,manifest=entry['manifest'],
                 checkpoint=entry['checkpoint'],arrays=29,bytes=sum((folder/n).stat().st_size for n in members))
    write_once(slot/'PUBLISHED.json',receipt)
    return receipt


def tick(root):
    cfg=read(root/'EXCHANGE.json')
    if cfg['plan']!=PLAN or not re.fullmatch(r'\d+\.stdct-mgmt-02',os.environ.get('PBS_JOBID','')):
        raise PermissionError('Expected a registered VANDA CPU transfer job')
    if time.time()>=min(cfg['grant_end'],DEADLINE)-600 or (root/'STOP').exists():
        return dict(status='STOP')
    (root/'endpoints').mkdir(exist_ok=True)
    with closing(OwnerLock(root/'exchange.lock')):
        command=shlex.join(['python3','-B',cfg['remote_program'],'list',cfg['line']])
        shell='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(cfg['remote_source'])+' && exec '+command
        state=json.loads(subprocess.check_output([*SSH,HOST,shell],text=True,timeout=40))
        if state['stop']:
            return dict(status='STOP')
        rows=state['rows']
        if len(rows)>7:
            raise ValueError('Deployment population exceeded')
        for entry in rows:
            done=root/'endpoints'/entry['key']/'PUBLISHED.json'
            if not done.exists():
                return dict(status='COPIED',public_checkpoint=publish(root,entry))
            if read(done)['manifest']!=entry['manifest']:
                raise ValueError('Published endpoint changed')
        return dict(status='EXCHANGE_COMPLETE' if state['complete'] else 'WAIT_SEAL',published=len(rows))


if __name__=='__main__':
    c=argparse.ArgumentParser()
    c.add_argument('mode',choices=('list','tick'))
    c.add_argument('root',type=Path)
    a=c.parse_args()
    os.umask(0o077)
    print(json.dumps(list_ready(a.root) if a.mode=='list' else tick(a.root)))
