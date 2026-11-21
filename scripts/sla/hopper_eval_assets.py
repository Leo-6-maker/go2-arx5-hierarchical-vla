"""CPU custody for the pre-existing engineering 48 and ordinary 288 evaluation inputs."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from hopper_parent_custody import HOST, PLAN, SSH, digest, remote, sync_dir, write_once

PANEL_SHA='068b73058d28b741c9ddeba6bcc0997e5ffb1453ceb349a7e4c2baa5f2bf4f57'
FORMAL_SHA='3136404c06f890a9dd830b2cf5f40e487ef436ede16e805e14ffc0cb06b5eefd'
EVAL_SPEC='4a99630a10f497417ddc1d1cf873a7b0339677b2f0b32c1f7b46937ec0c69e04'


def collect(base):
    pins={}
    def member(path,sha):
        if not path.resolve(strict=True).is_relative_to(base) or path.is_symlink():
            raise ValueError('Unexpected evaluation asset path')
        label=str(path.relative_to(base))
        if label not in pins:
            if digest(path)!=sha:
                raise ValueError('Evaluation member SHA mismatch: '+label)
            pins[label]=dict(sha256=sha,bytes=path.stat().st_size)
        elif pins[label]['sha256']!=sha:
            raise ValueError('Conflicting member identity')
    def manifest(ref):
        path=Path(ref['path'])
        member(path,ref['sha256'])
        return json.loads(path.read_bytes())
    def relative(root,row):
        member(root/row.get('file',row.get('path')),row['sha256'])
    panel_path=base/'sla-adapt5k-gt-v2-batch-offline-panel-48ad2421-20261120-01/panel/OFFLINE_PANEL.private.json'
    panel=manifest(dict(path=str(panel_path),sha256=PANEL_SHA))
    if panel['eval_input']['sha256']!=EVAL_SPEC or panel['original_panel']['sha256']!=FORMAL_SHA:
        raise ValueError('Unexpected engineering evaluation inputs')
    eval_spec=manifest(panel['eval_input'])
    formal=manifest(panel['original_panel'])
    if (formal['old_TEST_access'] or eval_spec['panel']!=panel['original_panel']
            or {k:len(v) for k,v in panel['rows'].items()}!=dict(target=32,robodojo=8,libero=8)
            or {k:len(v) for k,v in formal['formal'].items()}!=dict.fromkeys(('target','robodojo','libero'),96)):
        raise ValueError('Unexpected evaluation counts or population')
    member(Path(panel['noise']['path']),panel['noise']['sha256'])
    sources=eval_spec['original_source_dev']
    counts={}
    for name in ('robodojo','libero'):
        source=sources[name]
        root=Path(source['root'])
        data=manifest(source['manifest'])
        ids={r['identity']['episode'] for r in formal['formal'][name]+panel['rows'][name]}
        if name=='robodojo':
            relative(root,data['statistics'])
            raw=manifest(source['raw_binding'])
            catalogue={e['id']:e for e in data['episodes'] if e['split']=='SOURCE_DEV' and e['valid']}
            for eid in sorted(ids):
                ep=catalogue[eid]
                if ep['raw_sha256']!=raw['raw'][eid]['sha256']:
                    raise ValueError('RoboDojo evaluation RGB identity changed')
                relative(root,raw['raw'][eid])
                relative(root,ep['arrays'])
        else:
            catalogue={e['episode']:e for e in data['rows'] if e['split']=='SOURCE_DEV' and e['admitted']}
            for eid in sorted(ids):
                row=catalogue[eid]
                for ref in [row['data'],*row['cameras'].values()]:
                    relative(root,ref)
        counts[name]=len(ids)
    return dict(plan=PLAN,engineering_panel=PANEL_SHA,formal_panel=FORMAL_SHA,eval_input=EVAL_SPEC,
                engineering_rows=48,ordinary_rows=288,source_episodes=counts,entries=pins)


def receive(root,expected):
    inv=root/'FILES.private.json'
    if digest(inv)!=expected:
        raise ValueError('Evaluation file manifest changed')
    cfg=json.loads(inv.read_bytes())
    incoming=root/'inputs.incoming'
    final=root/'inputs.verified'
    tree=final if final.exists() else incoming
    if set(cfg['entries'])!={str(f.relative_to(tree)) for f in tree.rglob('*') if f.is_file()}:
        raise ValueError('Evaluation membership differs after transfer')
    for name,row in cfg['entries'].items():
        if ((root/'STOP').exists() or time.time()>=cfg['deadline_epoch'] or
                (tree/name).stat().st_size!=row['bytes'] or digest(tree/name,flush=True)!=row['sha256']):
            raise ValueError('Evaluation transfer stopped or changed: '+name)
    for directory in sorted((p for p in tree.rglob('*') if p.is_dir()),key=lambda p:-len(p.parts)):
        sync_dir(directory)
    sync_dir(tree)
    if tree==incoming:
        tree.rename(final)
        sync_dir(root)
    proof=dict(status='EVAL_INPUT_BYTES_VERIFIED',job=cfg['job'],plan=PLAN,
               source_revision=cfg['source_revision'],members=len(cfg['entries']),
               bytes=sum(v['bytes'] for v in cfg['entries'].values()),manifest_sha256=expected,
               engineering_panel=PANEL_SHA,formal_panel=FORMAL_SHA,
               new_predictions=0,independent_restores=0,optimizer_updates=0,native=0)
    write_once(root/'RESULT.private.json',proof)
    print(json.dumps(proof))


def transfer(root):
    mount=json.loads((root/'MOUNT.private.json').read_bytes())
    if (mount['job']!=os.environ.get('PBS_JOBID') or mount['plan']!=PLAN or
            time.time()>=mount['deadline_epoch'] or (root/'STOP').exists()):
        raise PermissionError('Invalid evaluation staging allocation')
    for name,sha in mount['scripts'].items():
        if digest(root/name)!=sha:
            raise ValueError('Evaluation transfer source changed')
    base=Path(mount['base'])
    cfg=collect(base)
    cfg.update(job=mount['job'],source_revision=mount['source_revision'],deadline_epoch=mount['deadline_epoch'])
    packet=root/'FILES.private.json'
    write_once(packet,cfg)
    dest=Path(mount['receiver'])
    remote('bash','-c','set -eu; umask 077; mkdir -- "$1" "$1/inputs.incoming"','bash',str(dest))
    sync=['rsync','-rt','--chmod=Fu=rw,Fgo=,Du=rwx,Dgo=','-e',shlex.join(SSH)]
    subprocess.run([*sync,str(packet),*[str(root/n) for n in mount['scripts']],HOST+':'+str(dest)+'/'],check=True)
    subprocess.run([*sync,'--from0','--files-from=-',str(base)+'/',HOST+':'+str(dest/'inputs.incoming')+'/'],
                   input=b'\0'.join(os.fsencode(n) for n in cfg['entries'])+b'\0',check=True)
    check=' && '.join('test "$(sha256sum '+name+' | cut -d" " -f1)" = '+value for name,value in mount['scripts'].items())
    msg=remote('bash','-c','set -eu; cd "$1"; '+check+' && exec python3 -B hopper_eval_assets.py receive "$1" "$2"',
               'bash',str(dest),digest(packet))
    result=json.loads(msg)
    if result['job']!=mount['job'] or result['manifest_sha256']!=digest(packet):
        raise ValueError('Destination evaluation packet differs')
    write_once(root/'RESULT.private.json',result)


if __name__=='__main__':
    os.umask(0o077)
    if sys.argv[1]=='receive':
        receive(Path(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=='transfer':
        root=Path.cwd()
        with (root/'OWNER.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                transfer(root)
            except Exception as error:
                write_once(root/'FAILURE.private.json',dict(job=os.environ.get('PBS_JOBID'),error=str(error)))
                raise
    else:
        raise ValueError('Unknown operation')
