"""Delete explicit retired large payloads only; preserve all small evidence."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def read(p):return json.loads(p.read_text())


def save(p,value):
    temp=p.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(p)


def free(scope):return int(subprocess.check_output(['df','-B1','--output=avail',scope],text=True).splitlines()[-1])


def opened():
    active=set();commands=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            commands.append((p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
            for fd in (p/'fd').iterdir():
                try:s=fd.stat();active.add((s.st_dev,s.st_ino))
                except OSError:pass
        except OSError:pass
    return active,commands


def main(plan_path,execute):
    plan=read(plan_path);root=Path(plan['root']).resolve();rows=[];skipped=[]
    active,commands=opened()
    seal=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0269/EXECUTION_FILESET.json')
    protected=[Path(r['path']).resolve() for r in read(seal)['files']] if seal.exists() else []
    for proposal in plan['paths']:
        cp=Path(proposal['path'])
        if not cp.exists():skipped.append(dict(**proposal,reason='ALREADY_ABSENT'));continue
        assert cp.resolve().is_relative_to(root) and cp.resolve()==cp and not cp.is_symlink(),str(cp)
        assert 'PI0269' not in cp.parts and not ('ROT' in cp.parts and cp.name=='checkpoint-2048'),str(cp)
        if any(p.is_relative_to(cp) for p in protected) or any(str(cp) in c for c in commands):
            skipped.append(dict(**proposal,reason='ACTIVE_SEAL_OR_PROCESS_REFERENCE'));continue
        index=cp/'model.safetensors.index.json'
        names=sorted(set(read(index)['weight_map'].values())|{'optimizer.pt'})
        for name in names:
            assert Path(name).name==name and (name.endswith('.safetensors') or name=='optimizer.pt')
            p=cp/name
            if not p.exists():continue
            s=p.lstat()
            if p.is_symlink() or not p.is_file() or s.st_uid!=os.getuid() or (s.st_dev,s.st_ino) in active:
                skipped.append(dict(path=str(p),reason='SYMLINK_SHARED_OR_ACTIVE_FILE'));continue
            rows.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,
                mtime_ns=s.st_mtime_ns,links=s.st_nlink,classification=proposal['classification'],evidence=proposal['evidence'],deleted=False))
    groups={}
    for row in rows:groups.setdefault((row['device'],row['inode']),[]).append(row)
    for group in groups.values():
        if any(r['links']!=len(group) for r in group):
            skipped.extend(dict(path=r['path'],reason='HARDLINK_OUTSIDE_EXPLICIT_RETIRED_SET') for r in group)
    rows=[r for r in rows if r['links']==len(groups[r['device'],r['inode']])]
    destination=plan_path.with_name(plan_path.stem+('-RESULT.json' if execute else '-PREVIEW.json'))
    assert not destination.exists(),str(destination)
    result=dict(status='PREVIEW',host=plan['host'],plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        before_free_bytes=free(plan['free_scope']),rows=rows,skipped=skipped,planned_bytes=sum(r['bytes'] for r in rows),
        quota_headroom='UNKNOWN',preserved=plan['preserve'],scope=plan['scope'],HF_history_deleted=False)
    save(destination,result)
    if execute:
        for row in rows:
            p=Path(row['path']);s=p.lstat();current,_=opened()
            retired=sum(r['deleted'] for r in groups[row['device'],row['inode']])
            assert not p.is_symlink() and s.st_nlink==row['links']-retired and (s.st_dev,s.st_ino) not in current
            assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns'])
            p.unlink();row['deleted']=True;save(destination,result)
        result.update(status='EXPLICIT_RETIRED_PAYLOADS_DELETED',after_free_bytes=free(plan['free_scope']),
            logical_deleted_bytes=sum(r['bytes'] for r in rows),unlinked_allocated_bytes=sum(g[0]['allocated_bytes'] for g in groups.values() if all(r['deleted'] for r in g)),
            local_status='METADATA_ONLY_RETIRED_WEIGHTS_NOT_LOCALLY_LOADABLE',concurrent_training_can_change_free_delta=True)
        result['free_delta_bytes']=result['after_free_bytes']-result['before_free_bytes'];save(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','preserved')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args();main(a.plan,a.execute)
