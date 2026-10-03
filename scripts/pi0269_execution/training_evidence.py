"""Seal only small, completed Hopper journals; never package Trainer weights."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0269')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    assert json.loads((ROOT/'evidence/MAIN_TRAINING_AUDIT.json').read_text())['status']=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY'
    files=[]
    for folder in ('code','owner_code','design','contracts','evidence','logs'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    files.extend(p for p in ROOT.iterdir() if p.is_file())
    for arm in ('UNIFORM','PREFIX'):
        files.extend(p for p in (ROOT/arm).iterdir() if p.is_file())
        for phase in ('continuous','split','main'):
            files.extend(p for p in (ROOT/arm/phase/'training').iterdir() if p.is_file())
    files=sorted(set(p for p in files if p.suffix in ('.json','.jsonl','.csv','.py','.pbs','.md','.log','.ER','.OU')))
    assert files and all(not p.is_symlink() and p.name not in ('token','stored_tokens') for p in files)
    members=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    tree=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest=dict(status='PASS_COMPLETE_SMALL_TRAINING_EVIDENCE_SEAL',files=members,tree_sha256=tree,model_weights=0,optimizer_payloads=0)
    out=ROOT/'archive';out.mkdir(exist_ok=True)
    target=out/'TRAINING_EVIDENCE_MANIFEST.json';content=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    if target.exists():assert target.read_text()==content
    else:target.write_text(content)
    package=out/'training-evidence.tar'
    if not package.exists():
        temp=package.with_suffix('.tar.partial')
        with tarfile.open(temp,'w') as tar:
            for p,row in zip(files,members):
                info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with p.open('rb') as stream:tar.addfile(info,stream)
            tar.add(target,arcname=target.name,recursive=False)
        temp.replace(package)
    print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in members),tree_sha256=tree,package=str(package),package_sha256=sha(package))),flush=True)


if __name__=='__main__':main()
