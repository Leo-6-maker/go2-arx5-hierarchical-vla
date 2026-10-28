"""Finite public numeric-evidence archive; private source/assets/logs excluded."""
import hashlib
import json
from pathlib import Path
import tarfile
import requests
from huggingface_hub import HfApi,hf_hub_url

ROOT=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0289')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    folder=ROOT/'archive';folder.mkdir(exist_ok=True)
    files=[]
    for e in json.loads((ROOT/'evidence/BUDGET.json').read_text())['episodes']:
        p=Path(e['out'])
        files += [q for q in p.iterdir() if q.is_file() and q.suffix in ['.json','.jsonl']]
        files += [q for q in (p/'cpu').glob('*') if q.suffix in ['.json','.jsonl']]
    files += [ROOT/'evidence'/name for name in ['BUDGET.json','GRIPPER_SELECTION.json','CLOCK_CONTRACT.json','GATE_RECEIPT.json','FINAL_CPU_RECEIPT.json','REPORT.md']]
    members=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))]
    assert members and all(x['path'].startswith('evidence/') and Path(x['path']).suffix in ['.json','.jsonl','.md'] for x in members)
    manifest=folder/'MANIFEST.json';write(manifest,dict(members=members,private_assets_included=False,private_wbc_logs_included=False))
    tree=sha(manifest);package=folder/'evidence.tar.gz'
    if not package.exists():
        with tarfile.open(package.with_suffix('.partial'),'w:gz') as tar:
            for row in members:
                info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with (ROOT/row['path']).open('rb') as f:tar.addfile(info,f)
        package.with_suffix('.partial').replace(package)
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0289/autonomous-startup-20261010'
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip());assert api.model_info(repo).private is False
    upload=folder/'UPLOAD.json'
    if upload.exists():
        u=json.loads(upload.read_text());assert u['tree_sha256']==tree;revision=u['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(folder),allow_patterns=['evidence.tar.gz','MANIFEST.json'],path_in_repo=prefix,commit_message='PI0289 actual startup and first hold failure evidence').oid
        write(upload,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree))
    print('UPLOADED_FIXED_REVISION',revision,flush=True)
    session=requests.Session();session.trust_env=False;checked=[]
    for p in [manifest,package]:
        h=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            for block in response.iter_content(8<<20):h.update(block);size+=len(block)
        assert size==p.stat().st_size and h.hexdigest()==sha(p);checked.append(dict(path=p.name,bytes=size,sha256=h.hexdigest()))
    remaining={x['path']:x for x in members}
    with session.get(hf_hub_url(repo,prefix+'/'+package.name,revision=revision),stream=True,timeout=(30,180)) as response:
        response.raise_for_status();response.raw.decode_content=True
        with tarfile.open(fileobj=response.raw,mode='r|gz') as tar:
            for member in tar:
                assert member.isfile() and member.name in remaining
                row=remaining.pop(member.name);h=hashlib.sha256();size=0
                with tar.extractfile(member) as f:
                    for block in iter(lambda:f.read(8<<20),b''):h.update(block);size+=len(block)
                assert size==row['bytes'] and h.hexdigest()==row['sha256'],member.name
    assert not remaining
    receipt=dict(status='PASS_PUBLIC_FIXED_REVISION_ALL_FILES_AND_MEMBERS',repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree,members=len(members),source_bytes=sum(x['bytes'] for x in members),checked_files=checked,local_retirement=False)
    write(ROOT/'evidence/HF_ARCHIVE_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
