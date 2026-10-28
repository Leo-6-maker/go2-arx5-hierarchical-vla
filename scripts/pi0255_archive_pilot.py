"""Bounded own-evidence archive, fixed commit and anonymous full member readback."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
import requests
from huggingface_hub import HfApi,hf_hub_url


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    with path.open('w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())


def main(root):
    summary=json.loads((root/'evidence/pilot_stage_summary.json').read_text());assert summary['status']=='PARTIAL_MULTITASK_PREPARATION_BLOCKED'
    assert summary['counters']['expert_physics_starts']==36
    out=root/'archive/pilot-v1';out.mkdir(parents=True,exist_ok=True)
    files=[]
    for folder in ('episodes','dataset','design','reports'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and not p.is_symlink())
    files.extend(p for p in (root/'evidence').glob('*.json') if p.is_file())
    files.extend(p for p in (root/'evidence/videos').glob('*.mp4') if p.is_file())
    files.extend(root/n for n in ('physics_attempts.jsonl','attempt_results.jsonl'))
    for runtime in ('runtime','runtime_v2'):
        files.extend(p for p in (root/runtime).glob('pi0255*.py') if p.is_file())
        files.extend(p for p in (root/runtime).glob('execution_seal.json') if p.is_file())
    files.append(root/'evidence/WBC_LICENSE_NOTICE.txt')
    files.extend(p for p in (root/'models').glob('*') if p.is_file() and (p.suffix=='.json' or p.name=='base_LICENSE'))
    files=sorted(set(files),key=lambda p:p.relative_to(root).as_posix())
    assert all(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() for p in files)
    assert all(p.suffix in ('.json','.jsonl','.npz','.md','.txt','.py','.mp4') or p.name=='base_LICENSE' for p in files)
    entries=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in files]
    tree=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest=dict(schema='PI0255_PILOT_PUBLIC_ARCHIVE_MEMBERS_V1',tree_sha256=tree,files=entries,files_count=len(entries),bytes=sum(x['bytes'] for x in entries),
        publication_scope='sanctioned synthetic observations and own numerical/source evidence; no private code/weights/assets/credentials',expert_starts=36,GR00T_queries=0)
    save(out/'MEMBER_MANIFEST.json',manifest)
    archive=out/'PI0255_fixed_pilot_audit.tar.gz'
    if not archive.exists():
        with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as tar:
            for path,row in zip(files,entries):
                info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with path.open('rb') as f:tar.addfile(info,f)
        with archive.open('rb') as f:os.fsync(f.fileno())
    token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip();api=HfApi(token=token)
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0255/fixed-pilot-audit-v1-20260929';assert api.model_info(repo).private is False
    record=root/'archive/pilot-v1-upload-commit.json'
    if record.exists():
        committed=json.loads(record.read_text());assert committed['tree_sha256']==tree and committed['archive_sha256']==digest(archive)
        revision=committed['revision']
    else:
        commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(out),path_in_repo=prefix,commit_message='PI0255 all36 fixed expert pilot failures and readiness evidence')
        revision=commit.oid;save(record,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree,archive_sha256=digest(archive),status='UPLOAD_COMMITTED_READBACK_PENDING'))
    session=requests.Session();session.trust_env=False;observed=[]
    for path in (out/'MEMBER_MANIFEST.json',archive):
        h=hashlib.sha256();size=0;readback=root/'archive'/('anonymous-'+path.name)
        with session.get(hf_hub_url(repo,filename=prefix+'/'+path.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            assert 'Authorization' not in response.request.headers
            with readback.open('wb') as f:
                for b in response.iter_content(8<<20):f.write(b);h.update(b);size+=len(b)
                f.flush();os.fsync(f.fileno())
        assert size==path.stat().st_size and h.hexdigest()==digest(path)
        observed.append(dict(path=path.name,bytes=size,sha256=h.hexdigest(),anonymous=True))
    expected={x['path']:x for x in entries};members=[]
    with tarfile.open(root/'archive/anonymous-PI0255_fixed_pilot_audit.tar.gz',mode='r|gz') as tar:
        for item in tar:
            assert item.isfile() and item.name in expected and not item.issym();row=expected.pop(item.name);h=hashlib.sha256();size=0
            f=tar.extractfile(item)
            for b in iter(lambda:f.read(1<<20),b''):h.update(b);size+=len(b)
            assert size==row['bytes'] and h.hexdigest()==row['sha256'];members.append(dict(row,anonymous_archive_member_readback=True))
    assert not expected
    receipt=dict(schema='PI0255_FIXED_PUBLIC_ANONYMOUS_FULL_CONTENT_READBACK_V1',status='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK',repo_id=repo,prefix=prefix,revision=revision,
        public=True,anonymous=True,source_tree_sha256=tree,source_files=len(entries),source_bytes=manifest['bytes'],HF_files=observed,members=members,
        model_tensors_uploaded=False,private_assets_uploaded=False,new_GR00T_loads=0,new_GR00T_queries=0)
    save(root/'evidence/pilot_HF_public_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','revision','source_tree_sha256','source_files','source_bytes')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
