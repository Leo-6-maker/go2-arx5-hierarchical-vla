"""Publish only owned PI0258 evidence, then verify immutable anonymous bytes."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tarfile
import requests
from huggingface_hub import HfApi,hf_hub_url
from pi0255_archive_pilot import digest,save

ROOT=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0258')


def run():
    status=json.loads((ROOT/'reports/terminal_status.json').read_text())
    assert status['B']=='KIN_DATA_BLOCKED' and status['training_updates']==0
    out=ROOT/'archive/upload';out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for folder in ('design','development','evidence','reports','delivery'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.name not in ('HF_PUBLIC_RECEIPT.json',))
    paths.extend(p for p in (ROOT/'runtime').glob('pi0258_*.py'))
    paths.append(ROOT/'runtime/eef_video_workflow.py')
    paths=sorted(set(paths),key=lambda p:p.relative_to(ROOT).as_posix())
    assert all(p.resolve().is_relative_to(ROOT) and not p.is_symlink() for p in paths)
    assert all(p.suffix in ('.json','.jsonl','.npz','.csv','.md','.txt','.py','.mp4','.png','.log','.html') for p in paths)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
    tree=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    save(out/'MEMBER_MANIFEST.json',dict(schema='PI0258_PUBLIC_EVIDENCE_V1',files=entries,tree_sha256=tree,
        rights_scope='Owned synthetic camera output, numerical evidence and project code; no private WBC source/assets, model/tokenizer weights or credentials.',
        bytes=sum(x['bytes'] for x in entries),files_count=len(entries)))
    archive=out/'PI0258_control_and_kinematic_development.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|') as tar:
        for p,row in zip(paths,entries):
            item=tarfile.TarInfo(row['path']);item.size=row['bytes'];item.mode=0o644;item.mtime=0
            with p.open('rb') as data:tar.addfile(item,data)
    for name in ('M3_fixed_pair_kinematic_reference.mp4','index.html'):
        shutil.copy2(ROOT/'delivery'/name,out/name)
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0258/control-audit-kinematic-development-v1-20260929'
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip())
    assert api.model_info(repo).private is False
    record=ROOT/'archive/upload_commit.json'
    if record.exists():
        prior=json.loads(record.read_text());assert prior['tree_sha256']==tree
        revision=prior['revision']
    else:
        revision=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(out),path_in_repo=prefix,
            commit_message='PI0258 control audit and fixed kinematic data admission negative with labelled video').oid
        save(record,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree))
    session=requests.Session();session.trust_env=False;readback=[]
    for p in sorted(out.iterdir()):
        target=ROOT/'archive'/('anonymous-'+p.name);h=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,filename=prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            with target.open('wb') as stream:
                for block in response.iter_content(8<<20):stream.write(block);h.update(block);size+=len(block)
        assert size==p.stat().st_size and h.hexdigest()==digest(p)
        readback.append(dict(path=p.name,bytes=size,sha256=h.hexdigest(),anonymous=True))
    expected={x['path']:x for x in entries};count=0
    with tarfile.open(ROOT/'archive'/('anonymous-'+archive.name),'r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected
            row=expected.pop(member.name);f=tar.extractfile(member);h=hashlib.sha256();size=0
            for block in iter(lambda:f.read(1<<20),b''):h.update(block);size+=len(block)
            assert size==row['bytes'] and h.hexdigest()==row['sha256'];count+=1
    assert not expected
    receipt=dict(status='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK',repo_id=repo,prefix=prefix,revision=revision,
        tree_sha256=tree,source_files=count,source_bytes=sum(x['bytes'] for x in entries),HF_files=readback,
        private_WBC_source_assets_uploaded=False,model_weights_uploaded=False,physics_steps=0,GR00T_queries=0)
    save(ROOT/'evidence/HF_PUBLIC_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':run()
