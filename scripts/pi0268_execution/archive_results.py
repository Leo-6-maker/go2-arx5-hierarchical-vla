"""Finite per-point/secondary tar seals; existing data and complete Trainer archives referenced."""
import argparse
import ast
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path
import requests
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0268';R1=ROOT/'R1'
os.environ.update(HF_HOME=str(ROOT/'runtime/hf_online'),HF_XET_CACHE=str(ROOT/'runtime/hf_online/xet'),HF_HUB_OFFLINE='0')
from huggingface_hub import HfApi,hf_hub_url
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0268/R1')
sys.path[:0]=[str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0255_motion3 import write,sha


def read(p):return json.loads(Path(p).read_text())


def admitted(part):
    if part not in ('OLD0','ROT0','OLD2048','ROT2048'):
        assert read(RAW/'delivery/PRIMARY_VERDICT.json')['status']=='PRIMARY_COMPLETE'
        return
    assert read(R1/'evidence/MAIN_TRAINING_AUDIT.json')['status']=='PASS_BOTH_2048_MATCHED_TRAINING_INTEGRITY'
    families=[f['family_id'] for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL'];assert len(families)==8
    expected={f'{f}-{t}-c{c}-{b}' for f in families for t in ('M1','M3') for c in (0,1) for b in ('E0','E1')}
    virtual=RAW.parents[1]/'PI0267/R1/virtual/donor' if part=='OLD0' else RAW/'virtual'/part
    paths=list(virtual.glob('*/result.json'));assert {p.parent.name for p in paths}==expected,'INCOMPLETE_FIXED_VIRTUAL_PART'
    for p in paths:
        r=read(p);assert r['validity']=='VALID' and r['point']==('donor' if part=='OLD0' else part)
        assert r['evaluation_card_sha256']==sha(BASE/'PI0267/R1/design/EVALUATION_CARD.json' if part=='OLD0' else R1/'design/EVALUATION_CARD.json') and r['arrays_sha256']==sha(p.parent/'trajectory.npz')
    card=R1/'design/FIXED_OFFLINE_CARD.json'
    for domain,key,n in (('fit','fit_rows',224),('factor','factor_rows',304),('legacy','legacy_rows',440)):
        rows=read(card)[key];assert len(rows)==n
        folder=RAW/'offline'/part/domain
        assert {p.parent.name for p in folder.glob('*/record.json')}=={r['uid'] for r in rows},'INCOMPLETE_FIXED_OFFLINE_PART'
        for row in rows:
            r=read(folder/row['uid']/'record.json');assert r['status']=='VALID' and r['point']==part and r['row']==row and r['card_sha256']==sha(card)
            assert r['tensor_sha256']==sha(Path(r['tensor']))


def main(part):
    admitted(part)
    out=RAW/'archive'/('results-'+part);out.mkdir(parents=True,exist_ok=True);files=[]
    roots=[(RAW/'virtual'/part,'virtual/'+part),(RAW/'offline'/part,'offline/'+part)] if part in ('OLD0','ROT0','OLD2048','ROT2048') else [(RAW/'diagnostics','secondary')]
    if part=='OLD0':roots=[(RAW/'offline'/part,'offline/'+part)]  # Exact old virtual package is referenced, not copied.
    if part=='summary':roots=[(RAW/'delivery','delivery'),(RAW/'factors','factors'),(R1/'contracts','contracts'),(R1/'design','design'),(R1/'evidence','evidence'),(R1/'model_bindings','model-bindings'),(R1/'code','code')]
    for folder,prefix in roots:
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                assert not p.is_symlink()
                # Never package credentials, cache, private URDF/controller payloads or Python binaries.
                if p.suffix not in ('.json','.jsonl','.npz','.csv','.md','.py','.pbs','.log','.mp4') or p.name=='RESULT_ARCHIVE_'+part+'.json':continue
                assert p.name not in ('token','stored_tokens') and 'private_source' not in p.parts
                files.append((p,prefix+'/'+p.relative_to(folder).as_posix()))
    assert files
    source=BASE/'PI0266/code/pi0266_report.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='archive_members')
    scope=dict(read=read,write=write,sha=sha,json=json);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    files,members,tree=scope['archive_members'](out,files);package=out/(part+'.tar')
    if not package.exists():
        temp=package.with_suffix('.tar.partial')
        with tarfile.open(temp,'w') as tar:
            for (path,name),row in zip(files,members):
                info=tarfile.TarInfo(name);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with path.open('rb') as stream:tar.addfile(info,stream)
        temp.replace(package)
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip());repo='leoyuer/pi0246-gate-pass-campaign'
    prefix='evidence/PI0268/R1/normalization-results-20261003/'+part;assert api.model_info(repo).private is False
    commit=RAW/'archive'/('RESULT_UPLOAD_'+part+'.json')
    if commit.exists():record=read(commit);assert record['tree_sha256']==tree;revision=record['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,commit_message='PI0268 R1 actual fixed results '+part).oid
        write(commit,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree))
    session=requests.Session();session.trust_env=False;checked=[]
    for path in sorted(p for p in out.iterdir() if p.is_file()):
        digest=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,prefix+'/'+path.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            for block in response.iter_content(8<<20):digest.update(block);size+=len(block)
        assert size==path.stat().st_size and digest.hexdigest()==sha(path);checked.append(dict(path=path.name,bytes=size,sha256=digest.hexdigest(),anonymous=True))
    remaining={r['path']:r for r in members}
    with session.get(hf_hub_url(repo,prefix+'/'+package.name,revision=revision),stream=True,timeout=(30,180)) as response:
        response.raise_for_status();response.raw.decode_content=True
        with tarfile.open(fileobj=response.raw,mode='r|') as tar:
            for member in tar:
                assert member.isfile() and member.name in remaining;row=remaining.pop(member.name);digest=hashlib.sha256();size=0
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(8<<20),b''):digest.update(block);size+=len(block)
                assert size==row['bytes'] and digest.hexdigest()==row['sha256'],member.name
    assert not remaining
    value=dict(status='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS',part=part,repo_id=repo,prefix=prefix,revision=revision,
        tree_sha256=tree,members=len(members),source_bytes=sum(r['bytes'] for r in members),files=checked,
        preserves_original_V2_FAIL=True,not_physical_generalization=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    write(R1/'evidence'/('RESULT_ARCHIVE_'+part+'.json'),value);print(json.dumps(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=('OLD0','ROT0','OLD2048','ROT2048','summary'));p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.check_only:admitted(a.part);print('PASS_COMPLETE_FROZEN_ARCHIVE_PART_NO_UPLOAD',a.part,flush=True)
    else:main(a.part)
