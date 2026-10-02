"""Public fixed-revision corpus seal; reuse the mature member manifest helper."""
import ast
import hashlib
import json
import sys
import tarfile
from pathlib import Path
import requests
from huggingface_hub import HfApi,hf_hub_url
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
sys.path[:0]=[str(BASE/'PI0250/runtime/reference-317a/src'),str(BASE/'PI0266/code')]
from pi0255_motion3 import write,sha


def read(p):return json.loads(Path(p).read_text())


def main():
    assert read(R1/'design/DATA_ADMISSION.json')['status']=='PASS_CONTINUE_ACTUAL_INPUT_AND_TRAINING'
    assert read(R1/'design/PROCESSED_PLAN_AUDIT.json')['status']=='PASS_PI0267_REAL_INPUT_COLOR_TEMPLATE_CODEC_RNG_BOUNDARY'
    assert read(R1/'evidence/DATA_CPU_REPLAY.json')['proposals']==10752
    out=RAW/'archive/data-public';out.mkdir(parents=True,exist_ok=True);files=[]
    for p in sorted((RAW/'S2').rglob('*')):
        if p.is_file():
            assert not p.is_symlink() and p.suffix in ('.json','.npz')
            files.append((p,'S2/'+p.relative_to(RAW/'S2').as_posix()))
    for p in sorted((R1/'design').glob('*.json')):files.append((p,'design/'+p.name))
    for name in ('FIXTURES.json','SAVED_ARRAY_READMISSION.json','V2_V3_64_ROWS.csv','RENDERER_FIXTURES.json','DATA_CPU_REPLAY.json'):
        files.append((R1/'evidence'/name,'evidence/'+name))
    files.append((ROOT/'design/SCENE_CARD.json','design/SCENE_CARD.json'))
    for name in ('pi0267_expert.py','pi0267_virtual_renderer.py','pi0267_virtual_data.py','pi0267_render_train.py',
                 'pi0267_data_admission.py','pi0267_training_plan.py','pi0267_data_replay.py','pi0267_metrics_v3.py','pi0267_r1_cpu.py'):
        files.append((ROOT/'code'/name,'code/'+name))
    source=BASE/'PI0266/code/pi0266_report.py';helper=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='archive_members')
    scope=dict(read=read,write=write,sha=sha,json=json)
    exec(compile(ast.Module(body=[helper],type_ignores=[]),str(source),'exec'),scope)
    files,members,tree=scope['archive_members'](out,files);package=out/'PI0267_R1_VIRTUAL_TRAIN_CORPUS.tar'
    if not package.exists():
        temp=package.with_suffix('.tar.partial')
        with tarfile.open(temp,'w') as tar:
            for (p,name),row in zip(files,members):
                info=tarfile.TarInfo(name);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with p.open('rb') as stream:tar.addfile(info,stream)
        temp.replace(package)
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip())
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0267/R1/virtual-train-corpus-20261002'
    assert api.model_info(repo).private is False;commit=RAW/'archive/DATA_UPLOAD_COMMIT.json'
    if commit.exists():
        a=read(commit);assert a['tree_sha256']==tree;revision=a['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,commit_message='PI0267 R1 actual10752 virtual TRAIN proposals and independent CPU/input admission').oid
        write(commit,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree))
    session=requests.Session();session.trust_env=False;checked=[]
    for p in sorted(x for x in out.iterdir() if x.is_file()):
        h=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            for block in response.iter_content(8<<20):h.update(block);size+=len(block)
        assert size==p.stat().st_size and h.hexdigest()==sha(p)
        checked.append(dict(path=p.name,bytes=size,sha256=h.hexdigest(),anonymous=True))
    remaining={r['path']:r for r in members}
    with session.get(hf_hub_url(repo,prefix+'/'+package.name,revision=revision),stream=True,timeout=(30,180)) as response:
        response.raise_for_status();response.raw.decode_content=True
        with tarfile.open(fileobj=response.raw,mode='r|') as tar:
            for member in tar:
                assert member.isfile() and member.name in remaining;row=remaining.pop(member.name);h=hashlib.sha256();size=0
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(8<<20),b''):h.update(block);size+=len(block)
                assert size==row['bytes'] and h.hexdigest()==row['sha256'],member.name
    assert not remaining
    receipt=dict(status='PASS_PUBLIC_FIXED_REVISION_DATA_FILES_AND_ALL_MEMBERS',repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree,
        members=len(members),source_bytes=sum(r['bytes'] for r in members),files=checked,virtual_proposals=10752,TRAIN_families=24,EVAL_payload_opened=False,
        dataset_domain='CONTROLLER_INDEPENDENT_VIRTUAL_TOOL_SUPERVISION',old_PI0267_archive_unchanged=True)
    write(R1/'evidence/DATA_PUBLIC_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
