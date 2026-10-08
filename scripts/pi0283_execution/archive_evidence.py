"""Seal compact evidence after terminal; full RGB and full Trainers stay protected."""
import argparse,gzip,hashlib,json,os,shutil,tarfile
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def allowed(p,roots):
    return any(p.resolve().is_relative_to(r.resolve()) for r in roots)


def main(root,host):
    root=Path(root);assert host in ('core','hop')
    assert json.loads((root/'evidence/TERMINAL_RECEIPT.json').read_text())['execution_closed']
    roots=[root]
    if host=='core':roots.append(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0283'))
    out=root/'archive/compact-terminal';out.mkdir(parents=True,exist_ok=True)
    tops=['authority','code','design','evidence','audit']
    tops+=['source','videos','models','transfers'] if host=='core' else ['data','delivery','learning','runtime']
    files=[];videos=[];excluded=[];links=[]
    for top in tops:
        for folder,dirs,names in os.walk(root/top,followlinks=True):
            p=Path(folder);assert allowed(p,roots),p
            for n in list(dirs):
                q=p/n
                if n in ('__pycache__','hf_online','hf_offline','cache') or (top=='data' and n=='episodes'):
                    dirs.remove(n);excluded.append(dict(path=str(q.relative_to(root)),class_='CACHE_OR_PREVIOUS_DATA_DEPENDENCY',retained_locally=q.exists()));continue
                if not allowed(q,roots):raise AssertionError(('UNAPPROVED_DIRECTORY',str(q)))
                if q.is_symlink():links.append(dict(path=str(q.relative_to(root)),resolved=str(q.resolve())))
            for n in sorted(names):
                q=p/n
                if not q.is_file():continue
                assert allowed(q,roots),q
                name=q.relative_to(root).as_posix()
                if q.is_symlink():links.append(dict(path=name,resolved=str(q.resolve())))
                if n=='observations.npz' or q.suffix in ('.safetensors','.pt','.pth','.bin'):
                    excluded.append(dict(path=name,bytes=q.stat().st_size,sha256=sha(q),class_='FULL_RGB_OR_TRAINER_PAYLOAD_RETAINED',retained_locally=True));continue
                if n=='COMPACT_ARCHIVE.json' or q.suffix not in ('.json','.jsonl','.npz','.npy','.md','.py','.sh','.pbs','.log','.mp4'):continue
                row=dict(path=name,bytes=q.stat().st_size,sha256=sha(q));files.append((q,row))
                if q.suffix=='.mp4':videos.append(q)
    rows=[v for _,v in files];assert len({v['path'] for v in rows})==len(rows)
    inventory=dict(host=host,members=rows,excluded=excluded,logical_symlinks=links,full_rgb='RETAINED_CORELAB_NOT_IN_COMPACT_ARCHIVE',full_Trainers='HOPPER_AND_PINNED_HF_NOT_IN_COMPACT_ARCHIVE',new_model_queries=0,new_training_updates=0,new_physics=0)
    (out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    package=out/'evidence.tar.gz'
    with package.with_suffix('.partial').open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for p,row in files:
            assert sha(p)==row['sha256']
            info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
            with p.open('rb') as f:tar.addfile(info,f)
    package.with_suffix('.partial').replace(package)
    for p in videos:
        # ponytail: these are a few MB; cross-filesystem hard links are unnecessary.
        q=out/p.name
        if q.exists():assert sha(q)==sha(p)
        else:shutil.copyfile(p,q)
    os.environ.update(HF_HUB_OFFLINE='0',CUDA_VISIBLE_DEVICES='',HF_HOME=str(root/'runtime/hf_online'))
    import requests
    from huggingface_hub import HfApi,hf_hub_url
    token=Path('/home/liuyu-nfs/.config/huggingface/token') if host=='core' else Path.home()/'.config/huggingface/token'
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0283/native-dose-event-20261008/'+host
    api=HfApi(token=token.read_text().strip());assert api.model_info(repo).private is False
    identity={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    upload=root/'archive/COMPACT_UPLOAD.json'
    if upload.exists():
        prior=json.loads(upload.read_text());assert prior['files']==identity;revision=prior['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,commit_message='PI0283 '+host+' immutable terminal evidence and actual videos').oid
        upload.write_text(json.dumps(dict(revision=revision,files=identity)))
    session=requests.Session();session.trust_env=False;checked=[]
    for p in sorted(out.iterdir()):
        h=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            for b in response.iter_content(8<<20):h.update(b);size+=len(b)
        assert h.hexdigest()==identity[p.name] and size==p.stat().st_size
        checked.append(dict(path=p.name,bytes=size,sha256=h.hexdigest()))
    result=dict(status='PASS_PUBLIC_FIXED_REVISION_FULL_FILE_READBACK',repo_id=repo,prefix=prefix,revision=revision,files=checked,members=len(rows),source_bytes=sum(v['bytes'] for v in rows),full_RGB='RETAINED_CORELAB_NOT_IN_COMPACT_ARCHIVE',full_Trainers='PINNED_HF_SEPARATE_FULL_READBACK_RECEIPTS',new_model_queries=0,new_training_updates=0,new_physics=0)
    (root/'evidence/COMPACT_ARCHIVE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',choices=('hop','core'),required=True);a=p.parse_args();main(a.root,a.host)
