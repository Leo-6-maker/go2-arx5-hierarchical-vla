"""Archive sealed compact PI0280 evidence; retain full image arrays on Corelab."""
import gzip,hashlib,json,os,tarfile
from pathlib import Path

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280')
os.environ.update(HF_HUB_OFFLINE='0',CUDA_VISIBLE_DEVICES='',HF_HOME=str(ROOT/'runtime/hf_online'))
import requests
from huggingface_hub import HfApi,hf_hub_url

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    assert json.loads((ROOT/'evidence/SOURCE_PHYSICS_RESULT.json').read_text())['status']=='FIXED_FOUR_SOURCE_CONDITIONS_EXECUTED'
    out=ROOT/'archive/compact-baseline';out.mkdir(parents=True,exist_ok=True)
    files=[];videos=[];excluded=[]
    for folder in ('design','evidence','native','source','wbc','attribution'):
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file():continue
            if p.name=='COMPACT_ARCHIVE.json':continue
            name=p.relative_to(ROOT).as_posix()
            if p.name=='observations.npz' or p.suffix=='.mp4':
                if p.name in ('source_ood_nonrealtime.mp4','hold.mp4','vertical.mp4','vertical-engineering-000.mp4'):videos.append(p)
                else:excluded.append(dict(path=name,bytes=p.stat().st_size,retained_on_corelab=True))
                continue
            if p.suffix not in ('.json','.jsonl','.npz','.md','.py'):continue
            assert not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()),name
            files.append((p,name))
    rows=[dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for p,n in files]
    assert sum(r['bytes'] for r in rows)<300*2**20
    assert len({r['path'] for r in rows})==len(rows)
    inventory=dict(members=rows,excluded=excluded,full_rgb_arrays_retained_locally=True,new_training_updates=0)
    (out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    package=out/'evidence.tar.gz'
    with package.with_suffix('.partial').open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for (p,name),row in zip(files,rows):
            assert sha(p)==row['sha256']
            info=tarfile.TarInfo(name);info.size=row['bytes'];info.mode=0o644;info.mtime=0
            with p.open('rb') as f:tar.addfile(info,f)
    package.with_suffix('.partial').replace(package)
    # ponytail: hard links avoid duplicating sealed videos in the upload directory.
    for p in videos:
        name=p.parent.parent.name+'-'+p.name if p.name.startswith('source_') else p.name
        q=out/name
        if q.exists():assert sha(q)==sha(p)
        else:os.link(p,q)
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0280/pristine-frozen-baseline-20261007'
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip())
    assert api.model_info(repo).private is False
    upload=ROOT/'archive/COMPACT_UPLOAD.json'
    identity={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    if upload.exists():
        prior=json.loads(upload.read_text());assert prior['files']==identity;revision=prior['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,commit_message='PI0280 pristine baseline compact evidence and actual videos').oid
        upload.write_text(json.dumps(dict(revision=revision,files=identity)))
    session=requests.Session();session.trust_env=False;checked=[]
    for p in sorted(out.iterdir()):
        h=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            for b in response.iter_content(8<<20):h.update(b);size+=len(b)
        assert h.hexdigest()==identity[p.name] and size==p.stat().st_size
        checked.append(dict(path=p.name,bytes=size,sha256=h.hexdigest()))
    result=dict(status='PASS_PUBLIC_FIXED_REVISION_FULL_FILE_READBACK',repo_id=repo,prefix=prefix,revision=revision,files=checked,members=len(rows),source_bytes=sum(r['bytes'] for r in rows),full_observation_arrays='RETAINED_CORELAB_NOT_IN_COMPACT_ARCHIVE',new_training_updates=0)
    (ROOT/'evidence/COMPACT_ARCHIVE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
