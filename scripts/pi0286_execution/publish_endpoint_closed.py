"""Publish the sealed endpoint evidence package and stream every fixed-revision byte back."""
import hashlib,json,os
from pathlib import Path

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    os.environ.update(HF_HUB_OFFLINE='0',HF_HUB_DISABLE_XET='1',CUDA_VISIBLE_DEVICES='',HF_HOME=str(ROOT/'runtime/hf_online'))
    from huggingface_hub import HfApi,hf_hub_url,CommitOperationAdd
    import requests
    ready=json.loads((ROOT/'archive/endpoint-closed/PACKAGE_READY.json').read_text())
    assert ready['status']=='PASS_FIXED_CLOSED_ENDPOINT_SMALL_EVIDENCE_AND_ALL32_VIDEO_PACKAGE'
    out=ROOT/'archive/endpoint-closed/bundle';repo=ready['repo_id'];prefix=ready['prefix']
    assert repo=='leoyuer/pi0246-gate-pass-campaign' and prefix=='evidence/PI0286/recovery-vs-replay-20261009/endpoint-closed'
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}
    assert files==ready['files']
    record=ROOT/'archive/endpoint-closed/UPLOAD.json'
    if record.exists():
        upload=json.loads(record.read_text());assert upload['files']==files and upload['prefix']==prefix
    else:
        token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip()
        api=HfApi(token=token);assert api.whoami()['name']=='leoyuer';assert api.model_info(repo).private is False
        operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=str(out/name)) for name in sorted(files)]
        commit=api.create_commit(repo_id=repo,operations=operations,commit_message='PI0286 actual C0 and C1 endpoint outcomes and all32 saved RGB conditions',num_threads=4)
        upload=dict(repo_id=repo,prefix=prefix,revision=commit.oid,files=files)
        record.write_text(json.dumps(upload,indent=2)+'\n')
    session=requests.Session();session.trust_env=False;checked=[]
    for name,v in sorted(files.items()):
        h=hashlib.sha256();total=0
        with session.get(hf_hub_url(repo,prefix+'/'+name,revision=upload['revision']),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            for block in response.iter_content(8<<20):h.update(block);total+=len(block)
        assert total==v['bytes'] and h.hexdigest()==v['sha256'],name
        checked.append(dict(path=name,bytes=total,sha256=h.hexdigest()))
        print(json.dumps(dict(verified=name,bytes=total)),flush=True)
    result=dict(status='PASS_PUBLIC_FIXED_REVISION_FULL_FILE_READBACK',scope='CLOSED_C0_C1_ENDPOINTS_NO_DEVELOPMENT_QUALIFIED_CANDIDATE',repo_id=repo,prefix=prefix,revision=upload['revision'],files=checked,full_RGB='RETAINED_ORIGINAL_CORELAB_SOURCE_HASH_INDEX_IN_PACKAGE',new_models=0,new_training_updates=0,new_physics=0,selected='NONE',deployment='NONE')
    (ROOT/'evidence/ENDPOINT_CLOSED_HF_READBACK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
