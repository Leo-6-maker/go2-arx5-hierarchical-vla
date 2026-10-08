"""Seal compact evidence after terminal; full RGB and full Trainers stay protected."""
import argparse,gzip,hashlib,json,os,shutil,tarfile,time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def allowed(p,roots):
    return any(p.resolve().is_relative_to(r.resolve()) for r in roots)


def readback(root,host):
    import requests
    from huggingface_hub import hf_hub_url
    out=root/'archive/compact-terminal'
    upload=json.loads((root/'archive/COMPACT_UPLOAD.json').read_text())
    inventory=json.loads((out/'INVENTORY.json').read_text());assert inventory['host']==host
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0285/common-scene-first-policy-20261008/terminal-R3/'+host
    session=requests.Session();session.trust_env=False;checked=[]
    for name,expected_sha in sorted(upload['files'].items()):
        p=out/name;assert p.parent==out and p.is_file() and sha(p)==expected_sha
        total=p.stat().st_size;h=hashlib.sha256();offset=0
        url=hf_hub_url(repo,prefix+'/'+name,revision=upload['revision'])
        while offset<total:
            end=min(offset+(128<<20),total)-1
            try:
                with session.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(30,180)) as response:
                    response.raise_for_status();assert 'Authorization' not in response.request.headers
                    assert (response.status_code==206 and response.headers.get('Content-Range')==f'bytes {offset}-{end}/{total}') or (response.status_code==200 and offset==0 and end+1==total)
                    block=b''.join(response.iter_content(8<<20));assert len(block)==end-offset+1
            except requests.RequestException as error:
                print('READBACK_NETWORK_RETRY',name,offset,type(error).__name__,flush=True);time.sleep(10);continue
            h.update(block);offset=end+1
            (root/'archive/READBACK_PROGRESS.json').write_text(json.dumps(dict(file=name,bytes_verified=offset,total=total,revision=upload['revision']))+'\n')
        assert h.hexdigest()==expected_sha
        checked.append(dict(path=name,bytes=total,sha256=h.hexdigest()))
    result=dict(status='PASS_PUBLIC_FIXED_REVISION_FULL_FILE_READBACK',repo_id=repo,prefix=prefix,revision=upload['revision'],files=checked,members=len(inventory['members']),source_bytes=sum(v['bytes'] for v in inventory['members']),full_RGB='RETAINED_CORELAB_NOT_IN_COMPACT_ARCHIVE',full_Trainers='PINNED_HF_SEPARATE_FULL_READBACK_RECEIPTS',new_model_queries=0,new_training_updates=0,new_physics=0)
    (root/'evidence/COMPACT_ARCHIVE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


def publish(root,host):
    os.environ.update(HF_HUB_OFFLINE='0',CUDA_VISIBLE_DEVICES='',HF_HOME=str(root/'runtime/hf_online'))
    from huggingface_hub import HfApi
    token=Path('/home/liuyu-nfs/.config/huggingface/token') if host=='core' else Path.home()/'.config/huggingface/token'
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0285/common-scene-first-policy-20261008/terminal-R3/'+host
    out=root/'archive/compact-terminal';identity={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    upload=root/'archive/COMPACT_UPLOAD.json'
    if upload.exists():assert json.loads(upload.read_text())['files']==identity
    else:
        api=HfApi(token=token.read_text().strip());assert api.model_info(repo).private is False
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,commit_message='PI0285 '+host+' immutable terminal evidence and actual videos').oid
        upload.write_text(json.dumps(dict(revision=revision,files=identity)))
    return readback(root,host)


def main(root,host,readback_only=False,publish_only=False):
    root=Path(root);assert host in ('core','hop')
    assert json.loads((root/'evidence/TERMINAL_RECEIPT.json').read_text())['execution_closed']
    if readback_only:return readback(root,host)
    if publish_only:return publish(root,host)
    roots=[root]
    if host=='core':roots.append(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0285'))
    out=root/'archive/compact-terminal';out.mkdir(parents=True,exist_ok=True)
    tops=['authority','code','design','evidence','audit']
    tops+=['data','diagnostics','natural','endpoint_diagnostics','retention','source','videos','models','transfers','runtime'] if host=='core' else ['data','delivery','learning','runtime']
    record_root=Path(json.loads((root/'design/RECORDING_TIER.json').read_text())['record_root']) if host=='core' else None
    if record_root is not None and record_root.resolve()!=(root/'source').resolve():tops.append('recorded-source')
    files=[];videos=[];excluded=[];links=[]
    for top in tops:
        top_root=record_root if top=='recorded-source' else root/top
        for folder,dirs,names in os.walk(top_root,followlinks=True):
            p=Path(folder);assert allowed(p,roots),p
            for n in list(dirs):
                q=p/n
                if n in ('__pycache__','hf_online','hf_offline','cache') or (top=='data' and n=='episodes'):
                    dirs.remove(n);excluded.append(dict(path=(Path(top)/q.relative_to(top_root)).as_posix(),class_='CACHE_OR_PREVIOUS_DATA_DEPENDENCY',retained_locally=q.exists()));continue
                if not allowed(q,roots):raise AssertionError(('UNAPPROVED_DIRECTORY',str(q)))
                if q.is_symlink():links.append(dict(path=(Path(top)/q.relative_to(top_root)).as_posix(),resolved=str(q.resolve())))
            for n in sorted(names):
                q=p/n
                if not q.is_file():continue
                assert allowed(q,roots),q
                name=(Path(top)/q.relative_to(top_root)).as_posix()
                if q.is_symlink():links.append(dict(path=name,resolved=str(q.resolve())))
                if n=='observations.npz' or q.suffix in ('.safetensors','.pt','.pth','.bin'):
                    row=dict(path=name,bytes=q.stat().st_size,class_='FULL_RGB_OR_TRAINER_PAYLOAD_RETAINED',retained_locally=True)
                    if n=='observations.npz':row['sha256']=sha(q)
                    else:row['identity_receipt']='Separate full Trainer/export/materialization readback receipt; not rehashed by compact packer'
                    excluded.append(row);continue
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
        q=out/(p.parent.parent.name+'-'+p.name)
        if q.exists():assert sha(q)==sha(p)
        else:shutil.copyfile(p,q)
    return publish(root,host)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',choices=('hop','core'),required=True);mode=p.add_mutually_exclusive_group();mode.add_argument('--readback-only',action='store_true');mode.add_argument('--publish-only',action='store_true');a=p.parse_args();main(a.root,a.host,a.readback_only,a.publish_only)
