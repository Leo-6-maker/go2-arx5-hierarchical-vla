"""Corelab pulls immutable exact model tensors; optimizer never leaves Hopper."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def pull(root,point,ready_only=False):
    root=Path(root);assert point in ('MORE1024','MORE2048','EVENT1024')
    folder=root/'transfers'/(point+'.partial');folder.mkdir(parents=True,exist_ok=True)
    key=root/'runtime/hopper_model_pull';assert key.stat().st_mode&0o777==0o600
    source='e1192548@hopper.nus.edu.sg:'+point+('/READY.json' if ready_only else '/')
    start=time.monotonic()
    subprocess.run(['rsync','-a','--partial','--append-verify','-e','ssh -i '+str(key)+' -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15',source,str(folder)+'/'],check=True)
    ready=json.loads((folder/'READY.json').read_text());assert ready['point']==point and not ready['optimizer_transfer']
    if ready_only:return ready
    assert {v['path'] for v in ready['files']}=={'exact_changed_tensors.safetensors','EXPORT.json','config.json','processor_config.json','statistics.json','embodiment_id.json'}
    for v in ready['files']:
        p=folder/v['path'];assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],v['path']
    final=folder.with_suffix('');assert not final.exists();folder.replace(final)
    v=dict(status='PASS_DIRECT_TRANSFER_ALL_FILES_FIXED_HASH_READBACK',point=point,files=ready['files'],elapsed_s=time.monotonic()-start,optimizer_transfer=False,HF_critical_path=False)
    (final/'DIRECT_READBACK.json').write_text(json.dumps(v,indent=2)+'\n');return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--point',required=True);p.add_argument('--ready-only',action='store_true');a=p.parse_args()
    print(json.dumps(pull(a.root,a.point,a.ready_only)))
