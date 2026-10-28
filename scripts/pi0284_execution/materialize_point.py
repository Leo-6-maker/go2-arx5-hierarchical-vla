"""Exact readonly changed-tensor transfer; full Trainer and Adam remain on Hopper."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from prepare_core import sha,write

POINTS=dict(CTRL1024='CTRL20481024',CTRL2048='CTRL20482048',LR5X1024='LR5X20481024',LR5X2048='LR5X20482048')

def run(root,point,ready_only=False):
    root=Path(root);source=POINTS[point];folder=root/'transfers'/(point+'.partial');folder.mkdir(parents=True,exist_ok=True)
    key=root/'runtime/hopper_model_pull';assert key.stat().st_mode&0o777==0o600
    dest=root/'transfers'/point
    if dest.exists():
        assert not ready_only
        proof=root/'models'/point/'MATERIALIZATION.json'
        assert proof.exists() and json.loads(proof.read_text())['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW','Recover incomplete CPU materialization before evaluation'
        return json.loads((dest/'DIRECT_READBACK.json').read_text())
    before=time.time();remote='e1192548@hopper.nus.edu.sg:'+source+('/READY.json' if ready_only else '/')
    ssh='ssh -i '+str(key)+' -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15'
    subprocess.run(['rsync','-a','--partial','--append-verify','-e',ssh,remote,str(folder)+'/'],check=True)
    ready=json.loads((folder/'READY.json').read_text());assert ready['point']==source and not ready['optimizer_transfer']
    if ready_only:return ready
    assert {v['path'] for v in ready['files']}=={'exact_changed_tensors.safetensors','EXPORT.json','config.json','processor_config.json','statistics.json','embodiment_id.json'}
    for v in ready['files']:
        p=folder/v['path'];assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],v['path']
    folder.replace(dest)
    readback=dict(status='PASS_DIRECT_TRANSFER_ALL_FILES_FIXED_HASH_READBACK',point=point,internal_snapshot_id=source,files=ready['files'],elapsed_s=time.time()-before,optimizer_transfer=False,HF_critical_path=False)
    write(dest/'DIRECT_READBACK.json',readback)
    sys.path.append(json.loads((root/'design/RUNTIME_CARD.json').read_text())['parent_code'])
    from model_view import materialize
    materialize(root,point,'/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B',dest)
    proof=json.loads((root/'models'/point/'MATERIALIZATION.json').read_text());assert proof['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW'
    export=json.loads((dest/'EXPORT.json').read_text());delta=dest/'exact_changed_tensors.safetensors'
    assert sha(delta)==export['exact_delta_sha256'] and delta.stat().st_size==export['exact_delta_bytes']
    retirement=dict(status='PREVIEW_CONSUMED_TRANSFER_DUPLICATE_ONLY',path=str(delta),bytes=delta.stat().st_size,sha256=export['exact_delta_sha256'],protected_full_Trainer=export['checkpoint'],full_master=proof['full_master_state_sha256'])
    write(dest/'DUPLICATE_RETIREMENT.json',retirement);delta.unlink();retirement['status']='DELETED_CONSUMED_TRANSFER_DUPLICATE_PROOFS_RETAINED';write(dest/'DUPLICATE_RETIREMENT.json',retirement)
    return readback

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--point',choices=tuple(POINTS),required=True);p.add_argument('--ready-only',action='store_true');a=p.parse_args();print(json.dumps(run(a.root,a.point,a.ready_only)))
