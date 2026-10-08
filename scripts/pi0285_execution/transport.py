"""Read-only exact model export pull; complete Trainer remains on Hopper."""
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path
from contract import ROOT,guard,sha,write


def pull(point,ready_only=False):
    from policy import tasks
    tasks(point,'TRAIN');guard()
    folder=ROOT/'transfers'/(point+'.partial');folder.mkdir(parents=True,exist_ok=True)
    final=ROOT/'transfers'/point
    key=ROOT/'runtime/hopper_model_pull';assert key.stat().st_mode&0o777==0o600
    before=time.time()
    if not final.exists():
        ssh=shlex.join(['ssh','-i',str(key),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(ROOT/'runtime/hopper_known_hosts'),'-o','ConnectTimeout=15'])
        source='e1192548@hopper.nus.edu.sg:'+point+('/READY.json' if ready_only else '/')
        subprocess.run(['rsync','-a','--partial','--append-verify','-e',ssh,source,str(folder)+'/'],check=True)
        ready=json.loads((folder/'READY.json').read_text())
        assert ready['point']==point and not ready['optimizer_transfer'] and ready['status']=='LOCAL_IMMUTABLE_MODEL_ONLY_SNAPSHOT'
        if ready_only:return ready
        assert {v['path'] for v in ready['files']}=={'exact_changed_tensors.safetensors','EXPORT.json','config.json','processor_config.json','statistics.json','embodiment_id.json'}
        for item in ready['files']:
            p=folder/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
        folder.replace(final)
        write(final/'DIRECT_READBACK.json',dict(status='PASS_DIRECT_TRANSFER_ALL_FILES_FIXED_HASH_READBACK',point=point,files=ready['files'],elapsed_s=time.time()-before,optimizer_transfer=False,HF_critical_path=False))
    assert not ready_only
    # Native master reconstruction and BF16 casting were already verified by this helper.
    sys.path.append(str(ROOT.parent/'PI0283/code'))
    from model_view import materialize
    base='/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B'
    proof=ROOT/'models'/point/'MATERIALIZATION.json'
    if not proof.exists():materialize(ROOT,point,base,final)
    result=json.loads(proof.read_text());assert result['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW'
    for name,h in result['files'].items():assert sha(ROOT/'models'/point/name)==h,name
    delta=final/'exact_changed_tensors.safetensors'
    if delta.exists():
        export=json.loads((final/'EXPORT.json').read_text());assert delta.stat().st_size==export['exact_delta_bytes'] and sha(delta)==export['exact_delta_sha256']
        retirement=dict(status='PREVIEW_CONSUMED_TRANSFER_DUPLICATE_ONLY',path=str(delta),bytes=delta.stat().st_size,sha256=export['exact_delta_sha256'],protected_full_trainer=export['checkpoint'],full_master_state_sha256=result['full_master_state_sha256'])
        write(final/'DUPLICATE_RETIREMENT.json',retirement);delta.unlink();assert not delta.exists()
        retirement['status']='DELETED_CONSUMED_TRANSFER_DUPLICATE_SMALL_PROOFS_RETAINED';write(final/'DUPLICATE_RETIREMENT.json',retirement)
    return json.loads((final/'DIRECT_READBACK.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True);p.add_argument('--ready-only',action='store_true');a=p.parse_args();print(json.dumps(pull(a.point,a.ready_only)))
