"""Finite READY-last publication of FD2's already audited actual videos only."""
import json
import os
from pathlib import Path
import resource
import signal
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.fd2 import validate

REPO='leoyuer/pi0294-public-artifacts'
NAME='FD2_DYNAMIC_TEACHER_CLEARANCE_ACTUAL_V1'
MAX_BYTES=32*1024**2


def inventory(root):
    root=Path(root);plan=read(root/'plans/PILOT_PLAN.json');rows=[];files={};sources={}
    terminal=read(root/'evidence/PILOT_RESULT.private.json')
    if terminal['status']!='TERMINAL_EIGHT_CELLS' or len(terminal['rows'])!=8:
        raise PermissionError('FD2_ALL_EIGHT_TERMINAL_ATTEMPTS_REQUIRED_NO_CHERRY_PICKING')
    for card in plan['cards']:
        folder=root/'episodes'/card['id'];out=root/'evidence/audits'/card['id']
        audit=read(out/'AUDIT.private.json');result=read(folder/'RESULT.json')
        if (audit['status']!='PASS_INDEPENDENT_SAVED_BYTE_AUDIT' or audit['errors']
                or not all(audit['checks'].values()) or audit['result_sha256']!=sha(folder/'RESULT.json')):
            raise PermissionError('FD2_AUDITED_EXACT_ORIGINAL_ATTEMPT')
        movie=audit['video'];path=root/movie['relative_path']
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_size!=movie['bytes'] or sha(path)!=movie['sha256']
                or movie['decoded_frames']!=movie['actual_paired_frames']
                or movie['decoded_frames']!=result['rgb_pairs'] or movie['padding_or_interpolation']):
            raise ValueError('FD2_EXISTING_ACTUAL_VIDEO_IDENTITY')
        name='movies/'+card['id']+'.mp4';files[name]=dict(bytes=movie['bytes'],sha256=movie['sha256']);sources[name]=path
        rows.append(dict(id=card['id'],group=card['group_id'],split='TRAIN',nominal_clearance_m=card['nominal_clearance_m'],
            status=result['status'],task_success=audit['score']['persistent_success'],
            S1_rejections=0 if audit['S1'] is None else audit['S1']['pre_execution_rejections'],
            actual_native_faults=0 if audit['S1'] is None else audit['S1']['live_native_failures'],
            native_sim_s=result['physics_steps']*.002,actual_paired_frames=movie['decoded_frames'],
            artifact_path=name,**files[name]))
    if sum(v['bytes'] for v in files.values())>MAX_BYTES:
        raise PermissionError('FD2_32MIB_PUBLIC_VIDEO_PAYLOAD_BOUND')
    from .public_checkpoint import reject_private
    reject_private(rows)
    return dict(schema='PI0294_FD2_ACTUAL_VIDEO_INDEX_V1',run_uuid=root.name,attempts=rows,
        planned=8,actual=8,scene_groups=4,Teacher=True,GR00T_inference=False,
        domain='FIXED_BASE_ARX5_NOT_GO2_WBC_OR_REAL_ROBOT',training_authorized=False,
        all_actual_attempts_included=True,no_padding_or_interpolation=True,selected='NONE',deployment='NONE'),files,sources


def run(root,token_file):
    root=Path(root);lock=OwnerLock(root/'FD2_OWNER.lock');m,_=validate(root)
    binding=read(root/'plans/ARCHIVE_BINDING.private.json')
    code=Path(binding['publisher_file']);ledger=read(root/'RESOURCE_LEDGER.json')
    if (Path(__file__).resolve()!=code.resolve() or sha(code)!=binding['publisher_sha256']
            or ledger['status']!='ARCHIVE_RESERVED' or ledger['CPU_reserved_s']!=900
            or ledger['unknown_effects']):
        raise PermissionError('FD2_EXACT_BOUND_CPU_ARCHIVE_OWNER')
    worker=read(root/'worker_processes/pilot.json');stat=Path('/proc',str(worker['pid']),'stat')
    f=stat.read_text().split() if stat.exists() else None
    if f and f[2]!='Z' and f[21]==str(worker['proc_start']):
        raise PermissionError('FD2_NO_ACTIVE_PHYSICS_CONSUMER_DURING_ARCHIVE')
    start=time.process_time();started=time.monotonic();out=root/'artifacts'/NAME;out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def expire(*args):raise TimeoutError('FD2_ARCHIVE_FINITE_ORIGINAL_DEADLINE')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(min(900,m['limits']['hard_close_epoch']-time.time()))))
    journal=dict(status='INVENTORY',publisher_source=binding['source_sha'],runtime_source=m['code']['vla_commit'],new_physics=0)
    try:
        index,files,sources=inventory(root);put(out/'VIDEO_INDEX.json',index)
        files['VIDEO_INDEX.json']=dict(bytes=(out/'VIDEO_INDEX.json').stat().st_size,sha256=sha(out/'VIDEO_INDEX.json'))
        sources['VIDEO_INDEX.json']=out/'VIDEO_INDEX.json'
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        from .checkpoint_delivery import verify_remote
        os.environ['HF_HUB_OFFLINE']='0';token=Path(token_file).read_text().strip();api=HfApi(token=token)
        info=api.repo_info(REPO,repo_type='dataset');assert info.private is False
        prefix='PI0294/'+root.name+'/'+NAME
        assert not any(n.startswith(prefix+'/') for n in api.list_repo_files(REPO,repo_type='dataset'))
        journal.update(status='PAYLOAD_COMMIT_INTENT',repo_id=REPO,prefix=prefix,files=files);write(out/'TRANSFER.private.json',journal)
        revision=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
            commit_message='FD2 all eight actual Teacher clearance videos; no training or policy selection',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(p)) for n,p in sources.items()]).oid
        journal.update(status='PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=revision);write(out/'TRANSFER.private.json',journal)
        def download(*a,**k):return hf_hub_download(*a,**k,local_dir=out/'small-readback')
        verify_remote(api,REPO,prefix,revision,files,download,token)
        ready=dict(status='ALL_EIGHT_ACTUAL_VIDEOS_FIXED_SHA_VERIFIED',repo_id=REPO,prefix=prefix,
            data_revision=revision,files=dict(files),READY_uploaded_last=True,private=False,selected='NONE',deployment='NONE')
        put(out/'READY.json',ready)
        final=api.create_commit(REPO,repo_type='dataset',parent_commit=revision,
            commit_message='FD2 READY after actual eight-video fixed-revision readback',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(out/'READY.json'))]).oid
        files['READY.json']=dict(bytes=(out/'READY.json').stat().st_size,sha256=sha(out/'READY.json'))
        verify_remote(api,REPO,prefix,final,files,download,token)
        if any(sha(sources[n])!=h['sha256'] for n,h in files.items() if n!='READY.json'):
            raise ValueError('FD2_LOCAL_ORIGINAL_VIDEO_CHANGED_DURING_UPLOAD')
        journal.update(status='PASS_ALL_EIGHT_PUBLIC_VIDEOS_FIXED_REVISION_READBACK',revision=final,
            READY_uploaded_last=True,files=files,payload_bytes=sum(h['bytes'] for h in files.values()),
            no_models_assets_or_private_paths_uploaded=True)
        return journal
    except BaseException as exc:
        journal.update(status='FAILED_OR_UNKNOWN_NO_AUTO_RETRY',error_type=type(exc).__name__)
        raise
    finally:
        signal.alarm(0);journal.update(CPU_s=time.process_time()-start,wall_s=time.monotonic()-started)
        write(out/'TRANSFER.private.json',journal)
        ledger.update(status='ARCHIVE_TERMINAL',Core_CPU_s=ledger['Core_CPU_s']+journal['CPU_s'],CPU_reserved_s=0)
        if journal['status']=='FAILED_OR_UNKNOWN_NO_AUTO_RETRY':ledger['unknown_effects'].append('ARCHIVE_REQUIRES_EXACT_REMOTE_RECONCILIATION')
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--existing-token-file',required=True)
    a=p.parse_args();v=run(a.root,a.existing_token_file)
    print(json.dumps({k:v[k] for k in ['status','revision','data_revision','payload_bytes']}),flush=True)
