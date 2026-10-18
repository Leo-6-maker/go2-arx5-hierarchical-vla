"""Final, bounded publication of existing confirmation videos after effect closure."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.relay_protocol import digest
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.public_checkpoint import reject_private
from . import confirmation_config as config,n1_limits as n1
from .confirmation_archive import inventory
from .scope import validate,charge_cpu

REPO='leoyuer/pi0294-public-artifacts'
NAME='SOURCE_CONFIRM40_ACTUAL_AUDITED_VIDEO_PUBLIC_V1'
BINDING='plans/PUBLIC_SOURCE_CONFIRM_VIDEO_BINDING.json'
CLOSURE='plans/SOURCE_CONFIRM_FINAL_EFFECT_CLOSURE.private.json'
MAX_BYTES=128*1024**2


def verify_videos(api,prefix,revision,files,download,token):
    if (not files or sum(v['bytes'] for v in files.values())>MAX_BYTES+1024**2
            or any(Path(n).is_absolute() or '..' in Path(n).parts
                or not (n in ['VIDEO_INDEX.json','READY.json'] or n.startswith('movies/PI0294-MP1-SOURCE-CONFIRM-') and n.endswith('.mp4')) for n in files)):
        raise PermissionError('ONLY_NEW_BOUNDED_CONFIRMATION_CLIPS_AND_TWO_INDEX_FILES_NO_MODEL_DOWNLOAD')
    remote={r.path:r for r in api.list_repo_tree(REPO,repo_type='dataset',revision=revision,path_in_repo=prefix,recursive=True) if hasattr(r,'size')}
    if set(remote)!={prefix+'/'+n for n in files}:raise ValueError('EXACT_FIXED_CONFIRMATION_VIDEO_FILE_SET')
    for name,expected in files.items():
        if remote[prefix+'/'+name].size!=expected['bytes']:raise ValueError('FIXED_CONFIRMATION_VIDEO_SIZE')
        path=Path(download(REPO,prefix+'/'+name,repo_type='dataset',revision=revision,token=token))
        if path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
            raise ValueError('ACTUAL_DOWNLOADED_CONFIRMATION_VIDEO_HASH')


def closed_snapshot(root,request):
    root=Path(root);child=root/'children'/config.plan.CHILD
    closure=root/CLOSURE;plan=config.validate_plan(child)
    if (closure.is_symlink() or closure.stat().st_mode&0o222
            or sha(closure)!=request['final_effect_closure_sha256']):
        raise PermissionError('SEALED_FINAL_EFFECT_CLOSURE_REQUIRED_NOT_LIVE_STATUS')
    proof=read(closure);rows=[read(p) for p in sorted((child/'episodes').glob('*/MP1_ADJUDICATION.json'))]
    ledger=read(child/'RESOURCE_LEDGER.json')
    native=proof['native_allocations']
    if (proof['schema']!='MP1_CONFIRM_FINAL_EFFECT_CLOSURE_V1' or proof['run_uuid']!=child.name
            or proof['plan_sha256']!=sha(child/config.PLAN) or proof['rows_sha256']!=digest(rows)
            or proof['unknown_effects'] or proof['pending_queries'] or proof['pending_optimizer_updates']
            or proof['all_Core_and_Hopper_consumers_terminal'] is not True or proof['all_action_effects_classified'] is not True
            or not 1<=len(native)<=2 or len({j['job_id'] for j in native})!=len(native)
            or any(j['state']!='F' or j['project']!='CFP04-CF-007' or type(j['actual_GPU_s']) is not int or j['actual_GPU_s']<0 for j in native)
            or not rows or len(rows)>80 or ledger['reservations'] or ledger['CPU_reserved_s']
            or request['attempt_ids']!=[r['attempt_id'] for r in rows]):
        raise PermissionError('ALL_TERMINAL_ATTEMPTS_NO_CHERRY_PICKING_OR_UNKNOWN_ACTION_EFFECTS')
    for folder in ['worker_processes','aux_worker_processes']:
        for p in (child/folder).glob('*.json'):
            worker=read(p);stat=Path('/proc',str(worker['pid']),'stat')
            fields=stat.read_text().split() if stat.exists() else None
            if fields is not None and fields[2]!='Z' and fields[21]==str(worker['proc_start']):
                raise PermissionError('NO_ACTIVE_CONFIRMATION_CONSUMER_DURING_FINAL_ARCHIVE')
    value,files,sources=inventory(child,plan,rows)
    if not files or sum(v['bytes'] for v in files.values())>MAX_BYTES:
        raise PermissionError('AT_MOST_EIGHTY_EXISTING_AUDITED_CLIPS_128MIB')
    value.update(parent_run=root.name,child=child.name,plan_sha256=proof['plan_sha256'],
        final_effect_closure_sha256=sha(closure),private=False,
        domain='FIXED_BASE_ARX5_TASK_SIM_NOT_GO2_WBC_OR_REAL_ROBOT',
        transport='FRESH_OBSERVATION_AFTER_PREVIOUS_ACK_PAUSED_SIM_DUAL_SSH_LOCKSTEP_NOT_REALTIME',
        realtime_control_claim=False,
        source_sha_scope='PUBLISHER_VERSION_NOT_ORIGINAL_PHYSICAL_EXECUTION_SOURCE',
        no_interpolation_or_synthetic_frames=True,all_original_terminal_attempts_in_snapshot=True,
        complete_confirmation=value['coverage']['status']=='COMPLETE_ACTUAL_AUDITED_CONFIRM40')
    reject_private(value)
    return value,files,sources


def checked_request(root,runtime,now):
    root=Path(root);binding=root/BINDING
    if (binding.is_symlink() or binding.stat().st_mode&0o222
            or runtime['execution'].get('confirmation_video_binding_sha256')!=sha(binding)):
        raise PermissionError('IMMUTABLE_EXACT_FINAL_VIDEO_BINDING')
    q=read(binding);adoption=root/n1.FIRST_ADOPTION
    if (adoption.is_symlink() or adoption.stat().st_mode&0o222 or sha(adoption)!=n1.ADOPTED_DELTA_SHA):
        raise PermissionError('ORIGINAL_N1_FIRST_ADOPTION_NO_RESET')
    delta=n1.validated_delta(read(adoption),read(root/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json'))
    if (q['parent_run']!=root.name or q['repo_id']!=REPO or q['child']!=config.plan.CHILD
            or q['N1_authority_release']!=n1.RELEASE or q['N1_adoption_sha256']!=n1.ADOPTED_DELTA_SHA
            or q['N1_T0_epoch']!=delta['autonomy_started_epoch']
            or q['maximum_CPU_s']!=1800 or q['worker_CPU_s']!=1620 or q['registration_CPU_upper_s']!=180
            or q['maximum_payload_bytes']!=MAX_BYTES or q['new_queries'] or q['new_physics'] or q['new_training_updates']
            or type(q['CPU_id']) is not int or q['CPU_id'] not in os.sched_getaffinity(0)
            or q['public_authority']!='DIRECT_USER_PUBLIC_GENERATED_SOURCE_VIDEO_PERMISSION'
            or not q['created_epoch']<=now<q['deadline_epoch']<=min(q['created_epoch']+3600,delta['hard_closeout_epoch'])):
        raise PermissionError('FINITE_ORIGINAL_N1_PUBLIC_VIDEO_ONLY_AUTHORITY')
    return q


def publish(root,token_file):
    root=Path(root);runtime=validate(root,'4080-cpu-data');q=checked_request(root,runtime,time.time())
    signal.alarm(max(1,int(q['deadline_epoch']-time.time())))
    # Reserve up to15 CPU seconds for each bounded single-core metadata subprocess.
    resource.setrlimit(resource.RLIMIT_CPU,(420,420));os.sched_setaffinity(0,{q['CPU_id']})
    lock=OwnerLock(root/'SOURCE_CONFIRM_FINAL_VIDEO_DELIVERY.lock')
    started=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    out=root/'artifacts'/NAME
    try:
        out.mkdir(mode=0o700)
        index,files,sources=closed_snapshot(root,q)
        for path in sources.values():
            metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format_tags:stream_tags','-of','json',str(path)],text=True,timeout=15))
            reject_private(metadata)
        index.update(publisher_source_sha=runtime['code']['vla_commit'])
        write(out/'VIDEO_INDEX.json',index);sources['VIDEO_INDEX.json']=out/'VIDEO_INDEX.json'
        files['VIDEO_INDEX.json']=dict(bytes=(out/'VIDEO_INDEX.json').stat().st_size,sha256=sha(out/'VIDEO_INDEX.json'))
        if sum(v['bytes'] for v in files.values())>MAX_BYTES:raise PermissionError('VIDEO_AND_INDEX_128MIB_UPPER')
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        os.environ['HF_HUB_OFFLINE']='0';token=Path(token_file).read_text().strip();api=HfApi(token=token)
        info=api.repo_info(REPO,repo_type='dataset');prefix='PI0294/'+root.name+'/'+NAME
        if info.private:raise PermissionError('EXISTING_OWNED_PUBLIC_REPOSITORY_NO_VISIBILITY_CHANGE')
        if any(n.startswith(prefix+'/') for n in api.list_repo_files(REPO,repo_type='dataset')):
            raise FileExistsError('IMMUTABLE_CONFIRMATION_VIDEO_PREFIX_NO_BLIND_REUPLOAD')
        state=dict(status='UPLOAD_INTENT_UNKNOWN_NO_BLIND_RETRY',repo_id=REPO,prefix=prefix,files=files,
            actual_videos=index['actual_videos'],complete_confirmation=index['complete_confirmation'])
        write(out/'TRANSFER.private.json',state)
        revision=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
            commit_message='All terminal audited source confirmation clips, preserving failures and gaps',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(p)) for n,p in sources.items()]).oid
        state.update(status='PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=revision);write(out/'TRANSFER.private.json',state)
        def download(*args,**kwargs):return hf_hub_download(*args,**kwargs,local_dir=out/'small-readback')
        verify_videos(api,prefix,revision,files,download,token)
        for name,path in sources.items():
            if sha(path)!=files[name]['sha256']:raise ValueError('SOURCE_VIDEO_OR_INDEX_CHANGED_DURING_UPLOAD')
        ready=dict(status='READY_ACTUAL_AUDITED_SOURCE_CONFIRMATION_VIDEO_BYTES',parent_run=root.name,
            files=files,data_revision=revision,actual_videos=index['actual_videos'],
            complete_confirmation=index['complete_confirmation'],READY_uploaded_last=True,selected='NONE',deployment='NONE')
        write(out/'READY.json',ready)
        final=api.create_commit(REPO,repo_type='dataset',parent_commit=revision,
            commit_message='READY after exact source confirmation clip hash readback',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(out/'READY.json'))]).oid
        state.update(status='READY_COMMITTED_FINAL_READBACK_PENDING',revision=final);write(out/'TRANSFER.private.json',state)
        final_files=dict(files,**{'READY.json':dict(bytes=(out/'READY.json').stat().st_size,sha256=sha(out/'READY.json'))})
        verify_videos(api,prefix,final,final_files,download,token)
        state.update(status='PASS_SOURCE_CONFIRMATION_VIDEOS_FIXED_REVISION_READBACK',READY_sha256=sha(out/'READY.json'),
            READY_uploaded_last=True,payload_bytes=sum(v['bytes'] for v in files.values()))
        write(out/'TRANSFER.private.json',state);return state
    finally:
        signal.alarm(0);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        charge_cpu(root,time.process_time()-started+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--existing-token-file',required=True);a=p.parse_args()
    def expire(signum,frame):raise TimeoutError('FINITE_N1_CONFIRMATION_VIDEO_ARCHIVE_DEADLINE')
    signal.signal(signal.SIGALRM,expire);v=publish(a.root,a.existing_token_file)
    print(json.dumps({k:v[k] for k in ['status','revision','prefix','actual_videos','payload_bytes','complete_confirmation']}),flush=True)
