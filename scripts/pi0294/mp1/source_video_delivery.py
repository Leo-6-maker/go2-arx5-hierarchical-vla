"""Deliver an exact frozen source panel or partial snapshot without replay."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.checkpoint_delivery import verify_remote
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.public_checkpoint import reject_private
from .audit import require_frozen_attempt
from .scope import validate,charge_cpu
from .video_delivery import REPO

NAME='SBC2048_TRAIN4_ACTUAL_AUDITED_VIDEO_PUBLIC_V1'
PANEL='S-BC-2048-TRAIN4'
IDS=[f'PI0294-MP1-{PANEL}-{f}-000-{t}' for f in ['F0','F1'] for t in ['T1','T2']]
DEV_PANEL='S-BC-2048-DEV8'
DEV_NAME='SBC2048_DEV8_ACTUAL_AUDITED_VIDEO_PUBLIC_V1'
DEV_IDS=[f'PI0294-MP1-{DEV_PANEL}-{f}-{n}-{t}'
    for f,n in [('F0','000'),('F1','000'),('F1','001'),('F2','000')] for t in ['T1','T2']]
CORR_PANEL='S-CORR-512-TRAIN4'
CORR_PLAN_IDS=[f'PI0294-MP1-{CORR_PANEL}-{f}-000-{t}' for f in ['F0','F1'] for t in ['T1','T2']]
CORR_IDS=CORR_PLAN_IDS[:3]+[CORR_PLAN_IDS[2]+'-REPLACEMENT-01']
CORR_NAME='SCORR512_FIRST4_ATTEMPTS_ACTUAL_AUDITED_VIDEO_PUBLIC_V1'


def delivery_scope(panel):
    if panel==PANEL:scope=dict(panel=PANEL,name=NAME,ids=IDS,split='TRAIN',suffix='TRAIN4')
    elif panel==DEV_PANEL:scope=dict(panel=DEV_PANEL,name=DEV_NAME,ids=DEV_IDS,split='DEV',suffix='DEV8')
    elif panel==CORR_PANEL:
        return dict(panel=CORR_PANEL,name=CORR_NAME,ids=CORR_IDS,plan_ids=CORR_PLAN_IDS,
            split='TRAIN',suffix='FIRST4_ATTEMPTS',step=512,tag='SCORR512_FIRST4_ATTEMPTS',
            binding='PUBLIC_SCORR512_FIRST4_ATTEMPTS_VIDEO_BINDING.json',complete_panel=False)
    else:raise PermissionError('EXACT_FIXED_SOURCE_VIDEO_DELIVERY_SCOPE')
    return dict(scope,step=2048,plan_ids=scope['ids'],tag='SBC2048_'+scope['suffix'],
        binding='PUBLIC_SBC2048_'+scope['suffix']+'_VIDEO_BINDING.json',complete_panel=True)


def audited_movies(root,*,panel=PANEL):
    scope=delivery_scope(panel);ids=scope['ids']
    child=Path(root)/'children/source-pair-01';plan=read(child/'plans/SOURCE_EVALUATION_PLAN.json')
    spec=plan['panels'][panel]
    if (plan['schema']!='MP1_FIRST_SOURCE_PAIR_FIXED_DEVELOPMENT_V1'
            or spec['attempt_ids']!=scope['plan_ids'] or spec['checkpoint_step']!=scope['step']
            or spec['split']!=scope['split'] or spec['teacher_takeover']):
        raise PermissionError('EXACT_PREDECLARED_SOURCE_PANEL_OR_PARTIAL_SNAPSHOT')
    rows,files,sources=[],{},{}
    for identifier in ids:
        actual=require_frozen_attempt(child,identifier)
        engineering=panel==CORR_PANEL and identifier in CORR_IDS[2:]
        allowed=['ENGINEERING_INCOMPLETE'] if engineering else ['VALID_TASK_FAIL'] if panel==CORR_PANEL else ['SUCCESS','VALID_TASK_FAIL']
        if (actual['model']!=panel or actual['split']!=scope['split']
                or actual['pipeline_trustworthy'] is not (not engineering)
                or actual['status'] not in allowed
                or actual['teacher_takeover'] or actual['live_native_failures']):
            raise PermissionError('FROZEN_EXACT_OUTCOME_CLASS_AND_NATIVE_SAFETY')
        folder=child/'evidence/audits'/identifier;audit=read(folder/'AUDIT.private.json');movie=audit['video']
        if (audit['status']!='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT'
                or audit['attempt_id']!=identifier or audit['adjudication']!=actual
                or not audit['checks'] or not all(audit['checks'].values()) or audit['errors']):
            raise PermissionError('COMPLETE_INDEPENDENT_RAW_EVIDENCE_AUDIT_REQUIRED')
        path=folder/'actual-body-wrist.mp4'
        if path.is_symlink() or not path.resolve().is_relative_to(child.resolve()):
            raise PermissionError('OWNED_ACTUAL_VIDEO_NO_SYMLINK_OR_ESCAPE')
        if (sha(path)!=movie['sha256'] or path.stat().st_size!=movie['bytes']
                or movie['decoded_frames']!=movie['actual_paired_frames']
                or movie['decoded_frames']!=actual['RGB_pairs']
                or movie['native_duration_s']!=actual['duration_s']
                or movie['padding_or_interpolation']):
            raise ValueError('ORIGINAL_VIDEO_HASH_FULL_DECODE_AND_DURATION_IDENTITY')
        name='movies/'+identifier+'.mp4';sources[name]=path
        files[name]=dict(bytes=movie['bytes'],sha256=movie['sha256'])
        rows.append(dict(id=identifier,status=actual['status'],task=actual['task'],
            scene_group=actual['scene_group'],native_sim_s=actual['duration_s'],
            actual_frames=movie['decoded_frames'],S1_rejections=actual['S1_rejections'],
            actual_native_faults=actual['live_native_failures'],teacher_takeover=False,
            failure_stage=actual['failure_stage'],artifact_path=name,initial_frame_included=True,
            task_success=actual['status']=='SUCCESS',
            pipeline_trustworthy=actual['pipeline_trustworthy'],
            qualified_policy_outcome=not engineering,
            engineering_incomplete=engineering,
            independent_new_condition=identifier!=CORR_IDS[-1],**files[name]))
    if sum(v['bytes'] for v in files.values())>16*1024**2:
        raise PermissionError('EXACT_PANEL_ACTUAL_CLIPS_16MIB_UPPER')
    reject_private(rows)
    return rows,files,sources


def publish(root,token_file,*,panel=PANEL):
    scope=delivery_scope(panel);name=scope['name'];ids=scope['ids'];suffix=scope['suffix']
    root=Path(root);runtime=validate(root,'4080-cpu-data');binding=root/'plans'/scope['binding']
    if sha(binding)!=runtime['execution']['source_video_binding_sha256']:
        raise PermissionError('SEALED_EXACT_SOURCE_VIDEO_BINDING')
    q=read(binding)
    if (q['parent_run']!=root.name or q['repo_id']!=REPO or q['attempt_ids']!=ids
            or q.get('panel',PANEL)!=panel
            or q['maximum_CPU_s']!=600 or q['worker_CPU_s']!=420
            or q['registration_control_CPU_upper_s']!=180 or q['new_queries'] or q['new_physics']
            or q['public_authority']!='DIRECT_USER_PUBLIC_GENERATED_SOURCE_VIDEO_PERMISSION'
            or not q['created_epoch']<=time.time()<q['deadline_epoch']<=q['created_epoch']+1800):
        raise PermissionError('BOUNDED_ORIGINAL_MP1_PUBLIC_SOURCE_CLIP_DELIVERY')
    signal.alarm(max(1,int(q['deadline_epoch']-time.time())))
    # Four one-core metadata probes each have a 15s wall limit; retain 60 CPU seconds for them.
    resource.setrlimit(resource.RLIMIT_CPU,(360,360));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    lock=OwnerLock(root/(name+'.lock'));started=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN);out=root/'artifacts'/name
    out.mkdir(mode=0o700)
    try:
        rows,files,sources=audited_movies(root,panel=panel)
        for path in sources.values():
            metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format_tags:stream_tags','-of','json',str(path)],text=True,timeout=15))
            reject_private(metadata)
        index=dict(schema='MP1_'+scope['tag']+'_ACTUAL_VIDEO_INDEX_V1',parent_run=root.name,
            source_sha=runtime['code']['vla_commit'],panel=panel,private=False,
            actual_videos=len(ids),rows=rows,all_videos_are_successes=all(r['task_success'] for r in rows),
            exposed_development_or_target_results_included=scope['split']=='DEV',
            no_interpolation_or_synthetic_frames=True,original_private_bytes_unchanged=True,
            transport='QUALIFIED_METADATA_READ_ASSISTED_PAUSED_SIM_LOCKSTEP_NOT_REALTIME',
            selected='NONE',deployment='NONE')
        if not scope['complete_panel']:
            index.update(complete_fixed_panel=False,unique_attempted_conditions=3,qualified_policy_conditions=2,
                engineering_incomplete_attempts=2,unattempted_original_conditions=[CORR_PLAN_IDS[-1]],
                third_F1_T1_attempt_authorized=False,partial_videos_are_policy_failures=False,
                all_videos_are_complete_episodes=False,checkpoint_step=512,
                same_condition_replacement_is_independent_example=False)
        if scope['split']=='DEV':index.update(exposed_development=True,independent_hidden_confirmation=False,target_results_included=False)
        reject_private(index);write(out/'VIDEO_INDEX.json',index)
        sources['VIDEO_INDEX.json']=out/'VIDEO_INDEX.json'
        files['VIDEO_INDEX.json']=dict(bytes=sources['VIDEO_INDEX.json'].stat().st_size,sha256=sha(sources['VIDEO_INDEX.json']))
        os.environ['HF_HUB_OFFLINE']='0'
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        token=Path(token_file).read_text().strip();api=HfApi(token=token)
        info=api.repo_info(REPO,repo_type='dataset')
        if info.private:raise PermissionError('EXISTING_PUBLIC_REPO_ONLY_NO_VISIBILITY_CHANGE')
        prefix='PI0294/'+root.name+'/'+name
        if any(n.startswith(prefix+'/') for n in api.list_repo_files(REPO,repo_type='dataset')):
            raise FileExistsError('NO_BLIND_REUPLOAD_OF_IMMUTABLE_SOURCE_VIDEO_PREFIX')
        state=dict(status='UPLOAD_INTENT_UNKNOWN_NO_BLIND_RETRY',repo_id=REPO,prefix=prefix,files=files,actual_videos=len(ids))
        write(out/'TRANSFER.private.json',state)
        revision=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
            commit_message='Audited '+scope['tag']+' actual source videos with original outcome classes',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(sources[n])) for n in files]).oid
        state.update(status='PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=revision);write(out/'TRANSFER.private.json',state)
        def download(*args,**kwargs):return hf_hub_download(*args,**kwargs,local_dir=out/'small-readback')
        verify_remote(api,REPO,prefix,revision,files,download,token)
        ready=dict(status='READY_ACTUAL_AUDITED_'+scope['tag']+'_VIDEOS',parent_run=root.name,
            actual_videos=len(ids),files=files,data_revision=revision,private=False,READY_uploaded_last=True)
        write(out/'READY.json',ready)
        final=api.create_commit(REPO,repo_type='dataset',parent_commit=revision,
            commit_message='READY after exact '+scope['tag']+' video hash readback',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(out/'READY.json'))]).oid
        final_files=dict(files,**{'READY.json':dict(bytes=(out/'READY.json').stat().st_size,sha256=sha(out/'READY.json'))})
        verify_remote(api,REPO,prefix,final,final_files,download,token)
        state.update(status='PASS_PUBLIC_'+scope['tag']+'_VIDEOS_FIXED_REVISION_READBACK',revision=final,
            READY_sha256=sha(out/'READY.json'),READY_uploaded_last=True,payload_bytes=sum(v['bytes'] for v in files.values()))
        write(out/'TRANSFER.private.json',state);return state
    finally:
        signal.alarm(0);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        charge_cpu(root,time.process_time()-started+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--existing-token-file',required=True);p.add_argument('--panel',choices=[PANEL,DEV_PANEL,CORR_PANEL],default=PANEL);a=p.parse_args()
    def expire(signum,frame):raise TimeoutError('ORIGINAL_SOURCE_VIDEO_DELIVERY_1800S_WALL')
    signal.signal(signal.SIGALRM,expire)
    value=publish(a.root,a.existing_token_file,panel=a.panel);print(json.dumps({k:value[k] for k in ['status','revision','prefix','actual_videos','payload_bytes']}),flush=True)
