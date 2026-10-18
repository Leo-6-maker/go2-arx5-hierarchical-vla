"""Fixed HF handoff of actual zero-step scene qualification, not action replay."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.public_checkpoint import reject_private
from . import confirmation_config as config,n1_limits as n1
from .scope import validate as validate_parent,charge_cpu
from .science import validate_runtime,Ledger

REPO='leoyuer/pi0294-public-artifacts'
NAME='SOURCE_CONFIRM_STATIC_TEN_GROUPS_V1'
PREFIX='PI0294/'+n1.PARENT_RUN+'/'+NAME
MAX_BYTES=32*1024**2


def verify_static(api,revision,files,download,token):
    if (not files or sum(v['bytes'] for v in files.values())>MAX_BYTES
            or any(Path(n).is_absolute() or '..' in Path(n).parts or not n.endswith(('.json','.png')) for n in files)):
        raise PermissionError('ONLY_NEW_BOUNDED_STATIC_JSON_PNG_NO_MODEL_DOWNLOAD')
    rows={r.path:r for r in api.list_repo_tree(REPO,repo_type='dataset',revision=revision,
        path_in_repo=PREFIX,recursive=True) if hasattr(r,'size')}
    if set(rows)!={PREFIX+'/'+n for n in files}:raise ValueError('EXACT_STATIC_REVISION_FILE_SET')
    for name,expected in files.items():
        if rows[PREFIX+'/'+name].size!=expected['bytes']:raise ValueError('STATIC_REVISION_MEMBER_SIZE')
        p=Path(download(REPO,PREFIX+'/'+name,repo_type='dataset',revision=revision,token=token))
        if p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:
            raise ValueError('ACTUAL_STATIC_REVISION_READBACK_MEMBER_HASH')


def inventory(root):
    root=Path(root);m=validate_runtime(root,'4080-simulation')
    if m['execution'].get('confirmation_phase')!='STATIC_PREPARATION':
        raise PermissionError('ONLY_COMPLETED_PREFREEZE_STATIC_QUALIFICATION_NO_POLICY')
    worker=read(root/'worker_processes/confirmation-static-01.json')
    state=Path('/proc',str(worker['pid']),'stat')
    if state.exists():
        fields=state.read_text().split()
        if fields[2]!='Z' and fields[21]==str(worker['proc_start']):
            raise PermissionError('STATIC_WORKER_MUST_BE_TERMINAL_BEFORE_PUBLICATION')
    if read(root/'RESOURCE_LEDGER.json')['reservations']:
        raise PermissionError('NO_STATIC_OR_PHYSICAL_RESERVATION_DURING_PUBLICATION')
    result=read(root/'evidence/CONFIRM_STATIC/RESULT.json');plan=read(root/config.PLAN);inputs=read(root/config.STATIC_INPUTS)
    if (result['status']!='PASS_TEN_NEW_SCENES_TWENTY_ORIGINAL_STATIC_TASK_QUALIFICATIONS_NO_POLICY'
            or result['scene_groups']!=10 or result['static_receipts']!=20 or result['actual_physics_integrations']
            or result['new_model_queries'] or result['new_positive_labels'] or result['Teacher_success_filtering']
            or result['policy_opened'] or result['plan_sha256']!=sha(root/config.PLAN)):
        raise PermissionError('ACTUAL_ZERO_STEP_TEN_GROUPS_TWENTY_TASK_STATIC_RECEIPTS_REQUIRED')
    expected=config.plan.evaluation_plan(plan['scene_groups'],plan['candidate'],plan['static_receipts'],inputs['exclusion_layouts'])
    if expected!=plan:raise PermissionError('EXACT_ORIGINAL_STATIC_AND_CANDIDATE_PLAN')
    sources={'SOURCE_CONFIRM_PLAN.json':root/config.PLAN,'STATIC_RESULT.json':root/'evidence/CONFIRM_STATIC/RESULT.json'}
    for row in plan['static_receipts']:
        images=row['actual_static_images']
        if set(images)!={'body_rgb','wrist_rgb'}:raise PermissionError('TWO_ACTUAL_STATIC_CAMERA_FILES')
        folders=set()
        for image in images.values():
            name=image['path'];p=root/name
            if (Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink()
                    or not p.resolve().is_relative_to((root/'evidence/CONFIRM_STATIC').resolve())
                    or sha(p)!=image['sha256'] or p.stat().st_size!=image['bytes']):
                raise PermissionError('OWNED_STATIC_PIXEL_IDENTITY')
            sources[name]=p;folders.add(p.parent)
        if len(folders)!=1:raise PermissionError('ONE_ACTUAL_STATIC_CARD_AND_CAMERA_PAIR')
        card=folders.pop()/'CARD.json'
        if sha(card)!=row['actual_card_file_sha256']:raise ValueError('ACTUAL_STATIC_CARD_HASH')
        sources[str(card.relative_to(root))]=card
    if len(sources)!=62:raise PermissionError('TWENTY_CARDS_FORTY_ACTUAL_IMAGES_TWO_METADATA_FILES_ONLY')
    for name,p in sources.items():
        if name.endswith('.json'):reject_private(read(p))
    return sources,plan,inputs


def publish(root,token_file):
    root=Path(root);ledger=Ledger(root,'core');ledger.guard();began=time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU,(600,600));signal.alarm(1800)
    lock=OwnerLock(root/'CONFIRM_STATIC_DELIVERY.lock');out=root/'artifacts'/NAME;out.mkdir(mode=0o700)
    try:
        sources,plan,inputs=inventory(root)
        sanitized={k:inputs[k] for k in ['schema','exclusion_layouts','exclusion_population_sha256','original_generator_sha256',
            'source_manifest_sha256','source_TRAIN_episodes','source_DEV_episodes','exact_raw_CARD_FILE_MANIFEST_verified',
            'recovery_parents_verified','legacy_source_file_sha256','original_master_plan_sha256','new_physics','new_model_queries']}
        reject_private(sanitized);write(out/'SCENE_EXCLUSIONS.json',sanitized);sources['SCENE_EXCLUSIONS.json']=out/'SCENE_EXCLUSIONS.json'
        files={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in sources.items()}
        if len(files)!=63 or sum(r['bytes'] for r in files.values())>MAX_BYTES:
            raise PermissionError('BOUNDED_63_FILE_STATIC_BUNDLE_32MIB')
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        os.environ['HF_HUB_OFFLINE']='0';token=Path(token_file).read_text().strip();api=HfApi(token=token)
        info=api.repo_info(REPO,repo_type='dataset')
        if info.private:raise PermissionError('EXISTING_PUBLIC_OWNED_ARTIFACT_REPOSITORY_ONLY')
        if any(n.startswith(PREFIX+'/') for n in api.list_repo_files(REPO,repo_type='dataset')):
            raise FileExistsError('NO_BLIND_STATIC_BUNDLE_REUPLOAD')
        state=dict(status='UPLOAD_INTENT_UNKNOWN_NO_RETRY',repo_id=REPO,prefix=PREFIX,files=files)
        write(out/'TRANSFER.private.json',state)
        rev=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
            commit_message='Original zero-step static qualification for ten new source confirmation groups',
            operations=[CommitOperationAdd(path_in_repo=PREFIX+'/'+n,path_or_fileobj=str(p)) for n,p in sources.items()]).oid
        state.update(status='PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=rev);write(out/'TRANSFER.private.json',state)
        def download(*args,**kwargs):return hf_hub_download(*args,**kwargs,local_dir=out/'readback')
        verify_static(api,rev,files,download,token)
        ready=dict(status='READY_ACTUAL_ZERO_STEP_SOURCE_CONFIRM_STATIC_NOT_POLICY_RESULTS',files=files,
            parent_run=n1.PARENT_RUN,child=config.plan.CHILD,data_revision=rev,groups=10,task_qualifications=20,
            actual_physics_integrations=0,new_model_queries=0,new_Teacher_labels=0)
        write(out/'READY.json',ready)
        final=api.create_commit(REPO,repo_type='dataset',parent_commit=rev,
            commit_message='READY after all actual static qualification payload hashes read back',
            operations=[CommitOperationAdd(path_in_repo=PREFIX+'/READY.json',path_or_fileobj=str(out/'READY.json'))]).oid
        final_files=dict(files,**{'READY.json':dict(bytes=(out/'READY.json').stat().st_size,sha256=sha(out/'READY.json'))})
        verify_static(api,final,final_files,download,token)
        state.update(status='PASS_SOURCE_CONFIRM_STATIC_FIXED_REVISION_READBACK',revision=final,
            READY_sha256=sha(out/'READY.json'),READY_uploaded_last=True,payload_bytes=sum(r['bytes'] for r in files.values()))
        write(out/'TRANSFER.private.json',state);return state
    finally:signal.alarm(0);ledger.cpu(time.process_time()-began);lock.close()


def finish(root,token_file):
    """Known pre-READY readback failure: verify existing commit, never reupload."""
    root=Path(root);ledger=Ledger(root,'core');ledger.guard();began=time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU,(600,600));signal.alarm(1800)
    owner=OwnerLock(root/'CONFIRM_STATIC_DELIVERY.lock');out=root/'artifacts'/NAME
    try:
        sources,_,_=inventory(root);sources['SCENE_EXCLUSIONS.json']=out/'SCENE_EXCLUSIONS.json'
        files={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in sources.items()}
        journal=out/'TRANSFER.private.json';state=read(journal)
        if (state['status']!='PAYLOAD_COMMITTED_READBACK_PENDING' or state['repo_id']!=REPO or state['prefix']!=PREFIX
                or state['files']!=files or len(files)!=63 or (out/'READY.json').exists()):
            raise PermissionError('ONLY_KNOWN_PRE_READY_FAILURE_EXACT_EXISTING_PAYLOAD_NO_BLIND_RETRY')
        archive=out/'TRANSFER_PRE_READY_FAILURE.private.json'
        if archive.exists():raise FileExistsError('ONE_EVIDENCED_PRE_READY_READBACK_RECOVERY_ONLY')
        write(archive,state);archive.chmod(0o400)
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        os.environ['HF_HUB_OFFLINE']='0';token=Path(token_file).read_text().strip();api=HfApi(token=token)
        info=api.repo_info(REPO,repo_type='dataset');rev=state['data_revision']
        if info.private or info.sha!=rev:raise PermissionError('NO_CONCURRENT_HF_WRITER_OR_PAYLOAD_REPLACEMENT')
        def download(*args,**kwargs):return hf_hub_download(*args,**kwargs,local_dir=out/'readback')
        verify_static(api,rev,files,download,token)
        ready=dict(status='READY_ACTUAL_ZERO_STEP_SOURCE_CONFIRM_STATIC_NOT_POLICY_RESULTS',files=files,
            parent_run=n1.PARENT_RUN,child=config.plan.CHILD,data_revision=rev,groups=10,task_qualifications=20,
            actual_physics_integrations=0,new_model_queries=0,new_Teacher_labels=0)
        write(out/'READY.json',ready)
        final=api.create_commit(REPO,repo_type='dataset',parent_commit=rev,
            commit_message='READY after exact bounded static JSON and camera evidence readback',
            operations=[CommitOperationAdd(path_in_repo=PREFIX+'/READY.json',path_or_fileobj=str(out/'READY.json'))]).oid
        state.update(status='READY_COMMITTED_FINAL_READBACK_PENDING',revision=final);write(journal,state)
        final_files=dict(files,**{'READY.json':dict(bytes=(out/'READY.json').stat().st_size,sha256=sha(out/'READY.json'))})
        verify_static(api,final,final_files,download,token)
        state.update(status='PASS_SOURCE_CONFIRM_STATIC_FIXED_REVISION_READBACK',READY_sha256=sha(out/'READY.json'),
            READY_uploaded_last=True,payload_bytes=sum(v['bytes'] for v in files.values()),payload_reuploaded=False,
            recovery='EXACT_EXISTING_PAYLOAD_BOUNDED_STATIC_METADATA_READBACK_NOT_LARGE_MODEL')
        write(journal,state);return state
    finally:signal.alarm(0);ledger.cpu(time.process_time()-began);owner.close()


def consume(parent,binding_relative,token_file):
    parent=Path(parent);relative=Path(binding_relative);p=parent/relative
    if relative.is_absolute() or '..' in relative.parts or p.is_symlink():raise PermissionError('SEALED_OWNED_STATIC_CONSUMPTION_BINDING')
    m=validate_parent(parent,'hopper-neural');q=read(p);first=parent/n1.FIRST_ADOPTION
    if first.is_symlink() or first.stat().st_mode&0o222 or sha(first)!=n1.ADOPTED_DELTA_SHA:
        raise PermissionError('ORIGINAL_FIRST_N1_STATIC_HANDOFF_AUTHORITY')
    delta=n1.validated_delta(read(first),read(parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json'))
    if (m['execution'].get('confirmation_static_consume_sha256')!=sha(p) or p.stat().st_mode&0o222
            or q['maximum_CPU_s']!=600 or q['new_queries'] or q['new_physics']
            or q['N1_authority_release']!=n1.RELEASE or q['N1_adoption_sha256']!=n1.ADOPTED_DELTA_SHA
            or q['N1_T0_epoch']!=delta['autonomy_started_epoch']
            or not delta['autonomy_started_epoch']<=q['created_epoch']<=time.time()<q['deadline_epoch']<=min(q['created_epoch']+1800,delta['new_submission_and_candidate_cutoff_epoch'])):
        raise PermissionError('BOUNDED_CPU_ONLY_STATIC_HF_CONSUMPTION_BEFORE_ORIGINAL_CUTOFF')
    t=q['transfer']
    if (t['status']!='PASS_SOURCE_CONFIRM_STATIC_FIXED_REVISION_READBACK' or t['repo_id']!=REPO or t['prefix']!=PREFIX
            or len(t['files'])!=63 or not t['READY_uploaded_last'] or sum(r['bytes'] for r in t['files'].values())>MAX_BYTES):
        raise PermissionError('EXACT_FIXED_SOURCE_CONFIRM_STATIC_DELIVERY')
    resource.setrlimit(resource.RLIMIT_CPU,(600,600));signal.alarm(max(1,int(q['deadline_epoch']-time.time())))
    owner=OwnerLock(parent/'CONFIRM_STATIC_CONSUME.lock');began=time.process_time();out=parent/'datasets'/NAME;out.mkdir(mode=0o700)
    try:
        from huggingface_hub import hf_hub_download
        os.environ['HF_HUB_OFFLINE']='0';token=Path(token_file).read_text().strip()
        def get(name,checksum):
            if (Path(name).is_absolute() or '..' in Path(name).parts or not name.endswith(('.json','.png'))):
                raise PermissionError('ONLY_STATIC_JSON_PNG_NO_MODEL_OR_OTHER_HISTORY_DOWNLOAD')
            f=Path(hf_hub_download(REPO,PREFIX+'/'+name,repo_type='dataset',revision=t['revision'],token=token,local_dir=out))
            if sha(f)!=checksum:raise ValueError('STATIC_FIXED_REVISION_MEMBER_HASH')
            return f
        ready=read(get('READY.json',t['READY_sha256']))
        if (ready['files']!=t['files'] or ready['data_revision']!=t['data_revision'] or ready['groups']!=10
                or ready['task_qualifications']!=20 or ready['actual_physics_integrations'] or ready['new_model_queries']):
            raise PermissionError('ACTUAL_ZERO_STEP_STATIC_READY_MEMBERSHIP')
        members={n:get(n,v['sha256']) for n,v in t['files'].items()}
        plan=read(members['SOURCE_CONFIRM_PLAN.json']);inputs=read(members['SCENE_EXCLUSIONS.json'])
        if plan!=config.plan.evaluation_plan(plan['scene_groups'],plan['candidate'],plan['static_receipts'],inputs['exclusion_layouts']):
            raise PermissionError('EXACT_FROZEN_40_CONDITIONS_AND_ORIGINAL_STATIC_CANDIDATE_RULES')
        for row in plan['static_receipts']:
            for image in row['actual_static_images'].values():
                if sha(members[image['path']])!=image['sha256']:raise ValueError('ACTUAL_STATIC_PIXEL_HASH_JOIN')
        result=dict(status='PASS_HOPPER_CPU_STATIC_FIXED_REVISION_CONSUMED_NOT_NEURAL_OR_POLICY',transfer=t,
            plan_sha256=sha(members['SOURCE_CONFIRM_PLAN.json']),exclusion_sha256=sha(members['SCENE_EXCLUSIONS.json']),
            new_model_queries=0,new_physics=0,new_Teacher_labels=0)
        write(out/'CONSUMED.private.json',result);return result
    finally:signal.alarm(0);charge_cpu(parent,time.process_time()-began);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['publish','finish','consume']);p.add_argument('--root',required=True)
    p.add_argument('--binding');p.add_argument('--existing-token-file',required=True);a=p.parse_args()
    def expire(signum,frame):raise TimeoutError('FINITE_STATIC_HF_DELIVERY_WALL')
    signal.signal(signal.SIGALRM,expire);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    v=publish(a.root,a.existing_token_file) if a.role=='publish' else finish(a.root,a.existing_token_file) if a.role=='finish' else consume(a.root,a.binding,a.existing_token_file)
    print(json.dumps(dict(status=v['status'],new_physics=0,new_queries=0)),flush=True)
