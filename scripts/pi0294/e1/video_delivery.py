"""Deliver only independently decoded actual E1 movies to an immutable private prefix."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.checkpoint_delivery import verify_remote
from .scope import validate,Ledger


def publish(root,repo,token_file):
    root=Path(root);runtime=validate(root,'4080-simulation');ledger=Ledger(root,'core');ledger.guard(closeout=True)
    if read(ledger.path)['reservations']:raise PermissionError('E1_NO_VIDEO_DELIVERY_WITH_UNSETTLED_PHYSICS')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});started=time.process_time();owner=OwnerLock(root/'E1_VIDEO_HF_PUBLISH.lock')
    output=root/'artifacts/actual-video-delivery';output.mkdir(parents=True,mode=0o700)
    try:
        main=read(root/'evidence/MAIN_EVIDENCE_AUDIT_INDEX.private.json');fork=read(root/'evidence/FORK_EVIDENCE_AUDIT_INDEX.private.json')
        if main['status']!='PASS_ALL_DERIVED_MAIN_EVIDENCE' or fork['status']!='PASS_ALL_SAVED_FORK_EVIDENCE':raise PermissionError('E1_INDEPENDENT_ACTUAL_VIDEO_AUDITS_REQUIRED')
        deviation=read(root/'evidence/PROTOCOL_SCOPE_DEVIATION_01.json') if (root/'evidence/PROTOCOL_SCOPE_DEVIATION_01.json').exists() else None
        excluded=deviation['attempt']['attempt_id'] if deviation else None
        index=[];files={};sources={}
        for row in main['rows']+fork['rows']:
            movie=row['video'];identifier=row.get('attempt_id',row.get('id'));name='movies/'+identifier+'.mp4'
            p=(root/movie['relative_path']).resolve()
            if (not p.is_relative_to(root.resolve()) or p.is_symlink() or sha(p)!=movie['sha256']
                    or p.stat().st_size!=movie['bytes'] or movie['decoded_frames']!=movie['actual_paired_frames']
                    or movie['padding_or_interpolation']):raise ValueError('E1_ACTUAL_DECODED_VIDEO_IDENTITY')
            files[name]=dict(bytes=p.stat().st_size,sha256=sha(p));sources[name]=p
            item=dict(movie,artifact_path=name,id=identifier,excluded_scope_deviation=identifier==excluded)
            if 'model_autonomous_evaluation' in row:item['model_autonomous_evaluation']=row['model_autonomous_evaluation']
            if 'restoration' in row:item['restoration']=row['restoration']
            index.append(item)
        if not index or sum(x['bytes'] for x in files.values())>128*1024**2:raise PermissionError('E1_VIDEO_NONEMPTY_128_MIB_CAP')
        value=dict(schema='E1_ACTUAL_VIDEO_INDEX_V1',run_uuid=root.name,actual_videos=len(index),rows=index,
            source_sha=runtime['code']['vla_commit'],private=True,not_all_videos_autonomous=True,
            no_interpolation_or_fabricated_final_camera=True,selected='NONE',deployment='NONE')
        write(output/'VIDEO_INDEX.json',value);sources['VIDEO_INDEX.json']=output/'VIDEO_INDEX.json'
        files['VIDEO_INDEX.json']=dict(bytes=(output/'VIDEO_INDEX.json').stat().st_size,sha256=sha(output/'VIDEO_INDEX.json'))
        os.environ['HF_HUB_OFFLINE']='0'
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        token=Path(token_file).read_text().strip();api=HfApi(token=token);info=api.repo_info(repo,repo_type='dataset')
        if not info.private:raise PermissionError('E1_ACTUAL_VIDEO_PRIVATE_ONLY')
        prefix='PI0294/'+root.name+'/E1_ACTUAL_AUDITED_VIDEO_V1'
        if any(n.startswith(prefix+'/') for n in api.list_repo_files(repo,repo_type='dataset')):raise FileExistsError('E1_IMMUTABLE_VIDEO_PREFIX')
        state=dict(status='UPLOAD_INTENT_UNKNOWN_NO_BLIND_RETRY',repo_id=repo,prefix=prefix,files=files);write(output/'TRANSFER.private.json',state)
        revision=api.create_commit(repo,repo_type='dataset',parent_commit=info.sha,commit_message='E1 actual audited movies, not autonomous success claims',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(sources[n])) for n in files]).oid
        state.update(status='PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=revision);write(output/'TRANSFER.private.json',state)
        verify_remote(api,repo,prefix,revision,files,hf_hub_download,token)
        ready=dict(status='READY_ACTUAL_AUDITED_VIDEO_BYTES',run_uuid=root.name,files=files,data_revision=revision,
            actual_videos=len(index),private=True,READY_uploaded_last=True)
        write(output/'READY.json',ready)
        final=api.create_commit(repo,repo_type='dataset',parent_commit=revision,commit_message='E1 actual videos READY after fixed SHA verification',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(output/'READY.json'))]).oid
        ready_files=dict(files);ready_files['READY.json']=dict(bytes=(output/'READY.json').stat().st_size,sha256=sha(output/'READY.json'))
        verify_remote(api,repo,prefix,final,ready_files,hf_hub_download,token)
        state.update(status='PASS_E1_ACTUAL_VIDEO_PRIVATE_FIXED_REVISION',revision=final,READY_sha256=sha(output/'READY.json'),actual_videos=len(index),READY_claim=True)
        write(output/'TRANSFER.private.json',state);return state
    finally:ledger.cpu(time.process_time()-started);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);p.add_argument('--existing-token-file',required=True);a=p.parse_args();v=publish(a.root,a.repo,a.existing_token_file)
    print(json.dumps({k:x for k,x in v.items() if k in ['status','revision','prefix','actual_videos']}),flush=True)
