"""Private owned procedural data only; verify archives and publish READY last."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_url
import requests
from go2arx5_vla.task_data.config import COMMISSIONING,PILOTS
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget


def stream_hash(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def verify_episode_archive(path,folder):
    files=json.loads((folder/'FILE_MANIFEST.json').read_text())
    expected={folder.name+'/'+x['path']:x['sha256'] for x in files}
    expected[folder.name+'/FILE_MANIFEST.json']=file_hash(folder/'FILE_MANIFEST.json')
    found={}
    with tarfile.open(path,'r|gz') as tar:
        for member in tar:
            if member.isdir():continue
            assert member.isfile() and member.name in expected
            digest=stream_hash(tar.extractfile(member));assert digest==expected[member.name]
            found[member.name]=digest
    assert found==expected
    return len(found)


def main(root):
    root=Path(root);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));budget=Budget(root)
    repo='leoyuer/pi0290-independent-dynamic-tool';prefix='PI0290/v1-c4853f3b'
    try:
        assert json.loads((root/'evidence/PILOT_SCORE.json').read_text())['status']=='INDEPENDENT_DATA_PILOT_READY'
        token=(Path.home()/'.config/huggingface/token').read_text().strip()
        api=HfApi(token=token);assert api.whoami()['name']=='leoyuer'
        api.create_repo(repo_id=repo,repo_type='dataset',private=True,exist_ok=True)
        info=api.repo_info(repo,repo_type='dataset');assert info.private
        assert not any(x.startswith(prefix+'/') for x in api.list_repo_files(repo,repo_type='dataset')),'Existing prefix: do not overwrite'
        archive=root/'archive';archive.mkdir(exist_ok=True)
        frozen=archive/'frozen_code_c4853f3.tar.gz';seal=json.loads((root/'TEACHER_AND_LAYOUT_SEAL.json').read_text())
        with tarfile.open(frozen,'r:gz') as tar:
            actual={m.name:stream_hash(tar.extractfile(m)) for m in tar if m.isfile()}
        assert actual==seal['code_files'],'Frozen source archive differs from prepilot code seal'
        payload={frozen.name:frozen};archive_proofs={}
        for identifier in COMMISSIONING+PILOTS:
            budget.guard();folder=root/'episodes'/identifier;out=archive/(identifier+'.tar.gz')
            if not out.exists():
                with tarfile.open(out,'w:gz',compresslevel=3) as tar:tar.add(folder,arcname=identifier)
            archive_proofs[identifier]=dict(member_files_verified=verify_episode_archive(out,folder),raw_manifest_sha256=file_hash(folder/'FILE_MANIFEST.json'))
            payload['episodes/'+out.name]=out
            video=folder/'ACTUAL_DUAL_RGB.mp4'
            if video.exists():payload['videos/'+identifier+'.mp4']=video
        for name in ['SCENE_AND_EXECUTOR_FREEZE.json','TEACHER_AND_LAYOUT_SEAL.json','LAYOUT_SEAL.json']:
            payload['receipts/'+name]=root/name
        for name in ['PILOT_SCORE.json','PILOT_RAW_MANIFEST.json','DATA_ADMISSION_AND_CODEC.json','CPU_TESTS.json','SOFTWARE_RENDER_PROBE.json','CAMERA_QUALIFICATION_X_SIDE.json','LAYOUT_PREFLIGHT.json']:
            payload['receipts/'+name]=root/'evidence'/name
        for name in ['audit_data.py','closeout_data.py','archive_private_hf.py']:
            payload['postpilot_audit_source/'+name]=root/'code/scripts/pi0290_task_data'/name
        payload['postpilot_audit_source/audit.py']=root/'code/src/go2arx5_vla/task_data/audit.py'
        files=[dict(path=prefix+'/'+n,bytes=p.stat().st_size,sha256=file_hash(p)) for n,p in payload.items()]
        license_proof=dict(destination='private dataset owned by authenticated repo owner',
            authorization='PI0290 #6127997973 private compliant raw artifacts',
            geometry='own new procedural boxes; no private WBC, robot mesh, model weights or tokenizer bytes copied',
            publication='research evidence only; no hardware or public robot-asset redistribution claim')
        write(root/'evidence/HF_LICENSE_AND_ARCHIVE_CHECK.json',dict(status='PASS',repo_id=repo,visibility='private',
            source_archive_matches_seal=True,episode_archives=archive_proofs,license=license_proof))
        payload['receipts/HF_LICENSE_AND_ARCHIVE_CHECK.json']=root/'evidence/HF_LICENSE_AND_ARCHIVE_CHECK.json'
        path=payload['receipts/HF_LICENSE_AND_ARCHIVE_CHECK.json'];files.append(dict(path=prefix+'/receipts/HF_LICENSE_AND_ARCHIVE_CHECK.json',bytes=path.stat().st_size,sha256=file_hash(path)))
        budget.guard()
        commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=info.sha,
            commit_message='PI0290 actual independent data, all attempts, fixed source and CPU proofs',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(p)) for n,p in payload.items()])
        revision=commit.oid
        def verify_remote(revision,rows):
            remote={x.path:x for x in api.list_repo_tree(repo,repo_type='dataset',revision=revision,path_in_repo=prefix,recursive=True) if hasattr(x,'size')}
            for row in rows:
                member=remote[row['path']];assert member.size==row['bytes']
                lfs=member.lfs
                if lfs:
                    digest=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256'];assert digest==row['sha256']
                else:
                    response=requests.get(hf_hub_url(repo,row['path'],repo_type='dataset',revision=revision),headers={'Authorization':'Bearer '+token},timeout=60)
                    response.raise_for_status();assert hashlib.sha256(response.content).hexdigest()==row['sha256']
            return True
        verify_remote(revision,files)
        ready=dict(status='READY',repo_id=repo,repo_type='dataset',visibility='private',data_revision=revision,prefix=prefix,
            frozen_source_sha=seal['source_sha'],files=files,license=license_proof,archive_members_verified=True,
            GR00T=0,WBC=0,Hopper_raw_transfer='DEFERRED_UNTIL_NEW_GO')
        ready_path=archive/'READY.json';write(ready_path,ready)
        final=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=revision,commit_message='PI0290 READY last after every artifact SHA readback',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(ready_path))]).oid
        ready_row=dict(path=prefix+'/READY.json',bytes=ready_path.stat().st_size,sha256=file_hash(ready_path))
        verify_remote(final,files+[ready_row]);assert api.repo_info(repo,repo_type='dataset',revision=final).private
        receipt=dict(status='HF_TRANSFER_READY',repo_id=repo,repo_type='dataset',visibility='private',complete_revision=final,
            data_revision=revision,prefix=prefix,files=files+[ready_row],READY_uploaded_last=True,
            readback='every uploaded LFS SHA and non-LFS content SHA at fixed final revision; each raw file verified inside its episode archive',
            frozen_source_sha=seal['source_sha'],Hopper_raw_transfer='DEFERRED_UNTIL_NEW_TRAINING_GO')
        write(root/'evidence/HF_TRANSFER_READY.json',receipt)
        print(json.dumps({'status':receipt['status'],'repo':repo,'revision':final,'file_count':len(files)+1}))
    except Exception as exc:
        response=getattr(exc,'response',None)
        write(root/'evidence/HF_BLOCKER.json',dict(status='BLOCKED_PRIVACY_OR_HF',error_type=type(exc).__name__,
            HTTP_status=None if response is None else response.status_code,raw_data_retained_private_Corelab=True,READY_claim=False))
        raise
    finally:budget.finish_cpu('D4_PRIVATE_HF_ARCHIVE_UPLOAD_AND_READBACK')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
