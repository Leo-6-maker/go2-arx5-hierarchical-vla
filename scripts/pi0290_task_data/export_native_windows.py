"""Measured policy inputs and actually committed desired actions; no neural batch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_url
import requests
from go2arx5_vla.task_data.config import PILOTS
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows,rot6d_rows_to_matrix

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--upload',action='store_true');a=p.parse_args()
root=Path(a.root);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));budget=Budget(root);records=[]
try:
    for identifier in PILOTS:
        folder=root/'episodes'/identifier;samples=json.loads((folder/'samples.json').read_text())
        assert len(samples)==641 and samples[-1]['supervision'] is None
        commands=[x['supervision'] for x in samples[:640]]
        assert all(x['actually_committed'] and x['valid'] for x in commands)
        assert all(not any(x['causal_history'][key] for key in ['reset','abort','takeover']) for x in samples)
        desired=np.asarray([x['committed'] for x in commands]);grip=np.asarray([x['committed_gripper'] for x in commands])
        anchors=np.asarray([x['policy_obs']['actual_tcp'] for x in samples[:601]])
        future=np.arange(601)[:,None]+np.arange(40)
        relative=np.linalg.inv(anchors)[:,None]@desired[future]
        actions=np.concatenate([relative[:,:,:3,3],rotation_matrix_to_rot6d_rows(relative[:,:,:3,:3]),grip[future,None]],axis=-1)
        reconstructed=np.repeat(np.eye(4)[None,None],601,axis=0);reconstructed=np.repeat(reconstructed,40,axis=1)
        reconstructed[:,:,:3,3]=actions[:,:,:3];reconstructed[:,:,:3,:3]=rot6d_rows_to_matrix(actions[:,:,3:9])
        error=float(np.max(np.abs(anchors[:,None]@reconstructed-desired[future])))
        assert actions.shape==(601,40,10) and error<1e-8 and np.array_equal(actions[:,:,-1],grip[future])
        states=np.concatenate([anchors[:,:3,3],rotation_matrix_to_rot6d_rows(anchors[:,:3,:3]),
            np.asarray([x['policy_obs']['actual_gripper'] for x in samples[:601]])[:,None]],axis=-1)
        paths=np.asarray([[x['policy_obs'][camera]['path'] for camera in ['body_rgb','wrist_rgb']] for x in samples[:601]])
        out=root/'evidence/codec_windows'/f'{identifier}.npz';out.parent.mkdir(exist_ok=True)
        if out.exists():raise RuntimeError('Existing derived labels; preserve without overwrite')
        np.savez_compressed(out,action_relative_H40=actions,policy_actual_state10=states,policy_rgb_paths=paths,
            anchor_time_s=np.arange(601)/20,offset_s=np.arange(40)/20,valid_window=np.ones(601,dtype=bool),
            label_source=np.asarray('ACTUAL_COMMITTED_DESIRED_EEF_NOT_ACHIEVED'),source_sha=np.asarray('c4853f3b7bd9747dff485d58780a0c0abb1e3402'))
        with np.load(out,allow_pickle=False) as loaded:
            assert loaded['action_relative_H40'].shape==(601,40,10) and loaded['policy_actual_state10'].shape==(601,10)
            assert loaded['policy_rgb_paths'].shape==(601,2) and loaded['valid_window'].all()
        records.append(dict(id=identifier,path=out.relative_to(root).as_posix(),bytes=out.stat().st_size,sha256=file_hash(out),
            window_count=601,roundtrip_max_error=error,labels='actual committed desired; not achieved or uncommitted plan',
            source_samples_sha256=file_hash(folder/'samples.json'),privileged_in_policy=False))
    receipt=dict(status='PASS_ACTUAL_NATIVE_CPU_WINDOWS',records=records,total_windows=4808,
        shape_per_episode=[601,40,10],input_RGB_shape=[2,256,256,3],all_exported_files_reopened=True,
        no_episode_boundary_crossing=True,no_training_statistics_fitted=True,official_GR00T_collator='NOT_RUN',model_loads=0,physics_integrations=0)
    write(root/'evidence/NATIVE_WINDOW_EXPORT.json',receipt)
    if a.upload:
        repo='leoyuer/pi0290-independent-dynamic-tool';prefix='PI0290/derived-native-codec-v1'
        token=(Path.home()/'.config/huggingface/token').read_text().strip();api=HfApi(token=token)
        info=api.repo_info(repo,repo_type='dataset');assert info.private and api.whoami()['name']=='leoyuer'
        assert not any(p.startswith(prefix+'/') for p in api.list_repo_files(repo,repo_type='dataset'))
        payload={r['id']+'.npz':root/r['path'] for r in records}
        payload['NATIVE_WINDOW_EXPORT.json']=root/'evidence/NATIVE_WINDOW_EXPORT.json'
        files=[dict(path=prefix+'/'+n,bytes=p.stat().st_size,sha256=file_hash(p)) for n,p in payload.items()]
        data_revision=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=info.sha,
            commit_message='PI0290 verified native CPU committed-action windows, no neural batch',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(p)) for n,p in payload.items()]).oid
        def verify(revision,rows):
            tree={p.path:p for p in api.list_repo_tree(repo,repo_type='dataset',revision=revision,path_in_repo=prefix,recursive=True) if hasattr(p,'size')}
            for row in rows:
                item=tree[row['path']];assert item.size==row['bytes']
                if item.lfs:
                    digest=item.lfs.sha256 if hasattr(item.lfs,'sha256') else item.lfs['sha256'];assert digest==row['sha256']
                else:
                    r=requests.get(hf_hub_url(repo,row['path'],repo_type='dataset',revision=revision),headers={'Authorization':'Bearer '+token},timeout=60)
                    r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==row['sha256']
        verify(data_revision,files)
        ready=root/'archive/CODEC_READY.json';write(ready,dict(status='READY',repo_id=repo,repo_type='dataset',visibility='private',prefix=prefix,data_revision=data_revision,files=files,
            raw_data_revision='0fd9247ba5426eb54ebf152912432302b193b479',derived_only=True,official_GR00T_collator='NOT_RUN'))
        final=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=data_revision,commit_message='Native codec READY last after all SHA checks',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(ready))]).oid
        files.append(dict(path=prefix+'/READY.json',bytes=ready.stat().st_size,sha256=file_hash(ready)));verify(final,files)
        write(root/'evidence/HF_CODEC_WINDOWS_READY.json',dict(status='HF_TRANSFER_READY',repo_id=repo,repo_type='dataset',visibility='private',
            complete_revision=final,data_revision=data_revision,prefix=prefix,files=files,READY_last=True,
            raw_data_revision='0fd9247ba5426eb54ebf152912432302b193b479',raw_files_not_overwritten=True,Hopper_raw_transfer='DEFERRED_NEW_GO'))
    print(json.dumps({'status':receipt['status'],'windows':4808,'HF_revision':final if a.upload else None}))
finally:budget.finish_cpu('D4_NATIVE_COMMITTED_WINDOW_EXPORT_AND_READBACK')
