"""Seal successful live episodes, then fit TRAIN-only latent statistics.

No simulator, policy, optimization or reconstruction of old RGB is performed.
The original current-query H40 and actual consumption masks remain distinct.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from scripts.pi0292.durable import write
from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, pad32, roundtrip_error
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec, VERSION, EPSILON

def read(path): return json.loads(Path(path).read_bytes())


def export(collection, destination):
    from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal
    from scripts.wm02.mfree_collection_v1 import capture_check
    collection=Path(collection); destination=Path(destination)
    snapshot=read(collection)
    if snapshot['schema']!='MFREE_NESTED_CAPTURE_SNAPSHOT_V1' or snapshot['state']!='SEALED':
        raise ValueError('ATOMIC_SEALED_CLOSED_EPISODE_SNAPSHOT_REQUIRED')
    success=[];failures=snapshot['diagnostic_failures'];requests=snapshot['physical_requests'];integrations=snapshot['native_integrations']
    for descriptor in snapshot['episodes']:
        pin=descriptor['result']
        if sha(pin['path'])!=pin['sha256']:raise ValueError('IMMUTABLE_SNAPSHOT_NATIVE_RESULT')
        if descriptor.get('spec')and sha(descriptor['spec']['path'])!=descriptor['spec']['sha256']:raise ValueError('IMMUTABLE_CAPTURE_EXECUTION_SPEC')
        if read(Path(pin['path']).parent/'control/LEASE.json')['status']!='CLOSED':raise ValueError('ONLY_CLOSED_CAPTURE_EPISODES')
        result=read(pin['path'])
        from go2arx5_vla.world_model.mobile_grasp_budget_v3 import qualification_record
        if not qualification_record(result)['qualified']:raise ValueError('QUALIFIED_COMPLETE_NATIVE_EPISODES_ONLY')
        success.append(dict(root=str(Path(pin['path']).parent),qualified=True))
    population=dict(train_attempts=[read(Path(r['root'])/'EXECUTION_SPEC.json')['collection_contract']for r in success])
    if len(success)<4: raise ValueError('AT_LEAST_FOUR_COMPLETE_LIVE_EPISODES_FOR_THIS_CANDIDATE')
    if len(success)>100 or requests>200: raise ValueError('ORIGINAL_PRODUCTION_CAP')
    destination.mkdir(exist_ok=False); episodes=[]; states=[]; actions=[]; work=[]
    known={r['episode_id']:r for r in population['train_attempts']}
    for result in success:
        root=Path(result['root']); spec=read(root/'EXECUTION_SPEC.json'); native=read(root/'RESULT.private.json')
        capture_check(native); manifest=read(native['detail']['capture']['path']); rows=read(manifest['labels']['path'])
        identity=spec['collection_contract']; eid=identity['episode_id']
        if known[eid]!=identity or identity['split']!='TRAIN': raise ValueError('PREDECLARED_TRAIN_EPISODE_ONLY')
        frame=np.asarray(rows[0]['episode_fixed_T_policy_from_mount']); codec=GripLogitCodec(frame)
        transform=np.asarray(spec['policy_from_world']); count=len(rows)
        state10=np.asarray([r['state_eef10'] for r in rows])
        nominal=np.stack([pose10(transform @ matrix_from10(r['nominal_world_H40']),np.asarray(r['nominal_world_H40'])[:,9]) for r in rows])
        state7=[]; latent7=[]
        for state, target in zip(state10,nominal,strict=True):
            s,a=codec.encode(state,target); state7.append(s);latent7.append(a)
        state7=np.asarray(state7);latent7=np.asarray(latent7);states.append(state7);actions.append(latent7.reshape(-1,7))
        folder=destination/eid;folder.mkdir(); shards=[]
        for descriptor in manifest['frame_shards']:
            path=folder/Path(descriptor['path']).name;shutil.copyfile(descriptor['path'],path)
            if sha(path)!=descriptor['sha256']:raise ValueError('EXACT_LIVE_RGB_TRANSFER')
            with np.load(path,allow_pickle=False)as z:
                start=descriptor['first_query'];end=start+descriptor['count']
                if (not np.array_equal(z['native_step'],[r['native_step']for r in rows[start:end]])
                        or not np.array_equal(z['state_eef10'],state10[start:end])
                        or z['rgb'].shape!=(descriptor['count'],2,256,256,3) or z['rgb'].dtype!=np.uint8):
                    raise ValueError('REAL_RGB_STATE_QUERY_CLOCK_EXACT_ALIGNMENT')
            shards.append(dict(file=str(path.relative_to(destination)),sha256=sha(path),
                first_frame=descriptor['first_query'],frames=descriptor['count']))
        label=folder/'LABELS.npz'
        with label.open('xb')as handle:
            np.savez(handle,state7=state7,latent7=latent7,state_eef10=state10,nominal_policy_H40=nominal,
                native_step=np.asarray([r['native_step']for r in rows]),frame_index=np.arange(count),
                phase=np.asarray([r['phase']for r in rows]),supervision_mask=np.asarray([r['supervision_mask']for r in rows],bool),
                consumed_mask=np.asarray([r['consumed_mask']for r in rows],bool),
                actual_consumed_ticks=np.asarray([r['actual_elapsed_prefix_ticks']for r in rows]),
                installed_reference_H40=np.asarray([r['installed_reference_H40']for r in rows]))
        episode=dict(id=eid,split='TRAIN',task='PICK_HOLD',family=identity['family'],layout=identity['layout_id'],
            scene_lineage=identity['scene_lineage'],session=identity['session'],restart=identity['reset'],
            windows=count,frames=count,commands_confirmed=10*count,
            labels=dict(file=str(label.relative_to(destination)),sha256=sha(label)),frame_shards=shards,
            mount=dict(T_policy_from_mount=frame.tolist()),language=spec['language'],
            expert_stage_audit_only=[r['phase']for r in rows],grasp_goal_raise_m=spec['grasp_goal_raise_m'],
            native_result=dict(path=str(root/'RESULT.private.json'),sha256=sha(root/'RESULT.private.json')),
            original_live_capture=native['detail']['capture'])
        commands=[json.loads(line)for line in(root/'native-01/COMMANDS.private.jsonl').read_text().splitlines()]
        holds={r['native_step']:r['measured_BASE_pose_hold']for r in commands if r['phase']=='proposed'}
        episodes.append(episode);work.append((spec,rows,state10,nominal,state7,latent7,codec,holds,folder))
    values=dict(state=np.concatenate(states),actions=np.concatenate(actions));statistics={k:Quantiles.fit(v)for k,v in values.items()}
    stats=dict(schema='MFREE_NESTED_TRAIN_LATENT_STATISTICS_V1',codec_version=VERSION,epsilon=EPSILON,
        split='TRAIN',fit_input='UNNORMALIZED_RELATIVE_XYZ_SO3_AND_LOGIT_GRIP',
        state_gripper_transformed=False,fit_unique_windows=sum(e['windows']for e in episodes),
        fit_episodes=len(episodes),**{k:q.summary(values[k])for k,q in statistics.items()})
    write(destination/'STATISTICS.json',stats)
    maximum=dict(xyz_m=0.,so3_rad=0.,gripper=0.); checks=0;state_error=0.
    for spec,rows,state10,nominal,state7,latent7,codec,holds,folder in work:
        check_folder=folder/'FP32_COMPILER';check_folder.mkdir()
        for index,row in enumerate(rows):
            # Exactly the FP32 normalized learner/deployment representation.
            normalized=statistics['actions'].normalize(latent7[index]).astype(np.float32)
            state_decoded=statistics['state'].unnormalize(statistics['state'].normalize(state7[index]).astype(np.float32))
            state_error=max(state_error,float(np.max(np.abs(state_decoded-state7[index]))))
            if state_error>1e-6:raise ValueError('FP32_MEASURED_STATE_ROUNDTRIP')
            physical=codec.decode(state10[index],pad32(statistics['actions'].unnormalize(normalized)))
            expected=nominal[index].copy();expected[:,9]=np.clip(expected[:,9],EPSILON,1-EPSILON)
            error=roundtrip_error(expected,physical)
            if error['xyz_m']>1e-6 or error['so3_rad']>1e-5 or error['gripper']>1e-6:
                raise ValueError('FP32_LATENT_STATS_CODEC_ROUNDTRIP:'+str(error))
            for key in maximum:maximum[key]=max(maximum[key],error[key])
            # Invoke the unchanged student compiler, including its original
            # finite-shape/hash input boundary. Export is not a hot control path.
            proposal=check_folder/f'{index:03d}.npz'
            with proposal.open('xb')as handle:
                np.savez(handle,physical_H40=physical,normalized=pad32(normalized)[None])
            compile_proposal(proposal,sha(proposal),source_from_target=spec['policy_from_world'],query_state10=state10[index],
                actual_TCP=np.asarray(row['actual_state']['tcp']),
                consumed_reference_TCP=_reference(row),base_hold=holds[row['native_step']])
            checks+=1
    manifest=dict(schema='MFREE_NESTED_LIVE_TRAIN_DATASET_V1',state='committed',
        dataset_id=snapshot['dataset_id'],requested_dataset_id=snapshot['intended_stage'],snapshot_descriptor_sha256=sha(collection),
        episodes=episodes,windows=sum(e['windows']for e in episodes),codec=VERSION,split='TRAIN',
        statistics=dict(file='STATISTICS.json',sha256=sha(destination/'STATISTICS.json')),
        phase_input_to_student=False,teacher_memory_input=False,BASE_head=False,
        plan_supervision_suffix=True,unconsumed_suffix_is_executed=False,
        actual_source_requests=requests,actual_source_native_integrations=integrations,
        failed_episodes_diagnostic_only=failures,
        population_sha256=sha(collection),old_RP8_passed=False)
    write(destination/'MANIFEST.private.json',manifest)
    audit=dict(schema='MFREE_TRAIN_FP32_DEPLOYMENT_AUDIT_V1',passes=True,
        manifest_sha256=sha(destination/'MANIFEST.private.json'),statistics_sha256=sha(destination/'STATISTICS.json'),
        episodes=len(episodes),windows=checks,maximum_FP32_roundtrip_error=maximum,
        sigmoid_count=1,state_gripper_transformed=False,maximum_FP32_state7_error=state_error,
        all_original_student_compiler_checks_pass=True,
        new_optimizer_updates=0,new_training_visits=0,new_physical_requests=0,new_integrations=0,
        new_statistics_prospective_native_requests=0,old_RP8_passed=False)
    write(destination/'AUDIT.private.json',audit);return audit


def _reference(row):
    from go2arx5_vla.trajectory.so3 import quat_to_matrix
    p=np.asarray(row['recent_reference']['accepted_pose7']);reference=np.eye(4)
    reference[:3,3]=p[:3];reference[:3,:3]=quat_to_matrix(p[3:]);return reference


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('collection');parser.add_argument('destination')
    args=parser.parse_args();print(json.dumps(export(args.collection,args.destination)))
