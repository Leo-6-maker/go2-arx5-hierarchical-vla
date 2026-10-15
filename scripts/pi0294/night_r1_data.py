"""Variable-duration full-body export. Failed/pilot data never become TRAIN.

This module consumes actual native records; it does not step, render, run a
Teacher, infer a joint action, or convert old fixed-base records into EEF13.
"""
import json
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import read,write
from scripts.pi0292.train_adapter import sha
from go2arx5_vla.task_data.codec import pose10

DOMAIN='FULL_GO2_ARX5_TASK_SIM_EEF13_STATE19_V1'


def counts(duration):
    if type(duration) is not int or duration not in (32,64,96,128):raise ValueError('FROZEN_DURATION_DOMAIN_REQUIRED')
    return 20*duration,20*duration+1,20*duration-39


def observed_state19(tcp,opening,mount,velocity,reset_mount):
    from .base_intent import mount_pose,rotation
    mount,reset_mount=mount_pose(mount),mount_pose(reset_mount)
    r=mount[:3,:3];yaw=np.arctan2(r[1,0],r[0,0]);yaw0=np.arctan2(reset_mount[1,0],reset_mount[0,0])
    xy=(rotation(-yaw0)@(mount[:3,3]-reset_mount[:3,3]))[:2]
    dyaw=np.arctan2(np.sin(yaw-yaw0),np.cos(yaw-yaw0))
    v=np.asarray(velocity,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all() or not 0<=opening<=1:raise ValueError('ACTUAL_MEASURED_STATE_REQUIRED')
    result=np.r_[pose10(tcp,opening),xy,dyaw,v,mount[2,3],np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))]
    if result.shape!=(19,) or not np.isfinite(result).all():raise ValueError('ACTUAL_STATE19_REQUIRED')
    return result


def admitted(folder):
    folder=Path(folder);card=read(folder/'CARD.json');audit=read(folder/'INDEPENDENT_CPU_AUDIT.json');result=read(folder/'RESULT.json')
    counts(card['duration_s'])
    if (card.get('fixture') is True or audit.get('fixture') is True or result.get('fixture') is True
        or card.get('split') not in ('TRAIN','DEV') or card.get('role')!='PRODUCTION'
        or audit.get('status')!='COMPLETE_TASK_PASS' or audit.get('complete_declared_duration') is not True
        or audit.get('quality',{}).get('persistent_success') is not True
        or audit.get('high_rate_ranges',{}).get('all_raw_samples_within_original_ranges') is not True
        or audit.get('illegal_contact_records')!=0 or result.get('native_stops')
        or audit.get('runtime_source_sha')!=result.get('source_sha')
        or audit.get('actual_state_and_FK_replay') is not True
        or audit.get('source_gripper20Hz_labels_exact') is not True
        or audit.get('pause_history_unchanged') is not True):
        raise PermissionError('ACTUAL_SUCCESS_SAFETY_INTEGRITY_AND_PRODUCTION_SPLIT_REQUIRED')
    return card,audit


def committed_series(samples,records,duration):
    nc,no,nw=counts(duration)
    if len(samples)!=no:raise ValueError('COMPLETE_VARIABLE_DURATION_OBSERVATIONS_REQUIRED')
    intents={r['sequence']:r for r in records if r['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK'}
    acks={r['wire_sha256']:r for r in records if r['phase']=='INSTALL_ACK'}
    commands=[]
    for i in range(nc):
        observed=samples[i+1];seq=observed['active_intent_sequence'];prefix=observed['executed_prefix_index']
        if type(prefix) is not int or not 0<=prefix<8:raise ValueError('ACTUAL_EXECUTED_PREFIX_REQUIRED')
        plan=intents[seq];action=np.asarray(plan['action_EEF13'],dtype=float)
        if (action.shape!=(40,13) or not np.isfinite(action).all() or plan.get('base_valid') is not True
            or plan['index']+prefix!=i or plan['wire_sha256'] not in acks
            or acks[plan['wire_sha256']]['native_step']!=samples[plan['index']]['native_step']
            or plan['canonical_safety_gate']['decision']=='REJECT'
            or abs(observed['time_s']-(i+1)*.05)>1e-8
            or observed['native_step']-samples[i]['native_step']!=25):
            raise ValueError('INSTALLED_FRESH_FULLBODY_PREFIX_IDENTITY_REQUIRED')
        commands.append(action[prefix].copy())
    return np.asarray(commands)


def export_episode(folder,output):
    import mujoco
    from PIL import Image
    from .task_state import ActualState
    from .base_intent import measured_base
    folder,output=Path(folder).resolve(),Path(output);card,audit=admitted(folder)
    if output.exists():raise FileExistsError('FIRST_EXPORT_PRESERVED')
    samples=[json.loads(l) for l in (folder/'samples.jsonl').read_text().splitlines()]
    records=[json.loads(l) for l in (folder/'committed_intents.jsonl').read_text().splitlines()]
    command=committed_series(samples,records,card['duration_s']);nc,no,nw=counts(card['duration_s'])
    model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'));engine=ActualState(model)
    task_start=read(folder/'TASK_START.json')
    imu=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'imu');states=[];rgb=[];rgb_sha=[];reset=None
    raw_members={};audit_rgb=read(folder/'RGB_HASHES.json')
    for i,s in enumerate(samples):
        native_time=task_start['time_s']+s['time_s']
        if s['native_step']!=task_start['native_step']+i*25:raise ValueError('NATIVE_TASK_CLOCK_IDENTITY')
        packet={k:np.asarray(s[k]) for k in ('qpos','qvel','ctrl')};packet.update(time_s=native_time,contacts=[],contact_time_s=native_time-.002,step=s['native_step'])
        # Replay only the saved state in a read-only FK clone, never integrate.
        state=engine.update(packet);d=engine.data;imu_pose=np.eye(4);imu_pose[:3,:3]=d.site_xmat[imu].reshape(3,3);imu_pose[:3,3]=d.site_xpos[imu]
        velocity=np.zeros(6);mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_SITE,imu,velocity,0)
        mount,measured=measured_base(imu_pose,imu_pose[:3,:3].T@velocity[3:],imu_pose[:3,:3].T@velocity[:3],[.07557,0,.05768])
        if reset is None:reset=mount.copy()
        if not np.allclose(state['tcp'],s['tcp'],atol=1e-12,rtol=0) or not np.allclose(mount,s['actual_BASE_mount'],atol=1e-12,rtol=0):raise ValueError('ACTUAL_FK_IDENTITY_CHANGED')
        states.append(observed_state19(state['tcp'],state['gripper_fraction'],mount,measured,reset))
        paths=[];hashes=[];cal=read(folder/'images'/f'{i:06d}-calibration.json')
        for camera in ('body_rgb','wrist_rgb'):
            rel=f'images/{i:06d}-{camera}.png';path=folder/rel;digest=sha(path)
            if digest!=audit_rgb[path.name] or abs(cal[camera]['timestamp_s']-packet['time_s'])>1e-8:raise ValueError('ACTUAL_RGB_SOURCE_BINDING')
            with Image.open(path) as im:
                if im.mode!='RGB' or im.size!=(256,256):raise ValueError('ACTUAL_RGB_FORMAT')
            paths.append(rel);hashes.append(digest);raw_members[rel]=digest
        raw_members[f'images/{i:06d}-calibration.json']=sha(folder/'images'/f'{i:06d}-calibration.json')
        rgb.append(paths);rgb_sha.append(hashes)
    for name in ('CARD.json','RESULT.json','INDEPENDENT_CPU_AUDIT.json','ALL_JOINT_RANGE_AUDIT.json','RGB_HASHES.json',
                 'samples.jsonl','committed_intents.jsonl','planner.jsonl','pause_intervals.jsonl','state.bin.journal','task-scene.private.mjb','TASK_START.json'):
        raw_members[name]=sha(folder/name)
    output.mkdir(parents=True);numeric=output/'episode.npz'
    np.savez_compressed(numeric,actual_state19=np.asarray(states),committed_absolute13=command,
        rgb_paths=np.asarray(rgb),rgb_sha256=np.asarray(rgb_sha),valid_window=np.ones(nw,dtype=bool),
        instruction=card['instruction'],label_representation='ABSOLUTE_COMMITTED_EEF13_BASEINTENT_V1')
    receipt=dict(schema='TFNR1_FULLBODY_EXPORT_V1',domain=DOMAIN,fixture=False,raw_root=str(folder),
        id=card['id'],split=card['split'],group_id=card['group_id'],task=card['task'],duration_s=card['duration_s'],
        commands=nc,observations=no,windows=nw,numeric_sha256=sha(numeric),raw_members=raw_members,
        actual_model_loaded=False,physics_steps=0,render_calls=0,official_consumer='NOT_RUN',statistics='NOT_FITTED')
    write(output/'EXPORT_RECEIPT.json',receipt);return receipt


def freeze_bank(bank,export_names,collection_receipt):
    """Fit once after the selected collection is terminal; never refit on DEV."""
    from scripts.pi0292.eval_contract import matrix10
    from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows
    bank=Path(bank).resolve();closed=read(collection_receipt)
    if (closed.get('status')!='COMPLETE_FROZEN_COLLECTION' or closed.get('all_first_attempts_terminal') is not True
        or not export_names or (bank/'statistics.json').exists() or (bank/'BANK_SEAL.json').exists()):
        raise ValueError('NONEMPTY_TERMINAL_COLLECTION_AND_ONE_FIT_REQUIRED')
    rows=[];state_values=[];action_values=[];relative_values=[];groups={'TRAIN':set(),'DEV':set()}
    for name in export_names:
        p=(bank/name).resolve()
        if not p.is_relative_to(bank):raise ValueError('EXPORT_PATH_ESCAPE')
        r=read(p/'EXPORT_RECEIPT.json');card,_=admitted(r['raw_root'])
        if r['fixture'] is not False or r['domain']!=DOMAIN or sha(p/'episode.npz')!=r['numeric_sha256']:raise ValueError('ACTUAL_EXPORT_IDENTITY')
        if r['split']!=card['split'] or r['group_id']!=card['group_id'] or r['id'] not in closed['admitted_ids']:raise ValueError('FROZEN_SCENE_SPLIT')
        if any(row['id']==r['id'] for row in rows):raise ValueError('DUPLICATE_EPISODE')
        groups[r['split']].add(r['group_id']);rows.append(dict(id=r['id'],split=r['split'],task=r['task'],export=name,windows=r['windows']))
        if r['split']!='TRAIN':continue
        with np.load(p/'episode.npz',allow_pickle=False) as f:s=f['actual_state19'];a=f['committed_absolute13']
        _,no,nw=counts(r['duration_s']);future=np.arange(nw)[:,None]+np.arange(40);target=a[future]
        if s.shape!=(no,19) or not np.isfinite(s).all() or not np.isfinite(target).all():raise ValueError('FINITE_FULLBODY_ARRAYS')
        transforms=np.linalg.inv(matrix10(s[:nw,:10]))[:,None]@matrix10(target[...,:10])
        relative=np.concatenate([transforms[...,:3,3],rotation_matrix_to_rot6d_rows(transforms[...,:3,:3])],axis=-1)
        state_values.append(s[:nw]);action_values.append(target);relative_values.append(relative)
    if not state_values or groups['TRAIN']&groups['DEV'] or {r['id'] for r in rows}!=set(closed['admitted_ids']):raise ValueError('TRAIN_REQUIRED_NO_SCENE_GROUP_LEAKAGE_NO_OMITTED_ADMITTED')
    s,a,r=map(np.concatenate,(state_values,action_values,relative_values))
    def statistics(v,axis):
        return {key:np.asarray(value).tolist() for key,value in dict(min=v.min(axis=axis),max=v.max(axis=axis),mean=v.mean(axis=axis),std=v.std(axis=axis),q01=np.quantile(v,.01,axis=axis),q99=np.quantile(v,.99,axis=axis)).items()}
    fitted={'new_embodiment':dict(state=dict(eef_9d=statistics(s[:,:9],0),gripper_position=statistics(s[:,9:10],0),base_observed9=statistics(s[:,10:19],0)),
        action=dict(eef_9d=statistics(a[...,:9],(0,1)),gripper_position=statistics(a[...,9:10],(0,1)),base_soft_velocity3=statistics(a[...,10:13],(0,1))),relative_action=dict(eef_9d=statistics(r,(0,1))))}
    write(bank/'statistics.json',fitted)
    write(bank/'STATISTICS_PROVENANCE.json',dict(TRAIN_ids=[r['id'] for r in rows if r['split']=='TRAIN'],DEV_fit=False,fit_count=1,
        rule='TRAIN_ONLY_ALL_FULL_H40_EQUAL_WINDOW_TIMESTEP_MINMAX',state_anchors=len(s),action_window_timesteps=int(np.prod(a.shape[:2]))))
    manifest=dict(domain=DOMAIN,fixture=False,statistics_fit_count=1,statistics_sha256=sha(bank/'statistics.json'),episodes=rows,
        collection_sha256=sha(collection_receipt),official_consumer='NOT_RUN',selected='NONE',deployment='NONE')
    write(bank/'DATASET_MANIFEST.json',manifest)
    members={p.relative_to(bank).as_posix():sha(p) for p in bank.rglob('*') if p.is_file()}
    write(bank/'BANK_SEAL.json',dict(manifest_sha256=sha(bank/'DATASET_MANIFEST.json'),file_sha256=members,fixture=False))
    return manifest
