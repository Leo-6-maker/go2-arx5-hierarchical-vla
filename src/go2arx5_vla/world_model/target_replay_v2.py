"""Versioned variable-length target replay; explicit episode-contained windows."""
from pathlib import Path
import json,hashlib
import numpy as np
from .storage import digest,write_json
from .target_replay import validate_windows,validate_partitions
from .target_manifest import model_fingerprint
SCHEMA='WM02_TARGET_NATIVE_REPLAY_V2'
MANIFEST_SCHEMA='WM02_NIGHT1_TARGET_REPLAY_MANIFEST_V2'


def window_indices(lengths):
    if any(type(n)is not int or n<24 or n>160 for n in lengths):raise ValueError('BOUNDED_COMPLETE_EPISODE_LENGTH_REQUIRED')
    return np.asarray([(i,k)for i,n in enumerate(lengths)for k in range(n-23)],dtype=np.int64)


def select_samples(data,meta,split,*,batch=None):
    selected={i for i,e in enumerate(meta['episodes'])if e['split']==split and(batch is None or e.get('batch')==batch)}
    index=data.get('episode_index',np.arange(len(meta['episodes'])))
    return np.flatnonzero(np.isin(index,list(selected)))


def materialize(raw,meta):
    states=0;atoms=0
    for e in meta['episodes']:
        n=e['task_atoms']
        if e['state_start']!=states or e['atom_start']!=atoms:raise ValueError('EXACT_NONOVERLAPPING_EPISODE_STORAGE_REQUIRED')
        sl=slice(states,states+n+1);steps=raw['flat_native_step'][sl];times=raw['flat_sim_time'][sl]
        if (len(steps)!=n+1 or not np.all(np.diff(steps)==25)
                or not np.allclose(times,steps*.002,atol=1e-8,rtol=0)
                or not np.allclose(raw['flat_contact_time'][sl],times-.002,atol=1e-8,rtol=0)):
            raise ValueError('NO_RESET_PARTIAL_OR_NATIVE_CLOCK_GAP_IN_EPISODE')
        states+=n+1;atoms+=n
    if len(raw['flat_state'])!=states or len(raw['flat_source_age'])!=atoms or not all(np.isfinite(a).all()for a in raw.values()):raise ValueError('COMPLETE_FINITE_EPISODE_STORAGE_REQUIRED')
    index=raw['window_index'];expected=window_indices([e['task_atoms']for e in meta['episodes']])
    if not np.array_equal(index,expected):raise ValueError('EXACT_EPISODE_CONTAINED_WINDOW_INDEX_REQUIRED')
    keys=('state','native_step','sim_time','contact_time','wall_monotonic','tcp','object','events')
    result={k:[]for k in keys};result.update(goal=[],source_age=[],window_start_s=[],window_start_atom=[],episode_index=[])
    for i,k in index:
        e=meta['episodes'][int(i)];start=e['state_start']+int(k);atom=e['atom_start']+int(k)
        for name in keys:result[name].append(raw['flat_'+name][start:start+25])
        result['source_age'].append(raw['flat_source_age'][atom:atom+24]);result['goal'].append(raw['episode_goal'][i]);result['episode_index'].append(i);result['window_start_atom'].append(k)
        result['window_start_s'].append(raw['flat_sim_time'][start]-raw['flat_sim_time'][e['state_start']])
    result={k:np.asarray(v)for k,v in result.items()}
    if result['state'].shape[1:]!=(25,meta['state_dimension']):raise ValueError('ACTUAL_WINDOW_STATE_SHAPE_REQUIRED')
    validate_partitions(meta['episodes']);return result,meta


def build(path,output):
    import mujoco
    from .native_replay import packets,forbid_integration
    from .init_trace_reader import read_trace
    from scripts.pi0294.task_state import ActualState
    manifest=json.loads(Path(path).read_text());entries=manifest['episodes'];validate_partitions(entries)
    if manifest.get('schema')!=MANIFEST_SCHEMA or not entries:raise ValueError('REGISTERED_VARIABLE_EPISODE_MANIFEST_REQUIRED')
    allow_path=Path(manifest['scene_allowlist']);assets=json.loads(allow_path.read_text())
    if digest(allow_path)!=manifest['scene_allowlist_sha256']:raise ValueError('SCENE_ALLOWLIST_CHANGED')
    scenes={s['model_sha256']:s for s in assets['scenes']};raw={k:[]for k in('state','source_age','native_step','sim_time','contact_time','wall_monotonic','tcp','object','events')};identities=[];goals=[];models={};state_start=0;atom_start=0;fields=None;joints=None
    executor=None
    for entry in entries:
        root=Path(entry['run']);run=root/'native-01';spec=json.loads((root/'EXECUTION_SPEC.json').read_text());result=json.loads((run/'RESULT.json').read_text());diag=json.loads((run/'DYNAMIC_DIAGNOSTIC.json').read_text());n=entry['task_atoms'];asset=digest(spec['model_path'])
        current_executor={key:digest(spec[key])for key in('native_binary','outer_library','observer_library','trace_library')}
        if current_executor['native_binary']!=result['native_binary_sha256']or current_executor['observer_library']!=result['observer_sha256']:raise ValueError('ACTUAL_EXECUTOR_RESULT_IDENTITY_CHANGED')
        if executor is not None and current_executor!=executor:raise ValueError('DIFFERENT_NATIVE_EXECUTORS_REQUIRE_SEPARATE_REPLAYS')
        executor=current_executor
        if (result['status']!='NATIVE_STAGE_PASSED'or not result['shutdown_clean']or result['physics_stops']or result['original_joint_or_warning_samples']
                or n not in(24,48,96,160)or diag['task_native_ticks']!=25*n or diag['partial_task_ticks']!=0
                or any(p['exit_code']!=0 for p in result['owned_process_exits'])or asset not in scenes):raise ValueError('WHOLE_QUALIFIED_EPISODE_REQUIRED_FAILED_PREFIX_EXCLUDED')
        if spec.get('parallel_batch_spec'):
            batch=Path(spec['parallel_batch_spec']).parent/'batch-01/RESULT.json'
            if json.loads(batch.read_text())['status']!='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED':raise ValueError('INVALIDATED_PEER_BATCH_EXCLUDED')
        scene=scenes[asset]
        if scene['parent']!=entry['parent']or scene['split']!=entry['split']:raise ValueError('PREDECLARED_SCENE_PARENT_SPLIT_REQUIRED')
        if asset not in models:
            model=mujoco.MjModel.from_binary_path(spec['model_path']);fp,_=model_fingerprint(model,assets.get('fingerprint_version','QPOS0_XY_V1'))
            if fp!=assets['robot_fingerprint']:raise ValueError('ROBOT_OR_FROZEN_MODEL_STRUCTURE_CHANGED')
            models[asset]=(model,ActualState(model))
        model,actual=models[asset];start=diag['actual_collection_start_step'];validate_windows(diag['windows'],start,n)
        by_step={p['step']:p for p in packets(run/'state.bin.journal')};selected=[by_step[start+25*i]for i in range(n+1)]
        if any(abs(p['time_s']-p['step']*.002)>1e-8 for p in selected):raise ValueError('ACTUAL_NATIVE_CLOCK_REQUIRED')
        trace=read_trace(run/'init.trace');wall={round(r['sim_time_s']/.002):r['monotonic_ns']/1e9 for r in trace if r['kind']==12}
        commands=[json.loads(l)for l in(run/'COMMANDS.private.jsonl').read_text().splitlines()];proposed=[p for p in commands if p['phase']=='proposed'];first=proposed[0];goal=np.r_[first['action_EEF10'][-1],first['measured_BASE_pose_hold']]
        if first['native_step']!=start or any(not np.allclose(np.r_[p['action_EEF10'][-1],p['measured_BASE_pose_hold']],goal,atol=1e-8,rtol=0)for p in proposed):raise ValueError('PREDECLARED_FIXED_GOAL_REQUIRED_NO_FUTURE_RELABEL')
        with forbid_integration(mujoco)as calls:
            for row in selected:
                value=actual.update(row);tp,op=value['tcp'],value['object_pose']
                raw['state'].append(np.r_[row['qpos'],row['qvel'],row['ctrl'],tp[:3,3],tp[:3,0],tp[:3,1],op[:3,3],op[:3,0],op[:3,1],value['finger_q']]);raw['tcp'].append(tp);raw['object'].append(op)
                raw['events'].append([value[k]for k in('left_object_contact','right_object_contact','native_grasp','robot_object_contact','opposing_contact','gripper_state_valid','safety_valid')])
                raw['native_step'].append(row['step']);raw['sim_time'].append(row['time_s']);raw['contact_time'].append(row['contact_time_s']);raw['wall_monotonic'].append(wall[row['step']])
        if calls:raise RuntimeError('NO_REPLAY_BUILD_INTEGRATION')
        raw['source_age'].extend(w['clock']['native_start_s']-w['clock']['source_mpc_s']for w in diag['windows']);goals.append(goal)
        loaded={p.name:digest(p)for p in run.glob('LOADED_EXECUTOR_*.private.json')}
        if not loaded:raise ValueError('ACTUAL_LOADED_EXECUTOR_IDENTITIES_REQUIRED')
        identities.append(dict(episode=root.name,parent=entry['parent'],split=entry['split'],batch=entry.get('batch'),task_atoms=n,state_start=state_start,atom_start=atom_start,window_count=n-23,scene_sha256=asset,scene_id=scene['scene_id'],source_sha=spec['source_sha'],result_sha256=digest(run/'RESULT.json'),state_journal_sha256=digest(run/'state.bin.journal'),command_sha256=digest(run/'COMMANDS.private.jsonl'),loaded_identity_sha256=loaded,terminal=False,collector_truncated=True,partial_ticks=0,full_task_success=False,truncation_reason='PREDECLARED_FINITE_COLLECTION_END'))
        state_start+=n+1;atom_start+=n
        current_fields=[('qpos',model.nq),('qvel',model.nv),('ctrl',model.nu),('tcp_position_m',3),('tcp_rotation6d',6),('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)]
        if fields is not None and fields!=current_fields:raise ValueError('STATE_LAYOUT_CHANGED')
        fields=current_fields
        joints=[dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i),type=int(model.jnt_type[i]),qpos_address=int(model.jnt_qposadr[i]),qvel_address=int(model.jnt_dofadr[i]))for i in range(model.njnt)]
    meta=dict(schema=SCHEMA,state_fields=[dict(name=k,size=n)for k,n in fields],state_dimension=sum(n for _,n in fields),original_model_sha256=assets['original_model_sha256'],robot_fingerprint=assets['robot_fingerprint'],scene_assets_sha256=sorted(models),joints=joints,actuator_gear=model.actuator_gear.tolist(),actuator_gain=model.actuator_gainprm.tolist(),episodes=identities,task_atoms=atom_start,windows=sum(e['window_count']for e in identities),history_burn_atoms=8,predict_atoms=16,action='PREDECLARED_RAMP8_FIXED_WORLD_GOAL_EEF10_BASE_POSE7_V1',window_clock='ACTUAL_EPISODE_RELATIVE_WINDOW_START_S',renewal_atoms=7,source_age_future='CURRENT_PREFIX_ONLY_THEN_MISSING_NO_FUTURE_ACTUAL',controller_hidden_state_exported=False,controller_selected_policy_id='DERIVED_NOT_DIRECT_EXPORT',contact_collection='NOT_QUALIFIED',events=['left_contact','right_contact','grasp','robot_object_contact','opposing','gripper_valid','safety_valid'],held_out_parent_generalization=any(e['split']=='DEV'for e in identities),overlapping_windows_independent=False)
    meta['executor_sha256']=executor
    arrays={'flat_'+k:np.asarray(a)for k,a in raw.items()};arrays['episode_goal']=np.asarray(goals);arrays['window_index']=window_indices([e['task_atoms']for e in identities]);arrays['metadata_utf8']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8)
    if not all(np.isfinite(a).all()for a in arrays.values()):raise ValueError('NONFINITE_ACTUAL_REPLAY')
    materialize({k:a for k,a in arrays.items()if k!='metadata_utf8'},meta)
    out=Path(output);out.mkdir(exist_ok=False)
    with(out/'REPLAY.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    result=dict(status='ACTUAL_TARGET_SEQUENTIAL_REPLAY_BUILT',schema=SCHEMA,replay_sha256=digest(out/'REPLAY.npz'),manifest_sha256=digest(path),episodes=len(entries),task_atoms=atom_start,windows=meta['windows'],state_dimension=meta['state_dimension'],robot_fingerprint=assets['robot_fingerprint'],scene_assets_sha256=meta['scene_assets_sha256'],physics_integrations=0,WM_updates=0,Actor_updates=0)
    write_json(out/'SCHEMA.private.json',meta);write_json(out/'RESULT.json',result);return result
