"""Actual native sequential replay. Each episode remains an indivisible sequence."""
from pathlib import Path
import json
import numpy as np
from .storage import digest,write_json

SCHEMA='WM02_TARGET_NATIVE_STATE_V1'


def validate_windows(windows,start,steps):
    if len(windows)!=steps:raise ValueError('COMPLETE_TASK_WINDOWS_REQUIRED')
    for i,row in enumerate(windows):
        if (row['native_start_step']!=start+25*i or row['native_end_step']!=start+25*(i+1)
                or row['actual_ticks']!=25 or row['mode']!='WBC'
                or not row['solver_policies_with_exact_installed_target']):
            raise ValueError('NO_CROSS_RESET_PARTIAL_MODE_OR_UNCONFIRMED_REFERENCE_WINDOWS')


def validate_partitions(entries):
    parents={}
    for row in entries:
        if row['split']not in('TRAIN','DEV'):raise ValueError('EXPLICIT_TRAIN_DEV_REQUIRED')
        if row['parent']in parents and parents[row['parent']]!=row['split']:
            raise ValueError('SAME_PARENT_SPLIT_LEAKAGE')
        parents[row['parent']]=row['split']


def build(manifest_path,output):
    import mujoco
    from .native_replay import packets,forbid_integration
    from .init_trace_reader import read_trace
    from scripts.pi0294.task_state import ActualState
    manifest=json.loads(Path(manifest_path).read_text());entries=manifest['episodes'];validate_partitions(entries)
    if not entries:raise ValueError('ACTUAL_EPISODES_REQUIRED')
    out=Path(output);out.mkdir(exist_ok=False);arrays={k:[]for k in('state','goal','source_age','native_step','sim_time','wall_monotonic','tcp','object','events')};identities=[];model=None
    for entry in entries:
        root=Path(entry['run']);run=root/'native-01';spec=json.loads((root/'EXECUTION_SPEC.json').read_text());result=json.loads((run/'RESULT.json').read_text());diag=json.loads((run/'DYNAMIC_DIAGNOSTIC.json').read_text())
        if (not spec['phase'].startswith('collect-target-')or result['status']!='NATIVE_STAGE_PASSED'
                or not result['shutdown_clean']or result['physics_stops']or diag['task_native_ticks']!=600
                or not all(p['exit_code']==0 for p in result['owned_process_exits'])):
            raise ValueError('QUALIFIED_ACTUAL_FINITE_COLLECTION_REQUIRED')
        if model is None:
            model=mujoco.MjModel.from_binary_path(spec['model_path']);asset=digest(spec['model_path']);actual=ActualState(model)
        if digest(spec['model_path'])!=asset:raise ValueError('MODEL_IDENTITY_CHANGED')
        start=diag['actual_collection_start_step'];validate_windows(diag['windows'],start,24)
        rows=packets(run/'state.bin.journal');by_step={p['step']:p for p in rows};selected=[by_step[start+25*i]for i in range(25)]
        if any(abs(p['time_s']-p['step']*.002)>1e-8 for p in selected):raise ValueError('NATIVE_CLOCK_MISMATCH')
        trace=read_trace(run/'init.trace');wall={round(r['sim_time_s']/.002):r['monotonic_ns']/1e9 for r in trace if r['kind']==12}
        commands=[json.loads(l)for l in(run/'COMMANDS.private.jsonl').read_text().splitlines()];proposed=[p for p in commands if p['phase']=='proposed'];first=proposed[0]
        goal=np.r_[first['action_EEF10'][-1],first['measured_BASE_pose_hold']]
        if first['native_step']!=start or any(not np.allclose(np.r_[p['action_EEF10'][-1],p['measured_BASE_pose_hold']],goal,atol=1e-8,rtol=0)for p in proposed):raise ValueError('FUTURE_ACTION_NOT_PREDECLARED_FIXED_GOAL')
        states=[];tcp=[];obj=[];events=[]
        with forbid_integration(mujoco)as integration_calls:
            for row in selected:
                if (len(row['qpos']),len(row['qvel']),len(row['ctrl']))!=(model.nq,model.nv,model.nu):raise ValueError('ACTUAL_MODEL_DIMENSIONS_REQUIRED')
                value=actual.update(row);tp,op=value['tcp'],value['object_pose']
                states.append(np.r_[row['qpos'],row['qvel'],row['ctrl'],tp[:3,3],tp[:3,0],tp[:3,1],op[:3,3],op[:3,0],op[:3,1],value['finger_q']]);tcp.append(tp);obj.append(op)
                events.append([value[k]for k in('left_object_contact','right_object_contact','native_grasp','robot_object_contact','opposing_contact','gripper_state_valid','safety_valid')])
        if integration_calls:raise RuntimeError('REPLAY_BUILD_MUST_NOT_INTEGRATE')
        arrays['state'].append(states);arrays['goal'].append(goal);arrays['source_age'].append([w['clock']['native_start_s']-w['clock']['source_mpc_s']for w in diag['windows']]);arrays['native_step'].append([r['step']for r in selected]);arrays['sim_time'].append([r['time_s']for r in selected]);arrays['wall_monotonic'].append([wall[r['step']]for r in selected]);arrays['tcp'].append(tcp);arrays['object'].append(obj);arrays['events'].append(events)
        pins={p.name:digest(p)for p in(run/'RESULT.json',run/'state.bin.journal',run/'COMMANDS.private.jsonl',run/'ROS_CONSUMPTION.private.jsonl',root/'EXECUTION_SPEC.json')}
        loaded={p.name:digest(p)for p in run.glob('LOADED_EXECUTOR_*.private.json')}
        if not loaded:raise ValueError('ACTUAL_LOADED_EXECUTOR_IDENTITY_REQUIRED')
        identities.append(dict(episode=root.name,parent=entry['parent'],split=entry['split'],source_sha=spec['source_sha'],files_sha256=pins,loaded_identity_sha256=loaded,executor_frozen_files_sha256=spec['frozen_files'],derived_codegen_seed_sha256=spec['codegen_seed_files_sha256']))
    joints=[dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i),type=int(model.jnt_type[i]),qpos_address=int(model.jnt_qposadr[i]),qvel_address=int(model.jnt_dofadr[i]))for i in range(model.njnt)]
    fields=[('qpos',model.nq),('qvel',model.nv),('ctrl',model.nu),('tcp_position_m',3),('tcp_rotation6d',6),('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)]
    meta=dict(schema=SCHEMA,state_fields=[dict(name=name,size=n)for name,n in fields],state_dimension=sum(n for _,n in fields),model_sha256=asset,joints=joints,actuator_gear=model.actuator_gear.tolist(),actuator_gain=model.actuator_gainprm.tolist(),episodes=identities,task_atoms=24*len(entries),history_burn_atoms=8,predict_atoms=16,action='PREDECLARED_FIXED_WORLD_EEF10_GOAL_AND_MEASURED_BASE_POSE7_NO_BASE_VELOCITY_LABELS',future_reference='Fixed goal already proposed before task t0; renewals preserve that suffix; no actual future achieved state is an input',controller_hidden_state_exported=False,controller_selected_policy_id='NOT_EXPORTED_DERIVED_ASSOCIATION_ONLY',source_age_future='Known within currently installed prefix; missing after future renewal, never read future recorded source',events=['left_contact','right_contact','grasp','robot_object_contact','opposing','gripper_valid','safety_valid'],contact_collection='NOT_QUALIFIED',held_out_parent_generalization=bool(any(e['split']=='DEV'for e in entries)))
    data={k:np.asarray(v)for k,v in arrays.items()};data['metadata_utf8']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8)
    if not all(np.isfinite(a).all()for a in data.values()):raise ValueError('NONFINITE_ACTUAL_REPLAY')
    with(out/'REPLAY.npz').open('xb')as f:np.savez_compressed(f,**data)
    result=dict(status='ACTUAL_TARGET_SEQUENTIAL_REPLAY_BUILT',schema=SCHEMA,replay_sha256=digest(out/'REPLAY.npz'),manifest_sha256=digest(manifest_path),episodes=len(entries),task_atoms=24*len(entries),state_dimension=meta['state_dimension'],model_sha256=asset,physics_integrations=0,WM_updates=0,Actor_updates=0)
    write_json(out/'SCHEMA.private.json',meta);write_json(out/'RESULT.json',result);return result


def load(path):
    with np.load(path,allow_pickle=False)as f:data={k:f[k].copy()for k in f.files}
    meta=json.loads(data.pop('metadata_utf8').tobytes())
    if meta['schema']!=SCHEMA or data['state'].shape[1:]!=(25,meta['state_dimension']):raise ValueError('REAL_TARGET_SCHEMA_REQUIRED')
    validate_partitions(meta['episodes']);return data,meta
