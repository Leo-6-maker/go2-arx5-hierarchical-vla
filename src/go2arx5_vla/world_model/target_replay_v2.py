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
    from .target_late_decision import SCHEMA as DECISION
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
    if meta.get('schema')==DECISION and raw['flat_goal'].shape!=(atoms,17):raise ValueError('COMPLETE_ACTUAL_PAST_GOAL_STORAGE_REQUIRED')
    index=raw['window_index'];expected=window_indices([e['task_atoms']for e in meta['episodes']])
    if not np.array_equal(index,expected):raise ValueError('EXACT_EPISODE_CONTAINED_WINDOW_INDEX_REQUIRED')
    keys=('state','native_step','sim_time','contact_time','wall_monotonic','tcp','object','events')
    result={k:[]for k in keys};result.update(goal=[],source_age=[],window_start_s=[],window_start_atom=[],episode_index=[])
    if meta.get('schema')==DECISION:result.update(history_goal=[],future_prefix_remaining=[])
    for i,k in index:
        e=meta['episodes'][int(i)];start=e['state_start']+int(k);atom=e['atom_start']+int(k)
        for name in keys:result[name].append(raw['flat_'+name][start:start+25])
        result['source_age'].append(raw['flat_source_age'][atom:atom+24]);result['goal'].append(raw['episode_goal'][i]);result['episode_index'].append(i);result['window_start_atom'].append(k)
        if meta.get('schema')==DECISION:
            if k!=0 or e['task_atoms']!=24:raise ValueError('LATE_DECISION_EXAM_USES_ONE_PRE_RESPONSE_HISTORY_WINDOW')
            result['history_goal'].append(raw['flat_goal'][atom:atom+8]);result['future_prefix_remaining'].append(7.)
        result['window_start_s'].append(raw['flat_sim_time'][start]-raw['flat_sim_time'][e['state_start']])
    result={k:np.asarray(v)for k,v in result.items()}
    if result['state'].shape[1:]!=(25,meta['state_dimension']):raise ValueError('ACTUAL_WINDOW_STATE_SHAPE_REQUIRED')
    validate_partitions(meta['episodes']);return result,meta


def build(path,output):
    import mujoco
    from .native_replay import packets,forbid_integration
    from .init_trace_reader import read_trace
    from scripts.pi0294.task_state import ActualState
    from .target_late_decision import SCHEMA as DECISION,MANIFEST_SCHEMA as DECISION_MANIFEST,goals as decision_goals
    from .next1_replay import MANIFEST as NEXT1_MANIFEST,sample_view
    manifest=json.loads(Path(path).read_text());entries=manifest['episodes'];validate_partitions(entries);pilot2=manifest.get('schema')=='WM02_PILOT2_COVERAGE_REPLAY_MANIFEST_V1';pilot1=pilot2 or manifest.get('schema')=='WM02_PILOT1_FEEDBACK_MANIFEST_V1';next2=manifest.get('schema')=='WM02_NEXT2_LATE_DECISION_MANIFEST_V1';next1=manifest.get('schema')==NEXT1_MANIFEST or next2 or pilot1;decision=next1 or manifest.get('schema')==DECISION_MANIFEST;schema=DECISION if decision else SCHEMA
    if manifest.get('schema')not in(MANIFEST_SCHEMA,DECISION_MANIFEST,NEXT1_MANIFEST,'WM02_NEXT2_LATE_DECISION_MANIFEST_V1','WM02_PILOT1_FEEDBACK_MANIFEST_V1','WM02_PILOT2_COVERAGE_REPLAY_MANIFEST_V1')or not entries:raise ValueError('REGISTERED_VARIABLE_EPISODE_MANIFEST_REQUIRED')
    if pilot1 and not pilot2 and any(e['split']!='TRAIN'for e in entries):raise ValueError('ONLY_NEW_TRAIN_MAY_FLOW_BACK')
    if pilot2 and any(e['split']not in('TRAIN','DEV')for e in entries):raise ValueError('PILOT2_COVERAGE_NEVER_IMPORTS_TEST')
    allow_path=Path(manifest['scene_allowlist']);assets=json.loads(allow_path.read_text())
    if digest(allow_path)!=manifest['scene_allowlist_sha256']:raise ValueError('SCENE_ALLOWLIST_CHANGED')
    scenes={s['model_sha256']:s for s in assets['scenes']};raw={k:[]for k in('state','source_age','native_step','sim_time','contact_time','wall_monotonic','tcp','object','events')};identities=[];goals=[];models={};state_start=0;atom_start=0;fields=None;joints=None
    if decision:raw['goal']=[]
    executor=None
    for entry in entries:
        root=Path(entry['run']);run=root/'native-01';spec=json.loads((root/'EXECUTION_SPEC.json').read_text());result=json.loads((run/'RESULT.json').read_text());diag=json.loads((run/'DYNAMIC_DIAGNOSTIC.json').read_text());n=entry['task_atoms'];asset=digest(spec['model_path'])
        commands=[json.loads(l)for l in(run/'COMMANDS.private.jsonl').read_text().splitlines()]
        if next1:diag,commands=sample_view(diag,commands)
        feedback=None
        if pilot1:
            dp=run/'PILOT1_DECISION.private.json';sp=run/'PILOT1_SCORE.private.json';dv=json.loads(dp.read_text());sv=json.loads(sp.read_text());event=[i for i,p in enumerate(commands)if p['phase']=='pilot1-choice-before-install'];boundary=[i for i,p in enumerate(commands)if p['phase']=='decision-boundary']
            if (entry['native_result_sha256']!=digest(run/'RESULT.json')or entry['decision_sha256']!=digest(dp)or entry['score_sha256']!=digest(sp)
                    or len(event)!=1 or len(boundary)!=1 or event[0]>=boundary[0]or commands[event[0]]['decision_sha256']!=digest(dp)
                    or dv['future_response_observed']is not False or dv['native_step']!=diag['actual_collection_start_step']+200
                    or dv['split']!=entry['split']or sv['split']!=entry['split']or sv['purpose']=='SCENE_QUALIFICATION'):
                raise ValueError('ACTUAL_TRAIN_CHOICE_BEFORE_INSTALL_AND_SCORE_REQUIRED')
            if pilot2:
                from .pilot2_coverage import coverage_contract
                from .pilot1_episode import OFFSETS
                episode=coverage_contract(spec,json.loads(Path(spec['episode_manifest']).read_text()))
                if (dv.get('selection_source')!='PREREGISTERED_RANDOMIZED_ACTION_MANIFEST'
                        or dv['selected_action']!=episode['forced_action']or sv['purpose']!='ACTION_COVERAGE'
                        or not np.array_equal(np.asarray(dv['selected_offset_xyz_m']),OFFSETS[episode['forced_action']])):
                    raise ValueError('EXECUTED_PREDECLARED_ACTION_REQUIRED_NO_TEACHER_OR_FUTURE_SELECTION')
            feedback=dict(action=dv['selected_action'],goal_offset_xy_m=dv['goal_offset_xy_m'],task_goal_xy_m=dv['task_goal_xy_m'],
                          local_XY_success=sv['local_XY_success'],actual_XY_error_m=sv['actual_XY_error_m'],
                          success_tolerance_m=sv['success_tolerance_m'],decision_sha256=digest(dp),score_sha256=digest(sp),
                          actual_outcome_not_counterfactual_oracle=True)
        if next2:
            predictions=run/'PREDICTIONS_BEFORE_ACTION.private.json';pv=json.loads(predictions.read_text());events=[p for p in commands if p['phase']=='predictions-sealed-before-action'];decision_events=[i for i,p in enumerate(commands)if p['phase']=='decision-boundary'];predict_events=[i for i,p in enumerate(commands)if p['phase']=='predictions-sealed-before-action']
            if len(events)!=1 or len(decision_events)!=1 or predict_events[0]>=decision_events[0]or events[0]['sha256']!=digest(predictions)or pv.get('future_response_observed')is not False or pv['native_step']!=diag['actual_collection_start_step']+200:raise ValueError('ACTUAL_FROZEN_PREDICTIONS_MUST_PRECEDE_CHOSEN_INSTALLATION')
        current_executor={key:digest(spec[key])for key in('native_binary','outer_library','observer_library','trace_library','controller_binary','solver_binary')}
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
        proposed=[p for p in commands if p['phase']=='proposed'];first=proposed[0];goal=np.r_[first['action_EEF10'][-1],first['measured_BASE_pose_hold']]
        if decision:
            if next1:
                from .next1_episode import episode_config,RULE as DECISION_RULE
                if next2:
                    from .next2_episode import episode_config
                if pilot1:
                    from .pilot1_episode import episode_config
            else:
                from .target_manifest import episode_config,DECISION_RULE
            e=episode_config(spec)
            if pilot1:e=dict(e,offset_xyz_m=dv['selected_offset_xyz_m'])
            if n!=24 or e['reference_rule']!=DECISION_RULE or diag.get('reference_rule')!=DECISION_RULE or diag.get('late_decision_applied')is not True:raise ValueError('ACTUAL_LATE_DECISION_EPISODE_REQUIRED')
            goal,per_atom=decision_goals(commands,diag['windows'],start,e['offset_xyz_m']);raw['goal'].extend(per_atom)
        elif first['native_step']!=start or any(not np.allclose(np.r_[p['action_EEF10'][-1],p['measured_BASE_pose_hold']],goal,atol=1e-8,rtol=0)for p in proposed):raise ValueError('PREDECLARED_FIXED_GOAL_REQUIRED_NO_FUTURE_RELABEL')
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
        if feedback is not None:
            actual_error=float(np.linalg.norm(actual.data.site_xpos[actual.tcp][:2]-feedback['task_goal_xy_m']))
            if abs(actual_error-feedback['actual_XY_error_m'])>1e-10 or bool(actual_error<=feedback['success_tolerance_m'])!=feedback['local_XY_success']:
                raise ValueError('ACTUAL_ENDPOINT_AND_FROZEN_LOCAL_SCORE_DIFFER')
            identities[-1]['pilot1_feedback']=feedback
        state_start+=n+1;atom_start+=n
        current_fields=[('qpos',model.nq),('qvel',model.nv),('ctrl',model.nu),('tcp_position_m',3),('tcp_rotation6d',6),('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)]
        if fields is not None and fields!=current_fields:raise ValueError('STATE_LAYOUT_CHANGED')
        fields=current_fields
        joints=[dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i),type=int(model.jnt_type[i]),qpos_address=int(model.jnt_qposadr[i]),qvel_address=int(model.jnt_dofadr[i]))for i in range(model.njnt)]
    meta=dict(schema=SCHEMA,state_fields=[dict(name=k,size=n)for k,n in fields],state_dimension=sum(n for _,n in fields),original_model_sha256=assets['original_model_sha256'],robot_fingerprint=assets['robot_fingerprint'],scene_assets_sha256=sorted(models),joints=joints,actuator_gear=model.actuator_gear.tolist(),actuator_gain=model.actuator_gainprm.tolist(),episodes=identities,task_atoms=atom_start,windows=sum(e['window_count']for e in identities),history_burn_atoms=8,predict_atoms=16,action='PREDECLARED_RAMP8_FIXED_WORLD_GOAL_EEF10_BASE_POSE7_V1',window_clock='ACTUAL_EPISODE_RELATIVE_WINDOW_START_S',renewal_atoms=7,source_age_future='CURRENT_PREFIX_ONLY_THEN_MISSING_NO_FUTURE_ACTUAL',controller_hidden_state_exported=False,controller_selected_policy_id='DERIVED_NOT_DIRECT_EXPORT',contact_collection='NOT_QUALIFIED',events=['left_contact','right_contact','grasp','robot_object_contact','opposing','gripper_valid','safety_valid'],held_out_parent_generalization=any(e['split']=='DEV'for e in identities),overlapping_windows_independent=False)
    meta['executor_sha256']=executor
    if next2:meta.update(prospective_prediction_first=True,new_panel_used_for_model_fitting=False,manifest_schema=manifest['schema'])
    if pilot1:meta.update(manifest_schema=manifest['schema'],actual_TASK_feedback=True,feedback_split='TRAIN_DEV_PREDECLARED_PARENT_PARTITIONS'if pilot2 else'TRAIN_ONLY')
    meta['schema']=schema
    if next1:meta.update(collection_protocol='FIXED_HOLD40_BURN8_DECIDE_RAMP8_V1',excluded_prehold_atoms_per_episode=40,raw_task_atoms_per_episode=64)
    if decision:meta.update(action='ACTUAL_HOLD8_PAST_GOALS_THEN_CHOSEN_RAMP8_FUTURE_GOAL_V3',decision_atom=8,current_installed_prefix_atoms=7,history_goal='ACTUAL_PAST_INSTALLED_GOALS_ONLY',future_reference='CHOSEN_GOAL_INSTALLED_AT_DECISION_BEFORE_RESPONSE')
    arrays={'flat_'+k:np.asarray(a)for k,a in raw.items()};arrays['episode_goal']=np.asarray(goals);arrays['window_index']=window_indices([e['task_atoms']for e in identities]);arrays['metadata_utf8']=np.frombuffer(json.dumps(meta,sort_keys=True,allow_nan=False).encode(),dtype=np.uint8)
    if not all(np.isfinite(a).all()for a in arrays.values()):raise ValueError('NONFINITE_ACTUAL_REPLAY')
    materialize({k:a for k,a in arrays.items()if k!='metadata_utf8'},meta)
    out=Path(output);out.mkdir(exist_ok=False)
    with(out/'REPLAY.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    result=dict(status='ACTUAL_TARGET_SEQUENTIAL_REPLAY_BUILT',schema=schema,replay_sha256=digest(out/'REPLAY.npz'),manifest_sha256=digest(path),episodes=len(entries),task_atoms=atom_start,windows=meta['windows'],state_dimension=meta['state_dimension'],robot_fingerprint=assets['robot_fingerprint'],scene_assets_sha256=meta['scene_assets_sha256'],physics_integrations=0,WM_updates=0,Actor_updates=0)
    write_json(out/'SCHEMA.private.json',meta);write_json(out/'RESULT.json',result);return result
