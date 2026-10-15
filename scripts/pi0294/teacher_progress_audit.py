"""Read-only task approach analysis from sealed native pilot arrays.

No dynamics integration, optimization, rendering, neural query or WBC call.
Stage targets, same-time previous reference and future previews stay separate.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .task_capture_runtime import cpu_stage,read,sha
from .task_clock import actual_source_age
from go2arx5_vla.trajectory.so3 import geodesic_angle,matrix_to_quat,quat_to_matrix,quat_slerp

CASES=('TASK-PILOT-W0-T2-engineering04','TASK-PILOT-W1-T1','TASK-PILOT-W1-T2')


def interpolate(points,elapsed):
    grid=np.asarray([p['time'] for p in points]);xyz=np.asarray([p['position'] for p in points])
    if not grid[0]-1e-8<=elapsed<=grid[-1]+1e-8:raise ValueError('REFERENCE_TIME_OUTSIDE_RECORDED_HORIZON')
    i=int(np.clip(np.searchsorted(grid,elapsed,side='right')-1,0,len(grid)-2))
    weight=float(np.clip((elapsed-grid[i])/(grid[i+1]-grid[i]),0,1))
    result=np.eye(4);result[:3,3]=[np.interp(elapsed,grid,xyz[:,k]) for k in range(3)]
    result[:3,:3]=quat_to_matrix(quat_slerp(points[i]['quaternion_xyzw'],points[i+1]['quaternion_xyzw'],weight))
    return result


def residual(target,actual):
    return float(np.linalg.norm(target[:3,3]-actual[:3,3])),float(geodesic_angle(target[:3,:3],actual[:3,:3]))


def run(root):
    import mujoco
    from .task_state import ActualState
    from .base_intent import measured_base
    from .teacher import PushTeacher
    from scripts.pi0291_arx5_data.teacher import Teacher as PickTeacher
    from scripts.pi0291_arx5_data.config import PARAMETERS as P
    from scripts.pi0292.durable import write
    root=Path(root)
    with cpu_stage(root,'TEACHER_APPROACH_PROGRESS_CPU_V1',cores=1,cpu_s_cap=900) as manifest:
        out=root/'analysis/TEACHER_APPROACH_PROGRESS_V1';out.mkdir(parents=True,exist_ok=False)
        capsule=root/manifest['code']['capsule_relative'];summaries=[];plotdata=[]
        for case in CASES:
            folder=root/'episodes'/case;card=read(folder/'CARD.json');start=read(folder/'TASK_START.json')
            execution=read(folder/'RESULT.json');prior=read(folder/'FULLBODY_CPU_AUDIT.json')
            sealed=root/'capsules'/execution['source_sha']/'code'
            names=('scripts/pi0291_arx5_data/teacher.py','scripts/pi0291_arx5_data/config.py',
                'scripts/pi0294/teacher.py','scripts/pi0294/scene.py','scripts/pi0294/task_plant.py',
                'scripts/pi0294/task_state.py','scripts/pi0294/base_intent.py',
                'src/go2arx5_vla/trajectory/so3.py','src/go2arx5_vla/task_data/codec.py')
            pins={n:sha(sealed/n) for n in names}
            if any(sha(capsule/n)!=h for n,h in pins.items()):raise ValueError('ACTUAL_CASE_NUMERICAL_SOURCE_MISMATCH')
            for name,digest in prior['raw_evidence'].items():
                if sha(folder/name)!=digest:raise ValueError('ORIGINAL_RAW_CHANGED:'+name)
            if sha(folder/'task-scene.private.mjb')!=read(folder/'PREEXECUTION.json')['model_sha256']:
                raise ValueError('ACTUAL_COMPILED_PLANT_CHANGED')
            samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
            commands=[json.loads(s) for s in (folder/'committed_intents.jsonl').read_text().splitlines() if 'action_EEF13' in s]
            planner=[json.loads(s) for s in (folder/'planner.jsonl').read_text().splitlines()]
            accepts={r['ref_id']:r for r in planner if r['kind']=='accept'}
            installs={key:[r for r in planner if r['kind']=='installed' and r.get('ref_id')==key] for key in accepts}
            if len(samples)!=641 or len(commands)!=640:raise ValueError('FULL_CASE_COUNTS_REQUIRED')
            model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'));engine=ActualState(model)
            s0=samples[0]
            engine.update(dict(s0,step=s0['native_step'],time_s=start['time_s'],contacts=[]))
            first={k:(np.asarray(v) if isinstance(v,list) else v) for k,v in s0.items()}
            teacher=PickTeacher(card,first['object_pose']) if card['task']=='T1' else PushTeacher(card,first['object_pose'],engine)
            first_mount=np.asarray(commands[0]['actual_BASE_mount']);station=first_mount[:3,3]+[card['base_translation_preference_m'],0,0]
            rows=[];path_replay_max=0.;angles_replay_max=0.;same_install_max=0.;geometry_rows=[]
            for index,saved in enumerate(samples):
                s={k:(np.asarray(v) if isinstance(v,list) else v) for k,v in saved.items()}
                teacher.observe(s);target,opening=teacher.target(s)
                distance,angle=residual(target,s['tcp'])
                engine.data.qpos[:]=s['qpos'];engine.data.qvel[:]=s['qvel'];engine.data.time=start['time_s']+s['time_s'];mujoco.mj_forward(model,engine.data)
                imu=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'imu');pose=np.eye(4);pose[:3,:3]=engine.data.site_xmat[imu].reshape(3,3);pose[:3,3]=engine.data.site_xpos[imu]
                vel=np.zeros(6);mujoco.mj_objectVelocity(model,engine.data,mujoco.mjtObj.mjOBJ_SITE,imu,vel,0)
                mount,measured=measured_base(pose,pose[:3,:3].T@vel[3:],pose[:3,:3].T@vel[:3],[.07557,0,.05768])
                row=dict(index=index,time_s=float(s['time_s']),native_time_s=float(engine.data.time),stage=teacher.stage,
                    stage_age_s=float(s['time_s']-teacher.stage_since),target_world=target.tolist(),actual_TCP_world=s['tcp'].tolist(),
                    stage_position_error_m=distance,stage_angle_error_rad=angle,
                    position_predicate=distance<=P['position_stage_tolerance_m'],
                    angle_predicate=angle<=.04,angle_is_transition_predicate=card['task']=='T2',
                    joint_transition_predicate=distance<=.004 and (card['task']=='T1' or angle<=.04),
                    actual_BASE_world=mount.tolist(),persistent_declared_station_world_xyz=station.tolist(),
                    station_xy_error_m=float(np.linalg.norm(station[:2]-mount[:2,3])),actual_BASE_velocity_yaw=measured.tolist(),
                    actual_TCP_object_distance_m=float(np.linalg.norm(s['tcp'][:3,3]-s['object_pose'][:3,3])),
                    consumer_identity='UNKNOWN_NO_REFERENCE_ID_IN_SAVED_MPC_POLICY_OR_NATIVE_SELECTED_COMMAND')
                if index<640:
                    command=commands[index];plan=teacher.plan(s);a=np.asarray(command['action_EEF13'])
                    path_replay_max=max(path_replay_max,float(np.max(np.abs(plan['poses'][:,:3,3]-a[:,:3]))))
                    for p,x in zip(plan['poses'],a):
                        # codec uses the same recorded six-dimensional convention.
                        from scripts.pi0292.eval_contract import matrix10
                        decoded=matrix10(x[None,:10])[0]
                        angles_replay_max=max(angles_replay_max,float(geodesic_angle(p[:3,:3],decoded[:3,:3])))
                    firstpoint=a[0,:3];preview=a[-1,:3]
                    row.update(new_H40_first_xyz=firstpoint.tolist(),new_H40_endpoint_xyz=preview.tolist(),
                        firstpoint_actual_gap_m=float(np.linalg.norm(firstpoint-s['tcp'][:3,3])),
                        preview_future_offset_s=1.95,preview_length_m=float(np.linalg.norm(preview-firstpoint)),
                        generated_BASE_velocity=a[0,10:13].tolist(),new_source_age_s=actual_source_age(row['native_time_s'],command['source_mpc_s']))
                if index:
                    prev=commands[index-1];accepted=accepts[prev['wire_sha256']];document=json.loads(accepted['raw'])
                    elapsed=row['native_time_s']-accepted['source_mpc_s'];reference=interpolate(document['ee']['waypoints'],elapsed)
                    base_reference=interpolate(document['base']['waypoints'],elapsed)
                    dp,da=residual(reference,s['tcp'])
                    candidates=[r for r in installs[prev['wire_sha256']] if r['monotonic_s']<
                        (accepts[commands[index]['wire_sha256']]['monotonic_s'] if index<640 else float('inf'))
                        and r['times'][0]-1e-8<=row['native_time_s']<=r['times'][-1]+1e-8]
                    if not candidates:raise ValueError('SAME_TIME_PREVIOUS_REFERENCE_INSTALL_MISSING')
                    installed=candidates[-1]
                    # Compare installed native xyz at this exact physical time;
                    # never use the new command generated from this observation.
                    installed_xyz=np.asarray([np.interp(row['native_time_s'],installed['times'],np.asarray(installed['poses'])[:,k]) for k in range(3)])
                    same_install_max=max(same_install_max,float(np.max(np.abs(installed_xyz-reference[:3,3]))))
                    row.update(previous_ref_id=prev['wire_sha256'],previous_reference_source_s=accepted['source_mpc_s'],
                        previous_reference_elapsed_s=elapsed,same_time_installed_reference_world=reference.tolist(),
                        same_time_installed_BASE_world=base_reference.tolist(),same_time_position_residual_m=dp,same_time_angle_residual_rad=da,
                        previous_reference_advance_from_first_m=float(np.linalg.norm(reference[:3,3]-np.asarray(document['ee']['waypoints'][0]['position']))))
                else:row.update(same_time_installed_reference_world=None,same_time_position_residual_m=None,same_time_angle_residual_rad=None)
                if index in (0,320,640):
                    pads=[]
                    for gid in engine.pad:
                        center=engine.data.geom_xpos[gid];axes=engine.data.geom_xmat[gid].reshape(3,3);size=model.geom_size[gid]
                        pads.append(dict(actual_center_world=center.tolist(),lowest_actual_z_m=float(center[2]-np.abs(axes[2])@size),
                            target_orientation_lowest_z_m=float(target[2,3]+(target[:3,:3]@(s['tcp'][:3,:3].T@(center-s['tcp'][:3,3])))[2]-np.abs((target[:3,:3]@s['tcp'][:3,:3].T@axes)[2])@size)))
                    geometry_rows.append(dict(index=index,pads=pads,table_top_m=.44,
                        note='GEOMETRIC_PAD_CLEARANCE_ONLY_NOT_IK_REACHABILITY_OR_SWEPT_ROBOT_COLLISION_PROOF'))
                rows.append(row)
            if teacher.events or teacher.stage!='approach' or path_replay_max>1e-12 or angles_replay_max>1e-6 or same_install_max>1e-6:
                raise ValueError('EXACT_OLD_TEACHER_OR_INSTALLED_REFERENCE_REPLAY_MISMATCH')
            for i,row in enumerate(rows):
                row['goal_distance_decrease_last_2s_m']=rows[max(0,i-40)]['stage_position_error_m']-row['stage_position_error_m']
            trace=out/(case+'.jsonl');trace.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
            dist=np.asarray([r['stage_position_error_m'] for r in rows]);angle=np.asarray([r['stage_angle_error_rad'] for r in rows])
            ref=[r for r in rows if r['same_time_position_residual_m'] is not None]
            stats=lambda x:dict(initial=float(x[0]),minimum=float(np.min(x)),final=float(x[-1]))
            summary=dict(case=case,task=card['task'],actual_execution_source=execution['source_sha'],numerical_source_files=pins,
                status='SAVED_APPROACH_ANALYSIS_COMPLETE_NOT_NEW_PHYSICAL_RESULT',observations=641,installed_transactions=640,
                stage_target_world=rows[0]['target_world'],stage_position_error_m=stats(dist),stage_angle_error_rad=stats(angle),
                position_pass_samples=int(np.count_nonzero(dist<=.004)),angle_pass_samples=int(np.count_nonzero(angle<=.04)),
                simultaneous_position_angle_pass_samples=int(np.count_nonzero((dist<=.004)&(angle<=.04))),
                actual_stage_transition_pass_samples=sum(r['joint_transition_predicate'] for r in rows),
                net_goal_distance_decrease_m=float(dist[0]-dist[-1]),
                two_second_windows_positive_decrease=sum(rows[i]['goal_distance_decrease_last_2s_m']>0 for i in range(40,641,40)),
                two_second_windows_total=16,
                same_time_previous_ref_position_residual_m=dict(median=float(np.median([r['same_time_position_residual_m'] for r in ref])),max=float(np.max([r['same_time_position_residual_m'] for r in ref]))),
                same_time_previous_ref_angle_residual_rad=dict(median=float(np.median([r['same_time_angle_residual_rad'] for r in ref])),max=float(np.max([r['same_time_angle_residual_rad'] for r in ref]))),
                reference_advance_before_replace_m=dict(median=float(np.median([r['previous_reference_advance_from_first_m'] for r in ref])),max=float(np.max([r['previous_reference_advance_from_first_m'] for r in ref]))),
                firstpoint_actual_gap_max_m=max(r['firstpoint_actual_gap_m'] for r in rows[:-1]),
                actual_replan_s=.05,declared_prefix_s=.4,prefix8_executed=False,
                preview_length_median_m=float(np.median([r['preview_length_m'] for r in rows[:-1]])),
                station_xy_error_m=stats(np.asarray([r['station_xy_error_m'] for r in rows])),
                actual_TCP_object_distance_m=stats(np.asarray([r['actual_TCP_object_distance_m'] for r in rows])),
                old_teacher_replay_position_max_m=path_replay_max,old_teacher_replay_angle_max_rad=angles_replay_max,
                same_time_installation_position_max_m=same_install_max,
                geometry=geometry_rows,reachability='UNPROVEN_NO_NEW_IK_OR_WBC_QUERY',
                consumer_identity='UNKNOWN_NO_SAVED_REFERENCE_TO_SOLVER_POLICY_ID',
                classifications=['STAGE_PREDICATE_NONATTAINMENT','REFERENCE_PROGRESS_OR_CADENCE_MISMATCH','REFERENCE_CONSUMPTION_UNOBSERVED'],
                unique_dynamic_root_cause_proved=False,raw_hashes={n:sha(folder/n) for n in ('samples.jsonl','committed_intents.jsonl','planner.jsonl','task-scene.private.mjb')},
                detailed_trace_sha256=sha(trace),new_physics=0,new_model_queries=0)
            summaries.append(summary);plotdata.append(rows)
        value=dict(status='COMPLETE_SAVED_ARRAY_APPROACH_PROGRESS_AUDIT',cases=summaries,
            new_physics=0,new_episodes=0,new_render=0,new_model_queries=0,new_IK=0,WBC_changes=0,
            source_sha=manifest['code']['vla_commit'],root_cause='SUPPORTED_TASK_LAYER_MISMATCHES_NOT_UNIQUE_CAUSAL_PROOF',
            physical_launch_allowed=False,selected='NONE',deployment='NONE')
        write(out/'TEACHER_APPROACH_PROGRESS.json',value);print(json.dumps(value))
        write(out/'PLOT_DATA.private.json',plotdata)
