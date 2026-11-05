"""Audit same-source native recording coverage; no integration or learning."""
def audit(config):
    import json,time
    from pathlib import Path
    import numpy as np
    from go2arx5_vla.world_model.native_replay import load,forbid_integration
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.target_replay import validate_windows
    from scripts.pi0294.task_state import ActualState
    start=time.monotonic();cpu=time.process_time()
    mujoco,model,_,rows=load(config);result_path=Path(config['native_result']['path'])
    if digest(result_path)!=config['native_result']['sha256']:raise ValueError('SEALED_NATIVE_RESULT_CHANGED')
    result=json.loads(result_path.read_text());diag=result.get('reference_qualification');ep_start=diag.get('actual_collection_start_step')if diag else None
    if result['scene_sha256']!=config['scene']['sha256']or result['source_sha']!=config['original_native_source_sha']or result['actual_target_integrations']!=len(rows):raise ValueError('SAME_ORIGINAL_NATIVE_RECORDING_REQUIRED')
    if not all(np.isfinite(np.r_[r['qpos'],r['qvel'],r['ctrl']]).all()for r in rows):raise ValueError('NONFINITE_NATIVE_RECORDING')
    commands=[]
    if config.get('commands'):
        if digest(config['commands']['path'])!=config['commands']['sha256']:raise ValueError('ORIGINAL_COMMAND_BYTES_CHANGED')
        commands=[json.loads(line)for line in Path(config['commands']['path']).read_text().splitlines()]
    proposed=[r for r in commands if r.get('phase')=='proposed'];installed={r['wire_sha256']for r in commands if r.get('phase')=='installed'}
    if ep_start is None and proposed:ep_start=proposed[0]['native_step']
    selected=[r for r in rows if ep_start is not None and r['step']>=ep_start and(r['step']-ep_start)%25==0]
    actual=ActualState(model)
    with forbid_integration(mujoco)as calls:states=[actual.update(row)for row in selected]
    if calls:raise RuntimeError('AUDIT_CANNOT_INTEGRATE')
    coverage={}
    if states:
        tcp=np.asarray([s['tcp'][:3,3]for s in states]);obj=np.asarray([s['object_pose'][:3,3]for s in states]);base=np.asarray([r['qpos'][:3]for r in selected]);fingers=np.asarray([s['finger_q']for s in states])
        coverage=dict(task_sampled_states=len(states),TCP_end_displacement_xyz_m=(tcp[-1]-tcp[0]).tolist(),TCP_max_distance_from_initial_m=float(np.linalg.norm(tcp-tcp[0],axis=1).max()),BASE_max_distance_from_initial_m=float(np.linalg.norm(base-base[0],axis=1).max()),object_max_distance_from_initial_m=float(np.linalg.norm(obj-obj[0],axis=1).max()),finger_peak_to_peak_m=np.ptp(fingers,axis=0).tolist(),contact_and_grasp_counts={k:sum(bool(s[k])for s in states)for k in ['left_object_contact','right_object_contact','native_grasp','robot_object_contact','opposing_contact']})
    if diag:
        validate_windows(diag['windows'],diag['actual_collection_start_step'],diag['complete_task_atoms'])
        coverage.update(complete_atoms=diag['complete_task_atoms'],solver_confirmed_atoms=diag['solver_target_confirmed_atoms'],expiry_margin_min_s=min(w['clock']['expiry_margin_s']for w in diag['windows']),controller_selected_policy=diag['controller_selected_reference'])
    violations=[];last=rows[-1]
    for j in range(model.njnt):
        if not model.jnt_limited[j]or int(model.jnt_type[j])not in(2,3):continue
        q=float(last['qpos'][model.jnt_qposadr[j]]);low,high=model.jnt_range[j];penetration=max(low-q,q-high,0.)
        if penetration:violations.append(dict(joint=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j),q=q,range=[float(low),float(high)],penetration=float(penetration)))
    output=dict(status='SAME_SOURCE_RECORDED_TARGET_AUDITED',episode_id=config['episode_id'],label=config['label'],original_native_source_sha=config['original_native_source_sha'],input_sha256={k:config[k]['sha256']for k in ['scene','journal','native_result','commands']if k in config},original_native_status=result['status'],native_steps=len(rows),native_clock_s=[rows[0]['time_s'],rows[-1]['time_s']],initialization_and_task_separate=True,proposed_references=len(proposed),proposed_with_actual_installation=sum(r['wire_sha256']in installed for r in proposed),coverage=coverage,last_native_joint_violations=violations,physics_stops=result['physics_stops'],trace_audit_errors=result.get('trace_audit',{}).get('errors'),valid_learning_atoms_added=0,failed_prefix_used_for_learning=False,full_task_success='NOT_EVALUATED_NO_BOUND_STANDING_TASK_SCORER',new_integrations=0,new_WM_updates=0,new_Actor_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    return output


def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args(argv)
    output=audit(json.loads(a.config.read_text()));write_json(a.output,output);print(json.dumps(output));return 0

if __name__=='__main__':raise SystemExit(main())
