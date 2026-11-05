"""Decompose sealed C08 motion and installed-reference errors without physics."""
def run(replay,references,expected_sha):
    import json,time
    from pathlib import Path
    import numpy as np
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.target_reference import trajectory_identity
    from go2arx5_vla.world_model.target_stability import motion,diameter,screen
    start=time.monotonic();cpu=time.process_time()
    if digest(replay)!=expected_sha:raise ValueError('SEALED_C08_REPLAY_CHANGED')
    reference=json.loads(Path(references).read_text());episodes={e['episode']:e for e in reference['episodes']}
    with np.load(replay,allow_pickle=False)as z:
        data={k:z[k]for k in z.files}
    meta=json.loads(data['metadata_utf8'].tobytes())
    if meta['schema']!='WM02_TARGET_LATE_DECISION_REPLAY_V3' or meta['executor_sha256']['native_binary']!='29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2':raise ValueError('EXACT_C08_EXECUTOR_AND_ACTION_PROTOCOL_REQUIRED')
    rows=[]
    for episode in meta['episodes']:
        old=episodes[episode['episode']];pins=old['input_sha256']
        if pins['COMMANDS.private.jsonl']!=episode['command_sha256'] or pins['RESULT.json']!=episode['result_sha256']:raise ValueError('SAME_ORIGINAL_COMMAND_AND_RESULT_REQUIRED')
        n=episode['task_atoms'];sl=slice(episode['state_start'],episode['state_start']+n+1);state=data['flat_state'][sl];tcp=data['flat_tcp'][sl,:3,3];times=data['flat_sim_time'][sl];base=state[:,:3];decomp=motion(base,state[:,3:7],tcp)
        events={trajectory_identity(e['times'],e['poses'],e['inputs']):e for e in old['reference_events']if e['kind']=='installed'}
        expected=[];windows=old['diagnostic']['windows']
        if n!=24 or len(windows)!=n or np.max(np.abs(np.diff(times)-.05))>1e-8:raise ValueError('EXACT_COMPLETE_50MS_C08_EPISODE_REQUIRED')
        for i in range(n+1):
            event=events[windows[min(i,n-1)]['target_trajectory_sha256']]
            t=np.asarray(event['times']);p=np.asarray(event['poses'])[:,:3]
            if len(t)<2 or np.any(np.diff(t)<=0)or times[i]<t[0]-1e-7 or times[i]>t[-1]+1e-7:raise ValueError('ACTUAL_INSTALLED_REFERENCE_TIME_COVERAGE_REQUIRED')
            expected.append([np.interp(times[i],t,p[:,j])for j in range(3)])
        expected=np.asarray(expected);relative=decomp['relative_tcp'];commands=[e for e in old['commands']if e['phase']=='proposed'];anchors=np.asarray([e['action_EEF10'][-1][:3]for e in commands]);offset=np.asarray(old['manifest']['offset_xyz_m'])
        curve=dict(time_s=(times-times[0]).tolist(),world_TCP_m=tcp.tolist(),BASE_m=base.tolist(),BASE_wxyz=state[:,3:7].tolist(),TCP_in_BASE_m=relative.tolist(),installed_reference_TCP_m=expected.tolist(),installed_reference_tracking_error_m=np.linalg.norm(tcp-expected,axis=-1).tolist(),base_translation_m=decomp['base_translation'].tolist(),base_rotation_common_motion_m=decomp['base_rotation'].tolist(),relative_arm_motion_world_m=decomp['relative_arm_world'].tolist(),qpos=state[:,:34].tolist(),qvel=state[:,34:66].tolist(),finger_m=state[:,-2:].tolist())
        rows.append(dict(episode=episode['episode'],parent=episode['parent'],split=episode['split'],scene_sha256=episode['scene_sha256'],offset_xyz_m=offset.tolist(),input_sha256=pins,predecision=screen(times[:9],relative[:9]),world_TCP_diameter_predecision_m=diameter(tcp[:9]),BASE_diameter_predecision_m=diameter(base[:9]),world_TCP_max_from_task_start_m=float(np.linalg.norm(tcp-tcp[0],axis=-1).max()),installed_reference_tracking_error_max_m=float(np.linalg.norm(tcp-expected,axis=-1).max()),reference_reanchor_error_max_m=float(np.linalg.norm(anchors-(anchors[0]+np.array([0,0,1,1,1])[:,None]*offset),axis=-1).max()),decision_state=dict(TCP_world_m=tcp[8].tolist(),BASE_m=base[8].tolist(),TCP_in_BASE_m=relative[8].tolist(),BASE_linear_velocity_m_s=state[8,34:37].tolist(),BASE_angular_velocity_rad_s=state[8,37:40].tolist()),postdecision={str(h):dict(delta_TCP_world_m=(tcp[8+h]-tcp[8]).tolist(),delta_TCP_BASE_m=(relative[8+h]-relative[8]).tolist(),delta_BASE_m=(base[8+h]-base[8]).tolist())for h in [1,8,16]},decomposition_residual_max_m=decomp['identity_residual_max_m'],TTL_margin_min_s=min(w['clock']['expiry_margin_s']for w in windows),mode_confirmed=all(w['mode']=='WBC'and w['solver_policies_with_exact_installed_target']for w in windows),controller_selected_policy='DERIVED_NOT_DIRECT_EXPORT',curve=curve))
    scatter=[]
    for parent in sorted({r['parent']for r in rows}):
        group=[r for r in rows if r['parent']==parent]
        scatter.append(dict(parent=parent,split=group[0]['split'],episodes=len(group),decision_TCP_world_diameter_m=diameter([r['decision_state']['TCP_world_m']for r in group]),decision_TCP_BASE_diameter_m=diameter([r['decision_state']['TCP_in_BASE_m']for r in group]),decision_BASE_diameter_m=diameter([r['decision_state']['BASE_m']for r in group]),HOLD_repetitions=sum(np.linalg.norm(r['offset_xyz_m'])<1e-10 for r in group),exact_same_state_restore=False))
    return dict(schema='WM02_NEXT1_STABILITY_CAUSALITY_AUDIT_V1',replay_sha256=expected_sha,references_sha256=digest(references),rows=rows,parent_restart_scatter=scatter,TRAIN_screen_passes=sum(r['predecision']['qualified']for r in rows if r['split']=='TRAIN'),TRAIN_screen_denominator=sum(r['split']=='TRAIN'for r in rows),old_window_screen_is_not_new_S0_qualification=True,motion_interpretation='Exact kinematic decomposition; magnitude alone does not identify causal command response',new_integrations=0,WM_updates=0,Actor_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)


def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['replay','references','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--replay-sha256',required=True);a=p.parse_args(argv)
    result=run(a.replay,a.references,a.replay_sha256);write_json(a.output,result);print(json.dumps({k:v for k,v in result.items()if k not in ['rows','parent_restart_scatter']}));return 0


if __name__=='__main__':raise SystemExit(main())
