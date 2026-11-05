"""Zero-integration first entry/exit and reference margins on sealed NEXT1 runs."""
def audit(spec):
    import time,json
    from pathlib import Path
    import numpy as np,mujoco
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.native_replay import packets,forbid_integration
    from go2arx5_vla.world_model.target_stability import motion
    from scripts.pi0294.task_state import ActualState
    from scripts.pi0294.persistent_teacher import geometry,pad_corners
    start=time.monotonic();cpu=time.process_time();episodes=[]
    for binding in spec['episodes']:
        root=Path(binding['run']);native=root/'native-01';s=json.loads((root/'EXECUTION_SPEC.json').read_text());result=json.loads((native/'RESULT.json').read_text())
        if digest(native/'RESULT.json')!=binding['native_result_sha256']or result['status']!='NATIVE_STAGE_PASSED'or not result['shutdown_clean']:raise ValueError('SEALED_SAFE_NATIVE_RECORDING_REQUIRED')
        if digest(s['model_path'])!=result['scene_sha256']:raise ValueError('SEALED_SCENE_CHANGED')
        engine=ActualState(mujoco.MjModel.from_binary_path(s['model_path']));commands=[json.loads(x)for x in(native/'COMMANDS.private.jsonl').read_text().splitlines()];proposed=[p for p in commands if p['phase']=='proposed'];windows={p['native_end_step']:p for p in(result['reference_qualification']or{}).get('windows',[])};timeline=[]
        with forbid_integration(mujoco):
            for p in packets(native/'state.bin.journal'):
                if p['step']<2503 or(p['step']-2503)%25:continue
                state=engine.update(p);g=geometry(engine,state,'T1');pads=pad_corners(engine);obj=state['object_pose'];local=(pads-obj[:3,3])@obj[:3,:3];lo=local.min(1);hi=local.max(1);neg,pos=np.argsort(local.mean(1)[:,1]);margin=dict(enclose_y_m=float(min(-.024-hi[neg,1],lo[pos,1]-.024)),overlap_x_m=float(min(np.min(.02-lo[:,0]),np.min(hi[:,0]+.02))),above_z_m=float(np.min(lo[:,2]-.024)),slab_z_m=float(np.min(.1-lo[:,2])))
                selected=[r for r in proposed if r['native_step']<p['step']or p['step']==2503 and r['native_step']==2503];ref=selected[-1];a=np.asarray(ref['action_EEF10']);at=(p['time_s']-ref['source_mpc_s'])/.05;left=max(0,min(int(np.floor(at)),39));right=min(left+1,39);fraction=min(1,max(0,at-left));desired=(1-fraction)*a[left,:3]+fraction*a[right,:3];w=windows.get(p['step'],{});v=np.zeros(3)if not timeline else(state['tcp'][:3,3]-np.asarray(timeline[-1]['TCP_world_m']))/.05
                timeline.append(dict(native_step=p['step'],time_s=p['time_s'],precontact=g['precontact_geometry_ready'],alignment=g['actual_contact_alignment'],margins=margin,TCP_world_m=state['tcp'][:3,3].tolist(),BASE_world_m=state['qpos'][:3].tolist(),BASE_wxyz=state['qpos'][3:7].tolist(),TCP_speed_m_s=float(np.linalg.norm(v)),reference_world_m=desired.tolist(),reference_source_s=ref['source_mpc_s'],reference_sequence=ref['sequence'],source_TTL_margin_s=w.get('clock',{}).get('expiry_margin_s'),solver_target_confirmed=bool(w.get('solver_policies_with_exact_installed_target')),controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',gripper_command_fraction=float(a[0,9]),gripper_command_source_s=ref['source_mpc_s'],actual_finger_m=state['finger_q'].tolist(),pad_object_contact=[state['left_object_contact'],state['right_object_contact']],native_grasp=state['native_grasp'],actual_qvel=state['qvel'].tolist()))
        enter=next((i for i,r in enumerate(timeline)if r['precontact']),None);leave=None if enter is None else next((i for i in range(enter+1,len(timeline))if not timeline[i]['precontact']),None);event=None
        if leave is not None:
            chosen=timeline[max(0,leave-4):min(len(timeline),leave+5)];b=np.asarray([r['BASE_world_m']for r in chosen]);q=np.asarray([r['BASE_wxyz']for r in chosen]);tcp=np.asarray([r['TCP_world_m']for r in chosen]);parts=motion(b,q,tcp);ref=np.asarray([r['reference_world_m']for r in chosen]);event=dict(first_enter_step=timeline[enter]['native_step'],first_leave_step=timeline[leave]['native_step'],failed_margins={k:val for k,val in timeline[leave]['margins'].items()if val<0},reference_max_change_m=float(np.linalg.norm(ref-ref[0],axis=1).max()),BASE_translation_m=parts['base_translation'][-1].tolist(),BASE_rotation_TCP_component_m=parts['base_rotation'][-1].tolist(),relative_arm_world_component_m=parts['relative_arm_world'][-1].tolist(),event_neighborhood=chosen,interpretation='Reference continuity, body/arm changes and pad margins observed; no direct selected-policy ID, no WBC causal attribution from geometry alone')
        ready=[r['TCP_speed_m_s']for r in timeline if r['precontact']];episodes.append(dict(run=root.name,source_sha=s['source_sha'],native_result_sha256=digest(native/'RESULT.json'),state_journal_sha256=digest(native/'state.bin.journal'),command_sha256=digest(native/'COMMANDS.private.jsonl'),actual_native_ticks=result['actual_target_integrations'],precontact_samples=len(ready),ready_speed_max_m_s=max(ready)if ready else None,first_exit=event,timeline=timeline,expired_or_rejected_commands=sum(r.get('phase')=='consumption_unconfirmed'for r in commands),physics_stops=result['physics_stops']))
    return dict(schema='WM02_NEXT2_PRECONTACT_FIRST_EXIT_AUDIT_V1',episodes=episodes,new_integrations=0,new_optimizer_updates=0,Actor_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)


def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=audit(json.loads(a.spec.read_text()));write_json(a.output,v);print(json.dumps({r['run']:{k:r[k]for k in ['precontact_samples','ready_speed_max_m_s']}for r in v['episodes']}));return 0
if __name__=='__main__':raise SystemExit(main())
