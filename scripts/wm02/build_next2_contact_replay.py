"""Versioned piecewise native transitions, actual commands and wrapper memory."""
def build(spec,out):
    from pathlib import Path
    import json
    import numpy as np,mujoco
    from go2arx5_vla.world_model.native_replay import packets,forbid_integration
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.next2_episode import state_vector
    from go2arx5_vla.world_model.target_replay import validate_windows
    from scripts.pi0294.task_state import ActualState
    root=Path(spec['run']);native=root/'native-01';s=json.loads((root/'EXECUTION_SPEC.json').read_text());r=json.loads((native/'RESULT.json').read_text());diag=json.loads((native/'DYNAMIC_DIAGNOSTIC.json').read_text());proof=json.loads((native/'PRECONTACT_CONTINUITY.private.json').read_text());out=Path(out)
    if r['status']!='NATIVE_STAGE_PASSED'or r['physics_stops']or not r['shutdown_clean']or diag['complete_task_atoms']!=400:raise ValueError('UNQUALIFIED_OR_PARTIAL_NATIVE_RECORDING_NOT_A_TRAINING_REPLAY')
    validate_windows(diag['windows'],2503,400);by_step={p['step']:p for p in packets(native/'state.bin.journal')};rows=[by_step[2503+25*i]for i in range(401)];engine=ActualState(mujoco.MjModel.from_binary_path(s['model_path']));states=[];events=[];commands=[json.loads(x)for x in(native/'COMMANDS.private.jsonl').read_text().splitlines()];proposed={p['wire_sha256']:p for p in commands if p['phase']=='proposed'};references=[];indices=[];ages=[]
    with forbid_integration(mujoco):
        for p in rows:
            state=engine.update(p);states.append(state_vector(state));events.append([state[k]for k in ['left_object_contact','right_object_contact','native_grasp','robot_object_contact','opposing_contact','gripper_state_valid','safety_valid']])
    for w in diag['windows']:
        p=proposed[w['reference_sha256']];index=(w['native_start_step']-p['native_step'])//25
        if not 0<=index<7:raise ValueError('EXACT_ACTUAL_INSTALLED_REFERENCE_WINDOW_REQUIRED')
        references.append(np.concatenate([np.asarray(p['action_EEF10']),np.repeat(np.asarray(p['measured_BASE_pose_hold'])[None],40,axis=0)],axis=1));indices.append(index);ages.append(w['clock']['native_start_s']-w['clock']['source_mpc_s'])
    memory=[]
    for p in proof['samples']:
        m=p['actual_gripper_internal_memory']
        if m['status']!='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'or m['values']['steps']!=p['native_step']:raise ValueError('ACTUAL_SAME_PAUSE_GRIPPER_MEMORY_REQUIRED')
        v=m['values'];memory.append([*v['target'],v['opening'],v['expiry'],v['sequence'],v['steps'],v['initialized'],v['has_command']])
    from go2arx5_vla.world_model.next2_native_events import contact_outcomes
    outcomes=[contact_outcomes([by_step[2503+25*i+k]for k in range(1,26)],engine.pad,engine.cube_geom)for i in range(400)]
    metadata=dict(native_event_schema='WITHIN_25_ACTUAL_NATIVE_TICKS_V1',native_event_positive_atoms=sum(bool(v[0].any())for v in outcomes),native_event_positive_ticks=sum(v[3]for v in outcomes),schema='WM02_NEXT2_PIECEWISE_NATIVE_V1',source_sha=s['source_sha'],state_dimension=106,model_sha256=r['scene_sha256'],native_binary_sha256=r['native_binary_sha256'],native_result_sha256=digest(native/'RESULT.json'),state_journal_sha256=digest(native/'state.bin.journal'),command_sha256=digest(native/'COMMANDS.private.jsonl'),piecewise_proof_sha256=digest(native/'PRECONTACT_CONTINUITY.private.json'),native_dt_s=.002,atom_dt_s=.05,atoms=400,reference_fields='WORLD_EEF10+INITIAL_MEASURED_BASE_POSE7',reference_semantics='ACTUAL_INSTALLED_H40_AND_EXECUTED_INDEX_NOT_FINAL_GOAL_LABEL',memory_fields=['target_left','target_right','opening','expiry','sequence','native_steps','initialized','has_command'],MPC_WBC_hidden_state='NOT_COMPLETE_RESTORE; sequential collection only',stage_semantics='FSM_AFTER_ACTUAL_OBSERVATION; executed action is reference_H40 plus executed_reference_index',controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',terminal=False,truncated=True,full_task_success=False,split='TRAIN_CONTACT_COMMISSIONING',contact_positives=sum(bool(p['left_object_contact']or p['right_object_contact'])for p in proof['samples']))
    arrays=dict(state=np.asarray(states),reference_H40=np.asarray(references),executed_reference_index=np.asarray(indices),source_age=np.asarray(ages),events=np.asarray(events),native_step=np.asarray([p['step']for p in rows]),native_time=np.asarray([p['time_s']for p in rows]),contact_time=np.asarray([p['contact_time_s']for p in rows]),actual_gripper_memory=np.asarray(memory),stage=np.asarray([p['stage']for p in proof['samples']]),table_support=np.asarray([p['table_support']for p in proof['samples']]),valid=np.ones(400,dtype=bool),metadata_utf8=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),dtype=np.uint8))
    arrays.update(native_contact_any_per_atom=np.asarray([v[0]for v in outcomes]),native_contact_peak_normal_force_N=np.asarray([v[1]for v in outcomes]),native_opposing_any_per_atom=np.asarray([v[2]for v in outcomes]))
    if arrays['state'].shape!=(401,106)or arrays['reference_H40'].shape!=(400,40,17)or arrays['actual_gripper_memory'].shape!=(400,8):raise ValueError('COMPLETE_PIECEWISE_ARRAY_LAYOUT_REQUIRED')
    for k in ['state','reference_H40','source_age','actual_gripper_memory']:
        if not np.isfinite(arrays[k]).all():raise ValueError('NONFINITE_PIECEWISE_TRANSITION')
    out.mkdir(exist_ok=False)
    with(out/'REPLAY.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    result=dict(status='ACTUAL_PIECEWISE_NATIVE_TRANSITIONS_SEALED',replay_sha256=digest(out/'REPLAY.npz'),task_atoms=400,contact_positives=metadata['contact_positives'],native_event_positive_atoms=metadata['native_event_positive_atoms'],native_event_positive_ticks=metadata['native_event_positive_ticks'],physics_integrations=0,WM_updates=0,Actor_updates=0,metadata=metadata);write_json(out/'RESULT.private.json',result);return result

def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);r=build(json.loads(a.spec.read_text()),a.output);print(json.dumps({k:r[k]for k in ['status','replay_sha256','task_atoms','contact_positives']}));return 0
if __name__=='__main__':raise SystemExit(main())
