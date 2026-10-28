"""Build the negative expert-data readiness record from first-valid raw evidence."""
from pathlib import Path
import argparse
import collections
import json
import shutil
import numpy as np
from pi0255_motion3 import sha,write


def main(root):
    replay=json.loads((root/'evidence/pilot_cpu_replay.json').read_text());assert replay['status']=='PILOTS_COMPLETE'
    assert replay['valid_results']==replay['physics_starts']==36 and replay['engineering_results']==0
    assert all(not x['admitted'] and x['valid']==12 and x['successful']==0 for x in replay['per_task'].values())
    assert len(replay['counterfactual_pairs'])==18 and all(x['matched'] for x in replay['counterfactual_pairs'])
    assert replay['completed_task_episodes']==0
    rows=[]
    for row in replay['rows']:
        directory=root/'episodes'/row['episode_id']/'attempt-00';result=json.loads((directory/'result.json').read_text())
        ticks=[json.loads(s) for s in (directory/'raw_ticks.jsonl').read_text().splitlines()]
        clean=[x for x in ticks if x['physical_clean']]
        p=np.asarray([np.asarray(x['achieved_world'])-np.asarray(x['desired_world']) for x in clean])
        q=np.asarray([x['achieved_quaternion'] for x in clean]);desired_q=np.asarray([x['desired_quaternion'] for x in clean])
        angle=2*np.arccos(np.clip(np.abs(np.sum(q/np.linalg.norm(q,axis=1)[:,None]*desired_q/np.linalg.norm(desired_q,axis=1)[:,None],axis=1)),0,1))
        rows.append(dict(row,clean_settling_same_tick_xyz_rms_m=float(np.sqrt(np.mean(np.sum(p*p,axis=1)))) if len(clean) else None,
            clean_settling_same_tick_so3_rms_rad=float(np.sqrt(np.mean(angle**2))) if len(clean) else None,clean_settling_ticks=len(clean),
            terminal_same_tick_tracking_excluded=not ticks[-1]['physical_clean'],actual_sim_duration_s=result['end_actual_sim_time_s'],
            task_outcome_measured=False,interpretation='FROZEN_SETTLING_FAILED_BEFORE_TASK',raw_attempt=str(directory.relative_to(root))))
    disk=shutil.disk_usage(root)
    summary=dict(schema='PI0255_FIXED_PILOT_NEGATIVE_READINESS_V1',status='PARTIAL_MULTITASK_PREPARATION_BLOCKED',
        pilot_cpu_replay_sha256=sha(root/'evidence/pilot_cpu_replay.json'),per_task=replay['per_task'],failure_classes=replay['failure_classes'],rows=rows,
        paired_reset_identity=dict(pairs=18,all_root_dof_rng_initial_camera_hashes_match=True,task_observability_proven=False),
        handoff_evidence=dict(original_V1_episodes=24,recording_V2_episodes=12,complete_gate_replayed_episodes=sum(x['complete_handoff_gate_replays']==x['handoff_count'] for x in rows),
            original_arrays_missing_retained=True,full_codec_window_qualification=False),
        counters=dict(expert_physics_starts=36,valid_expert_failures=36,valid_expert_successes=0,engineering_started_invalid=0,
            planned_not_executed_due_admission=300,TRAIN_episodes=0,MT_VAL_episodes=0,eligible_imitation_windows=0,TEST_executions=0,
            new_GR00T_loads=0,new_GR00T_queries=0,new_GR00T_backward=0,new_GR00T_optimizer=0,learned_policy_episodes=0),
        gates=dict(all_motion_family_admissions=False,TRAIN_C_T_support_audit='NOT_MEASURABLE_ZERO_ADMITTED_TRAIN',C_MT_numerical_statistics=None,
            model_metadata_headers='PASS_NOT_MODEL_LOAD',real6000_execution_and_full_Trainer_resume='UNRESOLVED_DATA_ADMISSION_FAILED'),
        resource=dict(Corelab_free_bytes=disk.free,Corelab_quota='UNKNOWN',new_Hopper_jobs=0,expert_gpu_index=0,other_workloads_modified=False),
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',scientific_stop='NO_TRAIN_MTVAL_COLLECTION_OR_MODEL_TRAINING; registered pilots exhausted, no replacement/tuning',
        limitations=['All failures precede task phase: no conclusion on reach/route/contour tracking feasibility or learned generalization.',
            '36 episodes share six reset families; task siblings are not independent reliability replications.',
            'Original24 full handoff arrays were not serialized; no reconstruction/full parity claim.',
            'Native physical termination can auto-reset; terminal post-step camera/pose may be post-reset. Event evidence is retained; terminal rows excluded from clean tracking.',
            'No usable motion TRAIN data, statistics, full processor/tokenizer proof or actual6000 launch/resume binding. Draft CPU fixtures do not close these dependencies.'])
    write(root/'evidence/pilot_stage_summary.json',summary)
    write(root/'dataset/audit_dataset_manifest.json',dict(id='PI0255_MOTION3_FIXED_PILOT_AUDIT_V1',status='AUDIT_ONLY_NOT_IMITATION_DATA',
        original_registry_sha256=sha(root/'design/task_registry_v1.json'),audit_attempts=[x['raw_attempt'] for x in rows],
        TRAIN=[],MT_VAL=[],successful_PILOT=[],future_TEST='MANIFEST_ONLY_UNEXECUTED',independent_scenario_families=6,task_families=3,
        usable_window_count=0,split_overlap=0,split_leakage_basis='registry CPU proof plus zero TRAIN/VAL/TEST capture',
        publication='sanctioned synthetic observations and own numerical evidence only; private code/weights/assets excluded'))
    print(json.dumps(dict(status=summary['status'],physics=36,per_task=summary['per_task'],eligible_windows=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
