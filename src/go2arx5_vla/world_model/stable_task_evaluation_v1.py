"""Independent learned candidate's real new-layout DEV native evaluation.

Teacher is absent from proposal generation. Ground truth remains in unchanged
safety and scoring only; actual observations feed the separately trained model.
"""
import copy
import json
from pathlib import Path
import uuid
import numpy as np
from scripts.pi0292.relay_protocol import digest as document_hash, unseal
from . import stable_task_scene_v1 as scene_contract
from .stable_task_scene_v1 import PHASE, KIND, B, E, need
from . import stable_start_short_task_v1 as common
from .storage import digest, write_json, jsonable

from . import stable_task_policy_v1 as policy
from .stable_task_data_v1 import capture_input,task_score_contract
PHASE='collect-target-stable-task-eval-v1'
KIND='STABLE_TASK_EVAL_V1'
B,E=34303,38703
SOURCE_MEMBERS=tuple(dict.fromkeys(common.SOURCE_MEMBERS+(
    'src/go2arx5_vla/world_model/stable_task_scene_v1.py',
    'src/go2arx5_vla/world_model/stable_task_layout_static_v1.py',
    'src/go2arx5_vla/world_model/stable_task_data_v1.py',
    'src/go2arx5_vla/world_model/stable_task_adapter_v1.py',
    'src/go2arx5_vla/world_model/stable_task_dataset_v1.py',
    'src/go2arx5_vla/world_model/stable_task_policy_v1.py',
    'src/go2arx5_vla/world_model/stable_task_evaluation_v1.py',
    'scripts/wm02/stable_task_evaluation_check_v1.py')))


def validate(spec):
    scene_contract.scene(spec)
    c=spec.get('stable_task_evaluation')or{}
    need(spec.get('phase')==PHASE and spec.get('kind')==KIND and spec.get('data_role')=='DEV'
         and spec.get('snapshot_binding_mode')=='PRE_REGISTERED_STABLE_TASK_ADAPTER_EVAL_V1'
         and c.get('method')=='LEARNED_TASK_ADAPTER'and c.get('step_m')==.004
         and c.get('suffix_atoms')==176 and c.get('minimum_course_clearance_m')==.14
         and c.get('producer')=='INDEPENDENT_TASK_ADAPTER_NOT_TEACHER'
         and not spec.get('paused_query_pair')and not spec.get('paused_query_gpu_spec')
         and spec.get('live_skill_geometry_scale')==1.
         and spec.get('live_skill_time_stretch_factor')==2,'EXPLICIT_REAL_DEV_CANDIDATE')
    validate_trained_candidate(c)
    return c


def validate_trained_candidate(c):
    trained=scene_contract.pinned(c['training_result']);root=Path(c['training_result']['path']).parent
    job=root.parent
    train_spec=json.loads((job/'EXECUTION_SPEC.json').read_bytes())
    lease=json.loads((job/'control/LEASE.json').read_bytes())
    worker=json.loads((job/'WORKER_EXIT.private.json').read_bytes())
    need(trained.get('schema')=='WM02_STABLE_TASK_ADAPTER_TRAIN_ACTUAL_V1'
         and trained.get('status')=='CANDIDATE_TRAINED_RELOADED_NOT_NATIVE_EVALUATED'
         and trained.get('TASK_ADAPTER_updates',0)>0 and trained.get('GR00T_updates')==0
         and trained.get('Actor_updates')==trained.get('WM_updates')==0
         and trained.get('optimizer_restored')is True and trained.get('sampler_restored')is True
         and trained.get('RNG_restored')is True and lease.get('status')=='CLOSED'
         and lease.get('worker_exit_code')==worker.get('exit_code')==0
         and lease.get('spec_sha256')==digest(job/'EXECUTION_SPEC.json')
         and train_spec.get('kind')=='TASK_ADAPTER'and train_spec.get('profile')=='wm02-pilot1-neural'
         and train_spec.get('TASK_ADAPTER_updates')==trained.get('TASK_ADAPTER_updates')
         and trained.get('export_sha256')==c['export']['sha256']
         and Path(c['export']['path'])==root/'TASK_ADAPTER_EXPORT.npz',
         'REAL_SEALED_NONZERO_TRAINING_AND_COMPLETE_RECOVERY')
    # Reject corrupt or substituted exports before starting the native prefix.
    policy.load_export(c['export'])
    return dict(result=trained,spec=train_spec,root=str(job))


def episode_config(spec):
    validate(spec)
    document=scene_contract.pinned(dict(path=spec['episode_manifest'],sha256=spec['episode_manifest_sha256']))
    episode=scene_contract.prefix_episode(document,spec)
    episode,source=common.operational.recipe_config(spec,episode,actual_phase=PHASE,
        actual_kind=KIND,source_members=SOURCE_MEMBERS)
    return episode,dict(source,stable_task_scene_reference=spec['stable_task_scene_reference'])


def episode_limit(spec,previous):
    episode_config(spec);common.audit_prefix_predecessor(spec,previous)
    return E

def validate_adapter_install(adapter, action):
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    need(adapter.spec.get('phase') == PHASE and adapter.spec.get('kind') == KIND, 'ACTUAL_NEW_DATA_PHASE')
    step = adapter.observe()['step']
    program = getattr(adapter, 'live_skill_program', None)
    if program is None:
        need(step < B, 'PREFIX_BEFORE_SUFFIX')
        return common.prefix.validate_install_preview(action, initial_tcp=adapter.trajectory_initial_tcp, program=adapter.trajectory_program, step=step, base=adapter.base_pose_hold, frozen_base=adapter.trajectory_initial_base, end_step=B)
    need(type(program) is TrajectoryAdaptation and B <= step < E and ((step - B) % 25 == 0) and (program.geometry_scale == 1.0) and (program.time_stretch_factor == 2) and np.array_equal(action, program.preview((step - B) // 25)) and np.array_equal(program.base_pose_hold, adapter.base_pose_hold) and np.array_equal(adapter.base_pose_hold, adapter.trajectory_initial_base), 'EXACT_FULL_H40_PREVIEW_AND_BASE')
    return dict(status='LEARNED_TASK_PREVIEW_REQUIRES_ORIGINAL_CERTIFICATE_AND_ACK', native_step=step, physics_advance_authorized=False)

def continue_native(adapter, report, samples, windows):
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .fusion_live_skill_segment import run_consumed_segment
    from .fusion_short_skill import _score_measured_samples
    c = validate(adapter.spec)
    ready = report['task_phase_initialization']['readiness']
    need(adapter.observe()['step'] == B and adapter.trace_sealed is False and all((v is True for (k, v) in ready['predicates'].items() if k != 'full_grade')) and (report['all_40_knots_consumed'] is True) and (len(windows) == (B - 2503) // 25), 'REAL_STABLE_NEW_LAYOUT_PREFIX')
    observation = capture_input(adapter)
    cube = np.asarray(samples[-1]['object_pose']).copy()
    arrays=policy.load_export(c['export'])
    delta,student=policy.predict_from_observation(adapter.spec,observation,arrays)
    program,proposal=policy.build_for_delta(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],adapter.base_pose_hold,delta)
    proposal.update(student=student,candidate_training_result=c['training_result'])
    proposal.update(data_parent=adapter.spec['data_parent'], data_role=adapter.spec['data_role'], student_observation=observation, expert_label_eligibility='DEV_CANDIDATE_NEVER_EXPERT_TRAINING_LABEL')
    path = adapter.output / 'STABLE_TASK_CANDIDATE_PROPOSAL.private.json'
    write_json(path, proposal)
    adapter.live_skill_program = program
    initial = dict(qpos=np.asarray(samples[-1]['qpos']).copy())

    def observe(sample):
        _noncontact(sample, cube)
        require_standing(sample, initial)
    (suffix, measured, actual_windows) = run_consumed_segment(adapter, program, start_step=B, end_step=E, past_samples=samples, past_windows=windows, sample_observer=observe)
    score = _score_measured_samples([s for s in measured if s['native_step'] >= B], common.score_contract(), expected_contract=common.score_contract())
    stable = all((v for (k, v) in score['predicates'].items() if k not in ('signed_progress', 'distance_reduction')))
    success = score['approach_step_success'] if adapter.spec['task_command'] != 'HOLD_OPEN' else stable and score['metrics']['maximum_TCP_excursion_m'] <= 0.002
    total = (E - 2503) // 25
    result = dict(copy.deepcopy(report), schema='WM02_STABLE_TASK_ADAPTER_NATIVE_EVALUATION_ACTUAL_V1', actual_spec_phase=PHASE, native_endpoint_step=E, stable_task_evaluation=c, data_role=adapter.spec['data_role'], data_parent=adapter.spec['data_parent'], actual_scene_sha256=digest(adapter.spec['model_path']), student_observation=observation, executed_reference_proposal_sha256=digest(path), desired_outcome_provenance=c['producer'], initialization_all40_knots_consumed=report['all_40_knots_consumed'], initialization_raw_knot_consumed=report['raw_knot_consumed'], initialization_adaptation=report.get('adaptation'), adaptation=program.evidence(), all_40_knots_consumed=False, raw_knot_consumed=[False] * 40, GR00T_raw_H40_consumed=False, capability_reference_all40_consumed=suffix['all_40_knots_consumed'], actual_reference_installations=len(suffix['install_attempts']), generated_segment=suffix, original_install_ACKs=list(report.get('original_install_ACKs', [])) + suffix['original_install_ACKs'], windows=actual_windows, complete_task_atoms=total, task_native_ticks=E - 2503, solver_target_confirmed_atoms=total, gripper_confirmed_atoms=total, actual_endpoint_TCP=np.asarray(measured[-1]['tcp']).tolist(), actual_VLA_commands=0, VLA_commands=0, VLA_action_installations=0, VLA_queries=0, VLA_inferences=0, actual_VLA_inferences=0, native_process_Processor_decodes=0, additional_action_decodes=0, unique_generated_action_chunks=0, WM_updates=0, GR00T_updates=0, Actor_updates=0, full_task_success=False, target_task_success=False, source_task_phase_equivalence=False, capability_score=score, hold_stability_pass=stable, task_condition_score_contract=task_score_contract(adapter.spec['task_command']), task_condition_success=bool(success), short_teacher_success=False, short_learned_policy_success=bool(adapter.spec['task_command'] != 'HOLD_OPEN' and score['approach_step_success']), expert_training_eligible=False, supervision_status='DEV_POLICY_EVALUATION_NOT_TRAINING_DATA', TASK_ADAPTER_inferences=1, TASK_ADAPTER_updates=0, candidate_training_result=c['training_result'], candidate_export=c['export'], raw_reference_is_GR00T=False)
    return jsonable(result)

def complete_consumption(spec, complete):
    validate(spec)
    suffix = complete.get('generated_segment') or {}
    return complete.get('stable_task_evaluation') == spec['stable_task_evaluation'] and complete.get('data_role') == spec['data_role'] and (complete.get('data_parent') == spec['data_parent']) and (complete.get('capability_reference_all40_consumed') is True) and (complete.get('GR00T_raw_H40_consumed') is False) and (complete.get('all_40_knots_consumed') is False) and (complete.get('raw_knot_consumed') == [False] * 40) and (suffix.get('all_40_knots_consumed') is True) and (suffix.get('raw_knot_consumed') == [True] * 40) and (suffix.get('complete_suffix_atoms') == len(suffix.get('windows', [])) == 176) and (suffix.get('solver_target_confirmed_atoms') == suffix.get('gripper_confirmed_atoms') == 176) and ((complete.get('student_observation') or {}).get('native_step') == B)

def run(adapter):
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    (episode, source) = episode_config(adapter.spec)
    return run_feedback_program(adapter, episode, source, recipe_module=common.prefix, actual_phase=PHASE, report_schema='WM02_EVALUATED_STABLE_TASK_CALIBRATION_PREFIX_V1', goal_schema='WM02_EVALUATED_STABLE_TASK_CALIBRATION_GOAL_V1', decision_schema='WM02_EVALUATED_STABLE_TASK_CALIBRATION_DECISION_V1', feedback_report_schema='WM02_EVALUATED_STABLE_TASK_CALIBRATION_FEEDBACK_V1', continuation=continue_native)
