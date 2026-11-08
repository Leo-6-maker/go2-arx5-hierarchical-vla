"""Explicit graded diagnostic composition; old half-grade readiness stays false.

The on-disk V9 episode describes only the slower V9 initialization prefix.
The actual LIVE V9 scope separately preregisters capture B and terminal E.
No source80mm equivalence, task success, motion or GPU grant is supplied here.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

from scripts.pi0292.relay_protocol import digest as document_hash, unseal
from scripts.pi0294.dec1.target_observation import transform_hash
from . import fusion_task_phase_pose_slow_feedback_v9 as prefix
from . import fusion_task_phase_pose_continuation_v7 as previous_recipe
from . import fusion_task_phase_pose_continuation_contract_v7 as predecessor
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_live_proposal_qualification_contract as common
from .fusion_task_phase_xyz_feedback_evidence import audit_completed_feedback
from .storage import digest

PHASE='collect-target-fusion-task-phase-live-skill-v9'
KIND='FUSION_TASK_PHASE_LIVE_SKILL_V9'
B,E=34303,66503
ROLE='GRADED_NONCONTACT_DIAGNOSTIC'
TASK_CONDITION='SOURCE71558_ORIGINAL_FULL_TASK_NO_PARAPHRASE'
SHORT_TASK_CONDITION='GRADED_TARGET_SHORT_APPROACH_HOLD_AT34303_V1'
TRACKING_TASK_CONDITION='GRADED_SHORT_LANGUAGE_FULL_H40_TRACKING_DIAGNOSTIC_AT34303_V1'
MECHANICAL_TASK_CONDITION='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1'
from .fusion_task_reference_diagnostic_v1 import TASK_CONDITION as TEACHER_DIAGNOSTIC_TASK_CONDITION
from .fusion_task_reference_diagnostic_v1 import SHORT_TASK_CONDITION as SHORT_TEACHER_DIAGNOSTIC_TASK_CONDITION
SHORT_LANGUAGE_TASK_CONDITIONS=(SHORT_TASK_CONDITION,TRACKING_TASK_CONDITION,MECHANICAL_TASK_CONDITION,TEACHER_DIAGNOSTIC_TASK_CONDITION,SHORT_TEACHER_DIAGNOSTIC_TASK_CONDITION)
from .fusion_source_phase_c01 import CONDITION as C01_QUERY_CONDITION,CPU_CONDITION as C01_CPU_CONDITION,CONDITIONS as C01_CONDITIONS
TASK_CONDITIONS=(TASK_CONDITION,*SHORT_LANGUAGE_TASK_CONDITIONS,C01_QUERY_CONDITION,C01_CPU_CONDITION)
PREDECESSOR_SOURCE='cdced494d46bd4df1930142317244b2da1ba9229'
PREDECESSOR_SPEC='79e5159de909c1d128bb14adc02ba61a2e0a83a142c1fb42d4b7e90c32d9ddad'
PREDECESSOR_RESULT='1e1b0c75626cefeaf89f417cb0c1d276ec14fbc4eb473cf2b64fc238581581ae'
SOURCE_MEMBERS=tuple(dict.fromkeys((*predecessor.SOURCE_MEMBERS,
 'src/go2arx5_vla/world_model/fusion_task_phase_pose_slow_feedback_v9.py',
 'src/go2arx5_vla/world_model/fusion_task_phase_live_skill_native_v8.py',
 'src/go2arx5_vla/world_model/fusion_task_phase_live_skill_contract_v9.py',
 'src/go2arx5_vla/world_model/fusion_task_phase_live_skill_native_v9.py',
 'src/go2arx5_vla/world_model/fusion_task_phase_live_skill_pair_v9.py',
 'src/go2arx5_vla/world_model/fusion_live_skill_segment.py',
 'src/go2arx5_vla/world_model/fusion_paused_query_bridge.py',
 'src/go2arx5_vla/world_model/fusion_paused_inference.py',
 'src/go2arx5_vla/world_model/fusion_paused_observation.py',
 'src/go2arx5_vla/world_model/fusion_source_camera_capture.py',
 'src/go2arx5_vla/world_model/fusion_source_camera_rig.py',
 'src/go2arx5_vla/world_model/fusion_trajectory_adapter.py',
 'src/go2arx5_vla/world_model/fusion_graded_short_skill_v1.py',
 'src/go2arx5_vla/world_model/fusion_bounded_reference_tracking_v1.py',
 'src/go2arx5_vla/world_model/fusion_program_forecast_v1.py',
 'src/go2arx5_vla/world_model/fusion_program_response_numpy_v1.py',
 'src/go2arx5_vla/world_model/fusion_program_response_data_v1.py',
 'src/go2arx5_vla/world_model/fusion_live_program_replay_v1.py',
 'src/go2arx5_vla/world_model/fusion_live_causal_features_v1.py',
 'src/go2arx5_vla/world_model/fusion_live_causal_features_v2.py',
 'src/go2arx5_vla/world_model/fusion_program_reference_intervention_v1.py',
 'src/go2arx5_vla/world_model/fusion_program_reference_selection_v1.py',
 'src/go2arx5_vla/world_model/fusion_task_reference_diagnostic_v1.py',
 'src/go2arx5_vla/world_model/fusion_source_phase_c01.py',
 'src/go2arx5_vla/world_model/fusion_source_phase_c01_capability_v1.py',
 'src/go2arx5_vla/world_model/fusion_short_skill.py',
 'scripts/wm02/fusion_task_phase_live_skill_check_v9.py')))

def need(ok,why):
    if not ok:raise ValueError('TASK_PHASE_LIVE_SKILL_V9_'+why)

def score_contract(*,time_stretch_factor=2,task_condition=TASK_CONDITION):
    # Keep the original time2 document byte-equivalent. A new time1 run binds
    # its distinct task/score hashes before capture, never after a rejection.
    need(type(time_stretch_factor)is int and time_stretch_factor in(1,2),
         'EXPLICIT_ORIGINAL_ADAPTER_TIME1_OR_TIME2')
    need(task_condition in TASK_CONDITIONS,'EXPLICIT_FULL_OR_NEW_SHORT_TASK')
    if task_condition==C01_CPU_CONDITION:
        from .fusion_source_phase_c01_capability_v1 import score_contract as c01_score
        need(time_stretch_factor==2,'C01_CPU_SAME_TIME2_ADAPTER')
        return c01_score()
    if task_condition==C01_QUERY_CONDITION:
        need(time_stretch_factor==2,'C01_SAME_TIME2_ADAPTER')
        return dict(schema='WM02_SOURCE_PHASE_C01_INITIALIZATION_AND_QUERY_ONLY_SCORE_V1',
            task_condition=task_condition,capture_boundary_step=B,registered_final_step=B,
            generated_action_execution_allowed=False,task_success_established=False,
            full_H40_saved=True,source_task_phase_equivalence=False,old_140mm_domain_reclassified=False,
            new_capture_phase_distance_bounds_m=[.07,.09],old_initialization_distance_predicate_not_relabelled=True)
    value=dict(schema=('WM02_GRADED_FULL_TASK_PROPOSAL_DIAGNOSTIC_SCORE_V9'
        +('_TIME1' if time_stretch_factor==1 else '')),
        purpose='FULL_GENERATED_H40_NATIVE_TRACKING_AND_INTENT_DIAGNOSIS_NOT_FULL_TASK_SCORE',
        task_condition=task_condition,capture_boundary_step=B,registered_final_step=terminal_step(task_condition),
        initialization_scale=.5,source_task_phase_equivalence=False,
        prefix_full_grade_required=False,old_prefix_query_ready_reclassified=False,
        initialization_recipe='12_SLOTS_MOVE28_SETTLE8_SAME_B34303',
        geometry_scale=1.,time_stretch_factor=time_stretch_factor,raw_knots_required=40,
        minimum_tail_atoms=8,extra_budget='FINAL_HOLD_ONLY_NO_TRAJECTORY_CLIP',
        candidate_minimum_TCP_cube_distance_m=.14,
        candidate_gripper_rule='ORIGINAL_MECHANICAL_TO_WORKING_MAPPING_ALL40_UNCHANGED',
        actual_contact_allowed=False,object_motion_allowed=False,
        measured_tracking_error_is_prediction_innovation=False,
        full_task_success_claim=False,native_safety_thresholds_changed=False)
    if task_condition in SHORT_LANGUAGE_TASK_CONDITIONS:
        from . import fusion_graded_short_skill_v1 as short
        actual=short.score_contract()
        value.update(schema=value['schema']+'_SHORT',
            purpose='FULL_GENERATED_SHORT_INTENT_AND_MEASURED_APPROACH_STEP_AT34303_NOT_PICK_PLACE',
            actual_short_skill_score_contract=actual,
            actual_short_skill_score_contract_sha256=document_hash(actual))
        if task_condition in(TEACHER_DIAGNOSTIC_TASK_CONDITION,SHORT_TEACHER_DIAGNOSTIC_TASK_CONDITION):
            from .fusion_task_reference_diagnostic_v1 import CONFIG,SHORT_CONFIG
            value.update(schema=value['schema']+'_PRIVILEGED_GEOMETRY',
                purpose='PRIVILEGED_GEOMETRIC_TASK_INTENT_DIAGNOSTIC_NOT_GR00T_OR_WM_TRANSFER',
                task_reference_diagnostic=SHORT_CONFIG if task_condition==SHORT_TEACHER_DIAGNOSTIC_TASK_CONDITION else CONFIG,GR00T_proposal_saved_but_not_executed=True)
        if task_condition==TRACKING_TASK_CONDITION:
            value.update(schema=value['schema']+'_TRACKING_DIAGNOSTIC',
                purpose='EXPLORATORY_FULL_H40_TRACKING_WITH_UNCHANGED_STRICT_SHORT_SCORE_REPORTED_SEPARATELY',
                strict_short_lateral_proposal_failure_is_not_tracking_admission_rejection=True,
                tracking_admission='OPEN_TOWARD_BOUNDED_NONCONTACT_THEN_ORIGINAL_NATIVE_CERTIFICATES',
                strict_short_task_success_required_for_native_engineering_result=False,
                strict_short_thresholds_changed=False)
        if task_condition==MECHANICAL_TASK_CONDITION:
            from .fusion_bounded_reference_tracking_v1 import domain_contract
            value.update(schema=value['schema']+'_MECHANICAL_ONLY',
                purpose='BOUNDED_FULL_H40_MECHANICAL_CONSUMPTION_ONLY_ALL_SEMANTIC_FAILURES_RETAINED',
                mechanical_domain=domain_contract(),strict_short_thresholds_changed=False,
                task_intent_is_reported_not_used_to_claim_mechanical_qualification=True)
    return value

def task_document(scene_sha256,language,*,time_stretch_factor=2,task_condition=TASK_CONDITION):
    from .fusion_paused_observation import TASK_SCHEMA
    score=score_contract(time_stretch_factor=time_stretch_factor,task_condition=task_condition)
    is_short=task_condition in SHORT_LANGUAGE_TASK_CONDITIONS
    if is_short:
        from .fusion_graded_short_skill_v1 import LANGUAGE,TASK_ID,TRACKING_TASK_ID
        need(language==LANGUAGE,'EXACT_DISTINCT_SHORT_TASK_LANGUAGE')
    task_id=((TRACKING_TASK_ID if task_condition==TRACKING_TASK_CONDITION else TASK_ID) if is_short else 'ORIGINAL71558_FULL_TASK_AT_GRADED_TARGET_STATE_V9'
        +('_TIME1' if time_stretch_factor==1 else ''))
    if task_condition==MECHANICAL_TASK_CONDITION:
        from .fusion_bounded_reference_tracking_v1 import TASK_ID as mechanical_task
        task_id=mechanical_task
    if task_condition in(TEACHER_DIAGNOSTIC_TASK_CONDITION,SHORT_TEACHER_DIAGNOSTIC_TASK_CONDITION):
        task_id=('FUSION_PRIVILEGED_GEOMETRIC_SHORT_DURATION_DIAGNOSTIC_AT34303_V2'if task_condition==SHORT_TEACHER_DIAGNOSTIC_TASK_CONDITION else'FUSION_PRIVILEGED_GEOMETRIC_APPROACH_DIAGNOSTIC_AT34303_V1')
    if task_condition==C01_CPU_CONDITION:
        from .fusion_short_skill import LANGUAGE
        need(language==LANGUAGE,'EXPLICIT_C01_CAPABILITY_LANGUAGE')
        task_id='C01_CPU_SMALL_APPROACH_CAPABILITY_V1'
    if task_condition==C01_QUERY_CONDITION:task_id='FUSION_SOURCE_PHASE_C01_ORIGINAL_LANGUAGE_QUERY_ONLY_V1'
    return dict(schema=TASK_SCHEMA,
        task_id=task_id,language=language,
        scene_sha256=scene_sha256,score_contract_sha256=document_hash(score),
        language_provenance='EXPLICIT_PRIVILEGED_C01_CAPABILITY_DIAGNOSTIC'if task_condition==C01_CPU_CONDITION else'NEW_TARGET_SHORT_SKILL_PROMPT'if is_short else'EXACT_SOURCE_TASK_PROMPT')

def recipe_config(spec):
    from .fusion_task_reference_diagnostic_v1 import validate_registration as validate_task_reference
    validate_task_reference(spec)
    from .fusion_program_forecast_v1 import validate_registration
    validate_registration(spec)
    from .fusion_program_reference_intervention_v1 import validate_scope
    validate_scope(spec)
    if spec.get('program_reference_selection') is not None:
        from .fusion_program_reference_selection_v1 import validate_registration as validate_selection
        validate_selection(spec)
    from .fusion_task_phase_live_skill_pair_v9 import PAIR_MODE
    from .fusion_source_posture_paused_query_v3 import SOURCE_CAMERA_IDENTITY,SOURCE_CAMERA_RECIPE
    from .fusion_source_camera_capture import SourceCameraCapture
    document=common._json(spec['episode_manifest'],spec['episode_manifest_sha256'])
    if spec.get('task_condition')in C01_CONDITIONS:
        from .fusion_source_phase_c01 import scene,prefix_episode
        scene(spec);episode=prefix_episode(document,spec)
    else:episode=prefix.episode_contract(document)
    need(spec.get('phase')==PHASE and spec.get('kind')==KIND
         and spec.get('experiment_role')==ROLE and spec.get('task_condition') in TASK_CONDITIONS
         and episode['geometry_scale']==.5 and prefix.endpoint(episode)==B
         and type(spec.get('live_skill_start_step'))is int and spec['live_skill_start_step']==B
         and type(spec.get('max_ticks'))is int and spec['max_ticks']==terminal_step(spec['task_condition'])
         and spec.get('snapshot_binding_mode')==('PRE_REGISTERED_C01_CPU_CAPABILITY_V1'if spec.get('task_condition')==C01_CPU_CONDITION else PAIR_MODE)
         and type(spec.get('live_skill_geometry_scale'))in(int,float) and spec['live_skill_geometry_scale']==1.
         and type(spec.get('live_skill_time_stretch_factor'))is int and spec['live_skill_time_stretch_factor'] in(1,2),
         'EXPLICIT_HALF_PREFIX_B_DISTINCT_FULL_H40_SUFFIX_E_NO_SCALE_CLIP')
    episode,source=operational.recipe_config(spec,episode,actual_phase=PHASE,actual_kind=KIND,
                                            source_members=SOURCE_MEMBERS)
    if spec['task_condition']in C01_CONDITIONS:
        binding=spec['source_phase_scene_binding']
        source=dict(source,nominal_calibration_reference=dict(path=binding['nominal_calibration_goal_binding'],sha256=binding['nominal_calibration_goal_binding_sha256']))
    language=unseal(json.loads(source['source_observation_bytes']))['payload']['language']
    condition=spec['task_condition']
    if condition in SHORT_LANGUAGE_TASK_CONDITIONS or condition==C01_CPU_CONDITION:
        from .fusion_graded_short_skill_v1 import LANGUAGE
        language=LANGUAGE
    task=common._json(spec['paused_target_task'],spec['paused_target_task_file_sha256'])
    score=common._json(spec['paused_score_contract'],spec['paused_score_contract_file_sha256'])
    factor=spec['live_skill_time_stretch_factor']
    need(task==task_document(digest(spec['model_path']),language,time_stretch_factor=factor,task_condition=condition)
         and score==score_contract(time_stretch_factor=factor,task_condition=condition)
         and document_hash(task)==spec.get('paused_target_task_sha256'),
         'EXACT_SOURCE_FULL_LANGUAGE_AND_DISTINCT_VERSIONED_DIAGNOSTIC_SCORE')
    need(transform_hash(spec['paused_source_from_target'])==spec.get('paused_frame_map_sha256')
         ==prefix.original.FRAME_MAP
         and np.array_equal(np.asarray(spec['paused_source_from_target']),source['T_source_from_target']),
         'UNCHANGED_FC347_TASK_AND_CAPTURE_FRAME')
    path=Path(spec['source_camera_identity']);raw=path.read_bytes()
    need(spec.get('capture_camera_recipe')==SOURCE_CAMERA_RECIPE
         and spec.get('source_camera_identity_sha256')==SOURCE_CAMERA_IDENTITY
         and not path.is_symlink() and path.is_absolute() and path.resolve()==path
         and hashlib.sha256(raw).hexdigest()==SOURCE_CAMERA_IDENTITY,
         'EXACT_DECLARED_SOURCE_CAMERA_RIG_NOT_EQUIVALENCE')
    camera=SourceCameraCapture(raw,source['T_source_from_target'])
    return episode,source,task,camera

def episode_config(spec):
    config=recipe_config(spec)
    from .fusion_paused_query_bridge import preflight_pair
    if spec.get('task_condition')==C01_CPU_CONDITION:
        from .fusion_source_phase_c01_capability_v1 import validate
        validate(spec)
    else:preflight_pair(Path(spec['paused_query_gpu_spec']),spec)
    return config

def episode_limit(spec,previous):
    episode_config(spec)
    path=Path(spec['previous_stage'])
    actual=common._json(path,spec['previous_stage_sha256'])
    prior,pin=common._closed(path.parent.parent)
    need(actual==previous and spec['previous_stage_sha256']==PREDECESSOR_RESULT
         and pin==PREDECESSOR_SPEC and prior['source_sha']==PREDECESSOR_SOURCE
         and prior['phase']==actual['phase']==previous_recipe.PHASE,
         'EXACT_CLOSED_QUALIFIED_HALF_EVIDENCE_NOT_REOPENED_PERMISSION')
    p=previous_recipe.episode_contract(common._json(prior['episode_manifest'],prior['episode_manifest_sha256']))
    need(p['geometry_scale']==.5 and previous_recipe.endpoint(p)==B,'EXACT_REAL_HALF_PREFIX_NOT_FULL1')
    checked=audit_completed_feedback(prior,actual,recipe=previous_recipe)
    need(checked['all_nine_non_full_grade_predicates_passed']is True
         and checked['readiness']['query_ready']is False
         and checked['readiness']['predicates']['full_grade']is False,
         'KEEP_OLD_HALF_QUERY_READY_FALSE_AND_MEASURED_NINE_PREDICATES')
    return terminal_step(spec['task_condition'])


def terminal_step(task_condition):
    if task_condition==C01_CPU_CONDITION:return 38703
    if task_condition==C01_QUERY_CONDITION:return B
    if task_condition==SHORT_TEACHER_DIAGNOSTIC_TASK_CONDITION:
        from .fusion_task_reference_diagnostic_v1 import endpoint
        return endpoint(task_condition)
    return E
