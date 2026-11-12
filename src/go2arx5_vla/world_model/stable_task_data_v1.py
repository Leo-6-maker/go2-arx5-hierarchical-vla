"""New TRAIN/DEV target episodes on statically checked real cube layouts.

Only submitted and solver/gripper-consumed commands can become supervision.
The final CLOSED0 data exporter supplies that eligibility, not this live loop.
"""
import copy
from pathlib import Path
import uuid
import numpy as np

from scripts.pi0292.relay_protocol import digest as document_hash, unseal
from . import stable_task_scene_v1 as scene_contract
from .stable_task_scene_v1 import PHASE,KIND,B,E,need
from . import stable_start_short_task_v1 as common
from .storage import digest,write_json,jsonable

SOURCE_MEMBERS = (*common.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/stable_task_scene_v1.py',
    'src/go2arx5_vla/world_model/stable_task_layout_static_v1.py',
    'src/go2arx5_vla/world_model/stable_task_data_v1.py',
    'src/go2arx5_vla/world_model/stable_task_adapter_v1.py',
    'src/go2arx5_vla/world_model/target_gr00t_collection_v1.py',
    'scripts/wm02/stable_task_data_check_v1.py')


def config(command,role):
    from .stable_task_adapter_v1 import COMMANDS
    need(command in COMMANDS and role in ('TRAIN','DEV'),'NEW_DATA_TASK_AND_ROLE')
    return dict(schema='WM02_STABLE_TASK_EXPERT_DATA_CONFIG_V1',task_command=command,
        method='HOLD' if command == 'HOLD_OPEN' else 'GEOMETRIC_TEACHER_4MM',
        producer='PRIVILEGED_NEW_TARGET_TASK_EXPERT',step_m=0. if command == 'HOLD_OPEN' else .004,
        suffix_atoms=176,minimum_course_clearance_m=.14,data_role=role,
        GR00T_queries=0,model_updates=0)


def validate(spec):
    scene_contract.scene(spec)
    if spec.get('target_gr00t_collection') is not None:
        from .target_gr00t_collection_v1 import validate as validate_vla
        c = validate_vla(spec)
    else:
        c = config(spec.get('task_command'),spec.get('data_role'))
    need(spec.get('stable_task_data') == c
         and spec.get('snapshot_binding_mode') == 'PRE_REGISTERED_STABLE_TASK_DATA_V1'
         and not spec.get('paused_query_pair') and not spec.get('paused_query_gpu_spec')
         and spec.get('live_skill_geometry_scale') == 1.
         and spec.get('live_skill_time_stretch_factor') == 2, 'EXPLICIT_NEW_CPU_TASK_DATA_WITHOUT_QUERY')
    return c


def task_score_contract(command):
    need(command in ('HOLD_OPEN','APPROACH_RED_4MM_OPEN'),'KNOWN_TASK_SCORE')
    return dict(schema='WM02_STABLE_TASK_CONDITION_SCORE_V1',task_command=command,
        original_short_contract=common.score_contract(),
        condition=('ORIGINAL_STRICT_SHORT_APPROACH' if command!='HOLD_OPEN' else
                   'ORIGINAL_NONPROGRESS_PREDICATES_AND_MAX_TCP_EXCURSION_2MM'),
        HOLD_maximum_TCP_excursion_m=.002,original_native_safety_changed=False)


def episode_config(spec):
    validate(spec)
    document = scene_contract.pinned(dict(path=spec['episode_manifest'],sha256=spec['episode_manifest_sha256']))
    episode = scene_contract.prefix_episode(document,spec)
    episode,source = common.operational.recipe_config(spec,episode,actual_phase=PHASE,
        actual_kind=KIND,source_members=SOURCE_MEMBERS)
    return episode,dict(source,stable_task_scene_reference=spec['stable_task_scene_reference'])


def episode_limit(spec,previous):
    episode_config(spec);common.audit_prefix_predecessor(spec,previous)
    return E


def validate_adapter_install(adapter,action):
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    need(adapter.spec.get('phase') == PHASE and adapter.spec.get('kind') == KIND,'ACTUAL_NEW_DATA_PHASE')
    step = adapter.observe()['step']; program = getattr(adapter,'live_skill_program',None)
    if program is None:
        need(step < B,'PREFIX_BEFORE_SUFFIX')
        return common.prefix.validate_install_preview(action,initial_tcp=adapter.trajectory_initial_tcp,
            program=adapter.trajectory_program,step=step,base=adapter.base_pose_hold,
            frozen_base=adapter.trajectory_initial_base,end_step=B)
    need(type(program) is TrajectoryAdaptation and B <= step < E and (step-B)%25 == 0
         and program.geometry_scale == 1. and program.time_stretch_factor == 2
         and np.array_equal(action,program.preview((step-B)//25))
         and np.array_equal(program.base_pose_hold,adapter.base_pose_hold)
         and np.array_equal(adapter.base_pose_hold,adapter.trajectory_initial_base),
         'EXACT_FULL_H40_PREVIEW_AND_BASE')
    return dict(status='NEW_TASK_PREVIEW_REQUIRES_ORIGINAL_CERTIFICATE_AND_ACK',
                native_step=step,physics_advance_authorized=False)


def capture_input(adapter, *, language=None, task_score=None):
    from .fusion_observation import TargetShadowBinding
    from .fusion_source_camera_capture import SourceCameraCapture
    from .fusion_source_camera_rig import SOURCE_IDENTITY_SHA256
    from .fusion_paused_observation import capture_paused_observation,verify_still_paused
    spec = adapter.spec
    language = language or ('Keep the gripper open and hold its current pose without touching the red cube.'
        if spec['task_command'] == 'HOLD_OPEN' else
        'Move the open gripper a little toward the red cube, then hold without touching it.')
    task = dict(schema='WM02_FUSION_PAUSED_TARGET_TASK_V2',task_id='STABLE_TASK_'+spec['task_command'],
        language=language,scene_sha256=digest(spec['model_path']),
        score_contract_sha256=document_hash(task_score if task_score is not None else task_score_contract(spec['task_command'])),language_provenance='NEW_TARGET_SHORT_SKILL_PROMPT')
    frame = spec['task_phase_source_from_target']
    from .native_body_camera_capture_v1 import session_configuration
    if session_configuration(spec):
        from .native_body_camera_capture_v1 import BodyBaseCameraCapture
        camera = BodyBaseCameraCapture(frame)
    elif spec.get('task_learning_role')in('NATIVE_BODY_SFT2000_TRAIN4_DIAGNOSTIC_V1','A_TARGET500_TRAIN4_DELIVERY_DIAGNOSTIC_V1','SLA_TARGET500_NATIVE_EVALUATION_V1'):
        from .native_body_camera_capture_v1 import BodyBaseCameraCapture
        from .native_camera_transfer_v1 import CAMERA_ID
        from .task_learning_grasp_v1 import native_body_profile
        need(spec.get('observation_camera_profile')==CAMERA_ID,'EXPLICIT_NATIVE_BODY_OBSERVATION_PROFILE')
        native_body_profile(spec['task_learning_role']).native_scene_admission(spec)
        camera = BodyBaseCameraCapture(frame)
    else:
        identity = Path(spec['source_camera_identity']).read_bytes()
        need(digest(spec['source_camera_identity']) == spec['source_camera_identity_sha256'] == SOURCE_IDENTITY_SHA256,
             'UNCHANGED_CAMERA_CALIBRATION_IDENTITY')
        camera = SourceCameraCapture(identity,frame)
    binding = TargetShadowBinding('MUJOCO_GO2_ARX5_WORLD',digest(spec['model_path']),
        digest(Path(spec['registered_root'])/'EXECUTION_SPEC.json'),frame,spec['task_phase_frame_map_sha256'])
    from .native_body_memory_v3 import configuration as memory_configuration
    if memory_configuration(spec):
        from .native_body_memory_v3 import capture_input as capture_memory
        return capture_memory(adapter,binding=binding,task=task,task_sha256=document_hash(task))
    query = str(uuid.uuid4())
    record = capture_paused_observation(adapter,binding=binding,task=task,
        task_sha256=document_hash(task),query_uuid=query,capture_index=0,camera_recipe=camera)
    pause = verify_still_paused(adapter,record)
    path = adapter.output/'PAUSED_OBSERVATIONS'/query/'PAUSED_OBSERVATION.private.json'
    body = unseal(record)
    return dict(path=str(path),sha256=digest(path),native_step=body['native_step'],
        native_time_s=body['native_time_s'],state_sha256=body['files']['STATE.private.bin'],
        pause_revalidated=pause,task_command=spec['task_command'],GR00T_queries=0,
        actual_cube_truth_is_student_input=False)


def continue_native(adapter,report,samples,windows):
    if adapter.spec.get('target_gr00t_collection') is not None:
        from .target_gr00t_collection_v1 import continue_native as collect_vla
        return collect_vla(adapter,report,samples,windows)
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .fusion_live_skill_segment import run_consumed_segment
    from .fusion_short_skill import _score_measured_samples
    c = validate(adapter.spec); ready = report['task_phase_initialization']['readiness']
    need(adapter.observe()['step'] == B and adapter.trace_sealed is False
         and all(v is True for k,v in ready['predicates'].items() if k != 'full_grade')
         and report['all_40_knots_consumed'] is True and len(windows) == (B-2503)//25,
         'REAL_STABLE_NEW_LAYOUT_PREFIX')
    observation = capture_input(adapter)
    cube = np.asarray(samples[-1]['object_pose']).copy()
    program,proposal = common.build_for_config(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
        adapter.base_pose_hold,cube,c)
    proposal.update(data_parent=adapter.spec['data_parent'],data_role=adapter.spec['data_role'],
                    student_observation=observation,expert_label_eligibility='ONLY_AFTER_REAL_FULL_CONSUMPTION_AND_CLOSED0')
    path = adapter.output/'STABLE_TASK_EXPERT_PROPOSAL.private.json';write_json(path,proposal)
    adapter.live_skill_program = program;initial = dict(qpos=np.asarray(samples[-1]['qpos']).copy())
    def observe(sample):
        _noncontact(sample,cube);require_standing(sample,initial)
    suffix,measured,actual_windows = run_consumed_segment(adapter,program,start_step=B,end_step=E,
        past_samples=samples,past_windows=windows,sample_observer=observe)
    score = _score_measured_samples([s for s in measured if s['native_step'] >= B],
        common.score_contract(),expected_contract=common.score_contract())
    stable = all(v for k,v in score['predicates'].items() if k not in ('signed_progress','distance_reduction'))
    success = (score['approach_step_success'] if c['method'] != 'HOLD' else
               stable and score['metrics']['maximum_TCP_excursion_m'] <= .002)
    total = (E-2503)//25
    result = dict(copy.deepcopy(report),schema='WM02_STABLE_TASK_DATA_NATIVE_ACTUAL_V1',
        actual_spec_phase=PHASE,native_endpoint_step=E,stable_task_data=c,
        data_role=adapter.spec['data_role'],data_parent=adapter.spec['data_parent'],
        actual_scene_sha256=digest(adapter.spec['model_path']),student_observation=observation,
        executed_reference_proposal_sha256=digest(path),desired_outcome_provenance=c['producer'],
        initialization_all40_knots_consumed=report['all_40_knots_consumed'],
        initialization_raw_knot_consumed=report['raw_knot_consumed'],
        initialization_adaptation=report.get('adaptation'),adaptation=program.evidence(),
        all_40_knots_consumed=False,raw_knot_consumed=[False]*40,GR00T_raw_H40_consumed=False,
        capability_reference_all40_consumed=suffix['all_40_knots_consumed'],
        actual_reference_installations=len(suffix['install_attempts']),generated_segment=suffix,
        original_install_ACKs=list(report.get('original_install_ACKs',[]))+suffix['original_install_ACKs'],
        windows=actual_windows,complete_task_atoms=total,task_native_ticks=E-2503,
        solver_target_confirmed_atoms=total,gripper_confirmed_atoms=total,
        actual_endpoint_TCP=np.asarray(measured[-1]['tcp']).tolist(),
        actual_VLA_commands=0,VLA_commands=0,VLA_action_installations=0,VLA_queries=0,VLA_inferences=0,
        actual_VLA_inferences=0,native_process_Processor_decodes=0,additional_action_decodes=0,
        unique_generated_action_chunks=0,WM_updates=0,GR00T_updates=0,Actor_updates=0,
        full_task_success=False,target_task_success=False,source_task_phase_equivalence=False,
        capability_score=score,hold_stability_pass=stable,
        task_condition_score_contract=task_score_contract(adapter.spec['task_command']),
        task_condition_success=bool(success),short_teacher_success=bool(c['method']!='HOLD' and score['approach_step_success']),
        expert_training_eligible=False,supervision_status='PENDING_FINAL_TRACE_AND_CLOSED_WORKER_PROOF',
        raw_reference_is_GR00T=False)
    return jsonable(result)


def complete_consumption(spec,complete):
    validate(spec);suffix = complete.get('generated_segment') or {}
    return (complete.get('stable_task_data') == spec['stable_task_data']
        and complete.get('data_role') == spec['data_role'] and complete.get('data_parent') == spec['data_parent']
        and complete.get('capability_reference_all40_consumed') is True
        and complete.get('GR00T_raw_H40_consumed') is False
        and complete.get('all_40_knots_consumed') is False and complete.get('raw_knot_consumed') == [False]*40
        and suffix.get('all_40_knots_consumed') is True and suffix.get('raw_knot_consumed') == [True]*40
        and suffix.get('complete_suffix_atoms') == len(suffix.get('windows',[])) == 176
        and suffix.get('solver_target_confirmed_atoms') == suffix.get('gripper_confirmed_atoms') == 176
        and (complete.get('student_observation') or {}).get('native_step') == B)


def run(adapter):
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    episode,source = episode_config(adapter.spec)
    return run_feedback_program(adapter,episode,source,recipe_module=common.prefix,actual_phase=PHASE,
        report_schema='WM02_NEW_STABLE_TASK_CALIBRATION_PREFIX_V1',
        goal_schema='WM02_NEW_STABLE_TASK_CALIBRATION_GOAL_V1',
        decision_schema='WM02_NEW_STABLE_TASK_CALIBRATION_DECISION_V1',
        feedback_report_schema='WM02_NEW_STABLE_TASK_CALIBRATION_FEEDBACK_V1',continuation=continue_native)
