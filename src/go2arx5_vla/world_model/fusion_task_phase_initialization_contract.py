"""Read-only admission recipe for a new, finite task-phase calibration.

The static episode contains no future cube measurement or runtime goal hash.
Actual AUTO1/P1 ownership and native safety are still checked by the original
worker. A separately persisted 3703 goal binding is created only by that worker.
"""
import hashlib
import json
from pathlib import Path

from scripts.pi0292.relay_protocol import unseal, validate_observation, digest as document_hash
from scripts.pi0294.dec1.target_observation import rigid, transform_hash
from . import fusion_task_phase_initialization as recipe
from . import fusion_live_proposal_qualification_contract as common
from . import fusion_trajectory_qualification_contract as geometry
from .storage import digest

PHASE = 'collect-target-fusion-task-phase-initialization-v1'
KIND = 'FUSION_TASK_PHASE_INITIALIZATION'
SCALES = tuple(recipe.RAMPS)
FIRST_PREDECESSOR = '82c5590d660202162802589916cb6361d691d21fcf74c2cce03ca63bbdacae46'
FIRST_PREDECESSOR_SPEC = 'c1a0878ae26ef1ea895cd2eb65d2ba938ecd792b04c314304c5787572f84033f'
FIRST_PREDECESSOR_SOURCE = '9459b60263c24f3b998b29e257a0aa35a9082f34'
TRACE_BUILD = common.TRACE_BUILD
SOURCE_MEMBERS = tuple(dict.fromkeys((*common.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_source_posture_qualification_contract.py',
    'src/go2arx5_vla/world_model/fusion_source_posture_paused_query_v3.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_initialization.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_initialization_contract.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_initialization_native.py',
    'scripts/wm02/fusion_task_phase_initialization_check.py')))


def need(value, reason):
    if not value:
        raise ValueError('TASK_PHASE_INITIALIZATION_NATIVE_' + reason)


def _json(path, pin=None):
    return common._json(path, pin)


def _source(spec):
    path = Path(spec['task_phase_source_observation'])
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
         'ORIGINAL_SOURCE_CANONICAL_PRIVATE_FILE')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == spec.get('task_phase_source_observation_sha256')
         == recipe.SOURCE_OBSERVATION, 'EXACT_ORIGINAL_71558_BYTES')
    payload = unseal(json.loads(raw))['payload']
    validate_observation(payload)
    need(document_hash(payload) == recipe.SOURCE_PAYLOAD
         and document_hash(payload['state_eef10']) == recipe.SOURCE_STATE,
         'EXACT_ORIGINAL_SOURCE_POLICY_STATE')
    frame = rigid(spec['task_phase_source_from_target'])
    need(transform_hash(frame) == spec.get('task_phase_frame_map_sha256') == recipe.FRAME_MAP,
         'UNCHANGED_EXPLICIT_FC347_FRAME')
    return dict(source_observation_bytes=raw, T_source_from_target=frame,
                scene_sha256=digest(spec['model_path']))


def episode_config(spec):
    """Check immutable inputs without constructing a native model or a grant."""
    return recipe_config(spec)


def recipe_config(spec, episode=None, *, actual_phase=PHASE, actual_kind=KIND, source_members=SOURCE_MEMBERS):
    """Reuse immutable operational checks with an explicitly checked new recipe.

    Callers retain their actual spec and episode identities; this is no phase
    substitution and gives no execution grant. The V1 entry above stays exact.
    """
    need((actual_phase, actual_kind) in ((PHASE, KIND),
         ('collect-target-fusion-task-phase-bias-calibration-v2', 'FUSION_TASK_PHASE_BIAS_CALIBRATION_V2'),
         ('collect-target-fusion-task-phase-bias-continuation-v3', 'FUSION_TASK_PHASE_BIAS_CONTINUATION_V3'),
         ('collect-target-fusion-task-phase-xyz-feedback-v4', 'FUSION_TASK_PHASE_XYZ_FEEDBACK_V4'),
         ('collect-target-fusion-task-phase-xyz-feedback-continuation-v5', 'FUSION_TASK_PHASE_XYZ_FEEDBACK_CONTINUATION_V5'),
         ('collect-target-fusion-task-phase-pose-feedback-v6', 'FUSION_TASK_PHASE_POSE_FEEDBACK_V6'),
         ('collect-target-fusion-task-phase-pose-continuation-v7', 'FUSION_TASK_PHASE_POSE_CONTINUATION_V7'),
         ('collect-target-fusion-task-phase-bias-fade-v8', 'FUSION_TASK_PHASE_BIAS_FADE_V8'),
         ('collect-target-fusion-task-phase-live-skill-v8', 'FUSION_TASK_PHASE_LIVE_SKILL_V8'),
         ('collect-target-fusion-task-phase-live-skill-v9', 'FUSION_TASK_PHASE_LIVE_SKILL_V9'),
         ('collect-target-stable-start-short-v1', 'STABLE_START_SHORT_TASK_V1'),
         ('collect-target-stable-task-data-v1', 'STABLE_TASK_DATA_V1'),
         ('collect-target-stable-task-eval-v1', 'STABLE_TASK_EVAL_V1'),
         ('collect-target-stable-task-closed-loop-v1', 'STABLE_TASK_CLOSED_LOOP_V1'))
         and isinstance(source_members, tuple) and set(SOURCE_MEMBERS) <= set(source_members),
         'EXPLICIT_KNOWN_RECIPE_CALLER_RETAINS_ALL_SHARED_SOURCE_CHECKS')
    need(spec.get('phase') == actual_phase and spec.get('kind') == actual_kind
         and spec.get('profile') == 'wm02-auto1-target'
         and spec.get('domain') == 'WM02_AUTO1_TARGET_COMMISSION_V1'
         and spec.get('site_profile') == '4080-simulation' and spec.get('physics') is True
         and spec.get('scope') == 'WM02_PILOT1_SIM_ONLY' and bool(spec.get('pilot1_binding')),
         'ORIGINAL_AUTO1_P1_NATIVE_BACKEND_REQUIRED_NOT_A_GRANT')
    live_skill = actual_phase in ('collect-target-fusion-task-phase-live-skill-v8',
                                 'collect-target-fusion-task-phase-live-skill-v9',
                                 'collect-target-stable-start-short-v1',
                                 'collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1')
    need(not any(spec.get(k) for k in ('fusion_binding', 'night1_binding', 'next1_binding', 'next2_binding',
        'parallel_batch_spec', 'standby_shutdown_diagnostic', 'reference_readiness_probe', 'diagnostic_debugger')),
         'SINGLE_WORLD_WITHOUT_ALTERNATIVE_BINDING_OR_DEBUGGER')
    new_data = actual_phase in ('collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1')
    if new_data:
        if actual_phase=='collect-target-stable-task-closed-loop-v1':
            from .stable_task_closed_loop_v1 import validate
        elif actual_phase=='collect-target-stable-task-eval-v1':
            from .stable_task_evaluation_v1 import validate
        else:
            from .stable_task_data_v1 import validate
        validate(spec)
    elif actual_phase == 'collect-target-stable-start-short-v1':
        from .stable_start_short_task_v1 import validate
        validate(spec)
    elif (actual_phase == 'collect-target-fusion-task-phase-live-skill-v9'
            and spec.get('task_condition') == 'SOURCE_PHASE_C01_CPU_HOLD_OR_5MM_TEACHER_V1'):
        from .fusion_source_phase_c01_capability_v1 import validate
        validate(spec)
        need(not spec.get('paused_query_pair'), 'CPU_CAPABILITY_HAS_NO_GPU_PAIR')
    elif live_skill:
        mode = ('PRE_REGISTERED_TASK_PHASE_LIVE_SKILL_V9' if actual_phase.endswith('-v9')
                else 'PRE_REGISTERED_TASK_PHASE_LIVE_SKILL_V8')
        need(spec.get('snapshot_binding_mode') == mode
             and bool(spec.get('paused_query_gpu_spec')) and bool(spec.get('paused_query_pair')),
             'EXPLICIT_LIVE_PAIR_CHECKED_BY_ITS_SEPARATE_CONTRACT')
    else:
        need(not any(spec.get(k) for k in ('paused_query_gpu_spec', 'paused_query_pair', 'snapshot_binding_mode')),
             'SINGLE_WORLD_INITIALIZATION_WITHOUT_NEW_GPU_OR_QUERY')
    need(spec.get('WBC_changes') is False and spec.get('physics_dt_s') == .002
         and type(spec.get('wall_limit_s')) in (int, float) and 0 < spec['wall_limit_s'] <= 1200,
         'FROZEN_WBC_2MS_FINITE_NATIVE_WALL')
    source, files = Path(spec['source_root']), spec.get('source_files', {})
    need(all(name in files and digest(source / name) == files[name] for name in source_members),
         'EXACT_NEW_AND_ORIGINAL_SHARED_SOURCE_BYTES')
    from .fusion_source_phase_c01 import CONDITIONS as c01_conditions,CPU_CONDITION
    c01=actual_phase=='collect-target-fusion-task-phase-live-skill-v9'and spec.get('task_condition')in c01_conditions
    if new_data:
        from .stable_task_scene_v1 import scene,prefix_episode
        scene(spec);view=prefix_episode(_json(spec['episode_manifest'],spec['episode_manifest_sha256']),spec)
        need(episode is None or episode==view,'EXPLICIT_NEW_SCENE_CALIBRATION_PREFIX_VIEW');episode=view
    elif c01:
        from .fusion_source_phase_c01 import scene,prefix_episode
        scene(spec);view=prefix_episode(_json(spec['episode_manifest'],spec['episode_manifest_sha256']),spec)
        need(episode is None or episode==view,'DECLARED_C01_CALIBRATION_PREFIX_VIEW');episode=view
    elif episode is None:
        episode = recipe.episode_contract(_json(spec['episode_manifest'], spec['episode_manifest_sha256']))
    else:
        need(_json(spec['episode_manifest'], spec['episode_manifest_sha256']) == episode,
             'ACTUAL_VERSIONED_ON_DISK_EPISODE_NOT_A_RELABELLED_VIEW')
    if actual_phase == 'collect-target-fusion-task-phase-bias-fade-v8':
        from .fusion_task_phase_bias_fade_v8 import endpoint as actual_endpoint
    elif actual_phase in ('collect-target-fusion-task-phase-live-skill-v9',
                          'collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1'):
        from .fusion_task_phase_pose_slow_feedback_v9 import endpoint as actual_endpoint
    elif actual_phase in ('collect-target-fusion-task-phase-pose-continuation-v7',
                          'collect-target-fusion-task-phase-live-skill-v8'):
        from .fusion_task_phase_pose_continuation_v7 import endpoint as actual_endpoint
    elif actual_phase == 'collect-target-fusion-task-phase-pose-feedback-v6':
        from .fusion_task_phase_pose_feedback_v6 import endpoint as actual_endpoint
    elif actual_phase == 'collect-target-fusion-task-phase-xyz-feedback-continuation-v5':
        from .fusion_task_phase_xyz_feedback_continuation_v5 import endpoint as actual_endpoint
    elif actual_phase == 'collect-target-fusion-task-phase-xyz-feedback-v4':
        from .fusion_task_phase_xyz_feedback_v4 import endpoint as actual_endpoint
    else:
        actual_endpoint = recipe.endpoint
    live_end = 66503
    if actual_phase in ('collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1'): live_end = 38703
    if actual_phase=='collect-target-stable-task-closed-loop-v1':live_end=43503
    if c01:live_end=38703 if spec.get('task_condition')==CPU_CONDITION else 34303
    if actual_phase == 'collect-target-fusion-task-phase-live-skill-v9':
        from .fusion_task_reference_diagnostic_v1 import SHORT_TASK_CONDITION, endpoint, validate_registration
        if spec.get('task_condition') == SHORT_TASK_CONDITION:
            validate_registration(spec)
            live_end = endpoint(SHORT_TASK_CONDITION)
    clock = (actual_endpoint(episode) == spec.get('live_skill_start_step') == 34303
             and episode['geometry_scale'] == .5 and spec.get('max_ticks') == live_end) if live_skill else (
             spec.get('max_ticks') == actual_endpoint(episode))
    need(episode['run'] == spec['registered_root'] and type(spec.get('max_ticks')) is int and clock
         and spec.get('reference_binding_version') == 'MEASURED_EEF_AND_BASE_POSE_HOLD_V2',
         'PREDECLARED_STATIC_EPISODE_RUN_AND_EXACT_ENDPOINT')
    catalog = _json(spec['scene_allowlist'], spec['scene_allowlist_sha256'])
    if new_data:
        from .stable_task_scene_v1 import scene
        proof=scene(spec);scene_id=spec['scene_id'];parent=proof['parent'];model_pin=proof['compiled']['scene_sha256']
    elif c01:
        from .fusion_source_phase_c01 import SCENE_ID as scene_id,PARENT as parent,SCENE as model_pin
    else:scene_id,parent,model_pin=common.SCENE_ID,common.PARENT,recipe.SCENE
    matches = [s for s in catalog['scenes'] if s['scene_id'] == scene_id]
    need(len(matches) == 1 and matches[0]['parent'] == parent and matches[0]['split'] == (spec['data_role']if new_data else'DIAGNOSTIC'if c01 else'TRAIN')
         and matches[0]['model_path'] == spec['model_path']
         and matches[0]['model_sha256'] == digest(spec['model_path']) == model_pin
         and matches[0].get('static_check') == 'PASSED_NO_INTEGRATION'
         and catalog['native_binary_sha256'] == digest(spec['native_binary']),
         'EXACT_STATIC_T06_AND_FROZEN_BINARY')
    need(spec.get('trajectory_trace_capacity') == 524288
         and type(spec.get('trajectory_trace_capacity')) is int
         and spec.get('trajectory_trace_capacity_build_sha256') == TRACE_BUILD,
         'EXACT_BUILD03_TRACE524_RECEIPT_PIN')
    if new_data:
        geometry.validate_trace_capacity_failure(spec,historical_scene_sha256=recipe.SCENE)
    elif c01:
        from .fusion_source_phase_c01 import CALIBRATION_SCENE
        geometry.validate_trace_capacity_failure(spec,historical_scene_sha256=CALIBRATION_SCENE)
    else:geometry.validate_trace_capacity_failure(spec)
    path = Path(spec['trajectory_trace_capacity_build'])
    build = _json(path, TRACE_BUILD)
    need(build.get('status') == 'TRACE_ONLY_524288_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
         and build.get('trace_capacity') == 524288 and build.get('new_target_integrations') == 0
         and build.get('trace_library_sha256') == digest(spec['trace_library'])
         and build.get('native_binary_sha256') == digest(spec['native_binary'])
         and build.get('observer_sha256') == digest(spec['observer_library'])
         and all(build.get('source_files', {}).get(name) == files[name] for name in (
             'src/go2arx5_vla/target_mujoco/init_trace.cpp', 'src/go2arx5_vla/target_mujoco/init_trace.hpp')),
         'ACTUAL_TRACE_ONLY_BUILD03_UNCHANGED_NATIVE_BYTES')
    common._closed(path.parent.parent)
    return episode, _source(spec)


def _previous(spec, episode, previous):
    path = Path(spec['previous_stage'])
    actual = _json(path, spec['previous_stage_sha256'])
    need(previous == actual, 'EXACT_PINNED_PREDECESSOR_RESULT')
    prior_spec, prior_pin = common._closed(path.parent.parent)
    report = actual.get('reference_qualification') or {}
    prior = report.get('episode') or {}
    need(_json(prior_spec['episode_manifest'], prior_spec['episode_manifest_sha256']) == prior,
         'PREDECESSOR_SPEC_EPISODE_REPORT_JOIN')
    index = SCALES.index(episode['geometry_scale'])
    if index == 0:
        from . import fusion_source_posture_paused_query_v3 as v3
        need(spec['previous_stage_sha256'] == FIRST_PREDECESSOR
             and prior_pin == FIRST_PREDECESSOR_SPEC
             and prior_spec.get('source_sha') == FIRST_PREDECESSOR_SOURCE
             and actual.get('phase') == prior_spec.get('phase') == v3.PHASE,
             'FIRST_GRADE_REQUIRES_EXACT_CLOSED_FULL1_V3_COLD_BOUNDARY')
        v3.episode_contract(prior)
        need(prior['geometry_scale'] == 1. and v3.endpoint(prior) == 16403,
             'COMPLETE_PRIOR_FULL1_NOT_NEW_TASK_PHASE_QUALIFICATION')
        expected_end = 16403
    else:
        recipe.episode_contract(prior)
        need(actual.get('phase') == prior_spec.get('phase') == PHASE
             and prior['geometry_scale'] == SCALES[index-1]
             and prior['source_observation_sha256'] == episode['source_observation_sha256']
             and prior['frame_map_sha256'] == episode['frame_map_sha256']
             and prior['scene_sha256'] == episode['scene_sha256'],
             'IMMEDIATELY_PRECEDING_SAME_RECIPE_GRADE')
        binding = report.get('task_phase_initialization') or {}
        need(binding.get('experiment_role') == 'NON_VLA_CALIBRATION'
             and binding.get('actual_VLA_commands') == binding.get('VLA_queries') == 0
             and binding.get('object_and_goal_are_policy_inputs') is False
             and digest(binding['goal_binding_path']) == binding.get('goal_binding_sha256')
             and digest(binding['readiness_path']) == binding.get('readiness_sha256'),
             'PREDECESSOR_RETAINS_SEPARATE_GOAL_BINDING_AND_REAL_READINESS')
        expected_end = recipe.endpoint(prior)
    need(prior['scene_id'] == common.SCENE_ID and prior['parent'] == common.PARENT,
         'UNCHANGED_T06_SCENE_LINEAGE')
    common._complete_previous(spec, actual, prior_spec, report, expected_end)


def episode_limit(spec, previous):
    episode, _ = episode_config(spec)
    _previous(spec, episode, previous)
    return recipe.endpoint(episode)


def wall_limit(spec):
    return 1200
