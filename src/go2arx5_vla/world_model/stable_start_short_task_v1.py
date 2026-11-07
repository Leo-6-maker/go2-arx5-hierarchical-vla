"""Finite T06 HOLD/4mm Teacher tasks on the unchanged qualified V9 prefix.

This is a new diagnostic lifecycle, not GR00T execution or TRAIN data. Source
state ranges are diagnostic only. Original native admission and safety remain.
"""
import copy
import json
from pathlib import Path
import numpy as np

from . import fusion_task_phase_pose_slow_feedback_v9 as prefix
from . import fusion_task_phase_live_skill_contract_v9 as old
from . import fusion_task_phase_initialization_contract as operational
from .storage import digest, write_json, jsonable

PHASE = 'collect-target-stable-start-short-v1'
KIND = 'STABLE_START_SHORT_TASK_V1'
B, E = 34303, 38703
CONDITION = 'STABLE_NATIVE_T06_HOLD_OR_4MM_V1'
PRODUCER = 'PRIVILEGED_STABLE_START_DIAGNOSTIC_NOT_GR00T'
SOURCE_MEMBERS = (*old.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/stable_start_short_task_v1.py',
    'scripts/wm02/stable_start_short_check_v1.py')


def need(ok, reason):
    if not ok:
        raise ValueError('STABLE_START_' + reason)


def config(method):
    need(method in ('HOLD', 'GEOMETRIC_TEACHER_4MM'), 'DECLARED_METHOD')
    return dict(schema='WM02_STABLE_START_DIAGNOSTIC_CONFIG_V1', method=method,
        producer=PRODUCER, step_m=0. if method == 'HOLD' else .004,
        suffix_atoms=176, minimum_course_clearance_m=.14,
        data_role='DIAGNOSTIC_NOT_AUTO_TRAIN', GR00T_queries=0, model_updates=0)


def score_contract():
    from .fusion_graded_short_skill_v1 import score_contract as original
    return original()


def validate(spec):
    need(spec.get('phase') == PHASE and spec.get('kind') == KIND
         and spec.get('task_condition') == CONDITION
         and spec.get('snapshot_binding_mode') == 'PRE_REGISTERED_STABLE_START_CPU_V1'
         and not spec.get('paused_query_gpu_spec') and not spec.get('paused_query_pair')
         and spec.get('live_skill_start_step') == B and spec.get('max_ticks') == E
         and spec.get('live_skill_geometry_scale') == 1.
         and spec.get('live_skill_time_stretch_factor') == 2,
         'EXACT_TYPED_CPU_SINGLE_WORLD_ENDPOINT')
    c = spec.get('stable_start_diagnostic') or {}
    need(c == config(c.get('method')), 'EXACT_FROZEN_METHOD')
    p = Path(spec['stable_start_panel'])
    panel = old.common._json(p, spec['stable_start_panel_sha256'])
    rows = panel.get('assignments', [])
    need(panel.get('schema') == 'WM02_STABLE_START_G1_PANEL_V1'
         and len(rows) == 3 and len({r['native_run'] for r in rows}) == 3
         and [r['method'] for r in rows] == ['HOLD','HOLD','GEOMETRIC_TEACHER_4MM']
         and all(r['split'] == 'DIAGNOSTIC' for r in rows), 'THREE_PREREGISTERED_DIAGNOSTICS')
    match = [r for r in rows if r['native_run'] == Path(spec['registered_root']).name]
    need(len(match) == 1 and match[0]['method'] == c['method'], 'PANEL_METHOD_RUN_JOIN')
    return c


def episode_config(spec):
    validate(spec)
    episode = prefix.episode_contract(old.common._json(
        spec['episode_manifest'], spec['episode_manifest_sha256']))
    need(episode['geometry_scale'] == .5 and prefix.endpoint(episode) == B,
         'UNCHANGED_T06_HALF_GRADE_PREFIX')
    return operational.recipe_config(spec, episode, actual_phase=PHASE,
        actual_kind=KIND, source_members=SOURCE_MEMBERS)


def episode_limit(spec, previous):
    episode_config(spec)
    # Reuse the exact original closed V7 qualification; this supplies evidence,
    # not permission. The new worker still passes actual current scope checks.
    path = Path(spec['previous_stage'])
    actual = old.common._json(path, spec['previous_stage_sha256'])
    prior, pin = old.common._closed(path.parent.parent)
    need(actual == previous and spec['previous_stage_sha256'] == old.PREDECESSOR_RESULT
         and pin == old.PREDECESSOR_SPEC and prior['source_sha'] == old.PREDECESSOR_SOURCE
         and prior['phase'] == actual['phase'] == old.previous_recipe.PHASE,
         'EXACT_CLOSED_HALF_PREFIX_EVIDENCE')
    checked = old.audit_completed_feedback(prior, actual, recipe=old.previous_recipe)
    need(checked['all_nine_non_full_grade_predicates_passed'] is True
         and checked['readiness']['query_ready'] is False,
         'OLD_SOURCE_EQUIVALENCE_NOT_RELABELLED')
    return E


def build(actual, reference, base, cube, method):
    from ..trajectory.so3 import rotation_matrix_to_rot6d_rows
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    c = config(method)
    actual, cube = np.asarray(actual, float), np.asarray(cube, float)
    need(actual.shape == cube.shape == (4,4) and np.isfinite(actual).all()
         and np.isfinite(cube).all(), 'FINITE_ACTUAL_GEOMETRY')
    ray = cube[:3,3] - actual[:3,3]; length = float(np.linalg.norm(ray))
    need(length > .14 + c['step_m'], 'FULL_STEP_ORIGINAL_NONCONTACT_DOMAIN')
    u = np.linspace(0., 1., 40); weight = u**3*(10+u*(-15+6*u))
    raw = np.empty((40,10)); raw[:,:3] = actual[:3,3] + weight[:,None]*c['step_m']*ray/length
    raw[:,3:9] = rotation_matrix_to_rot6d_rows(actual[:3,:3]); raw[:,9] = 1.
    program = TrajectoryAdaptation(raw, actual, np.eye(4), actual, reference, base, 1., time_stretch_factor=2)
    clearance = min(np.linalg.norm(program.desired_path[:,:3]-cube[:3,3],axis=1).min(),
                    np.linalg.norm(program.reference_path[:,:3]-cube[:3,3],axis=1).min())
    need(float(clearance) >= .14 and program.end_atom+8 <= 176,
         'FULL40_ORIGINAL_CLEARANCE_AND_TAIL_NO_CLIP')
    return program, dict(config=c, original_physical_H40=raw, adaptation=program.evidence(),
        actual_decision_TCP=actual, actual_cube=cube, minimum_full_path_clearance_m=float(clearance),
        native_execution_authorized=False, original_native_certificates_required=True)


def validate_adapter_install(adapter, action):
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    need(adapter.spec.get('phase') == PHASE and adapter.spec.get('kind') == KIND,
         'ACTUAL_TYPED_NATIVE_SCOPE')
    step = adapter.observe()['step']; program = getattr(adapter,'live_skill_program',None)
    if program is None:
        need(step < B, 'PREFIX_BOUNDARY')
        return prefix.validate_install_preview(action, initial_tcp=adapter.trajectory_initial_tcp,
            program=adapter.trajectory_program, step=step, base=adapter.base_pose_hold,
            frozen_base=adapter.trajectory_initial_base, end_step=B)
    need(type(program) is TrajectoryAdaptation and B <= step < E and (step-B)%25 == 0
         and program.geometry_scale == 1. and program.time_stretch_factor == 2
         and np.array_equal(np.asarray(action),program.preview((step-B)//25))
         and np.array_equal(program.base_pose_hold,np.asarray(adapter.base_pose_hold))
         and np.array_equal(adapter.base_pose_hold,adapter.trajectory_initial_base),
         'EXACT_COMPLETE_SUFFIX_CLOCK_BASE_AND_PREVIEW')
    return dict(status='FULL_H40_REQUIRES_ORIGINAL_NATIVE_CERTIFICATE',native_step=step,
                physics_advance_authorized=False)


def continue_native(adapter, report, samples, windows):
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .fusion_live_skill_segment import run_consumed_segment
    from .fusion_short_skill import _score_measured_samples
    c = validate(adapter.spec); ready = report['task_phase_initialization']['readiness']
    need(adapter.observe()['step'] == B and adapter.trace_sealed is False
         and all(v is True for k,v in ready['predicates'].items() if k != 'full_grade')
         and report['all_40_knots_consumed'] is True and len(windows) == (B-2503)//25,
         'ACTUAL_UNSEALED_T06_PREFIX_NOT_SOURCE_EQUIVALENCE')
    # Retain the actual decision packet for same-observation offline rendering.
    raw = (adapter.output/'state.bin').read_bytes()
    from scripts.pi0294.task_state import parse_state
    need(parse_state(raw)['step'] == B, 'SAVED_REAL_DECISION_PACKET_CLOCK')
    with (adapter.output/'STABLE_START_DECISION_STATE.bin').open('xb') as f:
        f.write(raw); f.flush()
    write_json(adapter.output/'STABLE_START_DECISION_MEASURED.private.json',samples[-1])
    cube = np.asarray(samples[-1]['object_pose']).copy()
    program, proposal = build(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
        adapter.base_pose_hold,cube,c['method'])
    path = adapter.output/'STABLE_START_PROPOSAL.private.json'; write_json(path,proposal)
    adapter.live_skill_program = program
    initial = dict(qpos=np.asarray(samples[-1]['qpos']).copy())
    def observe(sample):
        _noncontact(sample,cube); require_standing(sample,initial)
    suffix, measured, actual_windows = run_consumed_segment(adapter,program,start_step=B,end_step=E,
        past_samples=samples,past_windows=windows,sample_observer=observe)
    score = _score_measured_samples([s for s in measured if s['native_step'] >= B],
        score_contract(),expected_contract=score_contract())
    hold = all(v for k,v in score['predicates'].items() if k not in ('signed_progress','distance_reduction'))
    total = (E-2503)//25
    result = dict(copy.deepcopy(report),schema='WM02_STABLE_START_NATIVE_ACTUAL_V1',
        actual_spec_phase=adapter.spec['phase'],
        native_endpoint_step=E, stable_start_diagnostic=c, desired_outcome_provenance=PRODUCER,
        executed_reference_proposal_sha256=digest(path), actual_scene_sha256=digest(adapter.spec['model_path']),
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
        actual_VLA_commands=0,VLA_commands=0,VLA_action_installations=0,VLA_queries=0,
        VLA_inferences=0,actual_VLA_inferences=0,native_process_Processor_decodes=0,
        additional_action_decodes=0,unique_generated_action_chunks=0,
        WM_updates=0,GR00T_updates=0,Actor_updates=0,full_task_success=False,
        target_task_success=False,source_task_phase_equivalence=False,
        original_initialization_readiness_unchanged=True,capability_score=score,
        hold_stability_pass=hold,short_teacher_success=(c['method']=='GEOMETRIC_TEACHER_4MM'
            and score['approach_step_success']),
        Teacher_reference_commands=int(c['method']=='GEOMETRIC_TEACHER_4MM'),
        privileged_diagnostic_not_deployable_VLA=True,expert_training_eligible=False)
    return jsonable(result)


def complete_consumption(spec, complete):
    validate(spec); suffix = complete.get('generated_segment') or {}
    return (complete.get('stable_start_diagnostic') == spec['stable_start_diagnostic']
        and complete.get('capability_reference_all40_consumed') is True
        and complete.get('GR00T_raw_H40_consumed') is False
        and complete.get('all_40_knots_consumed') is False
        and complete.get('raw_knot_consumed') == [False]*40
        and complete.get('actual_VLA_commands') == complete.get('actual_VLA_inferences') == 0
        and suffix.get('all_40_knots_consumed') is True
        and suffix.get('raw_knot_consumed') == [True]*40
        and suffix.get('complete_suffix_atoms') == len(suffix.get('windows',[])) == 176
        and suffix.get('solver_target_confirmed_atoms') == suffix.get('gripper_confirmed_atoms') == 176
        and complete.get('actual_reference_installations') == len(suffix.get('install_attempts',[])) > 0)


def run(adapter):
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    episode, source = episode_config(adapter.spec)
    return run_feedback_program(adapter,episode,source,recipe_module=prefix,actual_phase=PHASE,
        report_schema='WM02_STABLE_START_PREFIX_ACTUAL_V1',
        goal_schema='WM02_STABLE_START_GOAL_BINDING_V1',
        decision_schema='WM02_STABLE_START_PREFIX_DECISION_V1',
        feedback_report_schema='WM02_STABLE_START_PREFIX_FEEDBACK_V1',continuation=continue_native)
