"""Two predeclared CPU-only capability diagnostics in the distinct C01 scene.

The original T06 140mm domain is unchanged. This separately qualified course
uses a 70mm clearance and a maximum 5mm privileged Teacher reference. It is
not a GR00T fallback, learned controller, deployment policy or pick/place task.
"""
import copy
import json
from pathlib import Path
import numpy as np
from scripts.pi0292.relay_protocol import digest as document_hash
from .storage import digest, write_json

CONDITION = 'SOURCE_PHASE_C01_CPU_HOLD_OR_5MM_TEACHER_V1'
B, E = 34303, 38703
MODE = 'PRE_REGISTERED_C01_CPU_CAPABILITY_V1'
PRODUCER = 'C01_PRIVILEGED_CAPABILITY_NOT_GR00T_NOT_WM'


def need(ok, reason):
    if not ok:
        raise ValueError('C01_CAPABILITY_' + reason)


def config(method):
    need(method in ('HOLD', 'GEOMETRIC_TEACHER_5MM'), 'PREDECLARED_METHOD')
    return dict(schema='WM02_C01_CPU_CAPABILITY_CONFIG_V1', method=method,
                producer=PRODUCER, step_m=0. if method == 'HOLD' else .005,
                suffix_atoms=176, minimum_course_clearance_m=.07,
                GR00T_queries=0, model_updates=0, data_role='DIAGNOSTIC_NOT_AUTO_TRAIN')


def score_contract():
    from .fusion_graded_short_skill_v1 import score_contract as original
    return dict(original(), schema='WM02_C01_DISTINCT_COURSE_CAPABILITY_SCORE_V1',
                task_id='C01_CPU_SMALL_APPROACH_CAPABILITY_V1',
                score_role='NEW_C01_70MM_COURSE_NOT_OLD_SHORT_OR_PICK_PLACE',
                precontact_offset_from_cube_m=.07, old_short_task_reclassified=False)


def validate(spec):
    from .fusion_source_phase_c01 import scene
    from .fusion_live_program_replay_v1 import assignment
    scene(spec)
    c = spec.get('c01_capability_diagnostic') or {}
    need(c == config(c.get('method')) and spec.get('snapshot_binding_mode') == MODE
         and spec.get('paused_query_gpu_spec') is None
         and spec.get('live_skill_time_stretch_factor') == 2
         and spec.get('live_skill_geometry_scale') == 1.
         and spec.get('max_ticks') == E and spec.get('live_skill_start_step') == B,
         'EXACT_CPU_ONLY_COMPLETE_HORIZON_NO_QUERY_OR_MODEL')
    a = assignment(spec)
    panel = Path(spec['response_learning_panel'])
    value = json.loads(panel.read_bytes())
    rows = [r for r in value['assignments'] if r['native_run'] == Path(spec['registered_root']).name]
    need(len(rows) == 1 and rows[0].get('c01_capability_diagnostic') == c
         and rows[0]['split'] == a['split'] == 'DIAGNOSTIC', 'FROZEN_METHOD_BEFORE_PHYSICS')
    return c


def build(actual, reference, base, cube, method):
    from ..trajectory.so3 import rotation_matrix_to_rot6d_rows
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    c = config(method)
    actual, cube = np.asarray(actual, float), np.asarray(cube, float)
    need(actual.shape == cube.shape == (4,4) and np.isfinite(actual).all()
         and np.isfinite(cube).all(), 'FINITE_ACTUAL_GEOMETRY')
    ray = cube[:3,3] - actual[:3,3]
    length = float(np.linalg.norm(ray))
    need(.07 <= length <= .09 and length > .07 + c['step_m'], 'NEW_COURSE_START_AND_FULL_STEP_CLEARANCE')
    u = np.linspace(0., 1., 40); weight = u**3*(10+u*(-15+6*u))
    raw = np.empty((40,10)); raw[:,:3] = actual[:3,3] + weight[:,None]*c['step_m']*ray/length
    raw[:,3:9] = rotation_matrix_to_rot6d_rows(actual[:3,:3]); raw[:,9] = 1.
    program = TrajectoryAdaptation(raw, actual, np.eye(4), actual, reference, base, 1., time_stretch_factor=2)
    clearance = min(np.linalg.norm(program.desired_path[:,:3]-cube[:3,3],axis=1).min(),
                    np.linalg.norm(program.reference_path[:,:3]-cube[:3,3],axis=1).min())
    need(float(clearance) >= .07 and program.end_atom+8 <= 176, 'FULL40_CLEARANCE_AND_TAIL_NO_CLIPPING')
    return program, dict(config=c, original_physical_H40=raw, adaptation=program.evidence(),
        actual_decision_TCP=actual, actual_cube=cube, minimum_full_path_clearance_m=float(clearance),
        native_execution_authorized=False, original_native_certificates_required=True,
        original_140mm_domain_unchanged=True, GR00T_or_WM_gain=False)


def continue_native(adapter, report, samples, windows):
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .fusion_live_skill_segment import run_consumed_segment
    from .fusion_short_skill import _score_measured_samples
    from .fusion_live_skill_segment import jsonable
    c = validate(adapter.spec)
    ready = report['task_phase_initialization']['readiness']
    need(adapter.observe()['step'] == B and adapter.trace_sealed is False
         and ready['query_ready'] is False and ready['predicates']['full_grade'] is False
         and all(v is True for k,v in ready['predicates'].items() if k not in ('full_grade','actual_cube_distance'))
         and report['all_40_knots_consumed'] is True and len(windows) == (B-2503)//25,
         'ACTUAL_C01_PREFIX_REQUIRED_OLD_READINESS_NOT_RELABELLED')
    cube = np.asarray(samples[-1]['object_pose']).copy()
    program, proposal = build(samples[-1]['tcp'], adapter.reference_history[-1]['pose'],
                              adapter.base_pose_hold, cube, c['method'])
    write_json(adapter.output/'C01_CAPABILITY_PROPOSAL.private.json', proposal)
    adapter.live_skill_program = program
    initial = dict(qpos=np.asarray(samples[-1]['qpos']).copy())
    def observe(sample):
        _noncontact(sample, cube)
        require_standing(sample, initial)
    suffix, actual_samples, actual_windows = run_consumed_segment(adapter, program,
        start_step=B, end_step=E, past_samples=samples, past_windows=windows, sample_observer=observe)
    actual_suffix = [s for s in actual_samples if s['native_step'] >= B]
    score = _score_measured_samples(actual_suffix, score_contract(), expected_contract=score_contract())
    total = (E-2503)//25
    complete = dict(copy.deepcopy(report), schema='WM02_C01_CPU_NATIVE_CAPABILITY_ACTUAL_V1',
        actual_spec_phase=adapter.spec['phase'], native_endpoint_step=E,
        actual_scene_id='FUSION_SOURCE_PHASE_C01_V1', actual_scene_sha256=digest(adapter.spec['model_path']),
        c01_capability_diagnostic=c, desired_outcome_provenance=PRODUCER,
        executed_reference_proposal_sha256=digest(adapter.output/'C01_CAPABILITY_PROPOSAL.private.json'),
        initialization_all40_knots_consumed=report['all_40_knots_consumed'],
        initialization_raw_knot_consumed=report['raw_knot_consumed'],
        initialization_adaptation=report.get('adaptation'), adaptation=program.evidence(),
        all_40_knots_consumed=False, raw_knot_consumed=[False]*40, GR00T_raw_H40_consumed=False,
        capability_reference_all40_consumed=suffix['all_40_knots_consumed'],
        actual_reference_installations=len(suffix['install_attempts']), generated_segment=suffix,
        original_install_ACKs=list(report.get('original_install_ACKs',[]))+suffix['original_install_ACKs'],
        windows=actual_windows, complete_task_atoms=total, task_native_ticks=E-2503,
        solver_target_confirmed_atoms=total, gripper_confirmed_atoms=total,
        actual_endpoint_TCP=np.asarray(actual_samples[-1]['tcp']).tolist(),
        actual_VLA_commands=0,VLA_commands=0,VLA_action_installations=0,VLA_queries=0,
        VLA_inferences=0,actual_VLA_inferences=0,native_process_Processor_decodes=0,
        additional_action_decodes=0,unique_generated_action_chunks=0,
        WM_updates=0,GR00T_updates=0,Actor_updates=0,full_task_success=False,target_task_success=False,
        source_task_phase_equivalence=False,original_initialization_readiness_unchanged=True,
        original_140mm_domain_unchanged=True,capability_score=score,
        Teacher_reference_commands=int(c['method']=='GEOMETRIC_TEACHER_5MM'),
        privileged_diagnostic_not_deployable_VLA=True,GR00T_or_WM_task_transfer=False)
    return jsonable(complete)


def complete_consumption(spec, complete):
    validate(spec)
    suffix = complete.get('generated_segment') or {}
    return (complete.get('c01_capability_diagnostic') == spec['c01_capability_diagnostic']
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
