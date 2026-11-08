"""New target VLA demonstrations, separate from WM/adapter data and old HOLD.

The existing native owner, finite worker, full-H40 installation, solver/gripper
joins and safety remain authoritative. Images are captured while the actual
world is paused, throughout the task. No future achieved pose is a label.
"""
import copy
from pathlib import Path
import numpy as np

from . import stable_start_short_task_v1 as common
from .storage import digest, write_json, jsonable

SCHEMA = 'WM02_TARGET_GR00T_COLLECTION_V1'
PANEL_SCHEMA = 'WM02_TARGET_GR00T_DEMONSTRATION_PANEL_V1'
COMMANDS = ('APPROACH_RED_4MM_OPEN', 'RETREAT_RED_4MM_OPEN')
B, E = 34303, 38703
CAPTURE_ATOMS = tuple(range(0, 176, 7)) + (176,)
LANGUAGES = {
    COMMANDS[0]: 'Move the open gripper a little toward the red cube, then hold without touching it.',
    COMMANDS[1]: 'Move the open gripper a little away from the red cube, then hold without touching it.',
}


def need(ok, reason):
    if not ok:
        raise ValueError('TARGET_GR00T_COLLECTION_' + reason)


def contract():
    return dict(schema=SCHEMA, profile='TARGET_GR00T_V1',
        commands=list(COMMANDS), native_phase='collect-target-stable-task-data-v1',
        decision_step=B, final_step=E, native_dt_s=.002, command_period_s=.05,
        camera_capture_atoms=list(CAPTURE_ATOMS), camera_state_synchrony='SAME_ACTUAL_PAUSED_PACKET',
        camera_maximum_gap_atoms=7, supervision='EXECUTED_TEACHER_POLICY_H40_WITH_PINNED_COMPILER_CONSUMPTION',
        policy_decisions_per_episode=1, compiler_preview_renewals_are_new_policy_decisions=False,
        incomplete_future='NO_UNEXECUTED_KNOTS_OR_RENEWED_PREVIEWS_AS_NEW_EXPERT_DECISIONS',
        initialization_is_demonstration=False, HOLD_controls_are_demonstrations=False,
        WM_transitions_are_demonstrations=False, rejected_or_unexecuted_suffix_is_demonstration=False,
        expert_training_eligible=False, admission_requires='SEALED_CLOSED0_AND_TASK_SUCCESS_AND_TRACE_AND_ALL_MEDIA',
        GR00T_updates=0, GR00T_queries=0, WBC_changes=False)


def config(command, role):
    need(command in COMMANDS and role in ('TRAIN', 'DEV', 'TEST', 'QUALIFICATION'), 'TASK_AND_ROLE')
    return dict(schema='WM02_TARGET_GR00T_EXPERT_CONFIG_V1', task_command=command,
        method='GEOMETRIC_TEACHER_4MM', producer='PRIVILEGED_TARGET_GR00T_EXPERT_V1',
        step_m=.004, direction_sign=1 if command == COMMANDS[0] else -1,
        suffix_atoms=176, minimum_course_clearance_m=.14, data_role=role,
        GR00T_queries=0, model_updates=0)


def validate(spec):
    from .stable_task_scene_v1 import pinned
    c = config(spec.get('task_command'), spec.get('data_role'))
    need(spec.get('target_gr00t_collection') == contract()
         and spec.get('stable_task_data') == c
         and spec.get('target_gr00t_attempt_role') in ('QUALIFICATION','DEMONSTRATION'),
         'EXPLICIT_NEW_PROFILE_NO_OLD_RELABEL')
    panel = pinned(dict(path=spec['stable_task_panel'], sha256=spec['stable_task_panel_sha256']))
    need(panel.get('schema') == PANEL_SCHEMA
         and panel.get('collection') == contract(), 'FROZEN_NEW_DEMONSTRATION_PANEL')
    rows = panel['assignments']
    need(len({r['native_run'] for r in rows}) == len(rows), 'UNIQUE_ATTEMPTS')
    parents = {}
    for row in rows:
        need(row['command'] in COMMANDS and row['split'] in ('TRAIN','DEV','TEST','QUALIFICATION'),
             'DECLARED_TASK_AND_SPLIT')
        # Qualification can use a TRAIN geometry, but is not admitted as TRAIN.
        if row['split'] != 'QUALIFICATION':
            need(row['parent'] not in parents or parents[row['parent']] == row['split'], 'PHYSICAL_PARENT_SPLIT')
            parents[row['parent']] = row['split']
    matches = [r for r in rows if r['native_run'] == Path(spec['registered_root']).name]
    need(len(matches) == 1 and matches[0]['purpose'] == spec['target_gr00t_attempt_role'],
         'QUALIFICATION_NOT_AUTO_DEMONSTRATION')
    return c


def score_contract(command):
    need(command in COMMANDS, 'SCORE_TASK')
    return dict(schema='WM02_TARGET_GR00T_DIRECTIONAL_SHORT_SCORE_V1', task_command=command,
        language=LANGUAGES[command], original_approach_contract=common.score_contract(),
        minimum_intent_progress_m=.002, minimum_intent_distance_change_m=.002,
        other_predicates='ALL_ORIGINAL_APPROACH_NONPROGRESS_PREDICATES_UNCHANGED',
        original_native_safety_changed=False, full_pick_place_success_claim=False)


def score(samples, command):
    from .fusion_short_skill import _score_measured_samples
    need(command in COMMANDS, 'SCORE_TASK')
    original = _score_measured_samples(samples, common.score_contract(), expected_contract=common.score_contract())
    sign = 1 if command == COMMANDS[0] else -1
    p = np.asarray([s['tcp'][:3,3] for s in samples])
    cube = np.asarray(samples[0]['object_pose'])[:3,3]
    ray = cube - p[0]
    progress = float(sign * ((p[-1]-p[0]) @ (ray/np.linalg.norm(ray))))
    distance_change = float(sign * (np.linalg.norm(p[0]-cube)-np.linalg.norm(p[-1]-cube)))
    predicates = {k:v for k,v in original['predicates'].items()
                  if k not in ('signed_progress','distance_reduction')}
    predicates.update(intent_progress=progress >= .002, intent_distance_change=distance_change >= .002)
    return dict(schema='WM02_TARGET_GR00T_DIRECTIONAL_SHORT_SCORE_ACTUAL_V1',
        contract=score_contract(command), original_approach_score=original,
        intent_progress_m=progress, intent_distance_change_m=distance_change,
        predicates=predicates, task_success=all(predicates.values()), full_pick_place_success=False)


def build(actual, reference, base, cube, c):
    # Reuse the qualified 4mm quintic/retiming implementation. Reflection is
    # Teacher-only construction of the retreat direction, never a fake scene or
    # a substituted student observation. Clearance is checked against REAL cube.
    construction_cube = np.asarray(cube).copy()
    if c['direction_sign'] == -1:
        construction_cube[:3,3] = 2*np.asarray(actual)[:3,3] - construction_cube[:3,3]
    program, proposal = common.build_for_config(actual, reference, base, construction_cube, c)
    real = np.asarray(cube)
    clearance = float(min(np.linalg.norm(program.desired_path[:,:3]-real[:3,3],axis=1).min(),
                          np.linalg.norm(program.reference_path[:,:3]-real[:3,3],axis=1).min()))
    need(clearance >= .14, 'REAL_CUBE_FULL_PATH_CLEARANCE')
    proposal.update(actual_cube=real, teacher_construction_cube=construction_cube,
        privileged_geometry_not_student_input=True, minimum_full_path_clearance_m=clearance)
    return program, proposal


def save_capture_index(path, captures, *, complete=False, task_success=None):
    """One live progress index; immutable observation directories stay intact."""
    from scripts.pi0292.durable import write
    value = dict(schema=SCHEMA+'_CAPTURES',captures=captures,
        expert_training_eligible=False,complete=complete)
    if task_success is not None:value['task_success']=task_success
    write(path,value)


def continue_native(adapter, report, samples, windows):
    from .stable_task_data_v1 import capture_input
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .fusion_live_skill_segment import run_consumed_segment
    c = validate(adapter.spec)
    ready = report['task_phase_initialization']['readiness']
    need(adapter.observe()['step'] == B and adapter.trace_sealed is False
         and all(v is True for k,v in ready['predicates'].items() if k != 'full_grade')
         and report['all_40_knots_consumed'] is True and len(windows) == (B-2503)//25,
         'REAL_STABLE_PREFIX_BEFORE_EXPERT_TASK')
    cube = np.asarray(samples[-1]['object_pose']).copy()
    program, proposal = build(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
        adapter.base_pose_hold,cube,c)
    proposal_path = adapter.output/'STABLE_TASK_EXPERT_PROPOSAL.private.json'
    write_json(proposal_path,proposal)
    adapter.live_skill_program = program
    initial = dict(qpos=np.asarray(samples[-1]['qpos']).copy())
    captures = []
    capture_path = adapter.output/'TARGET_GR00T_CAPTURE_INDEX.private.json'

    def observe(sample):
        _noncontact(sample,cube)
        require_standing(sample,initial)
        atom, remainder = divmod(sample['native_step']-B,25)
        need(remainder == 0 and 0 <= atom <= 176, 'ACTUAL_TASK_ATOM')
        if atom in CAPTURE_ATOMS:
            need(not captures or sample['native_step'] > captures[-1]['native_step'], 'NO_DUPLICATE_CAPTURE')
            record = capture_input(adapter, language=LANGUAGES[c['task_command']],
                                   task_score=score_contract(c['task_command']))
            need(record['native_step'] == sample['native_step'], 'NO_FUTURE_OR_STALE_IMAGE')
            captures.append(dict(record, atom=atom))
            # Partial index survives a later command/safety failure; eligibility
            # is still false until the independent sealed exporter verifies it.
            save_capture_index(capture_path,captures)

    suffix, measured, actual_windows = run_consumed_segment(adapter,program,start_step=B,end_step=E,
        past_samples=samples,past_windows=windows,sample_observer=observe)
    need([r['atom'] for r in captures] == list(CAPTURE_ATOMS), 'FULL_TASK_VISUAL_COVERAGE')
    task_score = score([s for s in measured if s['native_step'] >= B], c['task_command'])
    save_capture_index(capture_path,captures,complete=True,task_success=task_score['task_success'])
    total = (E-2503)//25
    return jsonable(dict(copy.deepcopy(report), schema='WM02_TARGET_GR00T_EXPERT_NATIVE_ACTUAL_V1',
        actual_spec_phase=adapter.spec['phase'], native_endpoint_step=E, stable_task_data=c,
        target_gr00t_collection=contract(), data_role=adapter.spec['data_role'], data_parent=adapter.spec['data_parent'],
        target_gr00t_attempt_role=adapter.spec['target_gr00t_attempt_role'],
        actual_scene_sha256=digest(adapter.spec['model_path']), student_observation=captures[0],
        demonstration_capture_index=dict(path=str(capture_path),sha256=digest(capture_path)),
        executed_reference_proposal_sha256=digest(proposal_path), desired_outcome_provenance=c['producer'],
        initialization_all40_knots_consumed=report['all_40_knots_consumed'],
        initialization_raw_knot_consumed=report['raw_knot_consumed'],
        initialization_adaptation=report.get('adaptation'), adaptation=program.evidence(),
        all_40_knots_consumed=False, raw_knot_consumed=[False]*40, GR00T_raw_H40_consumed=False,
        capability_reference_all40_consumed=suffix['all_40_knots_consumed'],
        actual_reference_installations=len(suffix['install_attempts']), generated_segment=suffix,
        original_install_ACKs=list(report.get('original_install_ACKs',[]))+suffix['original_install_ACKs'],
        windows=actual_windows, complete_task_atoms=total, task_native_ticks=E-2503,
        solver_target_confirmed_atoms=total, gripper_confirmed_atoms=total,
        actual_endpoint_TCP=np.asarray(measured[-1]['tcp']).tolist(),
        actual_VLA_commands=0,VLA_commands=0,VLA_action_installations=0,VLA_queries=0,VLA_inferences=0,
        actual_VLA_inferences=0,native_process_Processor_decodes=0,additional_action_decodes=0,
        unique_generated_action_chunks=0,WM_updates=0,GR00T_updates=0,Actor_updates=0,
        full_task_success=False,target_task_success=False,source_task_phase_equivalence=False,
        capability_score=task_score['original_approach_score'], target_gr00t_task_score=task_score,
        task_condition_score_contract=score_contract(c['task_command']), task_condition_success=task_score['task_success'],
        expert_training_eligible=False, supervision_status='REQUIRES_FINAL_TRACE_CLOSED0_AND_SEALED_EXPORT',
        raw_reference_is_GR00T=False))
