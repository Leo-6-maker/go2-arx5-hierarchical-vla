"""Read-only reconstruction of saved finite-feedback execution evidence.

No model/FK/physics/ROS/owner/permission entry is called. The recorded native
interpreter and BLAS environment are required for strict numerical replay.
Original aggregate execution records are rejoined; raw policy-selection logs
are not independently re-read and no controller-selected ID is invented.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

from scripts.pi0294.task_state import parse_state
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_live_proposal_qualification_contract as common
from .target_reference import require_live_window
from ..task_data.codec import pose10


def need(ok, reason):
    if not ok:
        raise ValueError('FINITE_FEEDBACK_SAVED_EVIDENCE_' + reason)


def audit_completed_feedback(spec, result, *, recipe):
    """Recompute one archived result, without supplying any execution grant."""
    from . import fusion_task_phase_xyz_feedback_v4 as v4
    from . import fusion_task_phase_xyz_feedback_continuation_v5 as v5
    from . import fusion_task_phase_pose_feedback_v6 as v6
    from . import fusion_task_phase_pose_continuation_v7 as v7
    from . import fusion_task_phase_bias_fade_v8 as v8
    need((spec.get('phase'), recipe) in ((v4.PHASE, v4), (v5.PHASE, v5), (v6.PHASE, v6), (v7.PHASE, v7), (v8.PHASE, v8)),
         'EXACT_KNOWN_RECIPE_CALLER')
    need(Path(sys.executable) == Path(spec['python']) and Path(spec['python']).is_absolute()
         and Path(sys.executable).resolve() == Path(spec['python']).resolve(),
         'RECORDED_ORIGINAL_NATIVE_PYTHON_REQUIRED_FOR_STRICT_RECOMPUTATION')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        need(os.environ.get(key) == spec['native_environment'].get(key) == '1',
             'RECORDED_SINGLE_THREAD_NATIVE_NUMERICAL_ENVIRONMENT:'+key)
    native = Path(spec['registered_root'])/'native-01'
    files = spec['source_files']
    episode = common._json(spec['episode_manifest'], spec['episode_manifest_sha256'])
    recipe.episode_contract(episode)
    end = recipe.endpoint(episode)
    steps = tuple(episode['feedback_decision_native_steps'])

    def native_artifact(path, pin=None, *, binary=False):
        path = Path(path)
        need(path.parent == native and not path.is_symlink(), 'EXACT_NATIVE_ARTIFACT_DIRECTORY')
        if binary:
            from .storage import digest
            need(digest(path) == pin, 'EXACT_NATIVE_BINARY_PACKET_FILE')
            return path.read_bytes(), pin
        value = common._json(path, pin)
        return value, pin

    report = result['reference_qualification']
    need(report['episode'] == episode and result.get('source_sha') == spec['source_sha'], 'RESULT_EPISODE_SOURCE_JOIN')
    common._complete_previous(spec, result, spec, report, end)
    need(report.get('experiment_role') == 'NON_VLA_CALIBRATION'
         and report.get('WM_updates') == report.get('Actor_updates') == report.get('VLA_queries') == 0,
         'ORIGINAL_NON_VLA_NONLEARNING_CALIBRATION_ROLE')
    binding = report['task_phase_initialization']
    goal, goal_pin = native_artifact(binding['goal_binding_path'], binding['goal_binding_sha256'])
    hold, hold_pin = native_artifact(binding['hold_samples_path'], binding['hold_samples_sha256'])
    readiness, readiness_pin = native_artifact(binding['readiness_path'], binding['readiness_sha256'])
    packet_raw, packet_pin = native_artifact(goal['saved_packet_path'], goal['saved_packet_sha256'], binary=True)
    measured, _ = native_artifact(goal['measured_state_path'], goal['measured_state_sha256'])
    inputs = recipe.load_inputs(**operational._source(spec), decision_packet_bytes=packet_raw,
                                measured_decision_state=measured)
    need(inputs['frozen'] == goal['frozen_input'] and goal.get('actual_spec_phase') == recipe.PHASE
         and goal.get('actual_spec_source_sha') == spec['source_sha']
         and goal.get('episode_sha256') == spec['episode_manifest_sha256']
         and goal.get('decision_step') == 3703
         and goal.get('frozen_before_generated_reference_install') is True,
         'EXACT_CURRENT_RESTART_3703_GOAL_AND_PACKET_BYTES')
    recomputed = recipe.query_readiness(hold, episode, inputs)
    need(recomputed == readiness == binding['readiness']
         and recomputed['actual_sample_steps'] == binding['readiness_sample_native_steps']
         == list(range(end-200, end+1, 25)), 'EXACT_RECOMPUTED_NINE_TERMINAL_SAMPLES')
    window_by_end = {row['native_end_step']: row for row in report['windows']}
    need(all(np.array_equal(sample['tcp'],window_by_end[sample['native_step']]['actual_TCP'])
             and np.array_equal(sample['finger_q'],window_by_end[sample['native_step']]['gripper']['actual_finger_q_m'])
             for sample in hold)
         and np.array_equal(hold[-1]['tcp'],report['actual_endpoint_TCP']),
         'NINE_ACTUAL_HOLD_SAMPLES_JOIN_ORIGINAL_CONSUMED_WINDOW_TCP_AND_FINGERS')
    frozen = goal['frozen_predecision_reference_calibration']
    windows = report['windows']
    base = np.asarray(windows[0]['actual_installed_consumed_EEF10_BASE7'][10:], float)
    program = recipe.make_program(inputs, episode, frozen['initial_TCP_target'],
                                   frozen['decision_TCP_target'], base)
    need(program.evidence()['frozen_predecision_reference_binding'] == frozen,
         'INITIAL_FIXED_BIAS_AND_STATIC_REFERENCE_PATH_RECOMPUTE')
    feedback = report['finite_xyz_feedback']
    decisions = feedback['decisions']
    need(len(decisions) == len(steps) and [d['native_step'] for d in decisions] == list(steps),
         'EXACT_PREDECLARED_FINITE_DECISIONS')
    acks = report['original_install_ACKs']
    ack_ids = {a['wire_sha256'] for a in acks if a.get('accepted', 0) > 0 and a.get('installed', 0) > 0}
    wires, decision_summaries, decision_index = {}, [], 0
    for index, row in enumerate(windows):
        step = row['native_start_step']
        if step in steps:
            receipt = decisions[decision_index]
            decision, decision_pin = native_artifact(receipt['decision_path'], receipt['decision_sha256'])
            need(decision.get('actual_spec_source_sha') == spec['source_sha']
                 and decision.get('actual_spec_phase') == recipe.PHASE
                 and decision.get('episode_sha256') == spec['episode_manifest_sha256']
                 and decision.get('original_goal_binding_sha256') == goal_pin
                 and decision.get('saved_before_program_commit_and_reference_install') is True
                 and decision['previous_complete_window'] == windows[index-1],
                 'SAME_GOAL_AND_PREVIOUS_COMPLETED_WINDOW_DECISION_JOIN')
            artifacts = decision['artifact_files']
            loaded = {}
            need(set(artifacts) == {'STATE.bin','MEASURED.private.json','REFERENCE_PATH.private.json',
                'REMAINING_PATH.private.json','H40.private.json'}, 'EXACT_SAVED_FEEDBACK_ARTIFACTS')
            for name, item in artifacts.items():
                loaded[name], _ = native_artifact(item['path'], item['sha256'], binary=name == 'STATE.bin')
            packet = parse_state(loaded['STATE.bin']); measured_now = loaded['MEASURED.private.json']
            need(packet['step'] == measured_now['native_step'] == step
                 and abs(packet['time_s']-step*.002) <= 1e-8
                 and measured_now['time_s'] == packet['time_s']
                 and all(np.array_equal(measured_now[k], packet[k]) for k in ('qpos','qvel','ctrl')),
                 'SAME_EXACT_FEEDBACK_PACKET_AND_SAVED_MEASURED_VECTOR')
            need(np.array_equal(measured_now['tcp'],windows[index-1]['actual_TCP']),
                 'FEEDBACK_MEASURED_TCP_EQUALS_PREVIOUS_REAL_COMPLETED_WINDOW')
            witness = decision['actual_fk_witness']
            need(witness.get('native_step') == step and witness.get('native_time_s') == packet['time_s']
                 and witness.get('model_sha256') == recipe.original.SCENE
                 and witness.get('reader_sha256') == files['scripts/pi0294/task_state.py']
                 and witness.get('packet_qpos_qvel_ctrl_exact') is True
                 and witness.get('original_reader_instance_retained') is True,
                 'PINNED_SAVED_ORIGINAL_FK_WITNESS_NOT_NEW_FK')
            update = program.prepare_feedback(step, measured_now['tcp'])
            atom = (step-3703)//25
            need(update.evidence == decision['decision']
                 and np.array_equal(update.reference_path, loaded['REFERENCE_PATH.private.json'])
                 and np.array_equal(update.reference_path[atom:], loaded['REMAINING_PATH.private.json'])
                 and np.array_equal(update.reference_path[np.minimum(atom+np.arange(40),program.end_atom)], loaded['H40.private.json']),
                 'RECOMPUTED_GAIN_DEADBAND_CAPS_COMPLETE_PATH_AND_H40')
            old_ack = decision['previous_original_install_ACK']
            need(old_ack.get('wire_sha256') == windows[index-1]['reference_sha256']
                 and old_ack.get('accepted',0) > 0 and old_ack.get('installed',0) > 0,
                 'PREVIOUS_WINDOW_ORIGINAL_AGGREGATE_ACK_JOIN')
            program.commit_feedback(update, saved_decision_sha256=decision_pin)
            need(row['reference_sha256'] not in wires, 'NEW_FORCED_REFERENCE_INSTALL_AT_EACH_DECISION')
            decision_summaries.append(dict(native_step=step, decision_sha256=decision_pin,
                error_before_mm=update.evidence['error_norm_m']*1000,
                applied_delta_norm_mm=update.evidence['applied_delta_norm_m']*1000,
                deadband_HOLD=update.evidence['deadband_hold']))
            decision_index += 1
        ref = row['reference_sha256']
        full = np.asarray(row['full_final_installed_H40'], float)
        need(ref in ack_ids, 'EVERY_ACTUAL_ROLLING_REFERENCE_HAS_ORIGINAL_AGGREGATE_ACK')
        if ref not in wires:
            expected_full = (np.repeat(pose10(np.asarray(frozen['initial_TCP_target']),recipe.original.OPEN)[None],40,axis=0)
                if step < 3703 else program.preview((step-3703)//25))
            need(np.array_equal(full, expected_full), 'ACTUAL_NEW_WIRE_FULL40_EQUALS_CURRENT_COMMITTED_PATH_SUFFIX')
            wires[ref] = (step, full.copy())
        installed_step, installed_full = wires[ref]
        wire_index = (step-installed_step)//25
        need(0 <= wire_index < 40 and np.array_equal(full, installed_full), 'UNCHANGED_ROLLING_WIRE_CONTENT_AND_INDEX')
        expected = (pose10(np.asarray(frozen['initial_TCP_target']),recipe.original.OPEN)
            if step < 3703 else program.preview((step-3703)//25)[0])
        need(np.allclose(installed_full[wire_index], expected, rtol=0, atol=1e-12)
             and np.array_equal(row['actual_installed_consumed_EEF10_BASE7'], np.r_[installed_full[wire_index],base])
             and np.array_equal(row['nominal_submitted_EEF10'], expected),
             'EVERY_INSTALLED_AND_CONSUMED_ATOM_MATCHES_CURRENT_REFERENCE_REVISION_UNCHANGED_BASE')
        if step >= 3703:
            elapsed = (step-3703)//25
            need(np.array_equal(row['desired_EEF10'],program.desired_at(elapsed))
                 and np.array_equal(row['desired_interval_end_EEF10'],program.desired_at(elapsed+1)),
                 'UNCHANGED_DESIRED_PATH_EVERY_ACTUAL_ATOM')
        clock = row['clock']
        need(abs(clock['native_start_s']-step*.002) <= 1e-8
             and require_live_window(clock['source_mpc_s'],clock['native_start_s'],25) == clock
             and abs(row['actual_installed_source_age_s']-(clock['native_start_s']-clock['source_mpc_s'])) <= 1e-9,
             'ORIGINAL_SOURCE_RELATIVE_TTL_AND_COMPLETE_ATOM_CLOCK')
    proof = program.evidence()
    need(decision_index == len(steps) and proof == report['adaptation'], 'FINAL_PROGRAM_ALL_REVISIONS_EXACT_RESULT_JOIN')
    nonlinear = proof['feedback_decisions']
    nonzero = sum(d['applied_delta_norm_m'] > 0 for d in nonlinear)
    holds = sum(d['deadband_hold'] for d in nonlinear)
    total = sum(d['applied_delta_norm_m'] for d in nonlinear)
    need(feedback['committed_feedback_decisions'] == feedback['actual_feedback_reference_updates'] == len(steps)
         and feedback['nonzero_XYZ_corrections'] == nonzero
         and feedback['deadband_HOLD_count'] == holds
         and feedback['sum_applied_XYZ_delta_norms_m'] == total
         and total <= recipe.CUMULATIVE_CAP_M+1e-12
         and feedback['final_reference_path_sha256'] == recipe.fixed._sha(program.reference_path)
         and feedback['unchanged_desired_path_sha256'] == recipe.fixed._sha(program.desired_path),
         'EXACT_ACTUAL_DOSE_COUNTS_AND_FINAL_REFERENCE_PATH')
    predicates = recomputed['predicates']
    need(predicates['full_grade'] is (episode['geometry_scale'] == 1.)
         and recomputed['query_ready'] is all(predicates.values())
         and recomputed['native_execution_authorized'] is recomputed['GPU_query_authorized'] is False,
         'CORRECT_GRADE_PREDICATE_NOT_A_NATIVE_OR_GPU_PERMISSION')
    value = dict(status='SAVED_CAUSAL_FEEDBACK_AND_ALL_AGGREGATE_ROLLING_CONSUMPTION_RECOMPUTED',
        readiness=recomputed, all_nine_non_full_grade_predicates_passed=all(
            value for key,value in predicates.items() if key != 'full_grade'),
        actual_feedback_decisions=decision_index, nonzero_XYZ_corrections=nonzero,
        deadband_HOLD_count=holds, sum_applied_delta_norms_m=total,
        actual_reference_renewals=len(wires), complete_task_atoms=len(windows),
        raw_policy_logs_independently_reread=False,
        controller_selected_policy_id='UNKNOWN_DERIVED_NOT_EXPORTED',
        saved_original_FK_witness_rejoined_without_new_FK=True,
        native_execution_authorized=False, GPU_query_authorized=False,
        new_integrations=0, model_loads=0, queries=0, source_writes=0)
    if recipe in (v6,v7,v8):
        angular_nonzero = sum(d['applied_rotation_norm_rad'] > 0 for d in nonlinear)
        angular_holds = sum(d['angular_deadband_hold'] for d in nonlinear)
        angular_total = sum(d['applied_rotation_norm_rad'] for d in nonlinear)
        need(feedback['nonzero_SO3_corrections'] == angular_nonzero
             and feedback['angular_deadband_HOLD_count'] == angular_holds
             and feedback['sum_applied_SO3_delta_norms_rad'] == angular_total
             and angular_total <= recipe.ANGULAR_CUMULATIVE_CAP_RAD+1e-12
             and feedback['rotation_error_frame'] == 'ACTUAL_TARGET_WORLD_LEFT_LOG_R_GOAL_R_ACTUAL_TRANSPOSE'
             and feedback['reference_rotation_from_previous_committed_endpoint'] is True,
             'EXACT_V6_ANGULAR_DOSE_AND_WORLD_LEFT_SEMANTICS')
        value.update(nonzero_SO3_corrections=angular_nonzero,angular_deadband_HOLD_count=angular_holds,
            sum_applied_SO3_delta_norms_rad=angular_total)
    return value
