"""Verify fixed D3 populations and actual mechanism evidence at their stage gates."""

import numpy as np

from go2arx5_vla.world_model.multitask_contact_teacher_d3 import MECHANISMS
from go2arx5_vla.world_model.multitask_scope_d2 import D3_PLAN, D3_TEACHER
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin


def verify_stage_manifest(spec, cards, entry):
    stage = spec['d2_stage']
    registered = read_pin(spec['registered_panel'])
    population = read_pin(entry['compiled_population'])
    rules = read_pin(entry['frozen_candidate_rules'])
    if (rules['plan_commit'] != D3_PLAN or rules['status'] != 'FROZEN_R0_PASS'
            or rules['teacher'] != D3_TEACHER or rules['source_sha'] != spec['source_sha']
            or not all(rules['mechanisms'].get(mechanism, {}).get('offline_regression_passed')
                       for mechanism in MECHANISMS)):
        raise PermissionError('D3_CANDIDATE_RULES_REQUIRE_COMPLETE_ZERO_INTEGRATION_REGRESSIONS')
    name = 'QUAL' if stage == 'Q' else stage
    expected = population['conditions'][name]
    if [r['id'] for r in registered] != [r['id'] for r in expected]:
        raise PermissionError('D3_EXACT_PREREGISTERED_POPULATION_ORDER')
    clusters = {c['id']: c for c in population['clusters']}
    for row, item, case in zip(registered, expected, cards, strict=True):
        condition = row['condition']
        scene = clusters[item['cluster']]
        if (row['instruction'] != item['instruction'] or condition['family'] != item['family']
                or condition['reset'] != item['reset'] or condition['seed'] != item['noise_seed']
                or condition['layout'] != item['cluster'] or row['scene'] != scene['scene']
                or row['geometry'] != scene['scoring_geometry']
                or case.get('d3_ablation') != item.get('ablated')
                or case.get('d3_pair') != item.get('pair') or case.get('d3_arm') != item.get('arm')
                or case.get('d3_mechanism') != item.get('mechanism')):
            raise PermissionError('D3_EXACT_SCENE_RESET_AND_SINGLE_REGISTERED_ABLATION')
    if stage == 'R2':
        earlier = read_pin(entry['causal_diagnostics_final'])
        treatment = [r for r in earlier['records'] if r.get('d3_arm') == 'TREATMENT']
        if (earlier['source_sha'] != spec['source_sha'] or earlier['profile'] != 'LEGACY_N1'
                or earlier['unresolved_case_launches'] or len(earlier['records']) != 12
                or {r.get('d3_mechanism') for r in treatment} != set(MECHANISMS)
                or len(treatment) != 6 or not all(r['d3_R1']['passed'] for r in treatment)):
            raise PermissionError('D3_ALL_SIX_ACTUAL_TREATMENT_MILESTONES_REQUIRED')
    if stage == 'Q':
        earlier = read_pin(entry['coverage_final'])
        if (earlier['source_sha'] != spec['source_sha'] or earlier['profile'] != spec['d2_profile']
                or earlier['unresolved_case_launches'] or len(earlier['records']) != 12
                or not all(r.get('G7_pass') and r['task_and_mobile_success'] for r in earlier['records'])):
            raise PermissionError('D3_EXACT_SAME_TEACHER_AND_BACKEND_COVERAGE_REQUIRED')


def consumed_prefix(spec, result):
    detail = result.get('detail') or {}
    if not detail.get('capture'):
        raise ValueError('D3_DIAGNOSTIC_HAS_NO_SEALED_NATIVE_CAPTURE')
    capture = read_pin(detail['capture'])
    labels = read_pin(capture['labels'])
    if (capture['writer_error'] or capture['planned_suffix_is_executed']
            or capture['queries'] != detail['expert_queries'] or len(labels) != capture['queries']
            or capture['final_native_step'] != result['actual_native_integrations']
            or capture['frozen_scene_sha256'] != spec['frozen_files'][spec['model_path']]):
        raise ValueError('D3_DIAGNOSTIC_CAPTURE_NOT_BOUND_TO_NATIVE_TERMINAL')
    cursor = 5703
    complete_atoms = 0
    for number, row in enumerate(labels):
        if (row['query'] != number or row['native_step'] != cursor
                or row['recent_reference']['native_step'] != cursor
                or not row['recent_reference']['exact_receiver_knot']
                or row['supervision_mask'] != [True]*40
                or row['rgb_provenance'] != 'CURRENT_OWNED_PAUSED_NATIVE_PACKET'):
            raise ValueError('D3_EXACT_INITIAL_QUERY_CLOCK_AND_NOMINAL_PLAN_REQUIRED')
        for key in ('nominal_world_H40', 'installed_reference_H40'):
            path = np.asarray(row[key])
            if path.shape != (40, 10) or not np.isfinite(path).all():
                raise ValueError('D3_FINITE_ARCHIVED_H40_REQUIRED')
        intervals = row['intervals']
        if len(intervals)>10 or row['consumed_mask'] != [True]*len(intervals)+[False]*(40-len(intervals)):
            raise ValueError('D3_CONSUMPTION_MASK_IS_NOT_A_CONFIRMED_PREFIX')
        for atom, interval in enumerate(intervals):
            if (interval['atom'] != atom or interval['start_native_step'] != cursor
                    or interval['confirmed_ticks'] != 25 or interval['end_native_step'] != cursor+25
                    or interval['boundary']['native_step'] != cursor+25
                    or not interval['boundary']['exact_receiver_knot']):
                raise ValueError('D3_EXACT_CONSUMED_INTERVAL_AND_RECEIVER_CLOCK_REQUIRED')
            cursor += 25
            complete_atoms += 1
        if row['actual_elapsed_prefix_ticks'] != 25*len(intervals) or row['partial_interval_ticks']:
            raise ValueError('D3_NONTERMINAL_OR_UNACCOUNTED_PARTIAL_ATOM')
    if cursor != result['actual_native_integrations'] or complete_atoms != detail['consumed_task_atoms']:
        raise ValueError('D3_CONSUMPTION_AND_NATIVE_DENOMINATOR_DISAGREE')
    return dict(H40_queries=len(labels), confirmed_atoms=complete_atoms, terminal_native_step=cursor)


def r1_result(spec, native, row):
    mechanism = spec['d3_mechanism']
    score = row.get('score') or {}
    events = score.get('events', {})
    if mechanism in ('INTERNAL_RELATIVE_CLEARANCE', 'EXTERNAL_CARRY_ROUTE'):
        milestone = bool(events.get('correct_region') and score.get('ever_target_lift20mm'))
    elif mechanism in ('APPROACH_TOPOLOGY', 'DESCEND_CLOSE_PREFIX'):
        milestone = bool(score.get('ever_target_grasp') and score.get('ever_target_lift20mm'))
    else:
        milestone = bool(row['task_and_mobile_success'])
    evidence, error = None, None
    if not row['hard_STOP'] and not row['technical_failure']:
        try:
            evidence = consumed_prefix(spec, native)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            error = type(exc).__name__ + ':' + str(exc)
    passed = bool(milestone and evidence and evidence['confirmed_atoms']
                  and not row['hard_STOP'] and not row['technical_failure']
                  and row['trace_status'] == 'RECORDED_TRACE_CHECKS_PASS')
    return dict(passed=passed, actual_milestone=milestone, consumption=evidence,
                first_evidence_failure=error, qualifies_as_training=False)
