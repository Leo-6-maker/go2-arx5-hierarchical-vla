"""P0 decision from completed, independently audited FD2 evidence only."""
import math

from scripts.pi0294.scene_cards import digest

AUTHORITY = '9eff29a44a193dd4d3620dc0bfe5764a402eff42'


def teacher_decision(evidence, geometry):
    rows = evidence['rows']
    if (evidence['status'] != 'PASS_ALL_EIGHT_SAVED_AUDITS_AND_FOUR_EXACT_INITIAL_PAIRS'
            or len(rows) != 8 or len({r['id'] for r in rows}) != 8
            or len(evidence['pairs']) != 4
            or any(not p['initial_native_and_scene_XML_equal']
                   or not p['initial_two_camera_pixels_equal'] for p in evidence['pairs'])):
        raise PermissionError('PV1_REQUIRES_COMPLETE_ACTUAL_FD2_EVIDENCE')
    indexed = {r['id']: r for r in geometry['rows']}
    if (geometry['status'] != 'PASS_SAVED_GEOMETRY_RECONSTRUCTION_NO_NEW_PHYSICS'
            or len(indexed) != 8 or set(indexed) != {r['id'] for r in rows}
            or geometry['integrations'] != 0):
        raise PermissionError('PV1_REQUIRES_QUALIFIED_FINITE_TABLE_GEOMETRY')
    pairs = []
    for pair in evidence['pairs']:
        cells = [r for r in rows if r['scene_group'] == pair['scene_group']]
        if len(cells) != 2 or {r['nominal_clearance_m'] for r in cells} != {.004, .012}:
            raise PermissionError('PV1_EXACT_TWO_REFERENCE_PAIR')
        cells.sort(key=lambda r: r['nominal_clearance_m'])
        gaps = []
        for row in cells:
            g = indexed[row['id']]
            if (row['audit_status'] != 'PASS_INDEPENDENT_SAVED_BYTE_AUDIT'
                    or not row['audit_checks'] or not all(row['audit_checks'].values())
                    or not row['independent_full32_predicates_qualified']
                    or row['actual_native_failures'] != 0):
                raise PermissionError('PV1_UNQUALIFIED_OR_UNSAFE_TEACHER_EVIDENCE')
            gap = row['minimum_vertical_gap_during_actual_robot_contact_m']
            qualified = bool(g['vertical_gap_is_exact_separation_for_inside_above_boxes']
                and g['all_pad_box_vertices_inside_finite_table_XY']
                and g['maximum_vertical_gap_reconstruction_error_m'] <= 1e-10
                and row['actual_robot_contact_ticks'] > 0 and gap is not None
                and len(gap) == 2 and all(math.isfinite(x) for x in gap))
            gaps.append(min(gap) if qualified else None)
        gain = None if None in gaps else gaps[1] - gaps[0]
        pairs.append(dict(scene_group=pair['scene_group'], direction=pair['direction'],
            original_contact_minimum_gap_m=gaps[0], alternate_contact_minimum_gap_m=gaps[1],
            paired_gain_m=gain, gain_at_least_4mm=gain is not None and gain >= .004,
            original_success=bool(cells[0]['original_score']['persistent_success']),
            alternate_success=bool(cells[1]['original_score']['persistent_success']),
            alternate_no_lift=bool(cells[1]['original_score']['history']['no_lift'])))
    original = sum(p['original_success'] for p in pairs)
    alternate = sum(p['alternate_success'] for p in pairs)
    gains = sum(p['gain_at_least_4mm'] for p in pairs)
    qualified = bool(alternate >= 3 and alternate >= original and gains >= 3
        and all(p['alternate_no_lift'] and p['alternate_success']
                for p in pairs if p['original_success']))
    route = 'CLEARANCE_PAIR' if qualified else 'COVERAGE_FIRST'
    if not qualified and original < 3:
        route = 'DATA_GENERATOR_REQUIRES_BOUNDED_TRAIN_QUALIFICATION'
    return dict(schema='PI0294_PV1_FD2_AND_TEACHER_DECISION_V1', authority_release=AUTHORITY,
        route=route, pairs=pairs, original_successes_out_of_4=original,
        alternate_successes_out_of_4=alternate, contact_gain_qualified_pairs=gains,
        FD2_evidence_sha256=digest(evidence), saved_geometry_sha256=digest(geometry),
        FD2_repeated=False, FD2_costs_recharged=False, neural_training_cause_proven=False,
        new_paired_training_requires_common_success_groups=64,
        insufficient_common_support_fallback='ORIGINAL_TEACHER_COVERAGE_ONLY_IF_BANK_QUALIFIED',
        selected='NONE', real_robot_deployment='NONE')
