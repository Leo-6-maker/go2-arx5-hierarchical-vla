"""Fixed V5.2 stage decisions from complete native episode results, no effects."""
from collections import Counter

from .mobile_selection_v52 import CONDITIONS, OFFLINE_GATES, summarize


def screen_decision(rows, offline):
    """Wait for all three controls before freezing the unique candidate or P."""
    students = [r for r in rows if r.get('role') == 'STUDENT']
    if (len(students) != 24 or
            Counter(r.get('arm') for r in students) != {'B0': 8, 'B1': 8, 'B2': 8}):
        raise ValueError('COMPLETE24_NATIVE_SCREEN_REQUIRED_BEFORE_SELECTION')
    for arm in ('B0', 'B1', 'B2'):
        items = [r for r in students if r['arm'] == arm]
        if {(r['family'], r['condition']['layout']) for r in items} != CONDITIONS:
            raise ValueError('ALL_THREE_PREREGISTERED_CONTROL_GRIDS_REQUIRED')
    result = summarize(students, offline)
    result['candidate_frozen'] = result['unique_confirmation_candidate'] is not None
    result['native_screen_requests'] = 24
    return result


def _counts(rows, arm, expected):
    ids = [r.get('panel_entry_id') for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('EXACT_COMPLETE_FROZEN_NATIVE_PANEL_REQUIRED')
    if any(r.get('arm') != arm or r.get('role') != 'STUDENT' for r in rows):
        raise ValueError('ONE_FROZEN_CANDIDATE_NO_ROLE_OR_MODEL_SWITCH')
    for r in rows:
        entry = expected[r['panel_entry_id']]
        condition = r['condition']
        if (r['family'] != entry['family'] or
                condition['family'] != entry['family'] or
                condition['layout'] != entry.get('layout', entry.get('layout_id')) or
                condition['reset'] != entry['reset'] or condition['seed'] != entry['noise_seed'] or
                r.get('scene_sha256') != entry['scene']['sha256'] or
                type(r.get('actual_native_requests')) is not int or r['actual_native_requests'] != 1 or
                any(type(r.get(k)) is not bool for k in
                    ('technical_failure', 'hard_STOP', 'completed', 'task_and_mobile_success'))):
            raise ValueError('ACTUAL_NATIVE_EPISODE_IDENTITY_AND_TYPED_OUTCOME_REQUIRED')
    successes = [r for r in rows if r['task_and_mobile_success']]
    return dict(attempts=len(rows), successes=len(successes),
        by_family={f: sum(r['family'] == f for r in successes) for f in ('F1', 'F2')},
        hard_STOP=sum(r['hard_STOP'] for r in rows),
        technical_failures=sum(r['technical_failure'] for r in rows),
        completed=sum(r['completed'] for r in rows))


def confirmation_decision(rows, panel, arm):
    """New eight-reset confirmation, preserving the original failed V5.1 pilot."""
    expected = {r['id']: r for r in panel}
    if (len(expected) != len(panel) or len(panel) != 8 or
            {(r['family'], r['layout']) for r in panel} != CONDITIONS):
        raise ValueError('FIXED_F1_F2_T0_T3_CONFIRMATION8_REQUIRED')
    counts = _counts(rows, arm, expected)
    passed = (counts['completed'] == 8 and counts['successes'] >= 6 and
        min(counts['by_family'].values()) >= 3 and
        counts['hard_STOP'] == counts['technical_failures'] == 0)
    return dict(schema='SLA_V52_CONFIRMATION_ACTUAL_DECISION', arm=arm,
        **counts, passed=passed, old_V51_pilot_GO=False,
        independent_layout_generalization=False, TEST_permission=False)


def generalization_decision(rows, split, stage, arm):
    """Score the sealed layout population; a failed/incomplete panel cannot pass."""
    if stage not in ('DEV', 'TEST'):
        raise ValueError('DECLARED_DEV_OR_TEST_STAGE_REQUIRED')
    count, layouts, minimum, family_min = (32, 8, 24, 12) if stage == 'DEV' else (40, 10, 32, 16)
    panel = [r for r in split['episodes'] if r['split'] == stage]
    expected = {r['episode_id']: r for r in panel}
    clusters = {(r['family'], r['layout_id']) for r in panel}
    if (split['frozen_before_training'] is not True or split['training_updates_before_freeze'] != 0 or
            len(panel) != count or len(expected) != count or len(clusters) != layouts * 2 or
            Counter(r['family'] for r in panel) != {'F1': count // 2, 'F2': count // 2} or
            any(sum((r['family'], r['layout_id']) == c for r in panel) != 2 for c in clusters)):
        raise ValueError('ORIGINAL_SEALED_GEOMETRY_CLUSTER_POPULATION_REQUIRED')
    counts = _counts(rows, arm, expected)
    passed = (counts['completed'] == count and counts['successes'] >= minimum and
        min(counts['by_family'].values()) >= family_min and
        counts['hard_STOP'] == counts['technical_failures'] == 0)
    by_layout = []
    for family, layout in sorted(clusters):
        group = [r for r in rows if r['family'] == family and r['condition']['layout'] == layout]
        by_layout.append(dict(family=family, layout=layout, attempts=2,
            successes=sum(r['task_and_mobile_success'] for r in group)))
    return dict(schema='SLA_V52_LAYOUT_EVALUATION_ACTUAL_DECISION', stage=stage, arm=arm,
        **counts, by_layout=by_layout, passed=passed,
        correlated_seeds_per_layout=2, TEST_permission=stage == 'DEV' and passed,
        population_success_probability_lower_bound_proved=False)


def authorize_next(stage, arm, *, screen, confirmation=None, development=None):
    """No authorization by intent: only the unique selected arm and passed predecessor."""
    if (not screen.get('candidate_frozen') or screen.get('native_screen_requests') != 24 or
            screen.get('unique_confirmation_candidate') != arm or
            not screen['by_arm'][arm]['qualified'] or
            not all(screen['by_arm'][arm]['promotion_checks'][k] for k in OFFLINE_GATES)):
        raise PermissionError('FULL_SCREEN_UNIQUE_QUALIFIED_CANDIDATE_REQUIRED')
    if stage not in ('CONFIRM', 'DEV', 'TEST'):
        raise ValueError('DECLARED_FINITE_SUCCESSOR_STAGE_REQUIRED')
    if stage in ('DEV', 'TEST') and (not confirmation or
            confirmation.get('arm') != arm or confirmation.get('passed') is not True):
        raise PermissionError('ACTUAL_PASSED_CONFIRMATION_REQUIRED')
    if stage == 'TEST' and (not development or development.get('stage') != 'DEV' or
            development.get('arm') != arm or development.get('passed') is not True or
            development.get('TEST_permission') is not True):
        raise PermissionError('ACTUAL_PASSED_DEV_REQUIRED_BEFORE_TEST')
    return dict(stage=stage, arm=arm, requests={'CONFIRM': 8, 'DEV': 32, 'TEST': 40}[stage])
