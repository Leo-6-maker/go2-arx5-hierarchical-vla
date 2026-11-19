"""Bounded GRASP24 selection and independent CONFIRM32 scoring."""
from collections import Counter


CONTROL = 'D4-B'
CANDIDATES = ('D5-W', 'D5-T')


def grasp_panel(rows, conditions, *, stage, candidate=None):
    """Use sealed 2 ms scores and the original mobile task result."""
    if stage == 'D5_GRASP24':
        arms, per_arm = (CONTROL, *CANDIDATES), 8
    elif stage == 'D5_CONFIRM32' and candidate in CANDIDATES:
        arms, per_arm = (CONTROL, candidate), 16
    else:
        raise ValueError('DECLARED_D5_GUARDED_PANEL_REQUIRED')
    expected = [(c['cluster'], c['instruction'], candidate if arm == 'SELECTED_D5' else arm)
                for c in conditions for arm in c['arm_order']]
    observed = [(r['cluster'], r['instruction'], r['arm']) for r in rows]
    if (len(expected) != len(arms)*per_arm or len(set(expected)) != len(expected)
            or expected[:len(observed)] != observed):
        raise ValueError('D5_RESULTS_MUST_MATCH_THE_SEALED_COMPLETE_OR_PREFIX_ORDER')
    by_condition = {key: row for key, row in zip(observed, rows, strict=True)}

    def won(row):
        return (row['task_and_mobile_success'] and not row['hard_STOP']
                and not row['technical_failure'] and bool((row.get('score') or {}).get('task_success')))

    def evidence_failed(row):
        return (row['technical_failure'] or row['actual_native_requests'] != 1
                or row['trace_status'] != 'RECORDED_TRACE_CHECKS_PASS'
                or bool(row['integrations_after_STOP'])
                or not row.get('score') or bool(row['score']['missing_native_intervals'])
                or row['score']['native_step'] != row['native_ticks'])

    totals = {}
    for arm in arms:
        arm_rows = [r for r in rows if r['arm'] == arm]
        wins = [r for r in arm_rows if won(r)]
        families = dict(Counter(r['family'] for r in wins))
        totals[arm] = dict(sampled=len(arm_rows), holds=len(wins), holds_by_family=families,
            actual_lift20mm=sum(bool((r.get('score') or {}).get('ever_target_lift20mm')) for r in arm_rows),
            STOP=sum(bool(r['hard_STOP']) for r in arm_rows),
            original_H40_or_seam_rejects=sum(bool(r.get('valid_policy_rejection')) for r in arm_rows),
            technical_or_evidence_failures=sum(bool(evidence_failed(r)) for r in arm_rows))
    complete = len(observed) == len(expected) and all(
        r['sampled'] == per_arm and not r['technical_or_evidence_failures'] for r in totals.values())
    paired = {}
    control = totals[CONTROL]
    for arm in arms[1:]:
        matrix = Counter({'both_success': 0, 'candidate_only': 0, 'control_only': 0, 'neither': 0})
        for cluster, instruction, _ in expected[::len(arms)]:
            b = by_condition.get((cluster, instruction, CONTROL))
            c = by_condition.get((cluster, instruction, arm))
            if b is None or c is None or evidence_failed(b) or evidence_failed(c):
                continue
            matrix[('neither', 'candidate_only', 'control_only', 'both_success')[2*bool(won(b))+bool(won(c))]] += 1
        paired[arm] = dict(matrix)
    eligible = []
    if complete:
        for arm in arms[1:]:
            other = totals[arm]
            family_min = min(other['holds_by_family'].get(f, 0) for f in ('F1', 'F2'))
            if other['STOP'] > control['STOP']:
                continue
            if stage == 'D5_GRASP24':
                passed = other['holds'] >= max(4, control['holds']) and family_min >= 1
            else:
                passed = other['holds'] >= 12 and other['holds'] > control['holds'] and family_min >= 6
            if passed:
                eligible.append(arm)

    def rank(arm):
        row = totals[arm]
        return (-row['holds'], -row['actual_lift20mm'], row['STOP'],
                row['original_H40_or_seam_rejects'], CANDIDATES.index(arm))

    return dict(stage=stage, complete_comparable=complete, dispatched=len(rows), planned=len(expected),
                unstarted=len(expected)-len(rows), arms=totals, paired=paired,
                selected_arm=min(eligible, key=rank) if stage == 'D5_GRASP24' and eligible else None,
                confirmation_goal_met=bool(eligible) if stage == 'D5_CONFIRM32' and complete else None)
