"""Independent JSON-only reconciliation of CR1 artifacts; no source-data or model read."""
from collections import Counter
import copy
import json
from pathlib import Path

from .rules import AUTH, CANDIDATE

GATES = dict(XYZ_m=1e-5, SO3_rad=1e-4, gripper=1e-5, relative_SE3_max_abs=1e-8)
PASS = 'PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'


def transitions(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('CR1_REVIEW_UNIQUE_NONEMPTY_PARENTS')
    fit = {r['lineage'] for r in rows if r['role'] == 'FIT'}
    diagnostic = {r['lineage'] for r in rows if r['role'] == 'DIAGNOSTIC'}
    if fit & diagnostic:
        raise ValueError('CR1_REVIEW_GEOMETRIC_LINEAGE_LEAKAGE')
    for row in rows:
        if row['role'] not in ('FIT', 'DIAGNOSTIC', 'KNOWN_CHALLENGE'):
            raise ValueError('CR1_REVIEW_EXPLICIT_TRAIN_ROLES_ONLY')
        flags = []
        for key in ('R0', 'R1'):
            result = row[key]
            if result['H40_windows_actually_checked'] != 601 or result['frozen_numeric_gates'] != GATES:
                raise ValueError('CR1_REVIEW_ALL_ORIGINAL_WINDOWS_AND_GATES')
            passed = all(0 <= result['maximum_error'][k] <= limit for k, limit in GATES.items())
            if passed != (result['status'] == PASS):
                raise ValueError('CR1_REVIEW_STATUS_MUST_MATCH_EVERY_NUMERIC_GATE')
            flags.append('PASS' if passed else 'FAIL')
        if row['transition'] != '->'.join(flags):
            raise ValueError('CR1_REVIEW_PAIRED_TRANSITION_CHANGED')
    counts = {role:dict(Counter(r['transition'] for r in rows if r['role'] == role))
              for role in ('FIT', 'DIAGNOSTIC', 'KNOWN_CHALLENGE')}
    recovered = {r['lineage'] for r in rows if r['role'] == 'DIAGNOSTIC' and r['transition'] == 'FAIL->PASS'}
    fit_good = all(r['transition'].endswith('->PASS') for r in rows if r['role'] == 'FIT')
    regressions = [r['id'] for r in rows if r['transition'] == 'PASS->FAIL'
                   and (r['role'] == 'DIAGNOSTIC' or r['frozen_139_member'])]
    return dict(counts=counts,fit_all_pass=fit_good,recovered_diagnostic_lineages=len(recovered),
                regressions=regressions,supported=bool(fit) and fit_good and bool(recovered) and not regressions,
                unique_parents=len(rows),retention=dict(Counter(r['transition'] for r in rows if r['frozen_139_member'])))


def review(root):
    root = Path(root)
    read = lambda n:json.loads((root/n).read_text())
    a = read('CR1_ADOPTION_AND_INPUTS.json'); rules = read('CR1_SPLIT_AND_RULE_SEAL.json')
    candidate = read('CR1_CANDIDATE_STATISTICS_AND_EFFECTIVE_PARAMS.json')
    paired = read('CR1_PAIRED_RESULTS_AND_READER.json')
    reproduction = read('CR1_REPRODUCTION_AND_CAUSAL_TRACE.json')
    if a['protocol_release'] != AUTH or rules['authority'] != AUTH or rules['results_seen_before_seal']:
        raise ValueError('CR1_REVIEW_EXACT_APPROVAL_AND_PREDECLARATION')
    if reproduction['status'] != 'PASS_EXACT_REPRODUCTION' or reproduction['diagnostic_windows'] > 32:
        raise ValueError('CR1_REVIEW_REPRODUCTION_FIRST')
    rows = paired['rows']; result = transitions(rows)
    sealed = {r['id']:r for r in rules['main']+rules['known']}
    if set(sealed) != {r['id'] for r in rows}:
        raise ValueError('CR1_REVIEW_SEALED_PARENT_COMPLETENESS')
    for row in rows:
        if any(row[k] != sealed[row['id']][k] for k in ('role','lineage','task','scene_group')):
            raise ValueError('CR1_REVIEW_SEALED_SPLIT_UNCHANGED')
    rr = candidate['range_report']
    if (rr['candidate'] != CANDIDATE or rr['padding_searches'] != 0
            or rr['fit_windows'] != 601*sum(r['role']=='FIT' for r in rows)
            or paired['diagnostic_feedback_refits'] != 0):
        raise ValueError('CR1_REVIEW_ONLY_ONE_FIT_CANDIDATE_NO_FEEDBACK')
    expected = copy.deepcopy(candidate['original_effective_params'])
    expected['new_embodiment']['action']['eef_9d'].update(min=rr['candidate_min'], max=rr['candidate_max'])
    if expected != candidate['candidate_effective_params']:
        raise ValueError('CR1_REVIEW_EFFECTIVE_STATE_GRIPPER_OTHER_PARAMS_FROZEN')
    singles = [r for r in paired['reader'] if 'id' in r]
    mixed = [r for r in paired['reader'] if 'cross_parent_ids' in r]
    if len(singles) != 2*len(rows) or len(mixed) != 2:
        raise ValueError('CR1_REVIEW_ACTUAL_READER_COVERAGE')
    for variant in ('R0', 'R1'):
        own = [r for r in singles if r['variant']==variant]
        if ({r['id'] for r in own} != set(sealed) or len(own) != len(rows)
                or any(r['actual_anchors'] != [0,80,160,320,480,600]
                       or not r['real_dual_RGB'] or not r['deterministic_repeat'] for r in own)):
            raise ValueError('CR1_REVIEW_SIX_REAL_ANCHORS_AND_REPEAT')
        cross = [r for r in mixed if r['variant']==variant]
        if len(cross)!=1 or len(set(cross[0]['cross_parent_ids']))!=2:
            raise ValueError('CR1_REVIEW_DISTINCT_PARENT_MIXED_BATCH')
    expected_conclusion = 'RANGE_SUPPORTED' if result['supported'] else 'RANGE_NOT_SUPPORTED'
    if paired['conclusion'] != expected_conclusion:
        raise ValueError('CR1_REVIEW_CONCLUSION_MUST_FOLLOW_PREDECLARED_GATES')
    return dict(status='PASS_JSON_ONLY_INDEPENDENT_RECONCILIATION',conclusion=expected_conclusion,**result)


if __name__ == '__main__':
    import sys
    print(json.dumps(review(sys.argv[1]),indent=2))
