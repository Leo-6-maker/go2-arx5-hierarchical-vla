"""Read-only development candidate review; never interprets DEV as final TEST."""
import argparse
import json
from pathlib import Path
from .supervisor import atomic, field, verify_file

SCHEMA = 'CORE_NIGHT10_DEV_POLICY_REVIEW_V1'


def review(card):
    if card.get('schema') != SCHEMA:
        raise ValueError('EXPLICIT_DEV_POLICY_REVIEW_SCHEMA')
    criteria = card.get('criteria', {})
    if (type(criteria.get('minimum_valid')) is not int or criteria['minimum_valid'] < 4
            or type(criteria.get('minimum_complete_successes')) is not int
            or criteria['minimum_complete_successes'] < 2
            or criteria.get('require_zero_hard_safety') is not True):
        raise ValueError('PREDECLARED_NONTRIVIAL_DEV_CANDIDATE_GATE')
    results = []
    seen = set()
    for row in card.get('candidates', []):
        cid = row['id']
        if cid in seen:
            raise ValueError('DUPLICATE_POLICY_CANDIDATE')
        seen.add(cid)
        checkpoint = row['checkpoint']
        if not verify_file(checkpoint):
            results.append(dict(id=cid, status='NO_QUALIFIED_CHECKPOINT'))
            continue
        report = row['evaluation']
        if not Path(report['path']).is_file() or Path(report['path']).is_symlink():
            results.append(dict(id=cid, status='NO_NATIVE_DEV_RESULT'))
            continue
        d = json.loads(Path(report['path']).read_bytes())
        mapping = report['fields']
        if set(mapping) != {'valid', 'successes', 'unrun', 'hard_safety'}:
            raise ValueError('EXACT_VALID_SUCCESS_UNRUN_AND_SAFETY_FIELDS')
        valid, successes, unrun, hard = [field(d, mapping[k]) for k in ('valid', 'successes', 'unrun', 'hard_safety')]
        if (any(type(x) is not int for x in (valid, successes, unrun, hard))
                or not 0 <= successes <= valid <= row['panel_conditions']
                or unrun != row['panel_conditions'] - valid or not 0 <= hard <= valid):
            raise ValueError('NATIVE_DEV_COUNTS_NOT_COHERENT:' + cid)
        qualified = (valid >= criteria['minimum_valid']
                     and successes >= criteria['minimum_complete_successes'] and hard == 0)
        results.append(dict(id=cid, status='DEVELOPMENT_GATE_PASS' if qualified else 'DEVELOPMENT_GATE_FAIL',
                            valid=valid, successes=successes, unrun=unrun, hard_safety=hard,
                            checked_native_dev_only=True, source_receipt=report['path']))
    eligible = [r for r in results if r['status'] == 'DEVELOPMENT_GATE_PASS']
    eligible.sort(key=lambda x: (-x['successes'], -x['successes']/x['valid'], x['id']))
    return dict(schema=SCHEMA, status='COMPLETE_DEV_REVIEW',
                provisional_policy=None if not eligible else eligible[0]['id'],
                candidates=results, policy_qualified_for_target_WBC=False,
                unseen_confirmation_not_opened=True, original_DEV_not_final_TEST=True,
                no_automatic_safety_or_score_relaxation=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--card',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    card=json.loads(Path(a.card).read_bytes())
    result=review(card)
    out=Path(a.output)
    if out.exists():
        raise FileExistsError('DO_NOT_REWRITE_PRIOR_DEV_SELECTION')
    atomic(out,result)
    print(json.dumps(dict(status=result['status'],provisional_policy=result['provisional_policy'])))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
