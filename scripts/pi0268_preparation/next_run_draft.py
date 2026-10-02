"""Pure config sketch and metadata validation. This release cannot launch anything."""
import argparse
import copy
import json
from pathlib import Path


def proposed_rotation_contract(statistics):
    """Not installed: theoretical bounds only for verified row-orthonormal rotation6D."""
    out=copy.deepcopy(statistics)
    for block in ('state','relative_action'):
        params=out['new_embodiment'][block]['eef_9d']
        for field,number in (('min',-1.),('q01',-1.),('max',1.),('q99',1.)):
            rows=[params[field]] if block=='state' else params[field]
            for row in rows:row[3:9]=[number]*6
    return out


def validate_panels(fit,factors):
    assert fit['launch_allowed'] is factors['launch_allowed'] is False
    rows=fit['rows'];assert len(rows)==112 and len({r['condition_sha256'] for r in rows})==112
    for task in ('M1','M3'):
        for command in (0,1):
            for variant in range(7):
                own=[r for r in rows if (r['task'],r['command'],r['variant'])==(task,command,variant)]
                assert len(own)==4 and all(r['split']=='TRAIN' and r['seen_by_R'] for r in own)
                if variant==0:assert all(r['seen_by_N'] for r in own)
    conditions=factors['conditions'];assert len(conditions)==304 and len({r['condition_id'] for r in conditions})==304
    assert all(r['base']['split']=='TRAIN' and r['matching_rgb']=='REQUIRES_FUTURE_AUTHORIZED_RENDER' for r in conditions)
    bases={(r['base']['family'],r['base']['task'],r['base']['command'],r['base']['anchor']) for r in conditions};assert len(bases)==16
    assert fit['proposed_queries']==672 and factors['proposed_queries']==912
    return dict(status='PASS_METADATA_ONLY_DRY_RUN',fit_inputs=112,factor_specs=304,proposed_queries=1584,attempt_cap=1664,head_cap=6656,render_pairs=304,launch_allowed=False)


def reject_launch(card,authorization_id=None):
    # No model import or submission path exists in a preparation release.
    raise PermissionError('NOT_AUTHORIZED: PI0268 is preparation only; a separate scientific release and approved run card are required.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--authorization-id');args=parser.parse_args()
    read=lambda name:json.loads((args.directory/name).read_text())
    if not args.dry_run:reject_launch(read('NEXT_RUN_DRAFT.json'),args.authorization_id)
    print(json.dumps(validate_panels(read('SEEN_TRAIN_112_MANIFEST.json'),read('FACTOR_304_MANIFEST.json')),indent=2))
