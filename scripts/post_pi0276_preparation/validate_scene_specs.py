"""Static split/factor checks; no native environment imports."""
import argparse,json
from pathlib import Path
from collections import Counter
from scene_specs import static_check,DOMAINS,SPLITS,camera

def validate(rows,protected_ids=()):
    if len(rows)!=456 or len({r['scene_id'] for r in rows})!=456:raise ValueError('Scene cardinality/duplicate')
    groups={}
    for row in rows:
        if row['native_reset_seed'] in protected_ids:raise ValueError('Protected split namespace collision')
        if row['native_reset_seed'] not in SPLITS[row['split']]:raise ValueError('Split range mismatch')
        if not row['original_geometry_physics_controls_and_wrist']:raise ValueError('Undeclared physical factor')
        groups.setdefault(row['parent_group'],[]).append(row);static_check(row)
        if row['domain']!='D1_APPEARANCE' and row['appearance'] is not None:raise ValueError('Appearance factor leaked')
        if row['split']!='TEST' and row['domain']!='D2_EXTERNAL_CAMERA' and row['camera']!=camera():raise ValueError('Camera factor leaked')
    for own in groups.values():
        if len({r['split'] for r in own})!=1 or Counter(r['domain'] for r in own)!=Counter(DOMAINS):raise ValueError('Correlated variants crossed split')
        if len({json.dumps(r['layout'],sort_keys=True) for r in own})!=1 or len({r['rng']['placement'] for r in own})!=1:raise ValueError('Visual variant changed placement RNG')
    return dict(status='CPU_STATIC_SCENE_FACTOR_AND_SPLIT_PASS',groups=len(groups),cards=len(rows),native_calls=0,render_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cards',type=Path);p.add_argument('--protected-manifest',type=Path);a=p.parse_args()
    rows=[json.loads(x) for x in a.cards.read_text().splitlines()];protected=[]
    if a.protected_manifest:protected=json.loads(a.protected_manifest.read_text())['protected_native_seeds']
    print(json.dumps(validate(rows,protected)))
