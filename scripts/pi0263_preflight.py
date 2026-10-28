"""Finite actual processor checks for the fixed PI0263 TRAIN schedule."""
import json
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0263')
DATA = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
CODE = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
COSMOS = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/inputs/frozen_v1/cosmos_snapshot')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  NO_ALBUMENTATIONS_UPDATE='1',
                  HF_HOME='/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/hf_home')
sys.path[:0] = [str(DATA / 'R3'), str(DATA / 'runtime'), str(DATA / 'qa'),
                str(CODE / 'scripts'), str(CODE / 'src'), '/home/liuyu-nfs/Projects/isaac-gr00t']
from pi0260_r3_inputs import batch, build, identity, inventory
from pi0263_plan import read, sha, write, language, validate, seed_for, NONLANGUAGE, TEMPLATES


def main():
    import torch
    plan = read(ROOT / 'PAIRING_PLAN.json')
    validate(plan)
    processor = build(read(DATA / 'dataset/statistics.json'), COSMOS)
    registry = read(DATA / 'design/registry.json')
    families = {r['family_id']: r for r in registry['families']}
    sequences = {r['sequence_id']: r for r in registry['sequences']}

    @lru_cache(maxsize=160)
    def source(name):
        path = DATA / 'dataset' / name
        rec = read(path.with_suffix('.json'))
        assert rec['sequence'] == sequences[name] and rec['sequence']['split'] == 'TRAIN'
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            return rec, z['state_v1'].copy(), z['raw_action'].copy()

    @lru_cache(maxsize=4000)
    def encoded(name, index, text):
        rec, states, labels = source(name)
        j = next(j for j,r in enumerate(rec['rows']) if r['index'] == index)
        row = rec['rows'][j]
        assert row['active'] and row['usable']
        assert sha(DATA / row['frame']) == row['frame_sha256']
        with np.load(DATA / row['frame'], allow_pickle=False) as z:
            packet = dict(body_rgb=z['body_rgb'], wrist_rgb=z['wrist_rgb'],
                          state_v1=states[j], language=text)
        value = batch(processor, packet, labels[j])['inputs']
        assert set(value) == set(NONLANGUAGE + ('input_ids', 'attention_mask', 'action', 'action_mask'))
        assert torch.isfinite(value['action']).all()
        assert value['action'].shape == value['action_mask'].shape == (1,40,132)
        assert (value['action_mask'][...,:10] > 0).all() and (value['action_mask'][...,10:] == 0).all()
        return dict(condition_sha256=identity({k:v for k,v in value.items() if k not in ('action','action_mask')}),
                    nonlanguage_sha256=identity({k:value[k] for k in NONLANGUAGE}),
                    labels_masks_sha256=identity({k:value[k] for k in ('action','action_mask')}),
                    input_inventory=inventory(value), frame_sha256=row['frame_sha256'])

    rows, counts = [], Counter()
    first_pair = None
    for unit in plan['units']:
        pair = []
        for micro, command in enumerate(unit['order']):
            item = unit['selected'][command]
            rec, _, _ = source(item['sequence'])
            assert rec['sequence']['command'] == command and rec['sequence']['task'] == unit['task']
            color = families[unit['family']]['factors']['colors'][command]
            both = []
            for arm in ('F','X'):
                text, template = language(arm, unit, rec['sequence'], color)
                actual = encoded(item['sequence'], item['index'], text)
                assert actual['frame_sha256'] == item['frame_sha256']
                counts[arm,unit['task'],template,color,command,unit['family']] += 1
                both.append(dict(arm=arm, unit=unit['unit'], original_unit=unit['original_unit'],
                    visit=unit['visit'], micro=micro, task=unit['task'], slot=unit['slot'],
                    family=unit['family'], command=command, color=color, template=template,
                    language=text, sequence=item['sequence'], index=item['index'],
                    condition_sha256=actual['condition_sha256'],
                    nonlanguage_sha256=actual['nonlanguage_sha256'], labels_masks_sha256=actual['labels_masks_sha256']))
            assert all(both[0][k] == both[1][k] for k in ('nonlanguage_sha256','labels_masks_sha256'))
            if unit['task'] == 'M3':
                assert both[0]['condition_sha256'] == both[1]['condition_sha256']
            pair.append(both)
            rows.extend(both)
        if unit['slot'] == 'PAIR_SLOT':
            assert pair[0][0]['nonlanguage_sha256'] == pair[1][0]['nonlanguage_sha256']
        if unit['task'] == 'M1' and first_pair is None:
            first_pair = pair
    assert len(rows) == 8000
    by_target = {}
    for r in rows:
        if r['arm'] == 'X' and r['task'] == 'M1':
            key = (r['original_unit'],r['command'],r['sequence'],r['index'],r['nonlanguage_sha256'],r['labels_masks_sha256'])
            by_target.setdefault(key,Counter())[r['template']] += 1
    assert len(by_target) == 1000 and all(v == {'T0':1,'T1':1} for v in by_target.values())
    assert Counter(r['template'] for r in rows if r['arm']=='X' and r['task']=='M1') == {'T0':1000,'T1':1000}
    for unit in range(1,2001):
        def random_packet():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed_for(unit))
                return identity({'epsilon':torch.randn(1,40,132),'tau':torch.rand(1),'S4_noise':torch.randn(1,40,132)})
        assert random_packet() == random_packet()

    # ponytail: negative controls use this real schedule and processor, no second test framework.
    negative = []
    def rejected(name, check):
        try:
            check()
        except (AssertionError, ValueError):
            negative.append(name)
        else:
            raise AssertionError(('NEGATIVE_CONTROL_ACCEPTED',name))
    sample = first_pair[0][1]
    rec, states, labels = source(sample['sequence'])
    unit = plan['units'][sample['unit']-1]
    wrong_color = next(c for c in families[unit['family']]['factors']['colors'] if c != sample['color'])
    rejected('wrong_color', lambda: language('X',unit,rec['sequence'],wrong_color))
    bad_units=[dict(u) for u in plan['units']]
    bad_units[sample['unit']-1]['x_template']='T1' if unit['x_template']=='T0' else 'T0'
    rejected('wrong_template', lambda: validate(dict(plan,units=bad_units)))
    item = unit['selected'][sample['command']]
    row = next(r for r in rec['rows'] if r['index']==item['index'])
    j = rec['rows'].index(row)
    with np.load(DATA / row['frame'],allow_pickle=False) as z:
        packet = dict(body_rgb=z['body_rgb'],wrist_rgb=z['wrist_rgb'],state_v1=states[j],language=sample['language'])
    rejected('audit_metadata_leak', lambda: batch(processor,dict(packet,source_id='PI0263'),labels[j]))
    for name, changed in [('duplicated_unit',plan['units'][:-1]+[plan['units'][-2]]),
                           ('skipped_unit',plan['units'][1:])]:
        rejected(name,lambda changed=changed:validate(dict(plan,units=changed)))
    def wrong_scale():
        x=torch.tensor(.3,requires_grad=True)
        a,b=(x-1.)**2,(x+.4)**2
        correct=torch.autograd.grad((a+b)/2,x,retain_graph=True)[0]
        wrong=torch.autograd.grad(a+b,x)[0]
        assert torch.equal(correct,wrong)
    rejected('wrong_half_loss',wrong_scale)
    other=first_pair[1][1]
    def false_cache():
        assert (sample['condition_sha256'],sample['labels_masks_sha256']) == (other['condition_sha256'],other['labels_masks_sha256'])
    rejected('false_cache_reuse',false_cache)
    write(ROOT / 'PROCESSED_PLAN_AUDIT.json',dict(schema='PI0263_ACTUAL_PROCESSOR_PLAN_V1',
        status='PASS_FINITE_ACTUAL_COLOR_INPUT_LABEL_TEMPLATE_RNG_CHECKS',
        plan_sha256=sha(ROOT/'PAIRING_PLAN.json'), processor_statistics_sha256=sha(DATA/'dataset/statistics.json'),
        conditioning_keys=list(NONLANGUAGE+('input_ids','attention_mask')),
        label_keys=['action','action_mask'],negative_controls=negative,rows=rows,
        counts=[dict(arm=a,task=t,template=v,color=c,command=k,family=f,n=n) for (a,t,v,c,k,f),n in sorted(counts.items())],
        CPU_packet_reproducible=True, GPU_actual_packet_requirement='compare canary and main S4 native epsilon/tau/stock noise bytes',
        model_queries=0,optimizer_updates=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(status='PASS_PI0263_PREFLIGHT',actual_examples=8000,negative_controls=negative)))


if __name__=='__main__':
    main()
