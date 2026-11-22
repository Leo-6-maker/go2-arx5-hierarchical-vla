"""Freeze a 40-window diagnostic from sealed native expert events and H40 labels."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


TRAIN052 = 'e0c5c481a95972e4ba5c9b712e1c3a1240d462ad28222a2c0e041e307fb42278'
SEED = 'MAIN3800-NATIVE-EVENT-PANEL-20261122-V1'
TAGS = ('LAST_APPROACH','P10_CLOSE_CROSSING','FIRST_REQUEST_AFTER_GRASP','BEFORE_LIFT20',
        'HOLD','CARRY','LOWER','RELEASE','RETREAT')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, sha=None):
    if sha is not None and digest(path) != sha:
        raise ValueError(f'Changed evidence: {path.name}')
    return json.loads(path.read_bytes())


def anchors(data, events):
    steps, phases = data['native_step'], data['phase']
    commands = .002 + .040*data['nominal_policy_H40'][:,:10,9]
    previous = data['previous_actual_command'][:,0]
    crosses = np.any((np.c_[previous,commands[:,:-1]] >= .022) & (commands < .022)
                     & data['consumed_mask'][:,:10],axis=1)
    first_close = np.flatnonzero(crosses)
    first_grasp = np.searchsorted(steps,events['target_grasp'],side='left')
    before_lift20 = np.searchsorted(steps,events['target_lift20mm'],side='left')-1
    found = {'FIRST_REQUEST_AFTER_GRASP':first_grasp,'BEFORE_LIFT20':before_lift20}
    if first_close.size:
        found['P10_CLOSE_CROSSING'] = close = first_close[0]
        approach = np.flatnonzero(np.isin(phases[:close],['PREGRASP','DESCEND']))
        if approach.size:
            found['LAST_APPROACH'] = approach[-1]
    for tag, names in {'HOLD':('HOLD',),'CARRY':('CARRY','TRANSPORT'),
                       'LOWER':('LOWER',),'RETREAT':('WITHDRAW','RETREAT')}.items():
        possible = np.flatnonzero(np.isin(phases,names))
        if possible.size:
            found[tag] = possible[0]
    if 'released' in events:
        found['RELEASE'] = np.searchsorted(steps,events['released'],side='right')-1
    return {tag:int(i) for tag,i in found.items() if 0 <= i < len(steps)}


def panel(population, manifest, ledgers):
    active = {r['native_result_sha256']:r['id'] for r in manifest['episodes']}
    entries = {r['id']:r for split in ('TRAIN','VAL') for r in population['conditions'][split]}
    rows = {'TRAIN':defaultdict(list),'VAL':defaultdict(list)}
    event_map = {}
    seen = set()
    for record in ledgers:
        if not record.get('admitted') or record['split'] not in rows:
            continue
        split, result = record['split'], record['result']
        key = result['sha256']
        if key in seen or (split == 'TRAIN' and key not in active):
            continue
        seen.add(key)
        actual = load(Path(result['path']),key)
        score = actual['detail']['score']
        if (actual['physical_requests'] != 1 or not actual['detail']['task_success']
                or score != record['score'] or not score['ever_target_lift20mm']):
            raise ValueError('Expected admitted successful native ground truth')
        events = {name:row['native_step'] for name,row in score['events'].items()}
        ep_pin = record['episode']
        ep_file = Path(ep_pin['path'])
        ep = load(ep_file,ep_pin['sha256'])
        condition = entries[ep['id']]
        if ep['split'] != split or ep['native_result_sha256'] != key or ep['id'] != record['condition']:
            raise ValueError('Split, episode or result binding differs')
        if split == 'TRAIN':
            event_map[key] = dict(events=events,condition=ep['id'],geometry=condition['cluster'])
        label = ep_file.parent.parent/ep['labels']['file']
        if digest(label) != ep['labels']['sha256']:
            raise ValueError('G7 label identity differs')
        with np.load(label,allow_pickle=False) as data:
            for tag, i in anchors(data,events).items():
                rows[split][tag].append(dict(tag=tag,anchor=i,tick=int(data['native_step'][i]),
                      instruction=ep['task'],family=ep['family'],scene=ep['scene_sha256'],
                      layout=condition['cluster'],episode=ep['id'],episode_pin=ep_pin,
                      result=result,label_sha256=ep['labels']['sha256']))
    selected = {}
    for split, limit in (('TRAIN',24),('VAL',16)):
        pool = rows[split]
        roster, used, layouts = [], set(), Counter()
        for _ in range(limit):
            inserted = len(roster)
            for tag in TAGS:
                if len(roster)==limit:
                    break
                options = [r for r in pool[tag] if (r['episode'],r['anchor']) not in used]
                if not options:
                    continue
                item = min(options,key=lambda r:(layouts[r['layout']],
                           hashlib.sha256(f"{SEED}/{split}/{tag}/{r['episode']}".encode()).hexdigest()))
                roster.append(item)
                used.add((item['episode'],item['anchor']))
                layouts[item['layout']] += 1
            if len(roster)==inserted:
                break
        selected[split] = roster
    train, val = selected.values()
    for key in ('layout','scene'):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError('A held-out scene overlaps the diagnostic TRAIN layout')
    if not 32<=len(train)+len(val)<=40 or not val:
        raise ValueError('Insufficient disjoint expert windows for the preregistered diagnostic')
    return dict(seed=SEED,training_manifest=TRAIN052,roster=selected,
                episode_events=event_map,new_native=0,new_generations=0,new_updates=0)


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('population',type=Path)
    cli.add_argument('manifest',type=Path)
    cli.add_argument('ledgers',type=Path,nargs='+')
    args = cli.parse_args()
    print(json.dumps(panel(load(args.population),load(args.manifest,TRAIN052),
                           [row for p in args.ledgers for row in load(p)]),sort_keys=True))
