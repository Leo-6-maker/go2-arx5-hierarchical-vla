"""Count actual phase and native-event exposures from the MAIN update journal."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


WIDTHS = {'target':64,'robodojo':32,'libero':32}
TRAIN052 = 'e0c5c481a95972e4ba5c9b712e1c3a1240d462ad28222a2c0e041e307fb42278'


def load(path, sha=None):
    raw = path.read_bytes()
    if sha is not None and hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f'Changed evidence: {path.name}')
    return json.loads(raw)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def episode_rows(manifests, events):
    rows = []
    for name, path, sha in manifests:
        for ep in load(path,sha)['episodes']:
            label = path.parent/ep['labels']['file']
            if hashlib.sha256(label.read_bytes()).hexdigest() != ep['labels']['sha256']:
                raise ValueError('Label SHA differs from frozen membership')
            key = ep.get('native_result_sha256') or ep.get('native_result',{}).get('sha256')
            record = events.get(key)
            with np.load(label,allow_pickle=False) as data:
                phases, steps = data['phase'], data['native_step']
                if len(phases) != ep['windows'] or phases.tolist() != ep['expert_stage_audit_only']:
                    raise ValueError('Label order differs from the actual reader')
                due = None
                if 'previous_actual_command' in data.files:
                    width = .002+.040*data['nominal_policy_H40'][:,:10,9]
                    previous = data['previous_actual_command'][:,0]
                    consumed = data['consumed_mask'][:,:10]
                    due = np.any((np.c_[previous,width[:,:-1]] >= .022) & (width < .022) & consumed,axis=1)
                grasp = record['events']['target_grasp'] if record else None
                lift = record['events']['target_lift20mm'] if record else None
                after = int(np.searchsorted(steps,grasp,side='left')) if grasp is not None else None
                for anchor, phase in enumerate(phases):
                    tags = [str(phase)]
                    if due is not None and due[anchor]:
                        tags.append('P10_CLOSE_CROSSING')
                    if grasp is not None and anchor == after:
                        tags.append('FIRST_REQUEST_AFTER_GRASP')
                    if grasp is not None and grasp <= steps[anchor] < lift:
                        tags.append('GRASP_TO_LIFT20')
                    rows.append((name,ep['id'],ep['family'],ep['task'],tuple(tags),record is not None))
    if len({(a,b) for a,b,*_ in rows}) != 78:
        raise ValueError('Expected the actual 26 D2 plus 52 adopted Core episodes')
    return rows


def fold(selected, rows):
    visits = defaultdict(Counter)
    episodes = defaultdict(set)
    cells = defaultdict(Counter)
    for i,count in selected.items():
        source, episode, family, task, tags, native = rows[i]
        values = ['ALL', *tags]
        if native:
            values.append('EXACT_NATIVE_EVENT_AVAILABLE')
        for tag in values:
            visits[source][tag] += count
            episodes[(source,tag)].add(episode)
            cells[(source,tag)][f'{family}/{task}'] += count
    return {source:{tag:dict(visits=count,episodes=len(episodes[source,tag]),cells=cells[source,tag])
                    for tag,count in sorted(counts.items())} for source,counts in sorted(visits.items())}


def auxiliary_design(bank, rows):
    """Dry-run a possible paired curriculum with exactly two ordinary slots retained."""
    public = dict(A=Counter(),B=Counter())
    selections = []
    for phase in range(3801,4301):
        start = (phase-2001)*WIDTHS['target']
        indices = bank[start:start+WIDTHS['target']]
        shift = 4*((2000+phase-1)%16)
        a = tuple(range(shift,shift+4))
        positions = [(shift+i)%64 for i in range(64)]
        keys = ('FIRST_REQUEST_AFTER_GRASP','P10_CLOSE_CROSSING')
        first = next((i for i in positions if i not in a[1:] and keys[0] in rows[indices[i]][4]),a[0])
        second = next((i for i in positions if i not in (first,*a[2:]) and keys[1] in rows[indices[i]][4]),a[1])
        b = (first,second,*a[2:])
        if len(set(b)) != 4:
            raise ValueError('Event design lost an ordinary or unique slot')
        selections.append(dict(update=phase,A=a,B=b,target_slice_sha256=digest(indices)))
        for arm, slots in (('A',a),('B',b)):
            for tag in keys:
                hits = sum(tag in rows[indices[i]][4] for i in slots)
                public[arm][tag+'_visits'] += hits
                public[arm][tag+'_updates'] += bool(hits)
    return dict(design_only=True,updates_per_arm=500,aux_per_arm=2000,target_FM_per_arm=32000,
                source_FM_per_arm=16000,counts=public,selections=selections)


def audit(root, custody, events):
    assets = custody/'assets.verified'
    banks_dir = custody/'banks.verified'
    spec = load(assets/'sla-adapt5k-d1-single-lineage-vanda-20261120-01/slaadp2250-02/train.SPEC.private.json')['data']['target']
    pins = {'C20':spec['legacy']['manifest'],'LEGACY_PLACE':spec['legacy']['admission'],
            'D1':spec['captures'][0]['manifest'],'D2':spec['captures'][1]['manifest']}
    manifests = [(name,assets/pin['path'].split('/runs/WM02/',1)[1],pin['sha256']) for name,pin in pins.items()]
    course = load(root/'curriculum/course02000.json')
    if course['adopted_at'] != 2000 or course['manifest_sha256'] != TRAIN052:
        raise ValueError('Expected the actual 2001 adoption')
    manifests.append(('TRAIN052',Path(course['folder'])/'MANIFEST.private.json',TRAIN052))
    rows = episode_rows(manifests,events)
    boundary = sum(row[0] != 'TRAIN052' for row in rows)
    if boundary != 2031:
        raise ValueError('Original D2 index boundary differs')
    original = load(banks_dir/'BANKS.private.json')['arrays']
    banks = {k:np.load(banks_dir/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in WIDTHS}
    for key, bank in banks.items():
        if digest(bank) != original[key]['sha256']:
            raise ValueError('Initial bank SHA changed')
    newer = Path(course['bank_root'])/'bank.verified'
    proof = load(newer/'BANK.private.json')
    target = np.load(newer/'TARGET.npy',mmap_mode='r',allow_pickle=False)
    if digest(target) != proof['bank_sha256'] or proof['target_manifest'] != TRAIN052:
        raise ValueError('Published TRAIN052 bank SHA changed')

    spans = []
    for first, last in ((1,2000),(2001,3800),(3801,3900)):
        fm = Counter()
        aux = Counter()
        output = dict.fromkeys(WIDTHS,0)
        aggregate = hashlib.sha256()
        for phase in range(first,last+1):
            path = root/'MAIN/updates'/f'phase{phase:05d}.json'
            raw = path.read_bytes()
            record = json.loads(raw)
            value = record['result']
            result_sha = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                                  allow_nan=False).encode()).hexdigest()
            if (record['status'] != 'COMPLETE' or record['message_id'] != path.stem or record['run_uuid'] != 'MAIN'
                    or record['result_sha256'] != result_sha or value['phase_step'] != phase
                    or value['visits'] != WIDTHS or value['full_H40'] != 4):
                raise ValueError('Update identity or realized denominator changed')
            aggregate.update(hashlib.sha256(raw).digest())
            for domain, width in WIDTHS.items():
                bank, offset = (target,2000) if domain == 'target' and phase > 2000 else (banks[domain],0)
                start = (phase-1-offset)*width
                indices = bank[start:start+width]
                stats = value['domains'][domain]
                if len(indices) != width or digest(indices) != stats['visit_slice_sha256']:
                    raise ValueError(f'Journal and visit bank differ at {phase}/{domain}')
                output[domain] += width
                if domain == 'target':
                    slots = stats['auxiliary_slots']
                    expected = 4*((2000+phase-1)%16)
                    if slots != list(range(expected,expected+4)) or stats['training_full_generations'] != 4:
                        raise ValueError('Recorded auxiliary routing or count differs')
                    fm.update(indices.tolist())
                    aux.update(indices[slots].tolist())
        spans.append(dict(first=first,last=last,journal_sha256=aggregate.hexdigest(),visits=output,
                          H40_auxiliary=sum(aux.values()),fm=fold(fm,rows),auxiliary=fold(aux,rows)))
    return dict(spans=spans,adopted=TRAIN052,new_updates=0,new_generations=0,new_native=0,
                possible_auxiliary_design=auxiliary_design(target,rows))


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('custody',type=Path)
    cli.add_argument('events',type=Path,nargs='?')
    args = cli.parse_args()
    print(json.dumps(audit(args.root,args.custody,load(args.events) if args.events else {}),sort_keys=True))
