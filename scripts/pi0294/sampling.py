"""Position-addressable task/family/episode/stage sampling; never model input."""
import hashlib
import json

import numpy as np

PHASES = ('approach', 'contact', 'transport', 'withdraw', 'hold')
WEIGHTS = (.20, .25, .25, .20, .10)


class BalancedSampler:
    def __init__(self, dataset, seed=29400, start=0, stop=6144*32, positioned=True, smoke=False):
        if not 0 <= start <= stop:
            raise ValueError('INVALID_CONSUMED_RANGE')
        self.dataset, self.seed, self.start, self.stop = dataset, int(seed), int(start), int(stop)
        self.positioned, self.smoke = positioned, smoke
        self.identity = dataset.manifest_sha
        self.bank = {}
        self.empty = []
        for index, (row, anchor) in enumerate(dataset.rows):
            task, family, origin = row.get('task', 'T1'), row.get('family', 'F0'), row.get('origin', 'legacy')
            stages = row.get('stage_anchors')
            if stages is None:
                if not smoke:
                    raise ValueError('FORMAL_STAGE_ANCHORS_REQUIRED')
                phase = PHASES[min(4, anchor*5//601)]
            else:
                matches = [p for p in PHASES if anchor in stages.get(p, [])]
                if len(matches) != 1:
                    raise ValueError('EACH_ANCHOR_REQUIRES_EXACTLY_ONE_OFFLINE_STAGE')
                phase = matches[0]
            key = (task, origin, family, row['id'])
            self.bank.setdefault(key, {}).setdefault(phase, []).append(index)
        tasks = {k[0] for k in self.bank}
        self.single_task_smoke = smoke and tasks == {'T1'}
        if tasks != {'T1', 'T2'} and not (smoke and tasks == {'T1'}):
            raise ValueError('BOTH_TASKS_REQUIRED_FOR_FORMAL_SAMPLING')
        if not smoke and not any(k[:2] == ('T1', 'new') for k in self.bank):
            raise ValueError('NEW_T1_REQUIRED')
        for key, phases in self.bank.items():
            self.empty.extend(dict(task=key[0],origin=key[1],family=key[2],episode_id=key[3],phase=p)
                              for p in PHASES if not phases.get(p))

    def __len__(self):
        return self.stop-self.start

    def index_at(self, cursor):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, int(cursor)])))
        task = 'T1' if self.single_task_smoke or cursor % 2 == 0 else 'T2'
        keys = [k for k in self.bank if k[0] == task]
        # Balance families before old/new exposure, otherwise legacy F0 alone
        # would absorb one third of all T1 samples before family balancing.
        families = sorted({k[2] for k in keys})
        family = families[int(rng.integers(len(families)))]
        keys = [k for k in keys if k[2] == family]
        if task == 'T1' and not self.single_task_smoke:
            # At most one third of T1 positions are eligible, and only within
            # the already sampled family; actual old F0 exposure is lower.
            origin = 'legacy' if (cursor//2) % 3 == 2 and any(k[1]=='legacy' for k in keys) else 'new'
            keys = [k for k in keys if k[1] == origin]
        keys = sorted(keys)
        key = keys[int(rng.integers(len(keys)))]
        phases = self.bank[key]
        weights = np.asarray([w if phases.get(p) else 0 for p,w in zip(PHASES,WEIGHTS)])
        phase = PHASES[int(rng.choice(5,p=weights/weights.sum()))]
        values = phases[phase]
        return values[int(rng.integers(len(values)))]

    def __iter__(self):
        for cursor in range(self.start,self.stop):
            index = self.index_at(cursor)
            yield (cursor,index) if self.positioned else index

    def state_dict(self, consumed):
        if not self.start <= consumed <= self.stop:
            raise ValueError('INVALID_CONSUMED_CURSOR')
        return dict(seed=self.seed,cursor=int(consumed),stop=self.stop,manifest_sha256=self.identity,
                    smoke=self.smoke,algorithm='PI0294_POSITION_TASK_FAMILY_EPISODE_STAGE_V1' if self.single_task_smoke else
                        'PI0294_POSITION_FAMILY_FIRST_OLD_EXPOSURE_CAP_V2',
                    empty_bucket_policy='RENORMALIZE_AVAILABLE_PHASES_WITH_EXPLICIT_EMPTY_LIST',
                    empty_bucket_sha256=hashlib.sha256(json.dumps(self.empty,sort_keys=True).encode()).hexdigest())
