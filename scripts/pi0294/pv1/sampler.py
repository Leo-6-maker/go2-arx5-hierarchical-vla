"""Matched task/C1/retained-original cursors for the fixed PV1 comparisons."""
import hashlib

import numpy as np

from scripts.pi0294.scene_cards import digest


ARMS = {'COVERAGE_FIRST':('A_CONTROL','B_COVERAGE'), 'CLEARANCE_PAIR':('A_4MM','B_12MM')}


def lane_at(cursor,route,arm):
    if type(cursor) is not int or cursor<0 or route not in ARMS or arm not in ARMS[route]:
        raise PermissionError('PV1_FIXED_ARM_AND_NONNEGATIVE_SAMPLE_CURSOR')
    slot=cursor%32
    task='T1' if slot%2==0 else 'T2'
    ordinal=slot//2
    if task=='T1' and ordinal%2:
        lane='C1_T1'
    elif route=='COVERAGE_FIRST' and arm=='A_CONTROL':
        lane='original_'+task
    elif task=='T1':
        lane='original_T1' if ordinal<8 else 'new_T1'
    else:
        lane='original_T2' if ordinal<8 else 'new_T2'
    return task,lane


class MatchedSampler:
    def __init__(self,dataset,start=0,stop=4096*32,seed=20261019,positioned=True):
        self.dataset=dataset
        self.route,self.arm=dataset.PV1_route,dataset.PV1_arm
        if (self.route not in ARMS or self.arm not in ARMS[self.route] or type(start) is not int
                or type(stop) is not int or not 0<=start<=stop<=4096*32
                or seed not in (20261019,20261020)):
            raise PermissionError('PV1_FIXED_MATCHED_SAMPLER_RECIPE_OR_EXACT_RESUME_CURSOR')
        self.start,self.stop,self.seed,self.positioned=start,stop,seed,positioned
        self.bank={}
        self.provenance={}
        for index,(row,anchor) in enumerate(dataset.rows):
            lane=row['PV1_lane']
            if row['split']!='TRAIN' or lane not in ['original_T1','original_T2','C1_T1','new_T1','new_T2']:
                raise PermissionError('PV1_SAMPLER_ONLY_FROZEN_TRAIN_BANKS')
            expected='T1' if lane.endswith('T1') else 'T2'
            if row['task']!=expected:
                raise PermissionError('PV1_LANE_MUST_MATCH_ACTUAL_TASK')
            # Teacher variants share scene identity, not pixels or physical labels.
            parent=row['scene_group'] if lane.startswith('new_') else row['id']
            bank=self.bank.setdefault(lane,{})
            bank.setdefault(parent,[]).append((int(anchor),index))
            identity=(lane,parent)
            if identity in self.provenance and self.provenance[identity]!=row['id']:
                raise PermissionError('PV1_ONE_EPISODE_PER_PARENT_IN_EACH_FROZEN_LANE')
            self.provenance[identity]=row['id']
        required={lane_at(c,self.route,self.arm)[1] for c in range(32)}
        if any(not self.bank.get(k) for k in required):
            raise PermissionError('PV1_REQUIRED_BANK_NOT_READY')
        for lane,parents in self.bank.items():
            for parent,values in parents.items():
                values.sort()
                if len({a for a,_ in values})!=len(values):
                    raise PermissionError('PV1_UNIQUE_LEGAL_WINDOW_ANCHORS')
        self.support_sha256=digest({k:{p:[a for a,_ in v] for p,v in sorted(parents.items())}
            for k,parents in sorted(self.bank.items())})

    def __len__(self):return self.stop-self.start

    def selection_at(self,cursor):
        if type(cursor) is not int or not self.start<=cursor<self.stop:
            raise IndexError('PV1_SAMPLE_CURSOR_OUTSIDE_FROZEN_SESSION')
        task,lane=lane_at(cursor,self.route,self.arm)
        lane_seed=int.from_bytes(hashlib.sha256(lane.encode()).digest()[:4],'big')
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,cursor,lane_seed])))
        bank=self.bank[lane]
        parents=sorted(bank)
        parent=parents[int(rng.integers(len(parents)))]
        windows=bank[parent]
        anchor,index=windows[int(rng.integers(len(windows)))]
        return dict(task=task,lane=lane,parent=parent,anchor=anchor,index=index)

    def index_at(self,cursor):return self.selection_at(cursor)['index']

    def __iter__(self):
        for cursor in range(self.start,self.stop):
            index=self.index_at(cursor)
            yield (cursor,index) if self.positioned else index

    def state_dict(self,consumed):
        if not self.start<=consumed<=self.stop:
            raise ValueError('PV1_CONSUMED_CURSOR_MUST_BE_ACTUAL_COMMITTED_RANGE')
        return dict(algorithm='PV1_MATCHED_ABSOLUTE_CURSOR_LANE_PARENT_WINDOW_V1',route=self.route,arm=self.arm,
            seed=self.seed,cursor=consumed,stop=self.stop,manifest_sha256=self.dataset.manifest_sha,
            support_sha256=self.support_sha256,
            lane_parents={k:sorted(v) for k,v in sorted(self.bank.items())},
            metadata_not_model_input=True,teacher_variants_same_parent_anchor_not_same_state=True)
