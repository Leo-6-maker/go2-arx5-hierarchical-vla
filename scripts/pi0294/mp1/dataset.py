"""Original TRAIN126 sampling; qualified correction never changes task exposure."""
import hashlib
import json

import numpy as np

from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0294.e1.correction import CorrectionDataset
from scripts.pi0294.e1.scope import DATA,STATS


class SourceDataset(NativeDataset):
    def __init__(self,*args,correction_root=None,correction_bindings=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.manifest_sha!=DATA or self.statistics_sha!=STATS:raise ValueError('MP1_FROZEN_SOURCE_DATA_AND_STATS')
        originals={row['id']:row for row,_ in self.rows}
        if len(originals)!=126 or any(r['split']!='TRAIN' for r in originals.values()):raise ValueError('MP1_EXACT_ORIGINAL_TRAIN126')
        self.original_rows=len(self.rows);self.original_manifest_sha=self.manifest_sha;self.correction=None
        if correction_root is not None:
            self.correction=CorrectionDataset(correction_root,self.source,self.cosmos,split='TRAIN',for_optimization=True,raw_bindings=correction_bindings)
            if self.correction.statistics_sha!=self.statistics_sha:raise ValueError('MP1_CORRECTION_REUSES_ORIGINAL_STATS')
            self.rows+=self.correction.rows
            self.manifest_sha=hashlib.sha256(json.dumps(dict(original=self.original_manifest_sha,correction=self.correction.manifest_sha),sort_keys=True).encode()).hexdigest()
        self.make_sampler=SourceSampler

    def raw(self,index):
        return super().raw(index) if index<self.original_rows else self.correction.raw(index-self.original_rows)


class SourceSampler:
    def __init__(self,dataset,start=0,stop=2048*32,seed=20261017,positioned=True):
        if not 0<=start<=stop:raise ValueError('MP1_EXACT_SAMPLE_CURSOR_RANGE')
        self.dataset,self.start,self.stop,self.seed,self.positioned=dataset,start,stop,seed,positioned
        self.original={t:{} for t in ['T1','T2']};self.corrected={t:{} for t in ['T1','T2']}
        for index,(row,anchor) in enumerate(dataset.rows):
            if row['split']!='TRAIN' or row['task'] not in self.original:raise ValueError('MP1_TRAIN_ONLY_TWO_TASKS')
            bank=self.original if index<dataset.original_rows else self.corrected
            bank[row['task']].setdefault(row['id'],[]).append(index)
        if any(len(v)!=63 for v in self.original.values()):raise ValueError('MP1_ORIGINAL_63_EPISODES_PER_TASK')
        if any(v and len(v)<2 for v in self.corrected.values()):raise ValueError('MP1_CORRECTION_REQUIRES_INDEPENDENT_EPISODES')

    def __len__(self):return self.stop-self.start

    def index_at(self,cursor):
        if not self.start<=cursor<self.stop:raise IndexError('MP1_SAMPLER_CURSOR_OUTSIDE_BOUND_RANGE')
        task='T1' if cursor%2==0 else 'T2'
        # Alternate within each task, so T1-only corrections occupy 25%, not 50%.
        corrected=bool(self.corrected[task]) and (cursor//2)%2==1
        bank=self.corrected[task] if corrected else self.original[task]
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,int(cursor),int(corrected)])))
        ids=sorted(bank);windows=bank[ids[int(rng.integers(len(ids)))]]
        return windows[int(rng.integers(len(windows)))]

    def __iter__(self):
        for cursor in range(self.start,self.stop):
            index=self.index_at(cursor);yield (cursor,index) if self.positioned else index

    def state_dict(self,consumed):
        if not self.start<=consumed<=self.stop:raise ValueError('MP1_INVALID_CONSUMED_CURSOR')
        return dict(algorithm='MP1_TASK_BALANCED_WITHIN_TASK_CORRECTION_HALF_V1',seed=self.seed,cursor=consumed,stop=self.stop,
            manifest_sha256=self.dataset.manifest_sha,original_episode_ids={t:sorted(v) for t,v in self.original.items()},
            correction_episode_ids={t:sorted(v) for t,v in self.corrected.items()},
            overall_correction_fraction=.25*sum(bool(v) for v in self.corrected.values()))
