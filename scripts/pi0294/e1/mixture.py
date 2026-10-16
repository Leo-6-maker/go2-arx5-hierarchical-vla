"""Exact half SMALL8 / half qualified correction sampling for the fixed M2 arm."""
import hashlib
import json
import numpy as np

from .dataset import SmallDataset,SmallSampler
from .correction import CorrectionDataset


class MixedDataset(SmallDataset):
    def __init__(self,*args,correction_root,correction_bindings,**kwargs):
        super().__init__(*args,**kwargs)
        self.small_rows=len(self.rows);self.original_manifest_sha=self.manifest_sha
        self.correction=CorrectionDataset(correction_root,self.source,self.cosmos,split='TRAIN',for_optimization=True,
            raw_bindings=correction_bindings)
        if self.statistics_sha!=self.correction.statistics_sha:raise ValueError('MIXTURE_MUST_REUSE_ORIGINAL_STATS')
        self.rows+=self.correction.rows
        self.manifest_sha=hashlib.sha256(json.dumps(dict(original=self.manifest_sha,correction=self.correction.manifest_sha),sort_keys=True).encode()).hexdigest()
        self.make_sampler=MixedSampler

    def raw(self,index):
        return super().raw(index) if index<self.small_rows else self.correction.raw(index-self.small_rows)


class MixedSampler:
    def __init__(self,dataset,start=0,stop=512*32,seed=20261016,positioned=True):
        self.dataset,self.start,self.stop,self.seed,self.positioned=dataset,start,stop,seed,positioned
        if not 0<=start<=stop:raise ValueError('INVALID_MIXED_CURSOR_RANGE')
        class View:pass
        view=View();view.rows=dataset.rows[:dataset.small_rows];view.manifest_sha=dataset.original_manifest_sha
        self.small=SmallSampler(view,stop=stop,seed=seed,positioned=False)
        self.bank={}
        for i,(row,anchor) in enumerate(dataset.rows[dataset.small_rows:],dataset.small_rows):
            if row['split']!='TRAIN':raise ValueError('TRAIN_ONLY_MIXTURE')
            self.bank.setdefault(row['task'],{}).setdefault(row['id'],[]).append(i)
        if not self.bank or any(len(v)<2 for v in self.bank.values()):raise ValueError('CORRECTION_INDEPENDENT_SOURCE_EPISODES_REQUIRED')

    def __len__(self):return self.stop-self.start
    def index_at(self,cursor):
        if cursor%4<2:return self.small.index_at((cursor//4)*2+cursor%4)
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,int(cursor),1])))
        tasks=sorted(self.bank);task=tasks[((cursor//4)*2+cursor%2)%len(tasks)]
        bank=self.bank[task];ids=sorted(bank);values=bank[ids[int(rng.integers(len(ids)))]]
        return values[int(rng.integers(len(values)))]
    def __iter__(self):
        for cursor in range(self.start,self.stop):
            index=self.index_at(cursor);yield (cursor,index) if self.positioned else index
    def state_dict(self,consumed):
        if not self.start<=consumed<=self.stop:raise ValueError('INVALID_MIXED_CONSUMED_CURSOR')
        return dict(algorithm='E1_EXACT_HALF_SMALL8_HALF_QUALIFIED_TASK_EPISODE_WINDOW_UNIFORM_V1',
            seed=self.seed,cursor=consumed,stop=self.stop,manifest_sha256=self.dataset.manifest_sha,
            correction_episode_ids={t:sorted(v) for t,v in self.bank.items()},small=self.small.state_dict((consumed//4)*2+min(consumed%4,2)))
