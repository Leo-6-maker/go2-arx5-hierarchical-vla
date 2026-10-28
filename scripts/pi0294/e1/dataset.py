"""Read-only SMALL8 selection and exact task/episode/window-uniform sampling."""
import hashlib
import json

import numpy as np

from scripts.pi0294.native_dataset import NativeDataset


def select_small8(manifest):
    groups={}
    for row in manifest['episodes']:
        if row['split']!='TRAIN' or row.get('origin')!='new':continue
        key=row.get('scene_group',row['id'].rsplit('-',1)[0])
        if key.startswith('PI0294-TRAIN-'):
            key='TRAIN/'+key.removeprefix('PI0294-TRAIN-').replace('-','/')
        elif key.startswith('PI0294/TRAIN/'):key=key.removeprefix('PI0294/')
        groups.setdefault(key,{})[row['task']]=row
    complete={k:v for k,v in groups.items() if set(v)=={'T1','T2'}}
    selected=[]
    for family in ('F0','F1'):
        mandatory='TRAIN/'+family+'/000'
        if mandatory not in complete:raise ValueError('MANDATORY_SMALL8_PAIR_MISSING:'+mandatory)
        candidates=sorted(k for k in complete if k.startswith('TRAIN/'+family+'/') and k!=mandatory)
        if not candidates:raise ValueError('NEXT_SMALL8_PAIR_MISSING:'+family)
        selected.extend([mandatory,candidates[0]])
    return [complete[group][task] for group in selected for task in ('T1','T2')]


class SmallDataset(NativeDataset):
    def __init__(self,*args,selected_ids,**kwargs):
        super().__init__(*args,**kwargs)
        ids=set(selected_ids)
        if len(ids)!=8:raise ValueError('EXACT_EIGHT_EPISODES_REQUIRED')
        self.rows=[(row,anchor) for row,anchor in self.rows if row['id'] in ids]
        if {row['id'] for row,_ in self.rows}!=ids:raise ValueError('SMALL8_IDS_NOT_IN_ORIGINAL_TRAIN')
        self.small8_sha=hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


class SmallSampler:
    def __init__(self,dataset,start=0,stop=1024*32,seed=20261016,positioned=True):
        if not 0<=start<=stop:raise ValueError('INVALID_SMALL_SAMPLER_RANGE')
        self.dataset,self.start,self.stop,self.seed,self.positioned=dataset,start,stop,seed,positioned
        self.bank={task:{} for task in ('T1','T2')}
        for i,(row,anchor) in enumerate(dataset.rows):
            if row['split']!='TRAIN' or row['task'] not in self.bank:raise ValueError('TRAIN_ONLY_SMALL8')
            self.bank[row['task']].setdefault(row['id'],[]).append(i)
        if any(len(bank)!=4 for bank in self.bank.values()):raise ValueError('FOUR_EPISODES_PER_TASK_REQUIRED')

    def __len__(self):return self.stop-self.start

    def index_at(self,cursor):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,int(cursor)])))
        bank=self.bank['T1' if cursor%2==0 else 'T2'];ids=sorted(bank)
        values=bank[ids[int(rng.integers(len(ids)))]]
        return values[int(rng.integers(len(values)))]

    def __iter__(self):
        for cursor in range(self.start,self.stop):
            index=self.index_at(cursor);yield (cursor,index) if self.positioned else index

    def state_dict(self,consumed):
        if not self.start<=consumed<=self.stop:raise ValueError('INVALID_SMALL_CONSUMED_CURSOR')
        return dict(algorithm='E1_TASK_ALTERNATE_EPISODE_UNIFORM_WINDOW_UNIFORM_V1',
            seed=self.seed,cursor=consumed,stop=self.stop,manifest_sha256=self.dataset.manifest_sha,
            episode_ids={k:sorted(v) for k,v in self.bank.items()})
