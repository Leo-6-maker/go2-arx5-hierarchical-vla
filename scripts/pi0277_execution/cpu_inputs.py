"""Exact CPU processor caching and ordered access; no Trainer or model imports."""
import hashlib, importlib, json, os, time
from functools import lru_cache
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

ITEM_CAP=65536
STATS_SHA='b3da3d500d514ce3d1ce83001de30846d9ab76d699dc48831db8e4367ce98572'

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.partial')
    with temp.open('w') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(p)

class ItemLedger:
    # ponytail: one owner plus its two workers; a shared counter and append log suffice.
    def __init__(self,path,context):
        self.path=str(path);Path(path).parent.mkdir(parents=True,exist_ok=True)
        rows=[json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []
        assert [r['ordinal'] for r in rows]==list(range(1,len(rows)+1))
        self.counter=context.Value('q',len(rows))
    def reserve(self,index,route,purpose):
        with self.counter.get_lock():
            if self.counter.value>=ITEM_CAP:raise RuntimeError('PI0277_CPU_ITEM_CAP_EXHAUSTED')
            ordinal=self.counter.value+1
            with open(self.path,'a') as f:
                f.write(json.dumps(dict(ordinal=ordinal,index=index,route=route,purpose=purpose,pid=os.getpid()))+'\n')
                f.flush();os.fsync(f.fileno())
            self.counter.value=ordinal

def plain(value):
    if isinstance(value,torch.Tensor):
        assert value.device.type=='cpu';return value.detach().contiguous()
    if isinstance(value,dict) or hasattr(value,'items'):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,list):return [plain(v) for v in value]
    if isinstance(value,tuple):return tuple(plain(v) for v in value)
    assert value is None or type(value) in (int,float,str,bool),type(value)
    return value

class Inputs(Dataset):
    def __init__(self,root,route,ledger,purpose,processor=None):
        assert route in ('REF_W0','FAST_CACHE0','FAST_CACHE2')
        self.root=Path(root);self.route=route;self.ledger=ledger;self.purpose=purpose
        self.config=read(self.root/'design/INPUT_BINDING.json');self.old=Path(self.config['old_root'])
        self.plan=read(self.root/'design/TRAIN_PLAN.json')['stream'];assert len(self.plan)==8192
        needed={x['uid'] for x in self.plan};card=read(self.old/'design/DATASET_CARD.json')
        admitted={e['seed'] for e in card['episodes'] if e['split']=='TRAIN' and e['admitted']}
        self.rows={r['uid']:r for r in card['rows'] if r['uid'] in needed and r['eligible'] and r['split']=='TRAIN' and r['seed'] in admitted}
        self.proof={r['uid']:r for r in read(self.old/'design/PROCESSED_INPUTS.json')['rows'] if r['uid'] in needed}
        assert set(self.rows)==set(self.proof)==needed
        self.processor=processor;self.helper=None
        if route!='REF_W0':
            self.cache=read(self.root/'cache/MANIFEST.json');assert self.cache['status']=='PASS_COMPLETE_PROCESSED_CACHE_PARITY'
            assert self.cache['binding_sha256']==sha(self.root/'design/INPUT_BINDING.json')
            assert self.cache['plan_sha256']==sha(self.root/'design/TRAIN_PLAN.json')
            assert set(self.cache['members'])==needed
    def __len__(self):return len(self.plan)
    def __getstate__(self):
        v=self.__dict__.copy();v['processor']=None;v['helper']=None;return v
    def initialize(self):
        if self.helper is not None:return
        import sys
        for row in self.config['sources']:assert sha(row['path'])==row['sha256'],row['path']
        sys.path[:0]=self.config['module_paths']
        self.helper=importlib.import_module('pi0260_r3_inputs')
        assert str(Path(self.helper.__file__).resolve())==self.config['input_helper']
        if self.route=='REF_W0' and self.processor is None:
            statistics=self.old/'contracts/TASK/source/dataset/statistics.json';assert sha(statistics)==STATS_SHA
            self.processor=self.helper.build(read(statistics),Path(self.config['cosmos']))
    @lru_cache(maxsize=96)
    def episode(self,seed,command_hash,observation_hash):
        import numpy as np
        folder=self.old/'source/data'/f'TRAIN-{seed}'
        assert sha(folder/'commands.npz')==command_hash and sha(folder/'observations.npz')==observation_hash
        with np.load(folder/'commands.npz',allow_pickle=False) as z:commands=z['action'].copy()
        with np.load(folder/'observations.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in ('state_v1','base_camera_96','hand_camera_96')}
        return commands,obs
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        self.ledger.reserve(index,self.route,self.purpose);self.initialize()
        visit=self.plan[index];row=self.rows[visit['uid']];proof=self.proof[row['uid']]
        if self.route=='REF_W0':
            import numpy as np
            commands,obs=self.episode(row['seed'],row['command_sha256'],row['observation_sha256']);i=row['anchor']
            packet=dict(body_rgb=obs['base_camera_96'][i],wrist_rgb=obs['hand_camera_96'][i],state_v1=obs['state_v1'][i],language='Stack the red cube on the green cube and release it.')
            value=self.helper.batch(self.processor,packet,commands[i:i+40].astype(np.float32))
        else:
            member=self.cache['members'][row['uid']];path=self.root/'cache'/member['file']
            assert sha(path)==member['sha256'];value=torch.load(path,map_location='cpu',weights_only=True)
        condition={k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')}
        labels={k:value['inputs'][k] for k in ('action','action_mask')}
        assert self.helper.identity(condition)==proof['condition_sha256']
        assert self.helper.identity(labels)==proof['labels_masks_sha256']
        assert value['inputs']['action'].shape==(1,40,132)
        mask=value['inputs']['action_mask'];assert torch.all(mask[...,:10]==1) and not mask[...,10:].any()
        value['_pi0264_meta']=dict(unit=visit['unit'],micro=visit['micro'],command=row['group'],condition_sha256=proof['condition_sha256'])
        value['_pi0276_visit']=dict(visit,seed=row['seed'],anchor=row['anchor'],condition_sha256=proof['condition_sha256'],labels_masks_sha256=proof['labels_masks_sha256'])
        # Prefetch is attempted input work only; the training parent owns consumption journals.
        if torch.utils.data.get_worker_info() is not None:assert not torch.cuda.is_initialized()
        return value
    def get_dataset_statistics(self):return read(self.old/'contracts/TASK/source/dataset/statistics.json')

def collate_one(rows):
    assert len(rows)==1;return rows[0]

def loader(dataset,start=0,stop=None,pin_memory=True):
    assert 0<=start<=len(dataset) and start%2==0
    workers=2 if dataset.route=='FAST_CACHE2' else 0
    options=dict(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True) if workers else {}
    return DataLoader(dataset,batch_size=1,sampler=range(start,len(dataset) if stop is None else stop),num_workers=workers,
                      collate_fn=collate_one,pin_memory=pin_memory,generator=torch.Generator().manual_seed(27703001),**options)

def build_cache(root,ledger):
    root=Path(root);dest=root/'cache';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'MANIFEST.json').exists():
        saved=read(dest/'MANIFEST.json')
        assert saved['status']=='PASS_COMPLETE_PROCESSED_CACHE_PARITY'
        for item in saved['members'].values():assert sha(dest/item['file'])==item['sha256']
        return saved
    source=Inputs(root,'REF_W0',ledger,'cache_original_and_parity');members={};total=0
    for i,v in enumerate(source.plan):
        uid=v['uid']
        if uid in members:continue
        # An interrupted cache build resumes only verified immutable members.
        target=dest/(uid+'.pt');receipt=target.with_suffix('.json')
        if receipt.exists():
            item=read(receipt);assert sha(target)==item['sha256']
            assert item['binding_sha256']==sha(root/'design/INPUT_BINDING.json')
            assert item['labels_masks_sha256']==source.proof[uid]['labels_masks_sha256']
        else:
            value=source[i];value={k:v for k,v in plain(value).items() if not k.startswith('_pi')}
            temp=target.with_suffix('.partial');torch.save(value,temp)
            if total+temp.stat().st_size>48*2**30:raise RuntimeError('PI0277_CACHE_CAP_EXHAUSTED')
            temp.replace(target);ledger.reserve(i,'CACHE_READBACK','cache_original_and_parity')
            restored=torch.load(target,map_location='cpu',weights_only=True)
            assert source.helper.identity(restored)==source.helper.identity(value)
            target.chmod(0o444)
            item=dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target),
                      binding_sha256=sha(root/'design/INPUT_BINDING.json'),**source.proof[uid]);write(receipt,item)
        members[uid]=item;total+=item['bytes'];assert total<=48*2**30
        if len(members)%256==0:print('CACHE_PARITY',len(members),total,flush=True)
    result=dict(status='PASS_COMPLETE_PROCESSED_CACHE_PARITY',members=members,bytes=total,unique_conditions=len(members),
                binding_sha256=sha(root/'design/INPUT_BINDING.json'),plan_sha256=sha(root/'design/TRAIN_PLAN.json'),
                neural_features=False,random_tensors_cached=False,actual_item_reservations=ledger.counter.value)
    write(dest/'MANIFEST.json',result);return result
