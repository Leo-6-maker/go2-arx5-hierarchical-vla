"""TRAIN-only CPU adapter. Importing this module imports neither torch nor a trainer."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib,importlib,json,os,sys,time

RELEASE='44aac6acce75c777f6777e5a5f4dc83a9d7f426b'
ITEM_CAP=32768

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

@dataclass(frozen=True)
class RunConfig:
    root:str
    cosmos:str
    module_paths:tuple
    sources:tuple
    statistics_sha256:str
    updates:int=8192
    loader_seed:int=27603001
    cache:str=''
    data_layout:str='staged_train_bank'
    def __post_init__(self):
        if type(self.updates) is not int or self.updates<=0 or not self.sources:raise ValueError('Explicit positive update limit and source bindings required')
        if self.data_layout not in ('staged_train_bank','card_episode'):raise ValueError('Explicit TRAIN layout required')
    @property
    def micro_stop(self):return 2*self.updates
    def next_micro(self,completed):
        if type(completed) is not int or not 0<=completed<=self.updates:raise ValueError('Optimizer-boundary cursor required')
        return 2*completed
    @classmethod
    def load(cls,path):
        v=read(path)
        if v.pop('authority_release')!=RELEASE:raise PermissionError('CPU preparation authority mismatch')
        v['module_paths']=tuple(v['module_paths']);v['sources']=tuple(v['sources'])
        return cls(**v)

def cpu_guard():
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('Preparation process already initialized CUDA')
    torch.set_num_threads(1)
    if not getattr(torch.nn.Module.__call__,'preparation_guard',False):
        original=torch.nn.Module.__call__
        def guarded(self,*args,**kwargs):
            if type(self).__module__.startswith('torchvision.transforms'):return original(self,*args,**kwargs)
            raise PermissionError('NO_NEW_NEURAL_FORWARD_IN_CPU_PREPARATION')
        guarded.preparation_guard=True;torch.nn.Module.__call__=guarded

class ItemLedger:
    """One shared ordinal counts attempted work, including discarded prefetch."""
    def __init__(self,path,context):
        self.path=str(path);Path(path).parent.mkdir(parents=True,exist_ok=True)
        previous=[json.loads(x)['ordinal'] for x in Path(path).read_text().splitlines()] if Path(path).exists() else []
        if previous!=list(range(1,len(previous)+1)):raise ValueError('Item ledger is not contiguous')
        self.counter=context.Value('q',len(previous))
    def reserve(self,index):
        with self.counter.get_lock():
            if self.counter.value>=ITEM_CAP:raise RuntimeError('Cumulative32768 item cap exhausted')
            ordinal=self.counter.value+1
            fd=os.open(self.path,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
            try:
                raw=(json.dumps(dict(ordinal=ordinal,index=index,pid=os.getpid()))+'\n').encode()
                if os.write(fd,raw)!=len(raw):raise IOError('Partial item-budget reservation')
            finally:os.close(fd)
            self.counter.value=ordinal

class TrainDataset:
    def __init__(self,config,ledger):
        self.config=config;self.ledger=ledger;root=Path(config.root)
        self.plan=read(root/'design/TRAIN_PLAN.json')['stream']
        if len(self.plan)!=config.micro_stop:raise ValueError('Schedule length mismatch')
        card=read(root/'design/DATASET_CARD.json')
        admitted={e['seed'] for e in card['episodes'] if e['split']=='TRAIN' and e['admitted']}
        self.episodes={e['seed']:e['path'] for e in card['episodes'] if e['split']=='TRAIN' and e['admitted']}
        self.rows={r['uid']:r for r in card['rows'] if r['split']=='TRAIN' and r['eligible'] and r['seed'] in admitted}
        self.proof={r['uid']:r for r in read(root/'design/PROCESSED_INPUTS.json')['rows'] if r['uid'] in self.rows}
        if any(v['uid'] not in self.rows or v['micro_index']!=i for i,v in enumerate(self.plan)):raise ValueError('Non-TRAIN or reordered schedule')
        self.processor=None;self.helper=None
    def __len__(self):return len(self.plan)
    def __getstate__(self):
        value=self.__dict__.copy();value['processor']=value['helper']=None;return value
    def initialize(self):
        if self.processor is not None:return
        cpu_guard()
        for row in self.config.sources:
            if sha(row['path'])!=row['sha256']:raise ValueError('Qualified CPU source changed: '+row['path'])
        sys.path[:0]=list(self.config.module_paths)
        self.helper=importlib.import_module('pi0260_r3_inputs')
        stat=Path(self.config.root)/'contracts/TASK/source/dataset/statistics.json'
        if sha(stat)!=self.config.statistics_sha256:raise ValueError('Frozen statistics changed')
        self.processor=self.helper.build(read(stat),Path(self.config.cosmos))
    @lru_cache(maxsize=96)
    def episode(self,seed,command_hash,observation_hash):
        import numpy as np
        folder=Path(self.episodes[seed]) if self.config.data_layout=='card_episode' else Path(self.config.root)/'source/data'/f'TRAIN-{seed}'
        if sha(folder/'commands.npz')!=command_hash or sha(folder/'observations.npz')!=observation_hash:raise ValueError('Episode source hash changed')
        if self.config.cache:
            target=Path(self.config.cache)/f'TRAIN-{seed}';receipt=read(target/'CACHE.json')
            if receipt['source_hashes']!=[command_hash,observation_hash]:raise ValueError('Memmap source identity mismatch')
            if receipt['statistics_sha256']!=self.config.statistics_sha256 or receipt['processor_sources']!=list(self.config.sources):raise ValueError('Memmap processor identity mismatch')
            arrays={k:np.load(target/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in receipt['arrays']}
            for key,row in receipt['arrays'].items():
                if sha(target/(key+'.npy'))!=row['sha256']:raise ValueError('Memmap bytes changed')
            return arrays['action'],{k:v for k,v in arrays.items() if k!='action'}
        with np.load(folder/'commands.npz',allow_pickle=False) as z:commands=z['action'].copy()
        with np.load(folder/'observations.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in ('state_v1','base_camera_96','hand_camera_96')}
        return commands,obs
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        self.ledger.reserve(index);self.initialize()
        import numpy as np
        begin=time.perf_counter();visit=self.plan[index];r=self.rows[visit['uid']];j=r['anchor']
        commands,obs=self.episode(r['seed'],r['command_sha256'],r['observation_sha256']);loaded=time.perf_counter()
        packet=dict(body_rgb=obs['base_camera_96'][j],wrist_rgb=obs['hand_camera_96'][j],state_v1=obs['state_v1'][j],language='Stack the red cube on the green cube and release it.')
        label=commands[j:j+40].astype(np.float32)
        if label.shape!=(40,10):raise ValueError('No genuine H40 future')
        value=self.helper.batch(self.processor,packet,label);processed=time.perf_counter()
        condition={k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')}
        labels={k:value['inputs'][k] for k in ('action','action_mask')};proof=self.proof[r['uid']]
        if self.helper.identity(condition)!=proof['condition_sha256'] or self.helper.identity(labels)!=proof['labels_masks_sha256']:raise ValueError('Actual processor/hash mismatch')
        value['_pi0264_meta']=dict(unit=visit['unit'],micro=visit['micro'],command=r['group'],condition_sha256=proof['condition_sha256'])
        value['_pi0276_visit']=dict(visit,seed=r['seed'],anchor=j,condition_sha256=proof['condition_sha256'],labels_masks_sha256=proof['labels_masks_sha256'])
        return dict(value=value,visit=value['_pi0276_visit'],timing=dict(io_s=loaded-begin,processor_s=processed-loaded,hash_s=time.perf_counter()-processed),raw_bytes=sum(x.nbytes for x in packet.values() if isinstance(x,np.ndarray))+label.nbytes)

def collate_one(rows):
    if len(rows)!=1:raise ValueError('CPU preparation preserves batch1')
    return rows[0]

@dataclass
class Consumption:
    next_micro:int=0
    pending:int=0
    def consume(self,index):
        if index!=self.next_micro+self.pending:raise ValueError('Out-of-order actual consumption')
        self.pending+=1
        if self.pending==2:self.next_micro+=2;self.pending=0
    def checkpoint(self):
        if self.pending:raise ValueError('Checkpoint must be an optimizer boundary')
        return dict(global_step=self.next_micro//2,next_micro=self.next_micro)

def prepare_memmap(config,destination):
    """Optional same-array bank; this never precomputes neural features."""
    import numpy as np
    root=Path(config.root);dst=Path(destination);dst.mkdir(parents=True,exist_ok=True)
    card=read(root/'design/DATASET_CARD.json');total=0;receipts=[]
    for e in card['episodes']:
        if e['split']!='TRAIN' or not e['admitted']:continue
        source=Path(e['path']) if config.data_layout=='card_episode' else root/'source/data'/f"TRAIN-{e['seed']}"
        out=dst/f"TRAIN-{e['seed']}";out.mkdir(exist_ok=True)
        source_hashes=[sha(source/n) for n in ('commands.npz','observations.npz')];arrays={}
        for filename,keys in [('commands.npz',['action']),('observations.npz',['state_v1','base_camera_96','hand_camera_96'])]:
            with np.load(source/filename,allow_pickle=False) as z:
                for key in keys:
                    a=z[key];total+=a.nbytes+256
                    if total>20*2**30:raise RuntimeError('Preparation cache ceiling exceeded')
                    p=out/(key+'.npy');np.save(p,a,allow_pickle=False)
                    b=np.load(p,mmap_mode='r',allow_pickle=False)
                    if b.dtype!=a.dtype or b.shape!=a.shape or not np.array_equal(a,b):raise ValueError('Memmap array parity failed')
                    arrays[key]=dict(sha256=sha(p),bytes=p.stat().st_size,dtype=str(a.dtype),shape=list(a.shape))
        receipt=dict(source_hashes=source_hashes,arrays=arrays,statistics_sha256=config.statistics_sha256,processor_sources=config.sources)
        (out/'CACHE.json').write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(dict(seed=e['seed'],receipt=sha(out/'CACHE.json')))
    return dict(status='CPU_SAME_ARRAY_MEMMAP',episodes=receipts,derived_bytes=total,neural_features=False)
