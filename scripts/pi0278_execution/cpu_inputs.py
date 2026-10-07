"""Qualified ordered CPU encoder/cache; add verified cold entries on demand."""
import fcntl,importlib.util,sys,time
from pathlib import Path
import torch

SOURCE=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R2_FOCUS/code/cpu_inputs.py')
spec=importlib.util.spec_from_file_location('qualified_cpu_inputs',SOURCE)
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
read,write,sha,plain,ItemLedger,collate_one=q.read,q.write,q.sha,q.plain,q.ItemLedger,q.collate_one
STATS_SHA=q.STATS_SHA

class Inputs(q.Inputs):
    def __init__(self,root,route,ledger,purpose,processor=None):
        assert route=='FAST_CACHE2'
        super().__init__(root,'REF_W0',ledger,purpose,processor)
        self.route=route;self.cache={'members':{}}
        self.previous=read(Path(read(self.root/'RUN_CONFIG.json')['reuse_cache'])/'MANIFEST.json')['members']
        (self.root/'cache').mkdir(exist_ok=True)
    def __getitem__(self,index):
        uid=self.plan[index]['uid'];target=self.root/'cache'/(uid+'.pt');receipt=target.with_suffix('.json');tick=time.perf_counter();hit=True
        with target.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not receipt.exists():
                if uid in self.previous:
                    item=self.previous[uid];source=Path(read(self.root/'RUN_CONFIG.json')['reuse_cache'])/item['file']
                    assert sha(source)==item['sha256'] and item['condition_sha256']==self.proof[uid]['condition_sha256'] and item['labels_masks_sha256']==self.proof[uid]['labels_masks_sha256']
                    if not target.exists():target.symlink_to(source)
                    assert target.resolve()==source.resolve()
                else:
                    hit=False;self.route='REF_W0';self.initialize()
                    if self.processor is None:self.processor=self.helper.build(read(self.old/'contracts/TASK/source/dataset/statistics.json'),Path(self.config['cosmos']))
                    value=super().__getitem__(index);self.route='FAST_CACHE2'
                    value={k:v for k,v in plain(value).items() if not k.startswith('_pi')}
                    temp=target.with_suffix('.partial');torch.save(value,temp);temp.replace(target);target.chmod(0o444)
                    self.ledger.reserve(index,'CACHE_READBACK',self.purpose)
                    restored=torch.load(target,map_location='cpu',weights_only=True)
                    assert self.helper.identity(restored)==self.helper.identity(value)
                    item=dict(file=target.name,sha256=sha(target),bytes=target.stat().st_size,**self.proof[uid])
                item=dict(item,file=target.name,binding_sha256=sha(self.root/'design/INPUT_BINDING.json'));write(receipt,item)
            item=read(receipt);assert item['binding_sha256']==sha(self.root/'design/INPUT_BINDING.json')
            self.cache['members'][uid]=item
        value=super().__getitem__(index)
        # ponytail: one bounded append per CPU worker; consumption still belongs to Trainer.
        with (self.root/'cache/ACCESS.jsonl').open('a') as f:
            import json,os
            f.write(json.dumps(dict(uid=uid,index=index,hit=hit,seconds=time.perf_counter()-tick,pid=os.getpid()))+'\n')
        assert not torch.cuda.is_initialized() if torch.utils.data.get_worker_info() is not None else True
        return value

def loader(dataset,start=0,stop=None,pin_memory=True):
    assert start%2==0 and 0<=start<=len(dataset)
    return torch.utils.data.DataLoader(dataset,batch_size=1,sampler=range(start,len(dataset) if stop is None else stop),num_workers=2,
        collate_fn=collate_one,pin_memory=pin_memory,generator=torch.Generator().manual_seed(27803001),multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True)
