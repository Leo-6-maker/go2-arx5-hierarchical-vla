"""R2 lazy verified CPU cache; preserve the original input construction verbatim."""
import importlib.util,os,sys
from pathlib import Path
import torch
parent=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pi0277_v1_cpu_inputs',parent/'code/cpu_inputs.py')
v1=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v1;spec.loader.exec_module(v1)
read,write,sha,plain,ItemLedger=v1.read,v1.write,v1.sha,v1.plain,v1.ItemLedger
STATS_SHA=v1.STATS_SHA

class Inputs(v1.Inputs):
    def __init__(self,root,route,ledger,purpose,processor=None):
        super().__init__(root,'REF_W0',ledger,purpose,processor)
        self.route=route
        self.cache_root=self.root/'cache';self.cache_root.mkdir(parents=True,exist_ok=True)
    def __getitem__(self,index):
        if self.route=='REF_W0':return super().__getitem__(index)
        import fcntl
        assert 0<=index<len(self)
        self.ledger.reserve(index,self.route,self.purpose);self.initialize()
        visit=self.plan[index];uid=visit['uid'];proof=self.proof[uid];row=self.rows[uid]
        target=self.cache_root/(uid+'.pt');receipt=target.with_suffix('.json')
        with target.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not receipt.exists():
                if self.processor is None:self.processor=self.helper.build(read(self.old/'contracts/TASK/source/dataset/statistics.json'),Path(self.config['cosmos']))
                route=self.route
                try:self.route='REF_W0';original=super().__getitem__(index)
                finally:self.route=route
                value={k:v for k,v in plain(original).items() if not k.startswith('_pi')}
                temporary=target.with_suffix('.partial');torch.save(value,temporary)
                with (self.cache_root/'capacity.lock').open('a') as capacity:
                    fcntl.flock(capacity,fcntl.LOCK_EX)
                    usage=sum(p.stat().st_size for p in self.cache_root.glob('*.pt'))
                    assert usage+temporary.stat().st_size<=48*2**30,'CPU_CACHE_CAP'
                    temporary.replace(target)
                self.ledger.reserve(index,'CACHE_READBACK',self.purpose)
                restored=torch.load(target,map_location='cpu',weights_only=True)
                assert self.helper.identity(restored)==self.helper.identity(value)
                item=dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target),binding_sha256=sha(self.root/'design/INPUT_BINDING.json'),**proof)
                target.chmod(0o444);write(receipt,item)
            item=read(receipt);assert item['binding_sha256']==sha(self.root/'design/INPUT_BINDING.json') and sha(target)==item['sha256']
            value=torch.load(target,map_location='cpu',weights_only=True)
        assert self.helper.identity({k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')})==proof['condition_sha256']
        assert self.helper.identity({k:value['inputs'][k] for k in ('action','action_mask')})==proof['labels_masks_sha256']
        mask=value['inputs']['action_mask'];assert tuple(mask.shape)==(1,40,132) and torch.all(mask[...,:10]==1) and not mask[...,10:].any()
        value['_pi0264_meta']=dict(unit=visit['unit'],micro=visit['micro'],command=row['group'],condition_sha256=proof['condition_sha256'])
        value['_pi0276_visit']=dict(visit,seed=row['seed'],anchor=row['anchor'],condition_sha256=proof['condition_sha256'],labels_masks_sha256=proof['labels_masks_sha256'])
        if torch.utils.data.get_worker_info() is not None:assert not torch.cuda.is_initialized()
        return value

def loader(*args,**kwargs):return v1.loader(*args,**kwargs)
