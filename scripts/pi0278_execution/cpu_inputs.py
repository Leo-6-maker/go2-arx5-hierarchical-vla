"""Reuse qualified FAST_CACHE2 cold-cache encoder; link matching old input entries."""
import importlib.util,json,sys,time
from pathlib import Path
import torch
SOURCE=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R2_FOCUS/code/cpu_inputs.py')
spec=importlib.util.spec_from_file_location('qualified_cpu_inputs',SOURCE)
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
read,write,sha,plain,ItemLedger=q.read,q.write,q.sha,q.plain,q.ItemLedger
STATS_SHA=q.STATS_SHA

class Inputs(q.Inputs):
    def __init__(self,root,route,ledger,purpose,processor=None):
        assert route=='FAST_CACHE2'
        super().__init__(root,route,ledger,purpose,processor)
        previous=Path(read(self.root/'RUN_CONFIG.json')['reuse_cache'])
        for uid,proof in self.proof.items():
            target=self.cache_root/(uid+'.pt');receipt=target.with_suffix('.json');source=previous/target.name;old_receipt=source.with_suffix('.json')
            if receipt.exists() or not old_receipt.is_file():continue
            item=read(old_receipt)
            assert item['condition_sha256']==proof['condition_sha256'] and item['labels_masks_sha256']==proof['labels_masks_sha256']
            assert source.stat().st_size==item['bytes']
            if not target.exists():target.symlink_to(source)
            assert target.resolve()==source.resolve()
            write(receipt,dict(item,binding_sha256=sha(self.root/'design/INPUT_BINDING.json'),reused_immutable_file=str(source),original_receipt_sha256=sha(old_receipt)))
    def __getitem__(self,index):
        uid=self.plan[index]['uid'];hit=(self.cache_root/(uid+'.json')).is_file();tick=time.perf_counter();value=super().__getitem__(index)
        with (self.cache_root/'ACCESS.jsonl').open('a') as f:f.write(json.dumps(dict(uid=uid,index=index,hit=hit,seconds=time.perf_counter()-tick))+'\n')
        return value

def loader(*a,**kw):return q.loader(*a,**kw)
