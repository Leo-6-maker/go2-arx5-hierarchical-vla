"""Actual two-worker cache path and byte identity; no neural model or optimizer."""
import argparse,json,os,hashlib
from pathlib import Path


def check(root):
    import torch
    from train import PlannedDataset,collate_with_metadata
    from cpu_inputs import tensor_hash,collate
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4);root=Path(root)
    ds=PlannedDataset(root/'data',root/'data/MORE_PLAN.jsonl')
    assert callable(ds.cached_item) and ds.base.namespace=='318f1f013bbfe42c24bb4546688db1ef872272c810c4738ff3d526fcf2a81e58'
    expected=[]
    for i in range(8):
        item=dict(ds[i]);assert item.pop('_pi0282_visit')==ds.plan[i]
        plain=ds.base[i];assert tensor_hash(item)==tensor_hash(plain)
        expected.append(item)
    loader=torch.utils.data.DataLoader(ds,batch_size=4,sampler=range(8),collate_fn=collate_with_metadata,num_workers=2,multiprocessing_context='spawn')
    hashes=[]
    for i,batch in enumerate(loader):
        assert batch.pop('_pi0282_visits')==ds.plan[i*4:i*4+4]
        h=tensor_hash(batch['inputs']);assert h==tensor_hash(collate(expected[i*4:i*4+4])['inputs']);hashes.append(h)
    assert len(hashes)==2
    out=dict(status='PASS_ACTUAL_TWO_WORKER_CACHE_BINDING_INPUT_LABEL_BYTES',batches=hashes,cache_namespace=ds.base.namespace,model_queries=0,optimizer_updates=0,physics=0)
    (root/'evidence/CACHE_BINDING_REPAIR_CPU.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();check(a.root)
