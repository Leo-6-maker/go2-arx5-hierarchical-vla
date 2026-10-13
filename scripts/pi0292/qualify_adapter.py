"""Compare real processed tensors in isolated reference and training processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile


def inventory(value):
    import numpy as np
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray):return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if hasattr(value,'items'):return {str(k):inventory(v) for k,v in sorted(value.items())}
    if isinstance(value,(tuple,list)):return [inventory(v) for v in value]
    return value


def passthrough(rows):return rows


def produce(args):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader,Subset
    original_call=torch.nn.Module.__call__
    torch.set_num_threads(1)
    from .train_adapter import NativeDataset,ProcessedCollator,CyclicSampler,read,sha
    before=sha(Path(args.dataset)/'statistics.json')
    if args.mode=='reference':
        sys.path.insert(0,str(Path(args.frozen_code)/'scripts/pi0291_arx5_data'))
        from native_dataset import DatasetLoader
        dataset=DatasetLoader(args.dataset,args.source,args.cosmos,args.split)
    else:
        dataset=NativeDataset(args.dataset,args.source,args.cosmos,args.split,for_optimization=args.split=='TRAIN')
    dataset.initialize()
    count=len(dataset)
    indices=[base+anchor for base in (0,601) for anchor in (0,80,160,320,480,600) if base+anchor<count]
    subset=Subset(dataset,indices)
    results={}
    for workers in (0,2):
        loader=DataLoader(subset,batch_size=2,num_workers=workers,collate_fn=passthrough,
            multiprocessing_context='spawn' if workers else None,generator=torch.Generator().manual_seed(29200))
        batches=[]
        for rows in loader:
            processed=[row['processed'] for row in rows]
            batch=dataset.processor.collator(processed)
            action=batch['inputs']['action'];mask=batch['inputs']['action_mask']
            assert tuple(action.shape)==(len(rows),40,132)
            assert bool((mask[...,:10]==1).all()) and bool((mask[...,10:]==0).all())
            assert bool((batch['inputs']['state'][...,10:]==0).all())
            batches.append(inventory(batch))
        results[str(workers)]=batches
    assert results['0']==results['2']
    negative={}
    if args.mode=='adapter':
        assert torch.nn.Module.__call__ is original_call
        assert 'native_dataset' not in sys.modules
        assert dataset.get_dataset_statistics()==read(Path(args.dataset)/'statistics.json')
        for name,call in {
            'DEV_optimizer':lambda:NativeDataset(args.dataset,args.source,args.cosmos,'DEV',for_optimization=True),
            'diagnostic_split':lambda:NativeDataset(args.dataset,args.source,args.cosmos,'DIAGNOSTIC'),
            'identity_collation':lambda:ProcessedCollator(dataset.processor)([{'processed':{},'identity':{}}])}.items():
            try:call()
            except ValueError:negative[name]='REJECTED'
            else:raise AssertionError(name)
        complete=list(CyclicSampler(count,stop=count*2))
        assert sorted(complete[:count])==sorted(complete[count:])==list(range(count))
        for cursor in (1,31,32,count-1,count,count+7):
            assert list(CyclicSampler(count,start=cursor,stop=count*2))==complete[cursor:]
        negative['sampler_all_windows_each_cycle']='PASS'
        negative['sampler_resume_cursors']='PASS'
    assert sha(Path(args.dataset)/'statistics.json')==before
    assert not torch.cuda.is_initialized()
    forbidden=[n for n in sys.modules if n.startswith('gr00t') and any(x in n for x in ('modeling_','.backbone.','.action_head.'))]
    assert not forbidden,forbidden
    own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(mode=args.mode,split=args.split,indices=indices,batches=results,negative=negative,
        statistics_sha256=before,CPU_s=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,
        model_load_forward=0,CUDA_initialized=False)


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','source','cosmos','frozen-code','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=['reference','adapter','compare'],default='compare')
    p.add_argument('--split',choices=['TRAIN','DEV'],default='TRAIN');a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    if a.mode!='compare':result=produce(a)
    else:
        directory=Path(a.output).parent
        directory.mkdir(parents=True,exist_ok=True)
        records={}
        for mode in ('reference','adapter'):
            dest=directory/(a.split+'_'+mode+'.json')
            if dest.exists():raise FileExistsError('Preserve earlier qualification evidence')
            cmd=[sys.executable,'-m','scripts.pi0292.qualify_adapter','--mode',mode,'--split',a.split]
            for key in ('dataset','source','cosmos','frozen_code'):cmd+=['--'+key.replace('_','-'),getattr(a,key)]
            subprocess.run(cmd+['--output',str(dest)],check=True,timeout=600)
            records[mode]=json.loads(dest.read_text())
        assert records['reference']['batches']==records['adapter']['batches']
        assert records['reference']['statistics_sha256']==records['adapter']['statistics_sha256']
        result=dict(status='PASS_REAL_SPLIT_TENSOR_EQUIVALENCE',split=a.split,records=records,
            CPU_s=sum(v['CPU_s'] for v in records.values()),model_qualification=False,
            source_reference_process_separate=True,no_legacy_guard_in_adapter=True)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','mode','split','CPU_s') if k in result}))


if __name__=='__main__':main()
