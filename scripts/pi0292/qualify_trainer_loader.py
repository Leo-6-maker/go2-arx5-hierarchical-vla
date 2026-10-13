"""Exercise the actual native Trainer subclass loader with real RGB, no model."""
import argparse
import json
import os
from pathlib import Path
import resource
from types import SimpleNamespace


def run(args):
    import torch
    from .train_adapter import NativeDataset,ProcessedCollator,official_import,sha
    from .trainer_adapter import make_trainer_class,tree_identity,validate_worker_trace
    official_import(args.source)
    trainer_type=make_trainer_class()
    records={}
    for workers in (0,2):
        records[str(workers)]={}
        for step in (0,1):
            directory=Path(args.output).parent/('workers-'+str(workers)+'-resume-'+str(step))
            if directory.exists():raise FileExistsError('Preserve previous loader evidence')
            dataset=NativeDataset(args.dataset,args.source,args.cosmos,for_optimization=True,trace_root=directory)
            dataset.initialize()
            # No Trainer/model initialization is needed to exercise this exact method.
            trainer=object.__new__(trainer_type)
            trainer.args=SimpleNamespace(ignore_data_skip=False,max_steps=2,dataloader_num_workers=workers,dataloader_pin_memory=False)
            trainer.state=SimpleNamespace(global_step=step)
            trainer.train_dataset=dataset;trainer.data_collator=ProcessedCollator(dataset.processor);trainer._train_batch_size=2
            loader=trainer.get_train_dataloader();iterator=iter(loader)
            batches=[tree_identity(next(iterator)) for _ in range(4 if step else 20)]
            count=len(batches)*2
            trace=validate_worker_trace(directory,dataset,step*32,step*32+count)
            records[str(workers)][str(step)]=dict(batches=batches,trace=trace)
            del iterator,loader,trainer,dataset
        assert records[str(workers)]['0']['batches'][16:20]==records[str(workers)]['1']['batches']
    assert records['0']==records['2']
    assert not torch.cuda.is_initialized()
    u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(status='PASS_REAL_TRAINER_LOADER_CPU_ONLY',records=records,statistics_sha256=sha(Path(args.dataset)/'statistics.json'),
        model_loaded=False,model_qualification=False,CUDA_initialized=False,
        CPU_s=u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime)


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','source','cosmos','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]))
    result=run(args)
    with Path(args.output).open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({k:result[k] for k in ('status','CPU_s','model_qualification')}))


if __name__=='__main__':main()
