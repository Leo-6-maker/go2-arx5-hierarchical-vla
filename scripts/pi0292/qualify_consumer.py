"""Cross-host actual RGB/processor comparison using fixed producer evidence."""
import argparse
import json
import os
from pathlib import Path
import resource


def qualify(args):
    import torch
    from torch.utils.data import DataLoader,Subset
    from .train_adapter import NativeDataset,ProcessedCollator,read,sha
    from .qualify_adapter import inventory,passthrough
    torch.set_num_threads(1)
    original_call=torch.nn.Module.__call__
    reference=read(args.reference)
    bindings=read(args.raw_bindings)
    dataset=NativeDataset(args.dataset,args.source,args.cosmos,args.split,raw_bindings=bindings)
    required={row['id'] for row,_ in dataset.rows}
    if not required.issubset(bindings):raise ValueError('ALL_SPLIT_RAW_BINDINGS_REQUIRED')
    if dataset.statistics_sha!=reference['statistics_sha256']:raise ValueError('PRODUCER_STATS_CHANGED')
    if reference['split']!=args.split:raise ValueError('WRONG_PRODUCER_SPLIT')
    dataset.initialize();before=sha(Path(args.dataset)/'statistics.json')
    results={}
    for workers in (0,2):
        loader=DataLoader(Subset(dataset,reference['indices']),batch_size=2,num_workers=workers,collate_fn=passthrough,
            multiprocessing_context='spawn' if workers else None,generator=torch.Generator().manual_seed(29200))
        results[str(workers)]=[inventory(ProcessedCollator(dataset.processor)(rows)) for rows in loader]
    if results!=reference['batches']:raise ValueError('CROSS_HOST_REAL_TENSOR_MISMATCH')
    if sha(Path(args.dataset)/'statistics.json')!=before:raise ValueError('FROZEN_STATISTICS_CHANGED')
    if torch.cuda.is_initialized() or torch.nn.Module.__call__ is not original_call or 'native_dataset' in __import__('sys').modules:
        raise ValueError('CONSUMER_MUST_BE_CPU_ONLY_WITHOUT_LEGACY_GUARD')
    own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(status='PASS_ACTUAL_CROSS_HOST_RGB_PROCESSOR_TENSORS',split=args.split,
        all_split_raw_bindings=len(required),indices=reference['indices'],batches=results,
        producer_receipt_sha256=sha(args.reference),manifest_sha256=dataset.manifest_sha,statistics_sha256=before,
        frozen_stats_unchanged=True,workers=[0,2],model_loaded=False,model_qualification=False,
        CPU_s=own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('dataset','source','cosmos','raw-bindings','reference','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=['TRAIN','DEV'],required=True);args=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    result=qualify(args)
    with Path(args.output).open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({k:result[k] for k in ('status','split','CPU_s')}))
