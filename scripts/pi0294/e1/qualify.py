"""Actual CPU-only original processing and consumed SMALL8 resume cursor checks."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import ProcessedCollator,official_import
from scripts.pi0294.native_train import tree_identity
from .dataset import SmallDataset
from .trainer import make_trainer_class,check_trace
from .scope import validate,Ledger


def run(root):
    root=Path(root);validate(root,'hopper-neural');started=time.process_time();ledger=Ledger(root,'hopper')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('E1_QUALIFICATION_CPU_ONLY')
    b=read(root/'NEURAL_BINDINGS.private.json');official_import(b['source'])
    import torch
    torch.set_num_threads(1);records=[];output=root/'evidence/SMALL8_LOADER_QUALIFICATION.json'
    if output.exists():raise FileExistsError('PRESERVE_PRIOR_LOADER_QUALIFICATION')
    try:
        for step in [0,1]:
            trace=root/'evidence'/('SMALL8_CPU_TRACE_'+str(step))
            dataset=SmallDataset(b['dataset'],b['source'],b['cosmos'],for_optimization=True,raw_bindings=read(b['raw_bindings']),
                trace_root=trace,selected_ids=[r['id'] for r in read(root/'plans/SMALL8_BINDING.json')['episodes']])
            dataset.initialize();trainer=object.__new__(make_trainer_class())
            trainer.args=SimpleNamespace(ignore_data_skip=False,max_steps=2);trainer.state=SimpleNamespace(global_step=step)
            trainer.train_dataset=dataset;trainer._train_batch_size=2;trainer.data_collator=ProcessedCollator(dataset.processor)
            loader=trainer.get_train_dataloader();batches=[]
            for batch in loader:
                action=batch['inputs']['action'];mask=batch['inputs']['action_mask']
                if tuple(action.shape)!=(2,40,132) or not bool((mask[...,:10]==1).all()) or not bool((mask[...,10:]==0).all()):
                    raise ValueError('ACTUAL_E1_NATIVE_ACTION_MASK_OR_BATCH_CHANGED')
                batches.append(tree_identity(batch))
            records.append(dict(step=step,batches=batches,trace=check_trace(trace,dataset,step*32,64)))
        if records[0]['batches'][16:]!=records[1]['batches']:raise ValueError('ACTUAL_E1_CURSOR_RESUME_DIFFERS')
        if torch.cuda.is_initialized():raise PermissionError('CPU_LOADER_INITIALIZED_CUDA')
        value=dict(status='PASS_ACTUAL_E1_SMALL8_NATIVE_CPU_LOADER_RESUME',records=records,model_loaded=False,optimizer_updates=0,
            process_death_fault_tests='scripts.pi0294.test_e1',CPU_s=time.process_time()-started)
        write(output,value);return dict(status=value['status'],CPU_s=value['CPU_s'],batches=[len(r['batches']) for r in records])
    finally:ledger.cpu(time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(run(a.root)),flush=True)
