"""Actual CPU processor/loader qualification of original TRAIN126 and fixed C1."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha,ProcessedCollator
from scripts.pi0294.native_train import tree_identity
from scripts.pi0294.e1.trainer import make_trainer_class,check_trace
from .dataset import SourceDataset,SourceSampler
from .scope import validate,charge_cpu,DATA,STATS


def prepare(root):
    root=Path(root);validate(root,'hopper-neural');started=time.process_time()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));out=root/'evidence/source-126-preparation';out.mkdir(mode=0o700)
    try:
        import torch
        torch.set_num_threads(1)
        b=read(root/'NEURAL_BINDINGS.private.json');bank=Path(b['dataset']);m=read(bank/'DATASET_MANIFEST.json')
        if sha(bank/'DATASET_MANIFEST.json')!=DATA or sha(bank/'statistics.json')!=STATS:raise ValueError('MP1_ORIGINAL_DATA_IDENTITY')
        train=[r for r in m['episodes'] if r['split']=='TRAIN'];dev=[r for r in m['episodes'] if r['split']=='DEV']
        if len(train)!=126 or len(dev)!=32:raise ValueError('MP1_ORIGINAL_TRAIN126_DEV32_MEMBERSHIP')
        for row in train:
            if sha(bank/row['file'])!=row['sha256']:raise ValueError('MP1_SOURCE_NUMERIC_CHANGED')
            with np.load(bank/row['file'],allow_pickle=False) as z:
                if (z['label_representation'].item()!='ABSOLUTE_COMMITTED_EEF10' or len(z['committed_absolute10'])<row['windows']+39
                        or not np.array_equal(np.flatnonzero(z['valid_window']),np.arange(row['windows']))):raise ValueError('MP1_SOURCE_FULL_H40_WINDOW_SEMANTICS')
        correction=read(root/'plans/SOURCE_CORRECTION_BINDING.private.json')
        if correction['status']!='PASS_ACTUAL_HOPPER_CORRECTION_PROCESSOR_LOADER' or sha(Path(correction['dataset'])/'DATASET_MANIFEST.json')!=correction['manifest_sha256']:
            raise ValueError('MP1_PREVIOUS_ACTUAL_CORRECTION_CONSUMPTION_REQUIRED')
        records=[]
        for branch,mixed in [('S-BC',False),('S-CORR',True)]:
            args=dict(for_optimization=True,raw_bindings=read(b['raw_bindings']))
            if mixed:args.update(correction_root=correction['dataset'],correction_bindings=read(correction['raw_bindings']))
            dataset=SourceDataset(bank,b['source'],b['cosmos'],**args);dataset.initialize();batches=[];traces=[]
            for step in [0,1]:
                dataset.trace_root=out/(branch+'-trace-'+str(step));dataset.trace_root.mkdir()
                trainer=object.__new__(make_trainer_class());trainer.args=SimpleNamespace(ignore_data_skip=False,max_steps=2)
                trainer.state=SimpleNamespace(global_step=step);trainer.train_dataset=dataset;trainer._train_batch_size=2;trainer.data_collator=ProcessedCollator(dataset.processor)
                actual=[tree_identity(v) for v in trainer.get_train_dataloader()]
                batches.append(actual);trace=check_trace(dataset.trace_root,dataset,step*32,64);counts=Counter();origin=Counter()
                for c,index in SourceSampler(dataset,start=step*32,stop=64):
                    counts[dataset.rows[index][0]['task']]+=1;origin['correction' if index>=dataset.original_rows else 'original']+=1
                traces.append(dict(resume_step=step,trace=trace,task_samples=dict(counts),origin_samples=dict(origin)))
            if batches[0][16:]!=batches[1]:raise ValueError('MP1_ACTUAL_NEW_SAMPLER_LOADER_RESUME_ORDER')
            records.append(dict(branch=branch,source_episodes=len(train),task_episodes=dict(Counter(r['task'] for r in train)),
                combined_manifest_sha256=dataset.manifest_sha,correction_manifest_sha256=correction['manifest_sha256'] if mixed else None,
                sampler=SourceSampler(dataset).state_dict(0),actual_loader=traces,actual_resume_batches_identical=True))
        if torch.cuda.is_initialized():raise PermissionError('MP1_CPU_DATA_LOADER_INITIALIZED_CUDA')
        old_recipe=read(root/'plans/PARENT_E1_RECIPE.json');recipe=dict(old_recipe,max_steps=2048,seed=20261017,
            parent='AUDITED_M1_AT_1024_INFERENCE_WEIGHTS_ONLY',new_optimizer=True,exact_resume=False,warmup_updates=32)
        write(root/'plans/SOURCE_LEARNING_RECIPE.json',recipe)
        write(root/'plans/SOURCE_TRAIN126_BINDING.json',dict(original_manifest_sha256=DATA,statistics_sha256=STATS,
            episodes=[{k:r[k] for k in ['id','task','split','file','sha256','windows']} for r in train],DEV_training_labels_used=False))
        value=dict(status='PASS_ACTUAL_MP1_TRAIN126_AND_CORRECTION_NATIVE_CPU_LOADERS',records=records,models_loaded=0,optimizer_updates=0,
            source_TRAIN=126,original_DEV=32,DEV_numeric_or_RGB_loaded=False,statistics_sha256=STATS,CPU_s=time.process_time()-started)
        write(out/'RESULT.json',value);return value
    finally:charge_cpu(root,time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=prepare(a.root);print(json.dumps(dict(status=v['status'],CPU_s=v['CPU_s'])),flush=True)
