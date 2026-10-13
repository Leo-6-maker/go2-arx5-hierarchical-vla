"""Bounded frozen-bank export and official CPU qualification, no model or physics."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from scripts.pi0292.durable import read,write
from .bank import validate_cpu_ticket,build
from .native_dataset import sha,official_import,NativeDataset,ProcessedCollator
from .native_train import tree_identity


def qualify(args,output):
    """Actual native TRAIN/DEV tensors plus all-window codec and exact new loader."""
    official_import(args.source)
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0291_arx5_data'))
    from scripts.pi0291_arx5_data.qualify_loader import codec_all_windows
    datasets={split:NativeDataset(args.output,args.source,args.cosmos,split,
        for_optimization=split=='TRAIN') for split in ('TRAIN','DEV')}
    checks={}
    for split,dataset in datasets.items():
        dataset.initialize()
        # Every admitted task/family/origin stratum gets real tensors, not only
        # the old first two T1 episodes at the beginning of the manifest.
        strata={}
        for index,(row,anchor) in enumerate(dataset.rows):
            if anchor==0:strata.setdefault((row['task'],row['family'],row['origin']),index)
        records=[]
        for stratum,index in sorted(strata.items()):
            for anchor in (0,300,600):
                value=dataset[index+anchor]
                batch=ProcessedCollator(dataset.processor)([value])
                inputs=batch['inputs'];action=inputs['action'];mask=inputs['action_mask']
                if (tuple(action.shape)!=(1,40,132) or not bool((mask[...,:10]==1).all())
                    or not bool((mask[...,10:]==0).all()) or not bool((inputs['state'][...,10:]==0).all())):
                    raise ValueError('ACTUAL_NATIVE_TASK_STRATUM_MASK_OR_PADDING_CHANGED')
                records.append(dict(stratum=stratum,identity=dataset.identity(index+anchor),batch=tree_identity(batch)))
        checks[split]=records
    codec=codec_all_windows(args.output,datasets['TRAIN'].processor)
    loader_path=output/'NATIVE_LOADER.json'
    command=[sys.executable,'-m','scripts.pi0294.native_train','--dataset',args.output,
        '--source',args.source,'--cosmos',args.cosmos,'--output',str(loader_path)]
    result=subprocess.run(command,timeout=max(1,args.deadline-time.time()))
    if result.returncode:raise RuntimeError('ACTUAL_MULTITASK_NATIVE_LOADER_FAILED')
    loader=read(loader_path)
    if loader['status']!='PASS_PI0294_ACTUAL_NATIVE_CPU_LOADER_0_2_4' or loader['legacy_smoke']:
        raise ValueError('NEW_MULTITASK_LOADER_AND_RESUME_QUALIFICATION_REQUIRED')
    if torch.cuda.is_initialized():raise PermissionError('CPU_PIPELINE_INITIALIZED_CUDA')
    return dict(status='PASS_PI0294_TRAIN_DEV_AND_TRAINER_LOADER_CPU',
        manifest_sha256=sha(Path(args.output)/'DATASET_MANIFEST.json'),
        statistics_sha256=sha(Path(args.output)/'statistics.json'),stratified_actual_tensors=checks,
        all_window_codec=codec,native_loader_sha256=sha(loader_path),
        actual_new_adapter_loader=str(loader_path),statistics_fit_count=1,DEV_refit=False,
        relative_conversion_count=1,model_loaded=False,CUDA_initialized=False)


def run(args):
    root=Path(args.root);ticket=validate_cpu_ticket(root,args.reservation,args.manifest)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        NO_ALBUMENTATIONS_UPDATE='1',PYTHONDONTWRITEBYTECODE='1')
    args.deadline=ticket['deadline_epoch']
    evidence=root/'evidence'/('bank-qualification-'+read(args.scene_plan)['version'].replace('.','_'))
    evidence.mkdir(exist_ok=False)
    started=time.monotonic();before=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    status='FAILED_PRESERVED';failure=None
    try:
        bank=build(root,args.scene_plan,args.old_native,args.old_admission,args.output)
        if bank.get('status')=='PI0294_BANK_QUALITY_GATE_FAIL':raise PermissionError('FIRST_ATTEMPT_QUALITY_GATE_FAILED')
        value=qualify(args,evidence);write(evidence/'CPU_QUALIFICATION.json',value)
        status=value['status'];return value
    except BaseException as error:
        failure=dict(type=type(error).__name__,error=str(error));raise
    finally:
        after=[resource.getrusage(k) for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
        cpu=sum(a.ru_utime+a.ru_stime-b.ru_utime-b.ru_stime for a,b in zip(after,before))
        write(evidence/'CPU_ATTEMPT.json',dict(status=status,CPU_s=cpu,wall_s=time.monotonic()-started,
            failure=failure,unreaped_children_unknown=status=='FAILED_PRESERVED',integrations=0,render_calls=0,model_loads=0,
            accounting_scope='WHOLE_PARENT_PLUS_REAPED_CHILDREN_NOT_ADDITIVE_TO_INNER_BANK_OR_LOADER_RECEIPTS'))
        signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','scene-plan','old-native','old-admission','output','reservation','source','cosmos'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--manifest')
    value=run(p.parse_args());print(json.dumps(dict(status=value['status'],model_loaded=False,CUDA_initialized=False)))
