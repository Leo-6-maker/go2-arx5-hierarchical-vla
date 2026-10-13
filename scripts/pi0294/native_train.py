"""Stateful loader and evidence instrumentation around the pinned native Trainer.

No loss, model forward, upstream source, or global torch function is replaced.
Import the verified official source before calling make_trainer_class().
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np

from scripts.pi0292.durable import read,write
from .sampling import BalancedSampler
from .native_dataset import sha


class MeasuredLoader:
    """Measure parent wait for actual yielded batches, not worker prefetch."""
    def __init__(self,loader):
        self.loader,self.wait_s,self.batches=loader,0.,0

    def __len__(self):return len(self.loader)

    def __getattr__(self,name):return getattr(self.loader,name)

    def __iter__(self):
        iterator=iter(self.loader)
        while True:
            started=time.monotonic()
            try:batch=next(iterator)
            except StopIteration:return
            self.wait_s+=time.monotonic()-started
            self.batches+=1
            yield batch


def tensor_identity(tensor):
    value=tensor.detach().cpu().contiguous()
    import torch
    raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(raw).hexdigest())


def tree_identity(value):
    import torch
    if torch.is_tensor(value):return tensor_identity(value)
    if hasattr(value,'items'):return {str(k):tree_identity(v) for k,v in sorted(value.items())}
    if isinstance(value,(list,tuple)):return [tree_identity(v) for v in value]
    return value


def rng_identity():
    import torch
    return dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        torch_cpu=tensor_identity(torch.get_rng_state()),
        torch_cuda=[tensor_identity(x) for x in torch.cuda.get_rng_state_all()])


def parameter_inventory(model):
    return {name:dict(**tensor_identity(p),requires_grad=p.requires_grad,numel=p.numel())
        for name,p in model.named_parameters()}


def validate_worker_trace(directory,dataset,start,stop):
    actual={}
    for path in Path(directory).glob('worker-*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line);cursor=row['cursor']
            if start<=cursor<stop:
                if cursor in actual:raise ValueError('DUPLICATE_CONSUMED_SAMPLE')
                actual[cursor]=row
    expected=list(BalancedSampler(dataset,start=start,stop=stop,positioned=True,smoke=dataset.sampling_smoke))
    for cursor,index in expected:
        if actual.get(cursor)!=dict(cursor=cursor,index=index,**dataset.identity(index)):
            raise ValueError('ACTUAL_WORKER_SAMPLE_ORDER_MISMATCH')
    return dict(status='PASS_ACTUAL_CONSUMED_WORKER_IDENTITIES',start=start,stop=stop,count=len(actual),
        consumed_sha256=hashlib.sha256(json.dumps([actual[n] for n in range(start,stop)],sort_keys=True).encode()).hexdigest(),
        prefetch_excluded=True)


class NativeNoiseAudit:
    """Observe native noise without consuming its RNG stream or altering inputs."""
    def __init__(self,model):
        self.head=model.action_head
        if self.head.state_dropout_prob!=0:raise ValueError('STATE_DROPOUT_MUST_BE_ZERO')
        self.pending=None;self.record=None
        self.handles=[self.head.register_forward_pre_hook(self.before_head),
            self.head.state_encoder.register_forward_hook(self.after_state),
            self.head.action_encoder.register_forward_pre_hook(self.before_action)]

    def before_head(self,module,args):
        self.actions=args[1]['action']

    def after_state(self,module,args,output):
        import torch
        actions=self.actions
        before=rng_identity()
        with torch.random.fork_rng(devices=[actions.device.index]):
            noise=torch.randn(actions.shape,device=actions.device,dtype=actions.dtype)
            t=self.head.sample_time(actions.shape[0],device=actions.device,dtype=actions.dtype)
            self.pending=((1-t[:,None,None])*noise+t[:,None,None]*actions,
                (t*self.head.num_timestep_buckets).long())
            self.record=dict(noise=tensor_identity(noise),time=tensor_identity(t),rng_before_noise=before)
        if rng_identity()!=before:raise ValueError('AUDIT_CHANGED_NATIVE_RNG')

    def before_action(self,module,args):
        import torch
        if self.pending is None:raise ValueError('MISSING_NATIVE_NOISE_AUDIT')
        if not torch.equal(args[0],self.pending[0]) or not torch.equal(args[1],self.pending[1]):
            raise ValueError('OBSERVED_NATIVE_NOISE_OR_TIME_DIFFERS')
        self.record.update(noisy_trajectory=tensor_identity(args[0]),time_bucket=tensor_identity(args[1]),exact=True)
        self.pending=None

    def close(self):
        for handle in self.handles:handle.remove()


def make_trainer_class():
    import torch
    from gr00t.experiment.trainer import Gr00tTrainer

    class NativeTrainer(Gr00tTrainer):
        def __init__(self,*args,evidence_dir,recipe,smoke=False,**kwargs):
            self.evidence_dir=Path(evidence_dir);self.evidence_dir.mkdir(parents=True,exist_ok=True)
            self.recipe=recipe;self.smoke=smoke
            self.consumed=0;self.start_cursor=0;self.last_native_loss=None
            self.noise_audit=None
            super().__init__(*args,**kwargs)
            if self.args.world_size!=1 or self.args.eval_strategy.value!='no':raise ValueError('SINGLE_GPU_NO_INLINE_EVAL')
            if self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps!=32:
                raise ValueError('EFFECTIVE_BATCH_MUST_BE_32')
            if not self.train_dataset.for_optimization or self.train_dataset.split!='TRAIN':
                raise ValueError('TRAIN_OPTIMIZATION_DATASET_REQUIRED')
            if self.args.remove_unused_columns or self.args.save_total_limit is not None:
                raise ValueError('PRESERVE_PROCESSED_SCHEMA_AND_ALL_CHECKPOINTS')
            if smoke:self.noise_audit=NativeNoiseAudit(self.model)

        def train(self,resume_from_checkpoint=None,**kwargs):
            if resume_from_checkpoint:
                checkpoint=Path(resume_from_checkpoint)
                saved=read(checkpoint/'SAMPLER_STATE.json')
                state=read(checkpoint/'trainer_state.json')
                expected=BalancedSampler(self.train_dataset,stop=self.args.max_steps*32,smoke=self.train_dataset.sampling_smoke).state_dict(state['global_step']*32)
                compatible=saved['sampler']==expected and saved['recipe']==self.recipe
                if not compatible:
                    raise ValueError('CHECKPOINT_SAMPLER_OR_RECIPE_MISMATCH')
                if saved['statistics_sha256']!=self.train_dataset.statistics_sha or saved['manifest_sha256']!=self.train_dataset.manifest_sha:
                    raise ValueError('CHECKPOINT_DATASET_MISMATCH')
            return super().train(resume_from_checkpoint=resume_from_checkpoint,**kwargs)

        def get_train_dataloader(self):
            # Worker prefetch never advances the authoritative consumed cursor.
            self.args.ignore_data_skip=True
            self.start_cursor=self.consumed=self.state.global_step*32
            self.sampler=BalancedSampler(self.train_dataset,start=self.consumed,
                stop=self.args.max_steps*32,positioned=True,smoke=self.train_dataset.sampling_smoke)
            generator=torch.Generator().manual_seed(29400)
            options=dict(batch_size=self._train_batch_size,sampler=self.sampler,collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=False,generator=generator,drop_last=True)
            if self.args.dataloader_num_workers:options.update(multiprocessing_context='spawn',prefetch_factor=2)
            self.measured_loader=MeasuredLoader(torch.utils.data.DataLoader(self.train_dataset,**options))
            return self.measured_loader

        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            loss,outputs=super().compute_loss(model,inputs,return_outputs=True,num_items_in_batch=num_items_in_batch)
            mask=outputs['action_mask'];masked=outputs['action_loss']
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(masked).all()):raise ValueError('NONFINITE_NATIVE_LOSS')
            if tuple(mask.shape[1:])!=(40,132) or not bool((mask[...,:10]==1).all()) or not bool((mask[...,10:]==0).all()):
                raise ValueError('NATIVE_ACTION_MASK_CHANGED')
            if not bool((masked[...,10:]==0).all()):raise ValueError('PADDED_ACTION_DIMENSIONS_ENTERED_LOSS')
            expected=masked.sum()/(mask.sum()+1e-6)
            if not torch.equal(loss,expected):raise ValueError('NON_NATIVE_LOSS_REDUCTION')
            self.last_native_loss=float(loss.detach())
            return (loss,outputs) if return_outputs else loss

        def training_step(self,model,inputs,num_items_in_batch=None):
            batch=int(inputs['inputs']['action'].shape[0])
            if batch!=self.args.per_device_train_batch_size:raise ValueError('PARTIAL_OPTIMIZATION_MICROBATCH')
            before=rng_identity() if self.smoke else None
            identity=tree_identity(inputs) if self.smoke else None
            result=super().training_step(model,inputs,num_items_in_batch)
            record=dict(optimizer_step=self.state.global_step,cursor=self.consumed,batch=batch,native_loss=self.last_native_loss,
                actual_exposure=[self.train_dataset.identity(self.sampler.index_at(c)) for c in range(self.consumed,self.consumed+batch)])
            if self.smoke:record.update(batch_identity=identity,rng_before=before,rng_after=rng_identity(),noise=self.noise_audit.record)
            self.consumed+=batch
            with (self.evidence_dir/'MICROBATCHES.jsonl').open('a') as handle:handle.write(json.dumps(record,sort_keys=True)+'\n')
            return result

        def _save_checkpoint(self,model,trial):
            if self.consumed!=self.state.global_step*32:raise ValueError('OPTIMIZER_AND_SAMPLE_CURSOR_DISAGREE')
            super()._save_checkpoint(model,trial)
            checkpoint=Path(self.args.output_dir)/('checkpoint-'+str(self.state.global_step))
            if sha(self.train_dataset.root/'statistics.json')!=self.train_dataset.statistics_sha:
                raise ValueError('STATISTICS_CHANGED_DURING_TRAINING')
            self.train_dataset.initialize()
            self.train_dataset.processor.save_pretrained(checkpoint/'processor')
            # The canonical original bytes, not JSON reformatting, define stats identity.
            import shutil
            shutil.copyfile(self.train_dataset.root/'statistics.json',checkpoint/'statistics.json')
            trace=validate_worker_trace(self.train_dataset.trace_root,self.train_dataset,self.start_cursor,self.consumed)
            write(checkpoint/'SAMPLER_STATE.json',dict(sampler=self.sampler.state_dict(self.consumed),recipe=self.recipe,
                manifest_sha256=self.train_dataset.manifest_sha,statistics_sha256=self.train_dataset.statistics_sha,
                worker_trace=trace,loader_RNG='ISOLATED_GENERATOR_29400_NO_STOCHASTIC_DATA_TRANSFORMS'))
            write(checkpoint/'CHECKPOINT_COMPLETE.json',dict(global_step=self.state.global_step,
                required_files={name:sha(checkpoint/name) for name in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','statistics.json','SAMPLER_STATE.json')},
                status='FULL_STATE_SAVED_MODEL_ARCHIVE_HASH_PENDING'))

    return NativeTrainer


def _qualify_cpu(args):
    """Exercise this exact loader without initializing Trainer or a model."""
    import os
    import resource
    from types import SimpleNamespace
    from scripts.pi0291_arx5_data.native_dataset import cpu_guard
    from .native_dataset import NativeDataset,ProcessedCollator,official_import
    cpu_guard()
    official_import(args.source)
    import torch
    trainer_type=make_trainer_class()
    raw=read(args.raw_bindings) if args.raw_bindings else None
    records={};throughput={}
    sample_checks=[]
    for workers in (0,2,4):
        records[str(workers)]={}
        for step in (0,1):
            directory=Path(args.output).parent/('pi0294-loader-'+str(workers)+'-resume-'+str(step))
            if directory.exists():raise FileExistsError('Preserve earlier CPU loader evidence')
            dataset=NativeDataset(args.dataset,args.source,args.cosmos,for_optimization=True,
                raw_bindings=raw,trace_root=directory)
            dataset.sampling_smoke=args.legacy_smoke
            dataset.initialize()
            if workers==0 and step==0:
                for index in (0,len(dataset)//2,len(dataset)-1):
                    # No trace entry is written for the separate first/middle/last inspection.
                    trace=dataset.trace_root;dataset.trace_root=None
                    value=dataset[index];dataset.trace_root=trace
                    batch=dataset.processor.collator([value['processed']])
                    action=batch['inputs']['action'];mask=batch['inputs']['action_mask']
                    if tuple(action.shape)!=(1,40,132) or not bool((mask[...,:10]==1).all()) or not bool((mask[...,10:]==0).all()):
                        raise ValueError('ACTUAL_CPU_BATCH_MASK_CHANGED')
                    sample_checks.append(dict(identity=dataset.identity(index),batch=tree_identity(batch)))
            trainer=object.__new__(trainer_type)
            trainer.args=SimpleNamespace(ignore_data_skip=False,max_steps=2,
                dataloader_num_workers=workers,dataloader_pin_memory=False)
            trainer.state=SimpleNamespace(global_step=step)
            trainer.train_dataset=dataset;trainer.data_collator=ProcessedCollator(dataset.processor)
            trainer._train_batch_size=2
            loader=trainer.get_train_dataloader();started=time.monotonic();iterator=iter(loader)
            batches=[tree_identity(next(iterator)) for _ in range(4 if step else 20)]
            consumed_elapsed=time.monotonic()-started;consumed_wait=loader.wait_s
            # Finish the finite iterator before worker teardown. Drained batches are
            # engineering qualification only, never optimizer/consumed-cursor input.
            drained_batches=sum(1 for _ in iterator)
            trace=validate_worker_trace(directory,dataset,step*32,step*32+len(batches)*2)
            throughput[str(workers)+'/'+str(step)]=dict(elapsed_s=consumed_elapsed,
                consumed_samples=len(batches)*2,parent_loader_wait_s=consumed_wait,
                consumed_samples_per_s=len(batches)*2/consumed_elapsed,
                drained_samples=drained_batches*2,natural_iterator_exhaustion=True,
                total_loader_elapsed_s=time.monotonic()-started)
            records[str(workers)][str(step)]=dict(batches=batches,trace=trace)
            del iterator,loader,trainer,dataset
        if records[str(workers)]['0']['batches'][16:20]!=records[str(workers)]['1']['batches']:
            raise ValueError('ACTUAL_RESUME_CONSUMPTION_DIFFERS')
    if records['0']!=records['2'] or records['0']!=records['4']:
        raise ValueError('ACTUAL_WORKER_ORDERED_CONSUMPTION_DIFFERS')
    if torch.cuda.is_initialized():raise PermissionError('CPU_QUALIFICATION_INITIALIZED_CUDA')
    u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(status='PASS_PI0294_ACTUAL_NATIVE_CPU_LOADER_0_2_4',records=records,
        actual_first_middle_last=sample_checks,manifest_sha256=sha(Path(args.dataset)/'DATASET_MANIFEST.json'),
        statistics_sha256=sha(Path(args.dataset)/'statistics.json'),model_loaded=False,neural_forward=False,
        CUDA_initialized=False,legacy_smoke=args.legacy_smoke,actual_loader_throughput=throughput,
        CPU_s=u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime)
    path=Path(args.output)
    if path.exists():raise FileExistsError('Preserve prior qualification')
    write(path,result)
    return result


def qualify_cpu(args):
    """Preserve per-attempt CPU accounting even when qualification raises."""
    import resource
    import traceback
    output=Path(args.output)
    receipt=output.with_name(output.name+'.CPU_ATTEMPT.json')
    if output.exists() or receipt.exists():raise FileExistsError('Preserve prior CPU qualification attempt')
    started=time.monotonic()
    before=[resource.getrusage(kind) for kind in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
    status='FAILED';failure=None
    try:
        result=_qualify_cpu(args)
        status='PASS'
        return result
    except BaseException as error:
        failure=dict(type=type(error).__name__,traceback=traceback.format_exc())
        raise
    finally:
        after=[resource.getrusage(kind) for kind in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN)]
        cpu=[(a.ru_utime+a.ru_stime)-(b.ru_utime+b.ru_stime) for a,b in zip(after,before)]
        evidence=dict(status=status,wall_s=time.monotonic()-started,
            self_CPU_s=cpu[0],reaped_children_CPU_s=cpu[1],CPU_s=sum(cpu),failure=failure,
            unreaped_children_unknown=status!='PASS',model_loaded=False,neural_forward=False)
        try:write(receipt,evidence)
        except Exception:
            # An accounting-write error must not replace the qualification failure.
            if failure is None:raise
            traceback.print_exc()


if __name__=='__main__':
    import argparse
    import os
    parser=argparse.ArgumentParser()
    for name in ('dataset','source','cosmos','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--raw-bindings');parser.add_argument('--legacy-smoke',action='store_true')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-6:]))
    value=qualify_cpu(args)
    print(json.dumps(dict(status=value['status'],CPU_s=value['CPU_s'],model_loaded=False)))

