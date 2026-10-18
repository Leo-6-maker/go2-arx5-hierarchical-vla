"""Versioned PV1 primary training around the mature official-loss Trainer."""
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.native_dataset import ProcessedCollator,sha
from scripts.pi0294.native_train import parameter_inventory
from scripts.pi0294.mp1.trainer import make_trainer_class
from scripts.pi0294.e1.scope import STATS
from .dataset import PV1Dataset
from .recipe import PARENT,validate_primary,begin_update,commit_update,run_key
from .pre_model_qualification import current_directory,require_current


def checked_resume(root,arm,checkpoint,recipe,dataset):
    root,checkpoint=Path(root),Path(checkpoint)
    if checkpoint.parent.resolve()!=(root/'training'/arm).resolve() or checkpoint.is_symlink():
        raise PermissionError('PV1_OWN_EXACT_COMPLETE_RESUME_ONLY')
    complete=read(checkpoint/'CHECKPOINT_COMPLETE.json')
    step=complete['global_step']
    required={'config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json','statistics.json'}
    if (not complete['full_resume'] or type(step) is not int or not 0<step<4096
            or checkpoint.name!='checkpoint-'+str(step) or not required<=set(complete['files'])
            or not any(n.endswith('.safetensors') for n in complete['files'])
            or not any(n.startswith('processor/') for n in complete['files'])):
        raise PermissionError('PV1_FULL_NATIVE_OPTIMIZER_RNG_SAMPLER_STATE_REQUIRED')
    for name,digest in complete['files'].items():
        p=checkpoint/name
        if p.is_symlink() or not p.resolve().is_relative_to(checkpoint.resolve()) or sha(p)!=digest:
            raise ValueError('PV1_RESUME_MEMBER_CHANGED:'+name)
    saved=read(checkpoint/'SAMPLER_STATE.json')
    state=read(checkpoint/'trainer_state.json')
    expected=dataset.make_sampler(dataset,stop=4096*32).state_dict(step*32)
    if (state['global_step']!=step or saved['recipe']!=recipe or saved['sampler']!=expected
            or saved['manifest_sha256']!=dataset.manifest_sha or saved['statistics_sha256']!=STATS
            or sha(checkpoint/'statistics.json')!=STATS):
        raise PermissionError('PV1_EXACT_RESUME_DATA_RECIPE_SEED_CURSOR_AND_STATS')
    return step


def run(root,instance,allocation,guard,ledger,*,arm,request_id,resume_checkpoint=None,stop_after_step=1024,replica=False):
    root=Path(root)
    guard()
    if (allocation.get('status')!='PASS_PV1_OWN_NATIVE_H200_ALLOCATION'
            or not os.environ.get('PBS_JOBID') or allocation['job_id']!=os.environ['PBS_JOBID']
            or stop_after_step not in (1024,4096) or type(replica) is not bool or not request_id
            or request_id in ('.','..') or Path(request_id).name!=request_id):
        raise PermissionError('PV1_NATIVE_ALLOCATION_BEFORE_ANY_TRAINING_AND_FIXED_PRIMARY_STOPS')
    directory=current_directory(root,allocation);require_current(root,allocation)
    submitted=read(directory/'REQUEST.json')
    block=submitted.get('training_block',{})
    resume_relative=Path(resume_checkpoint).resolve().relative_to(root.resolve()).as_posix() if resume_checkpoint else None
    if (submitted['request_id']!=request_id or submitted['stage']!=('P4_REPLICA' if replica else 'P2_PRIMARY')
            or block.get('arm')!=arm or block.get('replica')!=replica
            or block.get('stop_after_step')!=stop_after_step
            or block.get('resume_checkpoint_relative')!=resume_relative):
        raise PermissionError('PV1_ONLY_THE_EXACT_SUBMITTED_ARM_REPLICA_AND_TRAINING_BLOCK')
    consumed=read(root/'evidence/NEW_BANK_CONSUMPTION_READY.private.json')
    if (consumed['status']!='PASS_ACTUAL_FULL_NEW_BANK_CONSUMPTION' or consumed['full_member_hashes_verified'] is not True
            or consumed['delivery_binding_sha256']!=sha(root/'plans/PV1_HF_BANK_DELIVERY.json')
            or consumed['dataset_manifest_sha256']!=read(root/'PV1_STAGE_BINDING.json')['new_bank_manifest_sha256']):
        raise PermissionError('PV1_ACTUAL_FULL_FIXED_HF_BANK_CONSUMPTION_BEFORE_TRAINING')
    original=read(root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json')
    key=('REPLICA_' if replica else '')+arm
    recipe=read(root/'plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][key]
    validate_primary(recipe,original)
    if run_key(recipe)!=key or recipe['arm']!=arm:
        raise PermissionError('PV1_SEPARATE_PREBOUND_PRIMARY_AND_REPLICA_ARTIFACT_IDENTITY')
    b=read(root/'NEURAL_BINDINGS.private.json')
    inputs=read(root/'plans/PV1_TRAIN_INPUTS.private.json')
    evidence=root/'training'/key/('session-'+request_id)
    prior=list((root/'training'/key).glob('checkpoint-*/CHECKPOINT_COMPLETE.json'))
    if prior and (resume_checkpoint is None or Path(resume_checkpoint).resolve()!=
            max((p.parent for p in prior),key=lambda p:read(p/'CHECKPOINT_COMPLETE.json')['global_step']).resolve()):
        raise PermissionError('PV1_ONLY_LATEST_COMPLETE_RESUME_NO_FRESH_RESTART_OR_ROLLBACK')
    if evidence.exists():raise FileExistsError('PV1_PRESERVE_FIRST_NATIVE_TRAINING_SESSION')
    evidence.mkdir(parents=True)
    dataset=PV1Dataset(b['dataset'],b['source'],b['cosmos'],for_optimization=True,
        raw_bindings=read(b['raw_bindings']),trace_root=evidence/'worker_trace',
        correction_root=inputs['C1_dataset'],correction_bindings=read(inputs['C1_raw_bindings']),
        new_root=inputs['new_dataset'],new_raw_bindings=read(inputs['new_raw_bindings']),
        route=recipe['route'],arm=arm,seed=recipe['seed'])
    dataset.initialize()
    start=checked_resume(root,key,resume_checkpoint,recipe,dataset) if resume_checkpoint else 0
    if stop_after_step<=start:
        raise PermissionError('PV1_COMPLETED_TRAINING_BLOCK_NOT_REPEATED')
    current=read(ledger.path)
    if current['pending_updates']:
        raise PermissionError('PV1_UNKNOWN_NATIVE_OPTIMIZER_EFFECT_MUST_BE_RECONCILED')
    parent=read(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json')
    initial=parameter_inventory(instance.model)
    if (set(initial)!=set(parent)
            or any(p['requires_grad']!=parent[n]['requires_grad'] for n,p in initial.items())
            or any(p['requires_grad'] for n,p in initial.items() if n.startswith('backbone.'))
            or any(p['sha256']!=parent[n]['sha256'] for n,p in initial.items() if not p['requires_grad'])
            or (not start and any(p['sha256']!=parent[n]['sha256'] for n,p in initial.items()))):
        raise PermissionError('PV1_IDENTICAL_CORR_PARENT_AND_FROZEN_ORIGINAL_TRAINABLE_MAP')
    if (instance.identity['global_step']!=(start or 2048)
            or (not start and instance.identity['checkpoint_manifest_sha256']!=PARENT)):
        raise PermissionError('PV1_CORR2048_PARENT_NOT_M1_OR_A_PREVIOUS_ARM')
    import torch
    from transformers import TrainingArguments,TrainerCallback,set_seed
    write(evidence/'INITIAL_PARAMETERS.json',initial)
    write(evidence/'RECIPE.json',recipe)
    set_seed(recipe['seed'])
    instance.model.train()
    args=TrainingArguments(output_dir=str(root/'training'/key),max_steps=4096,per_device_train_batch_size=2,
        gradient_accumulation_steps=16,learning_rate=recipe['learning_rate'],weight_decay=recipe['weight_decay'],
        adam_beta1=recipe['adam_beta1'],adam_beta2=recipe['adam_beta2'],adam_epsilon=recipe['adam_epsilon'],
        max_grad_norm=recipe['max_grad_norm'],bf16=True,tf32=True,optim=recipe['optimizer'],
        dataloader_num_workers=0,dataloader_pin_memory=True,remove_unused_columns=False,
        seed=recipe['seed'],data_seed=recipe['seed'],eval_strategy='no',logging_strategy='steps',logging_steps=1,
        save_strategy='no',save_total_limit=None,save_only_model=False,gradient_checkpointing=False,
        report_to=[],disable_tqdm=True,ignore_data_skip=True,load_best_model_at_end=False)
    trainer=None
    times=[]
    began=None
    class Audit(TrainerCallback):
        def on_step_begin(self,arguments,state,control,**kwargs):
            nonlocal began
            guard()
            torch.cuda.synchronize()
            began=time.monotonic()
            with ledger.transaction() as l:
                updated=begin_update(l,arm=arm,local_step=state.global_step,request_id=request_id,recipe=recipe)
                l.clear();l.update(updated)

        def on_pre_optimizer_step(self,arguments,state,control,model=None,**kwargs):
            values=dict(projector=0.,diffusion=0.)
            for name,p in model.named_parameters():
                if p.grad is None:continue
                if not p.requires_grad or not bool(torch.isfinite(p.grad).all()):
                    raise ValueError('PV1_NONFINITE_OR_FROZEN_PARAMETER_GRADIENT')
                if name.startswith('action_head.'):
                    key='diffusion' if name.startswith('action_head.model.') else 'projector'
                    values[key]+=float(p.grad.float().square().sum())
            if not all(v>0 for v in values.values()):
                raise ValueError('PV1_ACTUAL_PROJECTOR_AND_DIT_GRADIENTS_REQUIRED')
            with (evidence/'GRADIENTS.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=state.global_step,norm_squared=values))+'\n')

        def on_step_end(self,arguments,state,control,**kwargs):
            with ledger.transaction() as l:
                updated=commit_update(l,arm=arm,local_step_after=state.global_step,request_id=request_id)
                l.clear();l.update(updated)
            torch.cuda.synchronize()
            times.append(time.monotonic()-began)
            row=dict(step=state.global_step,cursor=trainer.consumed,native_loss=trainer.last_native_loss,
                learning_rate=trainer.lr_scheduler.get_last_lr(),synchronized_s=times[-1],epoch=time.time())
            with (evidence/'OPTIMIZER_STEPS.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n');f.flush()
            write(evidence/'PROGRESS.json',row)
            if state.global_step in [1024,4096]:control.should_save=True
            if state.global_step==stop_after_step or time.time()>=allocation['allocation_end_epoch']-2730:
                control.should_save=True;control.should_training_stop=True
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    trainer=make_trainer_class()(model=instance.model,args=args,train_dataset=dataset,
        data_collator=ProcessedCollator(dataset.processor),callbacks=[Audit()],evidence_dir=evidence,recipe=recipe,smoke=False)
    result=trainer.train(resume_from_checkpoint=str(resume_checkpoint) if start else None)
    final=parameter_inventory(instance.model)
    unchanged=all(p['sha256']==initial[n]['sha256'] for n,p in final.items() if not p['requires_grad'])
    changed=[n for n,p in final.items() if p['requires_grad'] and p['sha256']!=initial[n]['sha256']]
    if not unchanged or not changed:
        raise ValueError('PV1_FROZEN_VLM_CHANGED_OR_NO_REAL_LEARNING')
    value=dict(status='PV1_PRIMARY_4096_TERMINAL' if trainer.state.global_step==4096 else 'PV1_EXACT_PARTIAL_STATE_PRESERVED',
        arm=arm,training_run_key=key,replica=replica,global_step=trainer.state.global_step,resumed_from_step=start,
        actual_updates_this_attempt=trainer.state.global_step-start,actual_samples_this_attempt=trainer.consumed-start*32,
        actual_consumed_cursor=trainer.consumed,sampler=trainer.sampler.state_dict(trainer.consumed),
        parent_checkpoint_manifest_sha256=PARENT,loaded_checkpoint_manifest_sha256=instance.identity['checkpoint_manifest_sha256'],
        source_manifest_sha256=dataset.manifest_sha,new_bank_manifest_sha256=dataset.new_reader.manifest_sha,
        statistics_sha256=dataset.statistics_sha,independent_new_support=dataset.support,
        recipe=recipe,native_metrics=result.metrics,synchronized_step_s=times,wall_s=time.monotonic()-started,
        frozen_parameters_unchanged=unchanged,changed_trainable_parameters=changed,final_parameters=final,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),development_evaluation_executed=False,
        selected='NONE',real_robot_deployment='NONE')
    write(evidence/'TRAINING_RUN.json',value)
    del trainer
    return value
