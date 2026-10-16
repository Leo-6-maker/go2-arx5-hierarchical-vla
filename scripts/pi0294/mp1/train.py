"""Real source learning around the unchanged official Flow Matching objective."""
import json
from pathlib import Path
import time

from scripts.pi0292.durable import write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.native_dataset import ProcessedCollator,sha
from scripts.pi0294.native_train import parameter_inventory
from .dataset import SourceDataset
from .trainer import make_trainer_class


def validate_recipe(recipe,branch):
    if branch not in ['S-BC','S-CORR']:raise PermissionError('MP1_FIRST_TWO_FIXED_SOURCE_BRANCHES_ONLY')
    expected=dict(seed=20261017,max_steps=2048,warmup_updates=32,effective_batch=32,
        per_device_train_batch_size=2,gradient_accumulation_steps=16,learning_rate=1e-5,end_learning_rate=1e-6,
        new_optimizer=True,exact_resume=False,parent='AUDITED_M1_AT_1024_INFERENCE_WEIGHTS_ONLY',VLM_frozen=True,statistics_fit=False)
    if any(recipe.get(k)!=v for k,v in expected.items()):raise PermissionError('MP1_FIXED_SOURCE_LEARNING_RECIPE')
    if recipe['gradient_checkpointing'] is not False or recipe['bf16'] is not True or recipe['tf32'] is not True:
        raise PermissionError('MP1_ORIGINAL_VERIFIED_TRAINING_NUMERICS')
    return expected


def checked_resume(root,branch,checkpoint,recipe,ledger):
    root=Path(root);checkpoint=Path(checkpoint);validate_recipe(recipe,branch)
    if checkpoint.parent.resolve()!=(root/'training'/branch).resolve():raise PermissionError('MP1_OWN_PARTIAL_SOURCE_STATE_ONLY')
    complete=read(checkpoint/'CHECKPOINT_COMPLETE.json');step=complete['global_step'];offset=0 if branch=='S-BC' else 2048
    if not complete['full_resume'] or not 0<step<2048 or checkpoint.name!='checkpoint-'+str(step):raise PermissionError('MP1_COMPLETE_NATIVE_PARTIAL_STATE_REQUIRED')
    required={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json','statistics.json'}
    if not required.issubset(complete['files']):raise PermissionError('MP1_NO_WEIGHTS_ONLY_RESTART')
    for name,digest in complete['files'].items():
        p=checkpoint/name
        if not p.resolve().is_relative_to(checkpoint.resolve()) or p.is_symlink() or sha(p)!=digest:raise ValueError('MP1_RESUME_NATIVE_MEMBER_CHANGED:'+name)
    saved=read(checkpoint/'SAMPLER_STATE.json');trainer=read(checkpoint/'trainer_state.json')
    from .scope import STATS
    if (saved['recipe']!=recipe or read(root/'training'/branch/'session-start/RECIPE.json')!=recipe
            or saved['sampler']['cursor']!=step*32 or trainer['global_step']!=step or sha(checkpoint/'statistics.json')!=STATS
            or ledger['pending_updates'] or ledger['optimizer_updates']!=offset+step):raise PermissionError('MP1_EXACT_STEP_RECIPE_CURSOR_AND_NO_UNKNOWN_UPDATE')
    return step


def run(root,instance,allocation,guard,ledger,*,branch,resume_checkpoint=None,stop_after_step=2048):
    import torch
    from transformers import TrainingArguments,TrainerCallback,set_seed
    root=Path(root);recipe=read(root/'plans/SOURCE_LEARNING_RECIPE.json');validate_recipe(recipe,branch)
    l=read(ledger.path);resume_step=checked_resume(root,branch,resume_checkpoint,recipe,l) if resume_checkpoint else 0
    if stop_after_step not in [512,2048] or stop_after_step<=resume_step:
        raise PermissionError('MP1_ONLY_FIXED_512_EVALUATION_BREAK_OR_TERMINAL_2048')
    output=root/'training'/branch;evidence=output/('session-resume-'+str(resume_step) if resume_step else 'session-start');evidence.mkdir(parents=True,exist_ok=False)
    expected_before=(0 if branch=='S-BC' else 2048)+resume_step
    if l['optimizer_updates']!=expected_before or l['pending_updates']:raise PermissionError('MP1_FIRST_SOURCE_PAIR_EXACT_UPDATE_ACCOUNTING')
    b=read(root/'NEURAL_BINDINGS.private.json');correction=read(root/'plans/SOURCE_CORRECTION_BINDING.private.json')
    options=dict(for_optimization=True,raw_bindings=read(b['raw_bindings']),trace_root=evidence/'worker_trace')
    if branch=='S-CORR':options.update(correction_root=correction['dataset'],correction_bindings=read(correction['raw_bindings']))
    dataset=SourceDataset(b['dataset'],b['source'],b['cosmos'],**options);dataset.initialize()
    initial=parameter_inventory(instance.model);parent=read(root/'plans/PARENT_PARAMETER_INVENTORY.json')
    if (set(initial)!=set(parent) or any(p['requires_grad']!=parent[n]['requires_grad'] for n,p in initial.items())
            or any(p['requires_grad'] for n,p in initial.items() if n.startswith('backbone.'))
            or any(p['sha256']!=parent[n]['sha256'] for n,p in initial.items() if not p['requires_grad'])):
        raise PermissionError('MP1_EXACT_PARENT_TRAINABLE_MAP_AND_FROZEN_VLM')
    if instance.identity['global_step']!=(resume_step or 1024):raise PermissionError('MP1_FIXED_PARENT_OR_EXACT_PARTIAL_STATE')
    write(evidence/'INITIAL_PARAMETERS.json',initial);write(evidence/'RECIPE.json',recipe)
    set_seed(recipe['seed']);instance.model.train()
    args=TrainingArguments(output_dir=str(output),max_steps=recipe['max_steps'],per_device_train_batch_size=2,gradient_accumulation_steps=16,
        learning_rate=recipe['learning_rate'],weight_decay=recipe['weight_decay'],adam_beta1=recipe['adam_beta1'],adam_beta2=recipe['adam_beta2'],
        adam_epsilon=recipe['adam_epsilon'],max_grad_norm=recipe['max_grad_norm'],bf16=True,tf32=True,optim=recipe['optimizer'],
        dataloader_num_workers=0,dataloader_pin_memory=True,remove_unused_columns=False,seed=recipe['seed'],data_seed=recipe['seed'],
        eval_strategy='no',logging_strategy='steps',logging_steps=1,save_strategy='no',save_total_limit=None,save_only_model=False,
        gradient_checkpointing=False,report_to=[],disable_tqdm=True,ignore_data_skip=True,load_best_model_at_end=False)
    times=[];step_start=None;trainer=None
    class Audit(TrainerCallback):
        def on_step_begin(self,arguments,state,control,**kwargs):
            nonlocal step_start
            guard();torch.cuda.synchronize();step_start=time.monotonic()
            with ledger.transaction() as l:
                if l['pending_updates'] or l['optimizer_updates']>=4096:raise PermissionError('MP1_NO_UNKNOWN_OR_EXCESS_SOURCE_UPDATE')
                l.update(pending_updates=1,pending_global_step=state.global_step,pending_branch=branch)
        def on_pre_optimizer_step(self,arguments,state,control,model=None,**kwargs):
            groups={'projector':0.,'diffusion':0.}
            for name,p in model.named_parameters():
                if p.grad is None:continue
                if not p.requires_grad or not bool(torch.isfinite(p.grad).all()):raise ValueError('MP1_INVALID_NATIVE_GRADIENT')
                if name.startswith('action_head.'):
                    key='diffusion' if name.startswith('action_head.model.') else 'projector';groups[key]+=float(p.grad.float().square().sum())
            if not all(v>0 for v in groups.values()):raise ValueError('MP1_PROJECTOR_AND_DIT_REAL_GRADIENTS_REQUIRED')
            with (evidence/'GRADIENTS.jsonl').open('a') as f:f.write(json.dumps(dict(step=state.global_step,norm_squared=groups))+'\n')
        def on_step_end(self,arguments,state,control,**kwargs):
            with ledger.transaction() as l:l['optimizer_updates']+=1;l['pending_updates']=0
            torch.cuda.synchronize();times.append(time.monotonic()-step_start)
            row=dict(step=state.global_step,cursor=trainer.consumed,native_loss=trainer.last_native_loss,learning_rate=trainer.lr_scheduler.get_last_lr(),synchronized_s=times[-1],epoch=time.time())
            with (evidence/'OPTIMIZER_STEPS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            write(evidence/'PROGRESS.json',row)
            if state.global_step in [512,2048]:control.should_save=True
            if state.global_step==stop_after_step or time.time()>=allocation['allocation_end_epoch']-2730:
                control.should_save=True;control.should_training_stop=True
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    trainer=make_trainer_class()(model=instance.model,args=args,train_dataset=dataset,data_collator=ProcessedCollator(dataset.processor),
        callbacks=[Audit()],evidence_dir=evidence,recipe=recipe,smoke=False)
    result=trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_step else None);final=parameter_inventory(instance.model)
    unchanged=all(initial[n]['sha256']==p['sha256'] for n,p in final.items() if not p['requires_grad'])
    changed=[n for n,p in final.items() if p['requires_grad'] and p['sha256']!=initial[n]['sha256']]
    if not unchanged or not changed:raise ValueError('MP1_FROZEN_PARAMETERS_CHANGED_OR_NO_LEARNING')
    value=dict(status=branch+'_ACTUAL_2048_UPDATES' if trainer.state.global_step==2048 else branch+'_PARTIAL_NATIVE_STATE_PRESERVED',branch=branch,
        global_step=trainer.state.global_step,resumed_from_step=resume_step,actual_updates_this_attempt=trainer.state.global_step-resume_step,
        predeclared_stop_step=stop_after_step,full_scheduler_horizon_steps=2048,
        consumed_samples=trainer.consumed,frozen_parameters_unchanged=unchanged,changed_trainable_parameters=changed,
        final_parameters=final,recipe=recipe,source_manifest_sha256=dataset.manifest_sha,statistics_sha256=dataset.statistics_sha,
        correction_manifest_sha256=correction['manifest_sha256'] if branch=='S-CORR' else None,
        parent_checkpoint_manifest_sha256=read(root/'plans/SOURCE_PARENT_CHECKPOINT.json')['checkpoint_manifest_sha256'],
        loaded_checkpoint_manifest_sha256=instance.identity['checkpoint_manifest_sha256'],sampler=trainer.sampler.state_dict(trainer.consumed),
        native_metrics=result.metrics,synchronized_step_s=times,wall_s=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        selected='NONE',real_robot_deployment='NONE')
    write(evidence/'TRAINING_RUN.json',value);del trainer;return value
