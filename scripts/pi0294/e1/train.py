"""Real fixed M1 learning, unchanged official FM forward and trainable map."""
import json
from pathlib import Path
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import ProcessedCollator
from scripts.pi0294.native_train import parameter_inventory,rng_identity
from .dataset import SmallDataset
from .scope import Ledger
from .trainer import make_trainer_class


def run(root,instance,allocation,guard):
    import torch
    from transformers import TrainingArguments,TrainerCallback,set_seed
    root=Path(root);ledger=Ledger(root,'hopper');recipe=read(root/'plans/M1_RECIPE_RESOLVED.json')
    b=read(root/'NEURAL_BINDINGS.private.json');selection=read(root/'plans/SMALL8_BINDING.json')
    output=root/'training/M1';evidence=output/'session-start';evidence.mkdir(parents=True,exist_ok=False)
    initial=parameter_inventory(instance.model)
    historical=read(Path(read(root/'E1_INPUTS.private.json')['original_recipe']).parent/'TRAINING_RUN.json')['final_parameters']
    if {n:p['requires_grad'] for n,p in initial.items()}!={n:p['requires_grad'] for n,p in historical.items()}:
        raise ValueError('E1_EXACT_ORIGINAL_TRAINABLE_PARAMETER_MAP_REQUIRED')
    if any(p['requires_grad'] for n,p in initial.items() if n.startswith('backbone.')):raise ValueError('E1_VLM_MUST_BE_FROZEN')
    write(evidence/'INITIAL_PARAMETERS.json',initial);write(evidence/'RECIPE.json',recipe)
    dataset=SmallDataset(b['dataset'],b['source'],b['cosmos'],for_optimization=True,
        raw_bindings=read(b['raw_bindings']),trace_root=evidence/'worker_trace',selected_ids=[r['id'] for r in selection['episodes']])
    dataset.initialize();set_seed(20261016);instance.model.train()
    args=TrainingArguments(output_dir=str(output),max_steps=1024,per_device_train_batch_size=recipe['per_device_train_batch_size'],
        gradient_accumulation_steps=recipe['gradient_accumulation_steps'],learning_rate=recipe['learning_rate'],
        weight_decay=recipe['weight_decay'],adam_beta1=recipe['adam_beta1'],adam_beta2=recipe['adam_beta2'],
        adam_epsilon=recipe['adam_epsilon'],max_grad_norm=recipe['max_grad_norm'],bf16=recipe['bf16'],tf32=recipe['tf32'],
        optim=recipe['optimizer'],dataloader_num_workers=0,dataloader_pin_memory=True,remove_unused_columns=False,
        seed=20261016,data_seed=20261016,eval_strategy='no',logging_strategy='steps',logging_steps=1,
        save_strategy='no',save_total_limit=None,save_only_model=False,gradient_checkpointing=False,
        report_to=[],disable_tqdm=True,ignore_data_skip=True,load_best_model_at_end=False)
    times=[];step_start=None;trainer=None
    class Audit(TrainerCallback):
        def on_step_begin(self,arguments,state,control,**kwargs):
            nonlocal step_start
            guard();torch.cuda.synchronize();step_start=time.monotonic()
            with ledger.transaction() as l:
                if l.get('pending_updates',0) or l['optimizer_updates']>=2048:raise PermissionError('E1_NO_UNKNOWN_OR_EXCESS_OPTIMIZER_UPDATE')
                l.update(pending_updates=1,pending_global_step=state.global_step,pending_branch='M1')
        def on_pre_optimizer_step(self,arguments,state,control,model=None,**kwargs):
            groups={'projector':0.,'diffusion':0.}
            for name,p in model.named_parameters():
                if p.grad is None:continue
                if not p.requires_grad or not bool(torch.isfinite(p.grad).all()):raise ValueError('E1_INVALID_NATIVE_GRADIENT')
                if name.startswith('action_head.'):
                    key='diffusion' if name.startswith('action_head.model.') else 'projector'
                    groups[key]+=float(p.grad.float().square().sum())
            if not all(v>0 for v in groups.values()):raise ValueError('E1_PROJECTOR_AND_DIT_REAL_GRADIENTS_REQUIRED')
            with (evidence/'GRADIENTS.jsonl').open('a') as f:f.write(json.dumps(dict(step=state.global_step,norm_squared=groups))+'\n')
        def on_step_end(self,arguments,state,control,**kwargs):
            with ledger.transaction() as l:l['optimizer_updates']+=1;l['pending_updates']=0
            torch.cuda.synchronize();times.append(time.monotonic()-step_start)
            row=dict(step=state.global_step,cursor=trainer.consumed,native_loss=trainer.last_native_loss,
                learning_rate=trainer.lr_scheduler.get_last_lr(),synchronized_s=times[-1],epoch=time.time())
            with (evidence/'OPTIMIZER_STEPS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            write(evidence/'PROGRESS.json',row)
            if state.global_step in [256,1024]:control.should_save=True
            if time.time()>=allocation['allocation_end_epoch']-2700:
                control.should_save=True;control.should_training_stop=True
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    trainer=make_trainer_class()(model=instance.model,args=args,train_dataset=dataset,
        data_collator=ProcessedCollator(dataset.processor),callbacks=[Audit()],evidence_dir=evidence,recipe=recipe,smoke=False)
    result=trainer.train();final=parameter_inventory(instance.model)
    unchanged=all(initial[n]['sha256']==p['sha256'] for n,p in final.items() if not p['requires_grad'])
    changed=[n for n,p in final.items() if p['requires_grad'] and p['sha256']!=initial[n]['sha256']]
    if not unchanged or not changed:raise ValueError('E1_FROZEN_PARAMETERS_CHANGED_OR_NO_LEARNING')
    receipt=dict(status='M1_ACTUAL_1024_UPDATES' if trainer.state.global_step==1024 else 'M1_PARTIAL_FULL_STATE_PRESERVED',
        global_step=trainer.state.global_step,consumed_samples=trainer.consumed,frozen_parameters_unchanged=unchanged,
        changed_trainable_parameters=changed,final_parameters=final,recipe=recipe,native_metrics=result.metrics,
        synchronized_step_s=times,wall_s=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        selected='NONE',deployment='NONE')
    write(evidence/'TRAINING_RUN.json',receipt)
    del trainer
    return receipt
