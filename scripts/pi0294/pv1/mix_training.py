"""C/D cold hook around the existing official Trainer and PV1 reader.

No hand-written optimizer/forward loop. The caller supplies actual registered
allocation, release and native-effect ledger; legacy entry is unchanged.
"""
from copy import deepcopy
import json,os,resource,time
from pathlib import Path
from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import ProcessedCollator,sha
from scripts.pi0294.native_train import parameter_inventory
from .dataset import PV1Dataset
from . import mix1024 as mix
from .recipe import PARENT
from scripts.pi0294.scene_cards import digest


def begin_update(v,*,arm,local_step,request_id,recipe):
    if (recipe.get('schema')!=mix.SCHEMA or recipe.get('arm')!=arm or arm not in mix.ARMS
        or type(local_step)is not int or not 0<=local_step<1024 or v['pending_updates'] or v['unknown_effects']
        or v['optimizer_updates']!=local_step or not request_id):raise PermissionError('V5_SINGLE_EXACT_KNOWN_UPDATE_INTENT')
    n=deepcopy(v);runs=n['training_runs'];expected=dict(recipe_sha256=digest(recipe),last_native_step=local_step)
    if not runs and local_step==0:runs[arm]=expected
    if runs!={arm:expected}:raise PermissionError('V5_NO_PRIOR_ARM_OR_OPTIMIZER_RESET')
    n['pending_updates']=[dict(arm=arm,run_key=arm,local_step_before=local_step,request_id=request_id)]
    return n


def commit_update(v,*,arm,local_step_after,request_id):
    from .recipe import commit_update as mature_commit
    # The existing commit primitive is arm-independent and requires the exact
    # durable pending step. Schema/1024 range is enforced at begin and ledger.
    if arm not in mix.ARMS or not 1<=local_step_after<=1024:raise PermissionError('V5_ONLY_1024_MATCHED_UPDATES')
    return mature_commit(v,arm=arm,local_step_after=local_step_after,request_id=request_id)


def dataset(root,arm,evidence):
    root=Path(root);b=read(root/'NEURAL_BINDINGS.private.json');i=read(root/'plans/PV1_TRAIN_INPUTS.private.json')
    # These are qualified full-bank READER roles, never request/arm identities.
    reader=PV1Dataset(b['dataset'],b['source'],b['cosmos'],for_optimization=True,
        raw_bindings=read(b['raw_bindings']),trace_root=Path(evidence)/'worker_trace',
        correction_root=i['C1_dataset'],correction_bindings=read(i['C1_raw_bindings']),
        new_root=i['new_dataset'],new_raw_bindings=read(i['new_raw_bindings']),
        route='COVERAGE_FIRST',arm='B_COVERAGE',seed=20261019)
    result=mix.bind_dataset_reader(reader,arm);result.initialize();return result


def checked_resume(root,arm,checkpoint,recipe,data):
    cp=Path(checkpoint);root=Path(root)
    if cp.is_symlink() or cp.parent.resolve()!=(root/'training'/arm).resolve():raise PermissionError('V5_OWN_EXACT_COMPLETE_RESUME')
    complete=read(cp/'CHECKPOINT_COMPLETE.json');hashes={}
    for n,h in complete['files'].items():
        p=cp/n
        if p.is_symlink() or not p.resolve().is_relative_to(cp.resolve()) or sha(p)!=h:raise PermissionError('V5_CHANGED_FULL_RESUME_MEMBER:'+n)
        hashes[n]=h
    sampler=data.make_sampler(data,stop=mix.VISITS)
    return mix.resume_state(read(cp/'SAMPLER_STATE.json'),complete,read(cp/'trainer_state.json'),recipe,sampler,hashes)


def run(root,instance,allocation,guard,ledger,*,arm,request_id,resume_checkpoint=None):
    from .mix_native import require_training,authorize_training
    root=Path(root);guard();directory=require_training(root,allocation)
    request=read(directory/'REQUEST.json');original=read(root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json');recipe=read(root/'plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][arm];mix.validate(recipe,original)
    if request['arm']!=arm or request['request_id']!=request_id:raise PermissionError('V5_EXACT_SUBMITTED_ARM_AND_REQUEST')
    evidence=root/'training'/arm/('session-'+request_id)
    if evidence.exists():raise FileExistsError('V5_NO_REPLAY_OF_NATIVE_TRAINING_SESSION')
    evidence.mkdir(parents=True);data=dataset(root,arm,evidence)
    start=checked_resume(root,arm,resume_checkpoint,recipe,data) if resume_checkpoint else 0
    if read(ledger.path)['optimizer_updates']!=start:raise PermissionError('V5_ACTUAL_EFFECT_STEP_AND_RESTORE_DIFFER')
    parent=read(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json');initial=parameter_inventory(instance.model)
    if (set(parent)!=set(initial) or any(p['requires_grad']!=parent[n]['requires_grad'] for n,p in initial.items())
        or any(p['requires_grad'] for n,p in initial.items() if n.startswith('backbone.'))
        or any(p['sha256']!=parent[n]['sha256'] for n,p in initial.items() if not p['requires_grad'] or not start)
        or instance.identity['global_step']!=(start or 2048)
        or not start and instance.identity['checkpoint_manifest_sha256']!=PARENT):raise PermissionError('V5_SAME_CORR_PARENT_AND_FROZEN_TRAINABLE_MAP')
    authorize_training(root,directory,allocation,guard)
    import torch
    from transformers import TrainingArguments,TrainerCallback,set_seed
    write(evidence/'INITIAL_PARAMETERS.json',initial);write(evidence/'RECIPE.json',recipe);set_seed(recipe['seed']);instance.model.train()
    args=TrainingArguments(output_dir=str(root/'training'/arm),**mix.trainer_arguments(recipe,original));times=[];began=None;trainer=None
    class Audit(TrainerCallback):
        def on_step_begin(self,arguments,state,control,**kwargs):
            nonlocal began
            guard();torch.cuda.synchronize();began=time.monotonic()
            with ledger.transaction() as v:
                n=begin_update(v,arm=arm,local_step=state.global_step,request_id=request_id,recipe=recipe);v.clear();v.update(n)
        def on_pre_optimizer_step(self,arguments,state,control,model=None,**kwargs):
            values=dict(projector=0.,diffusion=0.)
            for n,p in model.named_parameters():
                if p.grad is None:continue
                if not p.requires_grad or not bool(torch.isfinite(p.grad).all()):raise ValueError('V5_NONFINITE_OR_FROZEN_GRADIENT')
                if n.startswith('action_head.'):values['diffusion' if n.startswith('action_head.model.') else 'projector']+=float(p.grad.float().square().sum())
            if not all(v>0 for v in values.values()):raise ValueError('V5_REQUIRED_PROJECTOR_AND_DIT_GRADIENTS')
            with (evidence/'GRADIENTS.jsonl').open('a') as f:f.write(json.dumps(dict(step=state.global_step,norm_squared=values))+'\n')
        def on_step_end(self,arguments,state,control,**kwargs):
            torch.cuda.synchronize()
            with ledger.transaction() as v:
                n=commit_update(v,arm=arm,local_step_after=state.global_step,request_id=request_id);v.clear();v.update(n)
            v=read(ledger.path);mix.step_boundary(state.global_step,trainer.consumed,pending=v['pending_updates'],unknown=v['unknown_effects']);times.append(time.monotonic()-began)
            row=dict(step=state.global_step,cursor=trainer.consumed,native_loss=trainer.last_native_loss,learning_rate=trainer.lr_scheduler.get_last_lr(),synchronized_s=times[-1],epoch=time.time())
            with (evidence/'OPTIMIZER_STEPS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            write(evidence/'PROGRESS.json',row)
            # Resume-only checkpoint512, sole behavioral candidate1024. A
            # time/CPU orderly stop also saves exact committed full state.
            closing=time.time()>=allocation['allocation_end_epoch']-2730 or time.process_time()>=resource.getrlimit(resource.RLIMIT_CPU)[0]-120
            if state.global_step in (512,1024) or closing:control.should_save=True
            if state.global_step==1024 or closing:control.should_training_stop=True
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    base=mix.make_trainer_class()
    class BoundedSave(base):
        def _save_checkpoint(self,model,trial):
            from .mix_native import prewrite
            guard();prewrite(root);return super()._save_checkpoint(model,trial)
    trainer=BoundedSave(model=instance.model,args=args,train_dataset=data,data_collator=ProcessedCollator(data.processor),callbacks=[Audit()],evidence_dir=evidence,recipe=recipe,smoke=False)
    try:result=trainer.train(resume_from_checkpoint=str(resume_checkpoint) if start else None)
    except BaseException as e:
        from .training import save_known_completed_step
        closing=dict(original_error=type(e).__name__+': '+str(e),extra_optimizer_updates=0)
        try:
            if time.time()<allocation['allocation_end_epoch']:closing.update(save_known_completed_step(root,arm,trainer,ledger,torch))
            else:closing['status']='ALLOCATION_ENDED_NO_LATE_CUDA_OR_SAVE'
        except BaseException as q:closing.update(save_error_type=type(q).__name__,status='SAVE_FAILED_PRESERVED_NO_ROLLBACK')
        write(evidence/'INTERRUPTED_NATIVE_STATE.json',closing);raise
    final=parameter_inventory(instance.model);unchanged=all(p['sha256']==initial[n]['sha256'] for n,p in final.items() if not p['requires_grad']);changed=[n for n,p in final.items() if p['requires_grad'] and p['sha256']!=initial[n]['sha256']]
    if not unchanged or not changed:raise ValueError('V5_FROZEN_VLM_CHANGED_OR_NO_LEARNING')
    value=dict(status='V5_CD_1024_COMPLETE' if trainer.state.global_step==1024 else 'V5_CD_EXACT_PARTIAL_SAVED',arm=arm,global_step=trainer.state.global_step,resumed_from_step=start,actual_updates_this_attempt=trainer.state.global_step-start,actual_samples_this_attempt=trainer.consumed-start*32,actual_consumed_cursor=trainer.consumed,sampler=trainer.sampler.state_dict(trainer.consumed),parent_checkpoint_manifest_sha256=PARENT,statistics_sha256=mix.STATS,source_manifest_sha256=data.manifest_sha,new_bank_manifest_sha256=data.new_reader.manifest_sha,recipe=recipe,native_metrics=result.metrics,synchronized_step_s=times,wall_s=time.monotonic()-started,frozen_parameters_unchanged=unchanged,changed_trainable_parameters=changed,final_parameters=final,peak_GPU_bytes=torch.cuda.max_memory_allocated(),development_evaluation_executed=False)
    write(evidence/'TRAINING_RUN.json',value);del trainer;return value
