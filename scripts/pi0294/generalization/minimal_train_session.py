"""One qualified session through the original loss/Trainer/save methods.

Called only by the registered first512 compute entry. This is not a launcher,
owner, optimizer implementation or admission source. Each qualification and
formal run has its own raw model, optimizer, directory and local effect slice.
"""
import os,resource,time
from pathlib import Path

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import ProcessedCollator,sha
from . import minimal_training as training,minimal_execution as execution
from .minimal_adapter import frozen_fingerprint,parameter_state_fingerprint
from .minimal_effects import Effects
from .minimal_dataset import Dataset,PANEL_SHA

need=execution.need


def dataset(root,evidence):
    root=Path(root);binding=read(root/'NEURAL_BINDINGS.private.json')
    panel=read(root/'plans/MINIMAL_TWELVE_PANEL.json')
    need(training.study.digest(panel)==PANEL_SHA,'SESSION_FROZEN_TWELVE_PARENT_PANEL')
    value=Dataset(binding['dataset'],binding['source'],binding['cosmos'],split='TRAIN',
        for_optimization=True,raw_bindings=read(binding['raw_bindings']),
        trace_root=Path(evidence)/'worker_trace',panel=panel)
    value.initialize();return value


def save_headroom(root,arm,*,next_upper_bytes=28000000000,guard=None):
    """Original per-arm70GB retention; do not delete qualification states.

    The main request retains BOTH arm envelopes. Count the same arm's qual,
    formal and official export, without traversing source/input symlinks.
    """
    if guard is not None and hasattr(guard,'reserve'):guard.reserve(next_upper_bytes)
    root=Path(root);need(arm in training.ARMS and type(next_upper_bytes)is int
        and 0<next_upper_bytes<=28000000000,'SESSION_DECLARED_FULL_STATE_SAVE_BOUND')
    folders=[root/'training'/(arm+'_QUALIFY2'),root/'training'/(arm+'_FORMAL512'),root/'inference'/arm]
    used=0
    for folder in folders:
        need(not folder.is_symlink(),'SESSION_OWN_SAVE_ROOT_NOT_INPUT_ALIAS')
        if not folder.exists():continue
        for path in folder.rglob('*'):
            need(not path.is_symlink(),'SESSION_NO_SYMLINK_HIDDEN_RETENTION')
            if path.is_file():used+=path.stat().st_size
    fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize
    need(used+next_upper_bytes<=70000000000 and free>=next_upper_bytes,
        'SESSION_KEEP_QUALIFICATION_AND_FORMAL_STATE_WITHIN70GB_AND_REAL_FREE_SPACE')
    return dict(arm=arm,retained_bytes=used,next_full_save_upper_bytes=next_upper_bytes,free_bytes=free)


def build_trainer(root,instance,allocation,ledger,session_id,*,guard,restoring_qualification=False):
    """Construct the original Trainer; no train() or optimizer step here."""
    need(isinstance(ledger,Effects)and callable(guard),'SESSION_ORIGINAL_TYPED_LOCAL_EFFECTS_AND_GUARD')
    guard();spec=next(x for x in execution.native_sessions()if x['id']==session_id)
    need(spec['optimizer_limit']in(2,512)and instance.parameter_contract.arm==spec['arm']
        and instance.identity['mode']==spec['arm']and instance.identity['global_step']==0
        and instance.identity['checkpoint_manifest_sha256']==training.RAW_PARENT_FILES_ID,
        'SESSION_SEPARATE_FRESH_RAW_PARENT_NOT_QUALIFICATION_OR_OTHER_ARM')
    root=Path(root);original=read(root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json')
    recipe=read(root/'plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][spec['arm']]
    training.validate(recipe,original)
    known=ledger.known_effects(session_id)
    need(type(restoring_qualification)is bool and(not restoring_qualification or spec['optimizer_limit']==2)
        and known['optimizer_updates']==(2 if restoring_qualification else 0)
        and known['pending_updates']==known['unknown_effects']==[],
        'SESSION_EXACT_FRESH_START_OR_OWN_STEP2_AUDIT_NOT_FORMAL_RESUME')
    output=root/'training'/session_id;evidence=output/('restore-audit'if restoring_qualification else'session')
    need(not evidence.exists()and not evidence.is_symlink(),'SESSION_NO_NATIVE_TRAINING_OR_RESTORE_REPLAY')
    evidence.mkdir(parents=True,mode=0o700);data=dataset(root,evidence)
    import torch
    from transformers import TrainingArguments,TrainerCallback,set_seed
    set_seed(recipe['seed']);args=TrainingArguments(output_dir=str(output),**training.trainer_arguments(recipe,original))
    times=[];start={};holder={}
    from . import minimal_e2only_contract as e2
    recovery=ledger.request.get('platform_training_version')==e2.VERSION
    if recovery:
        from .minimal_e2only_execution import request as check_request
        check_request(ledger.request)
        need(spec==e2.sessions()[0] and not restoring_qualification,
            'SESSION_E2_ONLY_RECOVERY_IS_NOT_QUALIFICATION_OR_E1')

    class Audit(TrainerCallback):
        def on_step_begin(self,arguments,state,control,**kwargs):
            guard();torch.cuda.synchronize();start['wall']=time.monotonic()
            ledger.begin_update(session_id,state.global_step,recipe)

        def on_pre_optimizer_step(self,arguments,state,control,model=None,**kwargs):
            for name,p in model.named_parameters():
                if p.grad is not None:
                    need(p.requires_grad and bool(torch.isfinite(p.grad).all()),
                         'SESSION_FINITE_TRAINABLE_GRADIENT_ONLY:'+name)

        def on_step_end(self,arguments,state,control,**kwargs):
            torch.cuda.synchronize();ledger.commit_update(session_id,state.global_step)
            trainer=holder['trainer'];known=ledger.known_effects(session_id)
            need(trainer.consumed==state.global_step*32,'SESSION_ACTUAL_COMMITTED_SAMPLE_CURSOR')
            times.append(time.monotonic()-start['wall'])
            row=dict(step=state.global_step,cursor=trainer.consumed,native_loss=trainer.last_native_loss,
                learning_rate=trainer.lr_scheduler.get_last_lr(),synchronized_s=times[-1],epoch=time.time())
            with(evidence/'OPTIMIZER_STEPS.jsonl').open('a')as f:
                import json
                f.write(json.dumps(row)+'\n');f.flush()
            write(evidence/'PROGRESS.json',row)
            closing=(time.time()>=allocation['allocation_end_epoch']-2730 or
                     time.process_time()>=resource.getrlimit(resource.RLIMIT_CPU)[0]-120)
            if closing:control.should_save=True;control.should_training_stop=True
            need(known['pending_updates']==known['unknown_effects']==[],'SESSION_ORIGINAL_COMMIT_BEFORE_BOUNDARY')
            if recovery and e2.recovery_boundary(state.global_step,trainer.consumed,
                    pending=known['pending_updates'],unknown=known['unknown_effects']):
                control.should_save=True
            return control

    base=training.make_trainer_class()
    class BoundedSave(base):
        def _save_checkpoint(self,model,trial):
            guard()
            if recovery:
                from .minimal_e2only_worker import recovery_save_headroom
                recovery_save_headroom(root,ledger.request,self.state.global_step,guard=guard)
            else:save_headroom(root,spec['arm'],guard=guard)
            result=super()._save_checkpoint(model,trial)
            if recovery:
                from .minimal_worker import seal_checkpoint
                cp=output/('checkpoint-'+str(self.state.global_step))
                need(sum(p.stat().st_size for p in cp.rglob('*')if p.is_file())<=26000000000,
                    'SESSION_E2_COMPLETE_STATE_WITHIN_DECLARED26GB')
                proof=seal_checkpoint(root,session_id,self.state.global_step,sha(cp/'CHECKPOINT_COMPLETE.json'),
                    guard=guard,recovery_request=ledger.request)
                write(evidence/('FULL_RECOVERY_SAVE_'+str(self.state.global_step)+'.json'),proof)
            return result
    trainer=BoundedSave(model=instance.model,args=args,train_dataset=data,data_collator=ProcessedCollator(data.processor),
        callbacks=[Audit()],evidence_dir=evidence,recipe=recipe,smoke=False,
        parameter_contract=instance.parameter_contract,session_stop=spec['optimizer_limit'],
        original_recipe=original,known_effects=lambda:ledger.known_effects(session_id),
        qualification_only=not spec['formal_candidate'])
    holder['trainer']=trainer
    return trainer,dict(spec=spec,recipe=recipe,evidence=evidence,times=times)


def run_training(root,instance,allocation,ledger,session_id,*,guard,authorize_training):
    """Two qualification calls or fixed512; never a hand-written train loop."""
    need(callable(authorize_training),'SESSION_ORIGINAL_OWNER_FIRST_UPDATE_GATE_REQUIRED')
    from scripts.pi0294.dec1.a40_training_runtime import native_job
    job,_,_=native_job(ledger.request)
    need(job==allocation['job_id']==os.environ.get('PBS_JOBID'),'SESSION_ACTUAL_OWN_RUNNING_A40')
    trainer,context=build_trainer(root,instance,allocation,ledger,session_id,guard=guard)
    proof=authorize_training(session_id)
    need(proof['status']=='PASS_MINIMAL_SESSION_FIRST_UPDATE_NATIVE'
        and proof['session_id']==session_id and proof['job_id']==job
        and proof['request_sha256']==allocation['request_sha256']
        and 0<=time.time()-proof['observed_epoch']<=120,'SESSION_FRESH_NATIVE_FIRST_UPDATE_RELEASE')
    import torch
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();initial_frozen=frozen_fingerprint(instance.model)
    result=trainer.train()
    expected=context['spec']['optimizer_limit'];known=ledger.known_effects(session_id)
    need(trainer.state.global_step==expected and trainer.consumed==expected*32
        and known['optimizer_updates']==expected and known['pending_updates']==known['unknown_effects']==[],
        'SESSION_ORIGINAL_EXACT_COMMITTED_ENDPOINT_NO_PARTIAL_AS_COMPLETE')
    need(frozen_fingerprint(instance.model)==initial_frozen,'SESSION_FROZEN_PARAMETERS_AND_BUFFERS_UNCHANGED')
    cp=Path(trainer.args.output_dir)/('checkpoint-'+str(expected));complete=read(cp/'CHECKPOINT_COMPLETE.json')
    need(complete['global_step']==expected and complete['full_resume']is True
        and complete['qualification_only']is(not context['spec']['formal_candidate']),
        'SESSION_FULL_ORIGINAL_TRAINER_STATE_SAVED_NOT_INFERENCE_ONLY')
    from .minimal_model import bind_saved_checkpoint
    current=bind_saved_checkpoint(instance,cp,guard=guard)
    record=dict(status='ACTUAL_MINIMAL_ORIGINAL_TRAINER_SESSION_ENDPOINT_SAVED',session_id=session_id,
        arm=context['spec']['arm'],global_step=expected,cursor=trainer.consumed,
        checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),
        full_model_parameter_state=current['actual_parameter_identity'],
        scheduler_horizon=1024,sampler=trainer.sampler.state_dict(trainer.consumed),
        frozen_tensors_unchanged=True,wall_s=time.monotonic()-started,
        synchronized_step_s=context['times'],peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        native_metrics=result.metrics,not_a_behavior_result=True,
        qualification_requires_full_restore_and_output_equality=not context['spec']['formal_candidate'],
        official_inference_export_equality_still_required=context['spec']['formal_candidate'])
    write(context['evidence']/'TRAINING_RUN.json',record)
    return trainer,record
