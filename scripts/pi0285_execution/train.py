"""Instruction-aware adapter around the already verified native GR00T Trainer."""
import argparse,functools,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
from contract import guard,sha,write


def run(a):
    import torch
    from transformers import TrainingArguments,TrainerCallback
    from authority import allowed,completed_kind,stage_card
    root=Path(a.root);allowed(root,'update',60);card=stage_card(root,a.card);assert a.arm==card['stage']
    for path,h in json.loads((root/'design/CODE_SEAL.json').read_text()).get('native_dependencies',{}).items():assert sha(path)==h,path
    assert 0<=a.resume<a.stop<=card['updates'] and a.microbatch==4
    if a.phase=='main':assert json.loads((root/'learning'/a.arm/'CANARY_COMPARISON.json').read_text())['status']=='PASS_FULL_RESUME_CANARY'
    prior=root.parent/'PI0283/code';sys.path.append(str(prior))
    spec=importlib.util.spec_from_file_location('pi0283_native_trainer',prior/'train.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    import learning
    learning.completed_kind=completed_kind
    from scope import configure,frozen_hashes,assert_frozen,digest,is_projector
    from native_model import load_model
    from phase import install_audit,restore_model,restore_optimizer
    from data import Dataset,collate
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(28203);torch.cuda.manual_seed_all(28203)
    model,loading=load_model(a.base)
    reference=json.loads((root/'design/PRISTINE_LOADING_REFERENCE.json').read_text())
    for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):assert loading.get(key,[])==reference.get(key,[])==[]
    assert set(dict(model.named_parameters()))==set(reference['parameter_hashes'])
    for name,p in model.named_parameters():assert digest(p)==reference['parameter_hashes'][name],name
    groups,scope=configure(model,'ACTION2048');frozen=frozen_hashes(model)
    if card['parent'] is not None:
        restore_model(model,card);assert frozen_hashes(model)==frozen
    for group,lr in zip(groups,card['peak_LRs']):group['lr']=lr
    scope.update(stage=a.arm,parent=card['parent'],FP32_initial_row10_hashes={n:digest(p[10]) for n,p in model.named_parameters() if is_projector(n)},initial_DiT_hashes={n:digest(p) for n,p in model.named_parameters() if n.startswith('action_head.model.')})
    output=root/'learning'/a.arm/a.phase;output.mkdir(parents=True,exist_ok=True)
    record=output/f'PROCESS_resume{a.resume}_stop{a.stop}.json';assert not record.exists(),'Adopt existing process and actual completed work'
    write(record,dict(pid=os.getpid(),PBS_JOBID=os.environ['PBS_JOBID'],started=time.time(),resume=a.resume,stop=a.stop,card_sha256=sha(a.card),native_trainer_sha256=sha(prior/'train.py'),loader_workers=2,microbatch=4,accumulation=2,explicit_stage='TRAIN_FP32_MASTERS_BF16_AUTOCAST',deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
    write(output/'SCOPE.json',scope)
    args=TrainingArguments(output_dir=str(output),max_steps=card['updates'],per_device_train_batch_size=4,gradient_accumulation_steps=2,learning_rate=card['peak_LRs'][0],weight_decay=0.,warmup_steps=card['warmup'],max_grad_norm=1.,bf16=True,fp16=False,tf32=False,optim='adamw_torch',lr_scheduler_type='cosine',logging_steps=16,save_steps=1024,save_total_limit=None,save_only_model=False,report_to='none',dataloader_num_workers=2,dataloader_pin_memory=True,remove_unused_columns=False,ignore_data_skip=True,seed=28203,data_seed=28203)
    trainer=native.ConsumedTrainer(model=model,args=args,train_dataset=Dataset(a.data_root,card['plan_path']),data_collator=functools.partial(collate,root=a.data_root))
    trainer.warmup=card['warmup'];trainer.phase_updates=card['updates'];trainer.lineage_offset=card['parent_global_update']
    allocation=json.loads((root/'runtime/TRAINING_ALLOCATION.json').read_text());trainer.lease_deadline=allocation['deadline_epoch']-120
    trainer.groups=groups;trainer.start_step=a.resume;trainer.stop_step=a.stop;trainer.microbatch=4;trainer.accumulation=2;trainer.data_root=a.data_root;trainer.original_base=a.base
    trainer.root=root;trainer.arm=a.arm;trainer.phase=a.phase;trainer.cursor=a.resume*8;trainer.previous_end=time.perf_counter()
    trainer.journal=learning.Journal(output/'consumption.jsonl');trainer.forwards=learning.Journal(output/'native_training_forwards.jsonl')
    if card['parent'] is not None:
        trainer.create_optimizer();restore_optimizer(trainer.optimizer,card,output/'PARENT_RESTORE.json');trainer._load_rng_state(card['parent'])
    audit_handles=install_audit(trainer,model)

    class Provenance(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            assert trainer.cursor==state.global_step*8
            learning.append_completed(root,a.arm,a.phase,state.global_step,trainer.cursor)
            if state.global_step in (4,8) or state.global_step%1024==0:assert_frozen(kwargs['model'],frozen,trainer.optimizer)
            if time.time()+120>=allocation['deadline_epoch'] or state.global_step==a.stop:
                control.should_training_stop=True;control.should_save=True
            return control
        def on_save(self,args,state,control,**kwargs):
            checkpoint=Path(args.output_dir)/f'checkpoint-{state.global_step}'
            for name in ('processor_config.json','statistics.json','embodiment_id.json'):shutil.copyfile(Path(a.data_root)/name,checkpoint/name)
            learning.save_receipt(checkpoint,trainer.cursor,scope);assert_frozen(kwargs['model'],frozen,trainer.optimizer)
            rows={n:digest(p[10]) for n,p in kwargs['model'].named_parameters() if is_projector(n)}
            assert any(rows[n]!=scope['FP32_initial_row10_hashes'][n] for n in rows)
            assert any(digest(p)!=scope['initial_DiT_hashes'][n] for n,p in kwargs['model'].named_parameters() if n.startswith('action_head.model.'))
            # Same recovery ABI lets the mature disposable-canary retirement keep its proofs.
            write(checkpoint/'PI0282_SCOPE_PROOF.json',dict(status='FROZEN_EXACT_ROWS_MOMENTS_ZERO_ROW10_AND_DIT_CHANGED',authority_release=card['release'],row10=rows))
            if a.phase=='main' and state.global_step in (1024,2048,4096):
                snapshot=root/'delivery'/(a.arm+'-'+str(state.global_step))
                learning.export_point(checkpoint,a.base,snapshot,set(dict(kwargs['model'].named_parameters())))
                for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):shutil.copyfile(checkpoint/name,snapshot/name)
                write(snapshot/'READY.json',dict(status='LOCAL_IMMUTABLE_MODEL_ONLY_SNAPSHOT',point=snapshot.name,full_recovery=str(checkpoint),card_sha256=sha(a.card),files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in snapshot.iterdir() if p.is_file()],optimizer_transfer=False))
                # Immutable checkpoint and journal prefix are audited while this lease stays owned.
                with (root/'runtime'/f'{a.arm}-{state.global_step}_training_CPU.log').open('ab') as log:
                    subprocess.run([sys.executable,'-u',str(root/'code/training_audit.py'),'--arm',a.arm,'--step',str(state.global_step)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
                audits=root/'delivery/audits';audits.mkdir(exist_ok=True)
                shutil.copyfile(root/'evidence'/f'{a.arm}-{state.global_step}_TRAINING_CPU.json',audits/f'{a.arm}-{state.global_step}.json')
    trainer.add_callback(Provenance())
    resume=output/f'checkpoint-{a.resume}' if a.resume else None
    if resume:
        saved=json.loads((resume/'PI0282_RECOVERY.json').read_text());assert saved['cursor']==trainer.cursor
        assert saved['scope']==scope
    try:trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    finally:
        write(output/'JOURNAL.json',dict(consumption=trainer.journal.close(),native_training_forwards=trainer.forwards.close(),native_loss_modes_LR=trainer.mode_journal.close()))
        for handle in audit_handles:handle.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--data-root',required=True);p.add_argument('--card',required=True);p.add_argument('--arm',choices=('PLACE4096','SHARED4096','REPAIR_A2048','REPAIR_B2048'),required=True);p.add_argument('--phase',choices=('continuous','split','main','memory'),required=True);p.add_argument('--microbatch',type=int,choices=(4,),default=4);p.add_argument('--resume',type=int,default=0);p.add_argument('--stop',type=int,required=True);run(p.parse_args())
