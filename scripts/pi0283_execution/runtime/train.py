"""Isolated pristine native-forward HF Trainer, two scopes and exact cursor."""
import argparse,hashlib,json,math,os,time,shutil,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments,TrainerCallback,get_cosine_schedule_with_warmup
from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from scope import configure,frozen_hashes,assert_frozen,gradient_proof,digest,is_projector
from learning import Journal,native_rng,tensor_tree,append_completed,check_next_update,save_receipt


class ConsumedTrainer(Gr00tTrainer):
    def get_train_dataloader(self):
        assert self.state.global_step in (0,self.start_step)
        self.args.ignore_data_skip=True
        return DataLoader(self.train_dataset,batch_size=self.args.per_device_train_batch_size,
            sampler=range(self.start_step*8,len(self.train_dataset)),collate_fn=self.data_collator,
            num_workers=2,persistent_workers=True,pin_memory=True,
            generator=torch.Generator().manual_seed(28203),multiprocessing_context='spawn')

    def training_step(self,model,inputs,*args,**kwargs):
        index=self.cursor;step=self.state.global_step;micro=index//self.microbatch % self.accumulation
        assert index==8*step+micro*self.microbatch,(index,step,micro)
        if micro==0:check_next_update(self.root,self.arm,self.phase,step+1)
        # Dataset metadata is consumed here, never charged on worker prefetch.
        visit=inputs.pop('_pi0282_visits')
        assert [v['index'] for v in visit]==list(range(index,index+self.microbatch))
        before=time.perf_counter()
        # Native forward mutates prepared input dictionaries; audit encoded bytes
        # before the model receives them.
        encoded=tensor_tree(inputs)
        self.forwards.append(dict(kind='NATIVE_TRAINING_FORWARD_ATTEMPT',step=step+1,micro=micro,cursor=index,visits=visit,wall=time.time()))
        with native_rng(self.lineage_offset+step,micro) as observed:
            loss=super().training_step(model,inputs,*args,**kwargs)
        torch.cuda.synchronize()
        assert math.isclose(float(loss.detach()),self.actual_native_loss/self.accumulation,rel_tol=1e-6,abs_tol=1e-8),'NATIVE_TRAINER_ACCUMULATION_REDUCTION_CONTRADICTION'
        assert observed.noise is not None and observed.time is not None,'NATIVE_RANDOM_CAPTURE_MISSING'
        detailed=self.phase!='main' or step<8 or (step+1)%128==0 or step+1 in (128,512,1024,2048)
        proof=gradient_proof(model) if detailed else dict(status='DETAILED_GRADIENT_AUDIT_INTERVAL16_NATIVE_GRADCLIP_RETAINED')
        self.journal.append(dict(step=step+1,micro=micro,cursor=index,visits=visit,inputs=encoded,
          native_noise_sha256=digest(observed.noise),native_time_sha256=digest(observed.time),
          native_time=observed.time.float().cpu().tolist(),loss=float(loss.detach()),gradient=proof,
          forward_backward_s=time.perf_counter()-before,loader_wait_s=before-self.previous_end,
          allocated_GPU_bytes=torch.cuda.memory_allocated(),reserved_GPU_bytes=torch.cuda.memory_reserved()))
        self.cursor+=self.microbatch;self.previous_end=time.perf_counter()
        return loss

    def create_optimizer(self):
        if self.optimizer is None:self.optimizer=torch.optim.AdamW(self.groups,betas=(.9,.999),eps=1e-8)
        return self.optimizer

    def create_scheduler(self,num_training_steps,optimizer=None):
        if self.lr_scheduler is None:self.lr_scheduler=get_cosine_schedule_with_warmup(optimizer or self.optimizer,self.warmup,self.phase_updates)
        return self.lr_scheduler


class Provenance(TrainerCallback):
    def __init__(self,trainer,root,arm,phase,frozen,scope):
        self.trainer=trainer;self.root=root;self.arm=arm;self.phase=phase;self.frozen=frozen;self.scope=scope
    def on_step_end(self,args,state,control,**kwargs):
        assert self.trainer.cursor==state.global_step*8
        append_completed(self.root,self.arm,self.phase,state.global_step,self.trainer.cursor)
        if state.global_step in (4,8,128,1024,2048):
            assert_frozen(kwargs['model'],self.frozen,self.trainer.optimizer)
        if self.phase=='main' and state.global_step in (1024,self.trainer.phase_updates):control.should_save=True
        from authority import load
        if time.time()+60>=self.trainer.lease_deadline:
            control.should_training_stop=True;control.should_save=True
        if state.global_step==self.trainer.stop_step:
            control.should_training_stop=True;control.should_save=True
        return control
    def on_save(self,args,state,control,**kwargs):
        checkpoint=Path(args.output_dir)/f'checkpoint-{state.global_step}'
        for name in ('processor_config.json','statistics.json','embodiment_id.json'):
            source=Path(self.trainer.data_root)/name
            assert source.exists(),('DATA_CHECKPOINT_ARTIFACT_MISSING',str(source))
            shutil.copyfile(source,checkpoint/name)
        save_receipt(checkpoint,self.trainer.cursor,self.scope)
        assert_frozen(kwargs['model'],self.frozen,self.trainer.optimizer)
        row_hashes={n:digest(p[10]) for n,p in kwargs['model'].named_parameters() if is_projector(n)}
        assert any(row_hashes[n]!=self.scope['FP32_initial_row10_hashes'][n] for n in row_hashes),'ROW10_DID_NOT_CHANGE'
        if self.arm in ('MORE','EVENT'):
            assert any(digest(p)!=self.scope['initial_DiT_hashes'][n] for n,p in kwargs['model'].named_parameters() if n.startswith('action_head.model.')),'DIT_DID_NOT_CHANGE'
        (checkpoint/'PI0282_SCOPE_PROOF.json').write_text(json.dumps(dict(status='FROZEN_EXACT_ROWS_MOMENTS_ZERO_AND_ROW10_CHANGED',row10=row_hashes),indent=2)+'\n')
        if self.phase=='main' and state.global_step in (1024,self.trainer.phase_updates):
            from learning import export_point,file_digest
            snapshot=self.root/'delivery'/(self.arm+str(state.global_step))
            export_point(checkpoint,self.trainer.original_base,snapshot,
                set(dict(kwargs['model'].named_parameters())))
            for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):
                shutil.copyfile(checkpoint/name,snapshot/name)
            files=[dict(path=p.name,bytes=p.stat().st_size,sha256=file_digest(p)) for p in snapshot.iterdir() if p.is_file()]
            from training_launch import write
            write(snapshot/'READY.json',dict(status='LOCAL_IMMUTABLE_MODEL_ONLY_SNAPSHOT',point=self.arm+str(state.global_step),files=files,
                freeze_boundary=state.global_step,full_recovery=str(checkpoint),optimizer_transfer=False))


def collate_with_metadata(items):
    from cpu_inputs import collate
    visits=[v.pop('_pi0282_visit') for v in items]
    result=collate(items);result['_pi0282_visits']=visits
    return result


class PlannedDataset(torch.utils.data.Dataset):
    def __init__(self,data_root,plan_path=None):
        from cpu_inputs import FrozenDataset
        self.base=FrozenDataset(data_root,plan_path)
        self.plan=self.base.rows
        assert len(self.base)==len(self.plan) and len(self.plan) in (8192,16384)
    def __len__(self):return len(self.plan)
    def __getitem__(self,i):
        result=dict(self.cached_item(i));result['_pi0282_visit']=self.plan[i];return result


    def cached_item(self,i):
            from cpu_inputs import tensor_hash
            row=self.plan[i]
            key=hashlib.sha256(json.dumps({'namespace':self.base.namespace,'uid':row['uid'],'command_sha256':row['command_sha256'],'observation_sha256':row['observation_sha256']},sort_keys=True).encode()).hexdigest()
            old=Path(self.base.root).parent.parent/'PI0282/data/cache'/self.base.namespace/(key+'.pt')
            if old.exists():
                saved=torch.load(old,map_location='cpu',weights_only=True)
                assert saved['key']==key and saved['uid']==row['uid'] and tensor_hash(saved['inputs'])==saved['tensor_sha256']
                return saved['inputs']
            # ponytail: reuse the deterministic encoder; bound only the new private cache.
            if i%64==0:
                import fcntl
                folder=self.base.root/'cache'/self.base.namespace;folder.mkdir(parents=True,exist_ok=True)
                with (folder/'eviction.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    files=sorted(folder.glob('*.pt'),key=lambda p:p.stat().st_mtime);size=sum(p.stat().st_size for p in files)
                    for p in files:
                        if size<=3*1024**3:break
                        if time.time()-p.stat().st_mtime>60:size-=p.stat().st_size;p.unlink()
            try:return self.base[i]
            except FileNotFoundError:return self.base[i]


def continuation_proof(root,arm):
    # EVENT reuses the one authorized generic carry proof; no second canary is claimed.
    proof_arm='MORE' if arm=='EVENT' else arm
    path=root/'learning'/proof_arm/'CANARY_COMPARISON.json'
    v=json.loads(path.read_text());assert v['status']=='PASS_FULL_RESUME_CANARY'
    if arm=='EVENT':
        assert v['arm']=='MORE' and v['model_bitwise_equal'] and v['inputs_native_noise_time_exact'] and v['predictions_gates_exact']
        from learning import file_digest
        from training_launch import write
        record=dict(status='REUSED_MORE_CARRY_MECHANISM_PROOF_NOT_EVENT_CANARY',source_path=str(path),source_sha256=file_digest(path),EVENT_canary_updates=0,EVENT_parent_admission=str(root/'runtime/MORE_TRAINING_FINAL_CPU_AUDIT.json'))
        assert json.loads(Path(record['EVENT_parent_admission']).read_text())['status']=='PASS_ACTUAL_NATIVE_CONTINUATION_FULL_TRAINERS_ORDER_SCOPE_AND_MOMENTS_CPU'
        write(root/'runtime/EVENT_CARRY_PROOF_REUSE.json',record)


def run(a):
    root=Path(a.root)
    from authority import allowed
    allowed(root,'update',reserve_s=30)
    manifest=json.loads((root/'design/IMPLEMENTATION.json').read_text());assert manifest['release']=='8fdaab73b8b6686409317d9562833235c2839acc'
    assert manifest['published_commit']
    for name,sha in manifest['files'].items():assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==sha,name
    allocation=json.loads((root/'design/TRAINING_ALLOCATION.json').read_text())
    assert allocation['kind']=='PBS_H200' and os.environ.get('PBS_JOBID'),'Hopper model work requires owner legal allocation'
    layout=json.loads((root/'design/TRAINING_LAYOUT.json').read_text())
    assert a.microbatch==layout['microbatch'] and 8//a.microbatch==layout['accumulation']
    assert (root/'data/EXPERT_CODEC_ADMISSION.json').exists()
    admission=json.loads((root/'data/EXPERT_CODEC_ADMISSION.json').read_text());assert admission['status']=='PASS_FOUR_ORIGINAL_EXPERT_CODEC_REPLAYS'
    continuation=None
    assert a.continuation_card
    if a.continuation_card:
        from phase import validate
        continuation=validate(root,Path(a.continuation_card))
        assert a.arm==continuation['stage']
    if a.phase=='main':continuation_proof(root,a.arm)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(28203);torch.cuda.manual_seed_all(28203)
    # Stock model and stock forward/loss. All dependencies remain offline.
    from native_model import load_model
    model,loading=load_model(a.base)
    reference=json.loads((root/'learning/WEIGHT_LOADING_REFERENCE.json').read_text())
    for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
        assert loading.get(key,[])==reference.get(key,[])==[],(key,loading.get(key))
    parameters=dict(model.named_parameters());assert set(parameters)==set(reference['parameter_hashes'])
    for name,p in parameters.items():assert digest(p)==reference['parameter_hashes'][name],name
    groups,scope=configure(model,'ACTION2048')
    frozen=frozen_hashes(model)
    if continuation:
        from phase import restore_model
        restore_model(model,continuation)
        assert frozen_hashes(model)==frozen,'PARENT_FROZEN_IDENTITY_CHANGED'
        for i,group in enumerate(groups):group['lr']=continuation['peak_LRs'][i]
        scope.update(arm=a.arm,parent=continuation['parent'],phase_scheduler_warmup=continuation['warmup'])
    scope['FP32_initial_row10_hashes']={n:digest(p[10]) for n,p in model.named_parameters() if is_projector(n)}
    scope['initial_DiT_hashes']={n:digest(p) for n,p in model.named_parameters() if n.startswith('action_head.model.')}
    frozen=frozen_hashes(model)
    output=root/'learning'/a.arm/a.phase;output.mkdir(parents=True,exist_ok=True)
    process_receipt=output/f'PROCESS_resume{a.resume}_stop{a.stop}.json'
    assert not process_receipt.exists(),'Adopt existing process; do not repeat completed work'
    process_receipt.write_text(json.dumps(dict(pid=os.getpid(),PBS_JOBID=os.environ['PBS_JOBID'],argv=sys.argv,
       deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,
       cudnn_benchmark=torch.backends.cudnn.benchmark,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
       restore_global_step=a.resume,stop_global_step=a.stop,started=time.time(),resume_requires_fresh_process=a.resume>0),indent=2)+'\n')
    (output/'SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
    args=TrainingArguments(output_dir=str(output),max_steps=continuation['updates'],per_device_train_batch_size=a.microbatch,
       gradient_accumulation_steps=8//a.microbatch,learning_rate=continuation['peak_LRs'][0],weight_decay=0.,warmup_steps=continuation['warmup'],
       max_grad_norm=1.,bf16=True,fp16=False,tf32=False,optim='adamw_torch',lr_scheduler_type='cosine',
       logging_steps=16,save_steps=1024,save_total_limit=2,save_only_model=False,report_to='none',
       dataloader_num_workers=2,dataloader_pin_memory=True,remove_unused_columns=False,ignore_data_skip=True,seed=28203,data_seed=28203)
    trainer=ConsumedTrainer(model=model,args=args,train_dataset=PlannedDataset(a.data_root,continuation['plan_path'] if continuation else None),data_collator=collate_with_metadata)
    trainer.warmup=continuation['warmup'];trainer.phase_updates=continuation['updates'];trainer.lineage_offset=continuation['parent_global_update'];trainer.lease_deadline=json.loads((root/'runtime/TRAINING_ALLOCATION.json').read_text())['started_epoch']+allocation['training_GPU_reservation_s']-120
    trainer.groups=groups;trainer.start_step=a.resume;trainer.stop_step=a.stop;trainer.microbatch=a.microbatch;trainer.accumulation=8//a.microbatch;trainer.data_root=a.data_root;trainer.original_base=a.base
    trainer.root=root;trainer.arm=a.arm;trainer.phase=a.phase
    trainer.cursor=a.resume*8;trainer.previous_end=time.perf_counter();trainer.journal=Journal(output/'consumption.jsonl')
    trainer.forwards=Journal(output/'native_training_forwards.jsonl')
    if continuation:
        from phase import restore_optimizer
        trainer.create_optimizer()
        restore_optimizer(trainer.optimizer,continuation,output/'PARENT_RESTORE.json')
        trainer._load_rng_state(continuation['parent'])
        import random
        (output/'PARENT_RNG_RESTORE.json').write_text(json.dumps(tensor_tree(dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),indent=2))
    from phase import install_audit
    audit_hooks=install_audit(trainer,model)
    trainer.add_callback(Provenance(trainer,root,a.arm,a.phase,frozen,scope))
    resume=output/f'checkpoint-{a.resume}' if a.resume else None
    if resume:
        recovery=json.loads((resume/'PI0282_RECOVERY.json').read_text());assert recovery['cursor']==trainer.cursor
    try:trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    finally:
        receipt=dict(native_loss_modes_LR=trainer.mode_journal.close(),consumption=trainer.journal.close(),native_training_forwards=trainer.forwards.close())
        (output/'JOURNAL.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--arm',choices=('MORE','EVENT'),required=True);p.add_argument('--phase',choices=('continuous','split','main','memory'),required=True)
    p.add_argument('--continuation-card')
    p.add_argument('--microbatch',type=int,choices=(4,2,1),default=4);p.add_argument('--resume',type=int,default=0);p.add_argument('--stop',type=int,required=True)
    a=p.parse_args();assert 0<=a.resume<a.stop<=2048;run(a)
