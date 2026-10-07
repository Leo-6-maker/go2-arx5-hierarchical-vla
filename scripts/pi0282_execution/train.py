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
        with native_rng(step,micro) as observed:
            loss=super().training_step(model,inputs,*args,**kwargs)
        torch.cuda.synchronize()
        assert observed.noise is not None and observed.time is not None,'NATIVE_RANDOM_CAPTURE_MISSING'
        detailed=self.phase!='main' or step<8 or (step+1)%16==0 or step+1 in (128,512,1024,2048)
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
        if self.lr_scheduler is None:self.lr_scheduler=get_cosine_schedule_with_warmup(optimizer or self.optimizer,103,2048)
        return self.lr_scheduler


class Provenance(TrainerCallback):
    def __init__(self,trainer,root,arm,phase,frozen,scope):
        self.trainer=trainer;self.root=root;self.arm=arm;self.phase=phase;self.frozen=frozen;self.scope=scope
    def on_step_end(self,args,state,control,**kwargs):
        assert self.trainer.cursor==state.global_step*8
        append_completed(self.root,self.arm,self.phase,state.global_step,self.trainer.cursor)
        if state.global_step in (4,8,128,1024,2048):
            assert_frozen(kwargs['model'],self.frozen,self.trainer.optimizer)
        if self.phase=='main' and state.global_step in (128,512):control.should_save=True
        from authority import load
        if time.time()+30>=load(self.root)['training_cutoff_epoch']:
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
        if self.arm=='ACTION2048':
            assert any(digest(p)!=self.scope['initial_DiT_hashes'][n] for n,p in kwargs['model'].named_parameters() if n.startswith('action_head.model.')),'DIT_DID_NOT_CHANGE'
        (checkpoint/'PI0282_SCOPE_PROOF.json').write_text(json.dumps(dict(status='FROZEN_EXACT_ROWS_MOMENTS_ZERO_AND_ROW10_CHANGED',row10=row_hashes),indent=2)+'\n')
        if self.phase=='main' and state.global_step in (512,1024,2048):
            from learning import export_point
            export_point(checkpoint,self.trainer.original_base,Path(args.output_dir).parent/'snapshots'/f'step-{state.global_step}',
                set(dict(kwargs['model'].named_parameters())))
            for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):
                shutil.copyfile(checkpoint/name,Path(args.output_dir).parent/'snapshots'/f'step-{state.global_step}'/name)


def collate_with_metadata(items):
    from cpu_inputs import collate
    visits=[v.pop('_pi0282_visit') for v in items]
    result=collate(items);result['_pi0282_visits']=visits
    return result


class PlannedDataset(torch.utils.data.Dataset):
    def __init__(self,data_root):
        from cpu_inputs import FrozenDataset
        self.base=FrozenDataset(data_root)
        self.plan=self.base.rows
        assert len(self.base)==len(self.plan)==16384
    def __len__(self):return len(self.plan)
    def __getitem__(self,i):
        result=dict(self.base[i]);result['_pi0282_visit']=self.plan[i];return result


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
    if a.phase=='main':assert json.loads((root/'learning'/a.arm/'CANARY_COMPARISON.json').read_text())['status']=='PASS_FULL_RESUME_CANARY'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(28203);torch.cuda.manual_seed_all(28203)
    # Stock model and stock forward/loss. All dependencies remain offline.
    model,loading=Gr00tN1d7.from_pretrained(a.base,torch_dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
    reference=json.loads((root/'learning/WEIGHT_LOADING_REFERENCE.json').read_text())
    for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
        assert loading.get(key,[])==reference.get(key,[])==[],(key,loading.get(key))
    parameters=dict(model.named_parameters());assert set(parameters)==set(reference['parameter_hashes'])
    for name,p in parameters.items():assert digest(p)==reference['parameter_hashes'][name],name
    groups,scope=configure(model,a.arm)
    scope['FP32_initial_row10_hashes']={n:digest(p[10]) for n,p in model.named_parameters() if is_projector(n)}
    scope['initial_DiT_hashes']={n:digest(p) for n,p in model.named_parameters() if n.startswith('action_head.model.')}
    frozen=frozen_hashes(model)
    output=root/'learning'/a.arm/a.phase;output.mkdir(parents=True,exist_ok=True)
    process_receipt=output/f'PROCESS_resume{a.resume}_stop{a.stop}.json'
    assert not process_receipt.exists(),'Adopt existing process; do not repeat completed work'
    process_receipt.write_text(json.dumps(dict(pid=os.getpid(),PBS_JOBID=os.environ['PBS_JOBID'],argv=sys.argv,
       restore_global_step=a.resume,stop_global_step=a.stop,started=time.time(),resume_requires_fresh_process=a.resume>0),indent=2)+'\n')
    (output/'SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
    args=TrainingArguments(output_dir=str(output),max_steps=2048,per_device_train_batch_size=a.microbatch,
       gradient_accumulation_steps=8//a.microbatch,learning_rate=1e-4,weight_decay=0.,warmup_steps=103,
       max_grad_norm=1.,bf16=True,fp16=False,tf32=False,optim='adamw_torch',lr_scheduler_type='cosine',
       logging_steps=16,save_steps=1024,save_total_limit=2,save_only_model=False,report_to='none',
       dataloader_num_workers=2,dataloader_pin_memory=True,remove_unused_columns=False,ignore_data_skip=True,seed=28203,data_seed=28203)
    trainer=ConsumedTrainer(model=model,args=args,train_dataset=PlannedDataset(a.data_root),data_collator=collate_with_metadata)
    trainer.groups=groups;trainer.start_step=a.resume;trainer.stop_step=a.stop;trainer.microbatch=a.microbatch;trainer.accumulation=8//a.microbatch;trainer.data_root=a.data_root;trainer.original_base=a.base
    trainer.root=root;trainer.arm=a.arm;trainer.phase=a.phase
    trainer.cursor=a.resume*8;trainer.previous_end=time.perf_counter();trainer.journal=Journal(output/'consumption.jsonl')
    trainer.forwards=Journal(output/'native_training_forwards.jsonl')
    trainer.add_callback(Provenance(trainer,root,a.arm,a.phase,frozen,scope))
    resume=output/f'checkpoint-{a.resume}' if a.resume else None
    if resume:
        recovery=json.loads((resume/'PI0282_RECOVERY.json').read_text());assert recovery['cursor']==trainer.cursor
    try:trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    finally:
        receipt=dict(consumption=trainer.journal.close(),native_training_forwards=trainer.forwards.close())
        (output/'JOURNAL.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--arm',choices=('ACTION2048','INTERFACE2048'),required=True);p.add_argument('--phase',choices=('continuous','split','main','memory'),required=True)
    p.add_argument('--microbatch',type=int,choices=(4,2,1),default=4);p.add_argument('--resume',type=int,default=0);p.add_argument('--stop',type=int,required=True)
    a=p.parse_args();assert 0<=a.resume<a.stop<=2048;run(a)
