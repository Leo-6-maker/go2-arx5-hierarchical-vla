"""Finite TRAIN16 fitting study, using the unchanged native loss and Trainer.

The small checkpoint is a parent-dependent training delta, never a standalone
model. Every frozen tensor belongs to the exact retained full book128 parent.
"""
import hashlib
from pathlib import Path
import shutil

from scripts.pi0292.train_adapter import CyclicSampler
from scripts.pi0294.multitask.book_probe import plan as probe_plan


def plan():
    p=probe_plan()
    return dict(schema='BOOK128_TRAIN16_TARGET_FIT256_V1',
        checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],
        dataset_manifest_sha256=p['dataset_manifest_sha256'],statistics_sha256=p['statistics_sha256'],
        windows=p['windows'],max_steps=256,effective_batch_size=32,TRAIN_visits=8192,
        trainable='EXACT_EXISTING_FOURTEEN_TARGET_CATEGORY10_PARAMETERS',
        shared_DiT_frozen=True,backbone_frozen=True,optimizer_initialization='FRESH_ADAMW_DISTINCT_DIAGNOSTIC',
        learning_rate=1e-4,weight_decay=1e-5,warmup_updates=8,scheduler_horizon_steps=256,
        seed=2026103102,loss='UNCHANGED_OFFICIAL_MASKED_FLOW_MATCHING',
        checkpoint='PARENT_BOUND_TARGET_DELTA_WITH_ADAMW_SCHEDULER_RNG_CURSOR',
        denoise_steps=4,generation_queries=32,native_diagnostic_loss_forwards=32,
        matched_before_after_inputs_and_noise=True,statistics_refit=False,
        DEV_or_TEST_used=False,formal_DEV_conditions=0,physics_integrations=0,
        original_book_DEV10_unchanged=True,original_E1_E2_unchanged=True,
        purpose='Can the existing target interface fit sixteen actual TRAIN expert windows? Not a generalization or task-success claim.')


def validate_manifest(m,p):
    if p!=plan()or m['checkpoint_manifest_sha256']!=p['checkpoint_manifest_sha256']:
        raise PermissionError('EXACT_FROZEN_BOOK_TRAIN16_FIT_PLAN')
    for key,value in dict(optimizer_updates=256,physics_integrations=0,generation_queries=32,model_loads=1).items():
        if m['limits'].get(key)!=value:raise PermissionError('BOOK_TRAIN16_FIT_FINITE_'+key)


def select_rows(rows):
    lookup={(r['id'],a):(r,a)for r,a in rows}
    selected=[]
    for row in plan()['windows']:
        value=lookup[row['id'],row['anchor']]
        if value[0]['split']!='TRAIN':raise PermissionError('TRAIN16_FIT_NO_DEV_OR_TEST')
        selected.append(value)
    return selected


class FitSampler(CyclicSampler):
    def __init__(self,dataset,*,start=0,stop=8192):
        if len(dataset)!=16:raise ValueError('FIT_EXACT_SIXTEEN_TRAIN_WINDOWS')
        super().__init__(len(dataset),seed=plan()['seed'],start=start,stop=stop,positioned=True)

    def index_at(self,cursor):
        import numpy as np
        if not self.start<=cursor<self.stop:raise IndexError('FIT_FINITE_CURSOR')
        cycle,offset=divmod(cursor,self.size)
        return int(np.random.Generator(np.random.PCG64(self.seed+cycle)).permutation(self.size)[offset])


def validate_delta(delta,parameters,names):
    if set(delta)!=set(names)or len(names)!=14:raise ValueError('EXACT_FOURTEEN_TARGET_TENSORS_REQUIRED')
    for name in names:
        a,b=delta[name],parameters[name]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('TARGET_DELTA_SHAPE_DTYPE:'+name)


def measure(root,phase,instance,data,trainer,guard):
    import numpy as np
    import torch
    from scripts.pi0294.native_train import NativeNoiseAudit
    from scripts.pi0294.native_dataset import ProcessedCollator,sha
    from scripts.pi0294.e1.model import payload
    from scripts.pi0294.e1.artifacts import atomic_npz
    from scripts.pi0294.e1.smoke import put
    from scripts.pi0292.eval_contract import gate_action,error_metrics
    from scripts.pi0292.durable import write
    folder=root/phase;folder.mkdir(exist_ok=False);records=[]
    instance.model.eval()
    for index,row in enumerate(plan()['windows']):
        guard();key=row['id']+'-'+str(row['anchor']).zfill(3)
        state,expert,images,text=data.raw(index)
        seed=int.from_bytes(hashlib.sha256(f"BOOK128_TRAIN16/{row['id']}/{row['anchor']}".encode()).digest()[:8],'big')%(2**63)
        batch=ProcessedCollator(data.processor)([data[index]])
        batch={k:{n:v.to('cuda')if torch.is_tensor(v)else v for n,v in values.items()}for k,values in batch.items()}
        put(folder/(key+'.INTENT.json'),dict(window=row,seed=seed,denoise_steps=4))
        observer=NativeNoiseAudit(instance.model)
        try:
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,_=trainer.compute_loss(instance.model,batch,return_outputs=True)
            loss_value=float(loss.detach());noise=observer.record
        finally:observer.close()
        if not np.isfinite(loss_value)or noise is None:raise ValueError('FIT_ACTUAL_FINITE_NATIVE_LOSS')
        calls=[];hook=instance.model.action_head.model.register_forward_pre_hook(lambda module,args:calls.append(1))
        previous=instance.model.action_head.num_inference_timesteps
        try:
            instance.model.action_head.num_inference_timesteps=4
            value=instance.generate(payload(state,images,text),seed)
        finally:instance.model.action_head.num_inference_timesteps=previous;hook.remove()
        if len(calls)!=4:raise ValueError('FIT_ACTUAL_FOUR_DENOISE_CALLS')
        arrays=folder/(key+'.npz');atomic_npz(arrays,physical_H40=value['physical_H40'],normalized=value['normalized'],actual_state10=state,actual_completed_expert_H40=expert)
        r=dict(window=row,seed=seed,native_loss=loss_value,native_noise=noise,noise=value['noise'],
            observed_native_DiT_forward_calls=len(calls),arrays_sha256=sha(arrays),
            image_SHA256=[x['sha256']for x in payload(state,images,text)['images']],
            expert_gate=gate_action(expert,state,0.,key+'-EXPERT'),
            original_gate=gate_action(value['physical_H40'],state,0.,key),
            errors=error_metrics(value['physical_H40'],expert,state))
        put(folder/(key+'.RESULT.json'),r);records.append(r)
    result=dict(phase=phase,TRAIN_windows=16,generation_queries=16,native_loss_forwards=16,
        native_loss_mean=float(np.mean([r['native_loss']for r in records])),
        expert_gate_rejects=sum(r['expert_gate']['decision']=='REJECT'for r in records),
        gate_rejects=sum(r['original_gate']['decision']=='REJECT'for r in records),
        H40_XYZ_mean_m=float(np.mean([r['errors']['H40']['XYZ_euclidean_m']['mean']for r in records])),
        H40_SO3_mean_rad=float(np.mean([r['errors']['H40']['SO3_rad']['mean']for r in records])),
        H40_gripper_MAE=float(np.mean([r['errors']['H40']['gripper_MAE']for r in records])),records=records)
    write(folder/'SUMMARY.json',result);return result


def run(root,instance,contract,guard):
    import gc,json,time
    import torch
    from transformers import TrainingArguments,TrainerCallback,TrainerState,set_seed
    from scripts.pi0292.durable import read,write
    from scripts.pi0294.native_dataset import NativeDataset,ProcessedCollator,sha
    from scripts.pi0294.native_train import make_trainer_class,MeasuredLoader,tensor_identity,tree_identity,rng_identity
    from scripts.pi0294.e1.trainer import learning_rate_multiplier,check_trace
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    from scripts.pi0294.multitask.book_checkpoint import load_wrapped
    root=Path(root);p=read(root/'TRAIN_FIT_PLAN.json');validate_manifest(read(root/'RUN_MANIFEST.json'),p)
    model=instance.model;parameters=dict(model.named_parameters());names=contract.trainable_names
    for name in names:parameters[name].requires_grad_(True)
    contract.validate();frozen=frozen_fingerprint(model)
    before_parameters={n:tensor_identity(parameters[n])for n in names}
    binding=instance.bindings
    data=NativeDataset(binding['dataset'],binding['source'],binding['cosmos'],for_optimization=True,trace_root=root/'fit_trace')
    probe=NativeDataset(binding['dataset'],binding['source'],binding['cosmos'],for_optimization=False)
    for ds in(data,probe):
        if ds.manifest_sha!=p['dataset_manifest_sha256']or ds.statistics_sha!=p['statistics_sha256']:raise PermissionError('FIT_ORIGINAL_ADMITTED_TRAIN_DATA_AND_STATS')
        ds.rows=select_rows(ds.rows);ds.initialize()
    data.make_sampler=FitSampler;set_seed(p['seed'])

    class FitTrainer(make_trainer_class()):
        def get_train_dataloader(self):
            self.start_cursor=self.consumed=self.state.global_step*32
            self.sampler=FitSampler(self.train_dataset,start=self.consumed)
            return MeasuredLoader(torch.utils.data.DataLoader(self.train_dataset,batch_size=1,sampler=self.sampler,
                collate_fn=self.data_collator,num_workers=0,pin_memory=True,drop_last=True,
                generator=torch.Generator().manual_seed(p['seed'])))

        def create_optimizer(self):
            contract.validate();result=super().create_optimizer()
            if {id(x)for g in self.optimizer.param_groups for x in g['params']}!={id(x)for x in contract.parameters()}:raise ValueError('FIT_EXACT_OPTIMIZER_PARAMETERS')
            return result

        def create_scheduler(self,num_training_steps,optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer,lambda step:learning_rate_multiplier(step,256,8))
                self._created_lr_scheduler=True
            return self.lr_scheduler

        def training_step(self,model,inputs,num_items_in_batch=None):
            guard();contract.validate();return super().training_step(model,inputs,num_items_in_batch)

        def _save_checkpoint(self,model,trial):
            guard()
            if self.state.global_step!=256 or self.consumed!=8192:raise ValueError('FIT_ONLY_FIXED256_COMMITTED_ENDPOINT')
            cp=Path(self.args.output_dir)/'checkpoint-256';cp.mkdir(parents=True,exist_ok=False)
            if frozen_fingerprint(model)!=frozen:raise ValueError('FIT_FROZEN_PARENT_CHANGED')
            torch.save({n:parameters[n].detach().cpu().clone()for n in names},cp/'target_parameters.pt')
            self._save_optimizer_and_scheduler(str(cp));self._save_rng_state(str(cp));self.state.save_to_json(str(cp/'trainer_state.json'))
            self.train_dataset.processor.save_pretrained(cp/'processor')
            for name in('statistics.json','TRAIN_DEV_SEAL.json'):shutil.copyfile(data.root/name,cp/name)
            write(cp/'ADAPTER_PARAMETER_CONTRACT.json',contract.describe())
            write(cp/'FROZEN_PARENT_TENSORS.json',frozen)
            write(cp/'SAMPLER_STATE.json',dict(recipe=p,sampler=self.sampler.state_dict(self.consumed),
                worker_trace=check_trace(data.trace_root,data,0,8192),dataset_manifest_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha))
            members={str(x.relative_to(cp)):sha(x)for x in sorted(cp.rglob('*'))if x.is_file()}
            required={'target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','statistics.json','TRAIN_DEV_SEAL.json','SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json'}
            if not required<=set(members):raise ValueError('FIT_PARENT_DEPENDENT_FULL_RESUME_MEMBERS')
            write(cp/'CHECKPOINT_COMPLETE.json',dict(schema='BOOK_TRAIN16_PARENT_BOUND_TARGET_DELTA_V1',global_step=256,cursor=8192,
                full_resume_with_required_parent=True,standalone_model=False,required_parent_checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],
                recipe=p,files=members,actual_reload_check='PENDING'))

    args=TrainingArguments(output_dir=str(root/'training'),max_steps=256,per_device_train_batch_size=1,
        gradient_accumulation_steps=32,learning_rate=1e-4,weight_decay=1e-5,adam_beta1=.9,adam_beta2=.999,
        adam_epsilon=1e-8,max_grad_norm=1.,bf16=True,tf32=True,optim='adamw_torch',dataloader_num_workers=0,
        remove_unused_columns=False,seed=p['seed'],data_seed=p['seed'],eval_strategy='no',logging_strategy='steps',logging_steps=1,
        save_strategy='no',save_only_model=False,save_total_limit=None,gradient_checkpointing=False,report_to=[],disable_tqdm=True,ignore_data_skip=True)
    holder={}
    class Audit(TrainerCallback):
        def on_step_begin(self,args,state,control,**kwargs):guard()
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            for name,value in model.named_parameters():
                if value.grad is not None and(name not in names or not torch.isfinite(value.grad).all()):raise ValueError('FIT_ONLY_FINITE_DECLARED_GRADIENTS:'+name)
        def on_step_end(self,args,state,control,**kwargs):
            torch.cuda.synchronize();t=holder['trainer']
            if t.consumed!=state.global_step*32:raise ValueError('FIT_ACTUAL_COMMITTED_CURSOR')
            r=dict(step=state.global_step,cursor=t.consumed,native_loss=t.last_native_loss,learning_rate=t.lr_scheduler.get_last_lr(),epoch=time.time())
            with(root/'OPTIMIZER_STEPS.jsonl').open('a')as f:f.write(json.dumps(r)+'\n');f.flush()
            write(root/'FIT_PROGRESS.json',r)
            if state.global_step==256:control.should_save=True;control.should_training_stop=True
            return control
    trainer=FitTrainer(model=model,args=args,train_dataset=data,data_collator=ProcessedCollator(data.processor),callbacks=[Audit()],evidence_dir=root/'fit_evidence',recipe=p)
    holder['trainer']=trainer
    before=measure(root,'before',instance,probe,trainer,guard)
    # A probe uses fork_rng and deterministic preprocessing, but the training
    # seed is reset explicitly; no diagnostic random draw enters optimization.
    set_seed(p['seed']);trainer.train()
    if trainer.state.global_step!=256 or trainer.consumed!=8192 or frozen_fingerprint(model)!=frozen:raise ValueError('FIT_FIXED256_FROZEN_ENDPOINT')
    after_parameters={n:tensor_identity(parameters[n])for n in names}
    changed=[n for n in names if before_parameters[n]!=after_parameters[n]]
    if not changed:raise ValueError('FIT_NO_ACTUAL_PARAMETER_UPDATE')
    cp=root/'training/checkpoint-256';complete=read(cp/'CHECKPOINT_COMPLETE.json')
    for name,digest in complete['files'].items():
        if sha(cp/name)!=digest:raise ValueError('FIT_SAVED_MEMBER_CHANGED:'+name)
    expected_optimizer=tree_identity(trainer.optimizer.state_dict());expected_scheduler=trainer.lr_scheduler.state_dict();expected_rng=rng_identity()
    # Reconstruct the actual model from the retained full parent plus the delta.
    load_wrapped(model,Path(read(root/'RUN_MANIFEST.json')['checkpoint']))
    delta=torch.load(cp/'target_parameters.pt',map_location='cpu',weights_only=True);validate_delta(delta,parameters,names)
    with torch.no_grad():
        for name in names:parameters[name].copy_(delta[name])
    if {n:tensor_identity(parameters[n])for n in names}!=after_parameters or frozen_fingerprint(model)!=frozen:raise ValueError('FIT_ACTUAL_PARENT_PLUS_DELTA_RECONSTRUCTION')
    restored=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    if not restored['state']or len(restored['state'])!=14 or any(int(x['step'])!=256 for x in restored['state'].values()):raise ValueError('FIT_ACTUAL_ADAMW256_REQUIRED')
    trainer.optimizer.state.clear();trainer.optimizer.load_state_dict(restored)
    trainer.lr_scheduler.load_state_dict(torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False));trainer._load_rng_state(str(cp))
    if tree_identity(trainer.optimizer.state_dict())!=expected_optimizer or trainer.lr_scheduler.state_dict()!=expected_scheduler or rng_identity()!=expected_rng:raise ValueError('FIT_EXACT_OPTIMIZER_SCHEDULER_RNG_RESTORE')
    saved=read(cp/'SAMPLER_STATE.json')
    trainer.state=TrainerState.load_from_json(str(cp/'trainer_state.json'))
    restored_sampler=FitSampler(data,start=saved['sampler']['cursor'])
    if trainer.state.global_step!=256 or saved['sampler']!=restored_sampler.state_dict(8192):raise ValueError('FIT_ACTUAL_STEP_AND_SAMPLE_CURSOR_RESTORE')
    write(root/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json',dict(checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),
        required_parent_checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],model_reconstruction=True,optimizer_state_entries=14,
        optimizer_updates=256,scheduler_restored=True,RNG_restored=True,sampler_cursor=8192,frozen_parent_unchanged=True,
        changed_trainable_parameters=changed,standalone_model=False))
    after=measure(root,'after',instance,probe,trainer,guard)
    for a,b in zip(before['records'],after['records']):
        if a['window']!=b['window']or a['noise']!=b['noise']or a['image_SHA256']!=b['image_SHA256']:raise ValueError('FIT_MATCHED_BEFORE_AFTER_NOISE_INPUTS')
    trainer=None;gc.collect()
    return dict(status='COMPLETE_TRAIN16_FIT256_DIAGNOSTIC',optimizer_updates=256,TRAIN_visits=8192,
        checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),checkpoint_requires_retained_full_parent=True,
        generation_queries=32,native_diagnostic_loss_forwards=32,physics_integrations=0,DEV_conditions=0,
        before=before,after=after,learned_task_success_not_measured=True)
