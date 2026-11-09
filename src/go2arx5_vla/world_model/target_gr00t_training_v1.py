"""Target GR00T training on original native Trainer/loss and isolated profile.

Caller supplies the original registry/Journal/PBS owner. This module cannot
submit PBS, schedule another worker, run physics or train an external adapter.
"""
import json,os,time
from pathlib import Path

from . import target_gr00t_profile_v1 as profile
from .storage import digest,write_json


def site_context(spec, job_id, quota, *, uid, user):
    """PBS binds the allocation request; each child keeps its model generation."""
    return dict(schema='SHARED_RAW_VANDA_A40_EXECUTION_V1',job_id=job_id,owner_uid=uid,
        owner_user=user,science_run_id=spec['allocation_request_id'],
        submission_epoch=spec['submission_epoch'],submission_nonce=spec['submission_nonce'],
        authorization_mode='SAME_PBS_ALLOCATION_AFTER_FRESH_SUBMISSION',project_receipt=spec['project_receipt'],
        quota_receipt=quota,predicted_new_peak_bytes=spec['predicted_new_peak_bytes'],
        allocation_deadline_epoch=spec['deadline_epoch'])


def site(spec):
    """Validate the actual own single-A40 allocation BEFORE importing Torch."""
    if spec.get('execution_site')=='CORELAB_4080S':
        from .target_gr00t_core_v1 import site as core_site
        return core_site(spec)
    if spec.get('execution_site')=='HOPPER_SINGLE_GPU':
        from .task_learning_hopper_v1 import site as hopper_site
        return hopper_site(spec)
    import subprocess
    from scripts.pi0294.multitask.shared_site import validate_vanda,project_quota,mount_type,PROJECT_SCRATCH
    p=profile.validate(spec);now=time.time();root=Path(spec['registered_root'])
    need=profile.need
    need(os.getuid()==22182 and root.resolve()==root and not root.is_symlink()
         and root.is_relative_to(Path(PROJECT_SCRATCH)),'ACTUAL_PROJECT_SCRATCH')
    jid=os.environ.get('PBS_JOBID');need(bool(jid),'PBS_COMPUTE_REQUIRED_BEFORE_MODEL_IMPORT')
    q=subprocess.run(['/opt/pbs/bin/qstat','-f','-F','json',jid],capture_output=True,text=True,timeout=20,check=True)
    jobs=json.loads(q.stdout)['Jobs'];matches=[v for k,v in jobs.items()if k.split('.')[0]==jid.split('.')[0]]
    need(len(matches)==1,'ONE_ACTUAL_JOB')
    job=matches[0];need(job['Resource_List'].get('gpu_model')=='A40'
         or all(n.startswith('GN-A40-')for n in job.get('exec_host','').split('+')),'ACTUAL_A40_ONLY')
    quota_path=Path('/home/svu/vanda/quotas/CFP04-CF-007.proj_scratch.txt')
    quota=project_quota(quota_path.read_text(),user=os.environ['USER'],
        report_epoch=quota_path.stat().st_mtime,observed_epoch=now)
    context=site_context(spec,jid,quota,uid=os.getuid(),user=os.environ['USER'])
    validate_vanda(context,job,now=now,uid=os.getuid(),user=os.environ['USER'],
        hostname=os.uname().nodename,job_id=jid)
    # Existing site validator is the PBS/entitlement kernel, not the old M32
    # scientific recipe. Bind the actually observed mounts, not an old "nfs" guess.
    for value in (root,spec['bindings']['dataset'],spec['bindings']['cosmos'],spec['parent']['path']):
        path=Path(value).resolve(strict=True)
        need(any(path.is_relative_to(Path(r['root']).resolve(strict=True))and mount_type(path)==r['fstype']
                 for r in spec['storage_roots']),'PINNED_ACTUAL_STORAGE_MOUNT')
    lease_path=root/'control/LEASE.json';lease=json.loads(lease_path.read_bytes())
    need(lease.get('status')=='ACTIVE'and lease.get('spec_sha256')==digest(root/'EXECUTION_SPEC.json')
         and lease.get('job_id')==jid,'NEW_NONZERO_PROFILE_ACTIVE_LEASE')
    registry=json.loads(Path(spec['registry']).read_bytes())
    need(any(r.get('path')==str(root)and r.get('role')==spec['role']
             and r.get('scope')=='TARGET_GR00T_V1'for r in registry['entries']),'ORIGINAL_REGISTERED_TARGET_OWNER')
    for name,pin in spec['source_files'].items():
        path=Path(spec['source_root'])/name
        need(path.resolve().is_relative_to(Path(spec['source_root']))and digest(path)==pin,
             'IMMUTABLE_EXECUTION_SOURCE:'+name)
    need(now+spec['closeout_reserve_s']<spec['deadline_epoch'] and spec['closeout_reserve_s']>=1200,
         'FINITE_ALLOCATION_AND_TWENTY_MINUTE_CLOSEOUT')
    return p


def action_contract(model,variant,*,profile_name=profile.PROFILE):
    """Select the original action tensors; the profile names their provenance."""
    profile.need(profile_name in(profile.PROFILE,'TASK_LEARNING_GR00T_V1'),
        'EXPLICIT_ORIGINAL_OR_CONTINUOUS_ACTION_PARAMETER_IDENTITY')
    from scripts.pi0294.generalization.minimal_adapter import ParameterContract,LEAVES
    base=ParameterContract(model,'E1_ADAPTER_ONLY',10)
    names=profile.whitelist(tuple(dict(model.named_parameters())),base.trainable_names,variant)
    for name,p in model.named_parameters():p.requires_grad_(name in names)
    for path in LEAVES:model.get_submodule(path).retain_frozen_tables_on_cpu()
    class Contract:
        trainable_names=names
        def validate(self):
            profile.need(tuple(n for n,p in model.named_parameters()if p.requires_grad)==names
                and not any(p.requires_grad for n,p in model.named_parameters()if n.startswith('backbone.')),
                'DECLARED_INTERNAL_ACTION_SIDE_ONLY')
        def parameters(self):
            self.validate();return [model.get_parameter(n)for n in names]
        def describe(self):
            self.validate()
            rows=[dict(name=n,shape=list(model.get_parameter(n).shape),numel=model.get_parameter(n).numel(),
                dtype=str(model.get_parameter(n).dtype))for n in names]
            return dict(profile=profile_name,variant=variant,target_category=10,
                trainable=rows,trainable_numel=sum(r['numel']for r in rows),
                visual_and_language_backbone_frozen=True,external_adapter_trainables=0)
    return Contract()


def semantic_recipe(p):
    return {k:v for k,v in p.items()if k not in ('endpoint','initial_optimizer')}


def dataset_type(recipe):
    if recipe.get('profile')=='TASK_LEARNING_GR00T_V1':
        from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
        return TaskLearningNativeDataset
    profile.need(recipe.get('profile')==profile.PROFILE,'EXPLICIT_TARGET_DATASET_PROFILE')
    from .target_gr00t_dataset_v1 import TargetNativeDataset
    return TargetNativeDataset


def gpu_identity(torch,spec):
    from .runtime import cuda_uuid
    core=spec.get('execution_site')=='CORELAB_4080S'
    hopper=spec.get('execution_site')=='HOPPER_SINGLE_GPU'
    name=spec['GPU_name']if hopper else'NVIDIA GeForce RTX 4080 SUPER'if core else'NVIDIA A40'
    profile.need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)==name,
        'ACTUAL_SINGLE_DECLARED_GPU')
    properties=torch.cuda.get_device_properties(0)
    if hopper:
        from .task_learning_hopper_v1 import actual_gpu,allocation_model
        actual_gpu(dict(count=torch.cuda.device_count(),name=name,major=properties.major,
            total_memory=properties.total_memory),allocation_model(spec))
    actual=cuda_uuid(properties.uuid)
    profile.need(actual==cuda_uuid(spec['GPU_uuid']),'ALLOCATED_EXACT_GPU_UUID')
    if core:torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/properties.total_memory)
    return actual


def _load_original_parent(spec,root,label):
    from transformers import AutoModel
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    from scripts.pi0292.durable import write
    bindings=spec['bindings']
    write(root/(label+'_INTENT.json'),dict(parent_manifest_sha256=profile.PARENT,epoch=time.time()))
    model,loading=AutoModel.from_pretrained(spec['parent']['path'],local_files_only=True,trust_remote_code=True,
        output_loading_info=True,model_name=local_model_name(root,bindings),
        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
            cache_dir=bindings['cosmos_hub'],revision=COSMOS_REVISION),
        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
        state_dropout_prob=0.,backbone_trainable_params_fp32=True,load_bf16=False)
    write(root/(label+'_CONSTRUCTED_ACTUAL.json'),dict(actual_model_loads=1,epoch=time.time(),loading=loading))
    profile.need(not any(loading.get(k)for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
                 'STRICT_PINNED_PARENT_LOAD')
    profile.need(model.action_head.num_inference_timesteps==4,'ORIGINAL_FOUR_STEP_FREE_GENERATION')
    return model


def _package_and_reconstruct(root,cp,spec,restore,model,trainer,instance,data,guard):
    """Production only: fresh original model + packaged delta + saved Processor."""
    import gc,torch
    from . import target_gr00t_delivery_v1 as delivery
    from .target_gr00t_probes_v1 import measure
    from scripts.pi0292.durable import write
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    guard();package=delivery.create(cp,root/'inference-package',spec,restore,guard=guard)
    # The complete optimizer archive is already restored and checked on disk.
    # Free its device moments before constructing the independent inference model.
    trainer.optimizer.state.clear();trainer.lr_scheduler=None;trainer.optimizer=None
    model.to('cpu');instance.model=None;gc.collect();torch.cuda.empty_cache()
    guard();rebuilt=_load_original_parent(spec,root,'PACKAGE_MODEL_LOAD')
    delivery.install(rebuilt,package['path'],package['sha256'],spec['parent']['files'])
    processor=Gr00tN1d7Processor.from_pretrained(Path(package['path'])/'processor',
        model_name=local_model_name(root,spec['bindings']),local_files_only=True,
        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
            cache_dir=spec['bindings']['cosmos_hub'],revision=COSMOS_REVISION))
    processor.eval()
    profile.need(processor.embodiment_id_mapping['new_embodiment']==10,'PACKAGE_ORIGINAL_EMBODIMENT_MAPPING')
    rebuilt.to('cuda').eval();instance.model=rebuilt;instance.processor=processor
    instance.identity['inference_package_sha256']=package['sha256']
    measure(root,'package-reconstructed',instance,data)
    proof=delivery.compare_generations(root/'restored',root/'package-reconstructed')
    write(root/'PACKAGE_RECONSTRUCTION_ACTUAL.json',dict(**proof,package=package))
    guard();return dict(**package,reconstruction=proof)


def load_checkpoint(path,pin,*,model,contract,trainer,spec,guard):
    import torch
    from scripts.pi0294.native_train import tensor_identity,tree_identity,rng_identity
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    path=Path(path);guard();profile.need(digest(path/'CHECKPOINT_COMPLETE.json')==pin,'CHECKPOINT_HEADER')
    c=json.loads((path/'CHECKPOINT_COMPLETE.json').read_bytes())
    profile.need(c['schema']=='WM02_TARGET_GR00T_FULL_STATE_DELTA_V1'
        and c['generation_id']==spec['generation_id'] and c['parent_manifest_sha256']==profile.PARENT
        and semantic_recipe(c['recipe'])==semantic_recipe(spec['recipe'])
        and c['cursor']==32*c['global_step'],'SAME_GENERATION_DATA_SCHEDULE_AND_CURSOR')
    for name,pin in c['files'].items():
        member=path/name
        profile.need(member.resolve().is_relative_to(path)and digest(member)==pin,'CHECKPOINT_MEMBER:'+name)
    saved_contract=json.loads((path/'PARAMETER_CONTRACT.json').read_bytes())
    profile.need(saved_contract==contract.describe(),'EXACT_TRAINABLE_ARCHITECTURE')
    frozen=json.loads((path/'FROZEN_PARENT_TENSORS.json').read_bytes())
    profile.need(frozen_fingerprint(model)==frozen,'UNCHANGED_FROZEN_PARENT_BEFORE_RESTORE')
    values=torch.load(path/'target_parameters.pt',map_location='cpu',weights_only=True)
    profile.need(set(values)==set(contract.trainable_names),'ALL_INTERNAL_ACTION_PARAMETERS')
    with torch.no_grad():
        for n,v in values.items():
            target=model.get_parameter(n)
            profile.need(v.shape==target.shape and v.dtype==target.dtype and torch.isfinite(v).all(),
                         'FINITE_EXACT_PARAMETER:'+n)
            target.copy_(v)
    if trainer is not None:
        trainer.optimizer.state.clear()
        trainer.optimizer.load_state_dict(torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False))
        trainer.lr_scheduler.load_state_dict(torch.load(path/'scheduler.pt',map_location='cpu',weights_only=False))
        trainer._load_rng_state(str(path))
        sampler=json.loads((path/'SAMPLER_STATE.json').read_bytes())
        profile.need(sampler['cursor']==c['cursor']and sampler['dataset_sha256']==spec['recipe']['dataset_sha256'],
                     'REAL_SAMPLER_AND_DATA_RESTORE')
    profile.need(frozen_fingerprint(model)==frozen,'FROZEN_PARENT_AFTER_RESTORE')
    return c


def run(spec):
    p=site(spec)
    Dataset=dataset_type(p)
    from scripts.pi0294.native_dataset import official_import,ProcessedCollator
    from scripts.pi0294.numerics import prepare_environment,configure_torch,MODE
    from scripts.pi0292.durable import write
    root=Path(spec['registered_root']);bindings=spec['bindings'];parent=Path(spec['parent']['path'])
    start=time.monotonic();cpu=time.process_time();stop_requested=[]
    import signal
    def stop(sig,frame):stop_requested.append(sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    def guard():
        profile.need(time.time()<spec['deadline_epoch']and not(root/'STOP.json').exists(),'FINITE_DEADLINE_OR_STOP')
    guard()
    for name,pin in spec['parent']['files'].items():
        path=parent/name
        profile.need(path.resolve().is_relative_to(parent)and digest(path)==pin,'ACTUAL_ORIGINAL_CORR_PARENT:'+name)
    data=Dataset(bindings['dataset'],bindings['source'],bindings['cosmos'],
        for_optimization=True,trace_root=root/'sample-trace',optimization_site=spec['execution_site'])
    profile.need(data.manifest_sha==p['dataset_sha256']and data.statistics_sha==p['statistics_sha256'],
                 'SEALED_REAL_TARGET_TRAIN_AND_PROCESSOR')
    bundle=('FORMAL_TASK_LEARNING_V1'if p['profile']=='TASK_LEARNING_GR00T_V1'else
        'FORMAL'if p['mode']=='FORMAL'else'FIT4_DIAGNOSTIC')
    profile.need(data.manifest.get('bundle_kind')==bundle,
                 'FIT4_QUALIFICATION_NOT_FORMAL_MODEL')
    prepare_environment(MODE);official_import(bindings['source'])
    import torch
    import numpy as np
    from transformers import TrainingArguments,TrainerCallback,set_seed
    from gr00t.experiment.trainer import Gr00tTrainer
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from scripts.pi0294.native_train import make_trainer_class,MeasuredLoader,tensor_identity,rng_identity,tree_identity
    from scripts.pi0292.train_adapter import CyclicSampler
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    from scripts.pi0294.e1.trainer import learning_rate_multiplier
    configure_torch(torch,MODE);torch.set_num_threads(4);set_seed(p['seed'])
    actual_uuid=gpu_identity(torch,spec)
    torch.cuda.reset_peak_memory_stats()
    model=_load_original_parent(spec,root,'MODEL_LOAD')
    if p.get('memory_policy')in('CORE_UNUSED_LOGITS1_CPU_ACTIVATIONS_2G_V2','CORE_FROZEN_BACKBONE_OPTIMIZER_SWAP_V3'):
        from .target_gr00t_core_v1 import trim_unused_logits
        trim_unused_logits(model)
    contract=action_contract(model,p['variant'],profile_name=p['profile']);model.to('cuda').eval();data.initialize()
    activation_storage=None
    if p.get('memory_policy')in('CORE_LOSSLESS_CPU_ACTIVATIONS_2G_V1','CORE_UNUSED_LOGITS1_CPU_ACTIVATIONS_2G_V2','CORE_FROZEN_BACKBONE_OPTIMIZER_SWAP_V3'):
        from .target_gr00t_core_v1 import activation_policy
        from scripts.pi0294.multitask.activation_offload import SavedActivationOffload
        activation_storage=SavedActivationOffload(model,policy=activation_policy())
    backbone_swap=None
    if p.get('memory_policy')=='CORE_FROZEN_BACKBONE_OPTIMIZER_SWAP_V3':
        from .target_gr00t_backbone_swap_v1 import BackboneOptimizerSwap
        backbone_swap=BackboneOptimizerSwap(model,activation_storage)
    write(root/'PARAMETER_WHITELIST.json',contract.describe())
    frozen=frozen_fingerprint(model);names=contract.trainable_names
    before={n:tensor_identity(model.get_parameter(n))for n in names}
    memory=[]
    def measured(label):
        torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
        memory.append(dict(label=label,free_bytes=free,total_bytes=total,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        write(root/'GPU_MEMORY.json',dict(GPU_uuid=actual_uuid,measurements=memory))
        if activation_storage is not None:write(root/'ACTIVATION_STORAGE_ACTUAL.json',activation_storage.audit())
        if backbone_swap is not None:write(root/'BACKBONE_OPTIMIZER_SWAP_ACTUAL.json',backbone_swap.audit())
        profile.need(free>=2*1024**3,'ORIGINAL_TWO_GIB_MEMORY_MARGIN')
    if spec.get('resume'):
        restored=load_checkpoint(spec['resume']['path'],spec['resume']['sha256'],model=model,
            contract=contract,trainer=None,spec=spec,guard=guard)
        profile.need(restored['global_step']==spec['start_step'],'EXACT_RESUME_START_BEFORE_PROBES')
    measured('AFTER_MODEL_LOAD')
    from scripts.pi0294.e1.model import E1ResidentModel
    from .target_gr00t_probes_v1 import measure as probe
    instance=E1ResidentModel.__new__(E1ResidentModel)
    instance.root,instance.bindings,instance.model,instance.processor=root,bindings,model,data.processor
    instance.identity=dict(kind=p['kind'],parent_manifest_sha256=profile.PARENT,
        statistics_sha256=data.statistics_sha,dataset_sha256=data.manifest_sha,
        source_sha=spec['source_sha'],generation_id=spec['generation_id'],mode=p['mode'],variant=p['variant'],
        source_checkpoint_sha256=spec['resume']['sha256'] if spec.get('resume') else profile.PARENT,
        global_step=spec['start_step'])
    before_probes=probe(root,'before',instance,data)
    if p.get('memory_policy')in('CORE_UNUSED_LOGITS1_CPU_ACTIVATIONS_2G_V2','CORE_FROZEN_BACKBONE_OPTIMIZER_SWAP_V3')and spec['start_step']==0:
        from .target_gr00t_core_v1 import verify_memory_predictions
        write(root/'MEMORY_EXECUTION_QUALIFICATION_ACTUAL.json',
            verify_memory_predictions(spec['memory_reference'],root/'before'))
    dev=None
    if p['mode']=='FORMAL':
        dev=Dataset(bindings['dataset'],bindings['source'],bindings['cosmos'],'DEV')
        dev.processor=data.processor
        probe(root,'before-DEV',instance,dev)
    set_seed(p['seed'])
    checkpoints=[];holder={}
    class Sampler(CyclicSampler):
        def index_at(self,cursor):
            import numpy as np
            cycle,offset=divmod(cursor,self.size)
            return int(np.random.Generator(np.random.PCG64(self.seed+cycle)).permutation(self.size)[offset])
    class TargetTrainer(make_trainer_class()):
        def training_step(self,model,inputs,num_items_in_batch=None):
            from scripts.pi0294.multitask.activation_offload import context
            with context(activation_storage):
                return super().training_step(model,inputs,num_items_in_batch)
        def train(self,resume_from_checkpoint=None,**kwargs):
            return Gr00tTrainer.train(self,resume_from_checkpoint=resume_from_checkpoint,**kwargs)
        def get_train_dataloader(self):
            # Trainer can construct the loader before loading trainer_state.
            # The independently checked checkpoint step is authoritative here.
            self.start_cursor=self.consumed=spec['start_step']*32
            self.sampler=Sampler(len(data),seed=p['seed'],start=self.consumed,stop=p['endpoint']*32,positioned=True)
            return MeasuredLoader(torch.utils.data.DataLoader(data,batch_size=p['microbatch'],sampler=self.sampler,
                collate_fn=self.data_collator,num_workers=0,pin_memory=True,drop_last=True,
                generator=torch.Generator().manual_seed(p['seed'])))
        def create_optimizer(self):
            result=super().create_optimizer();contract.validate()
            profile.need({id(t)for g in self.optimizer.param_groups for t in g['params']}==
                {id(t)for t in contract.parameters()},'REAL_OPTIMIZER_WHITELIST')
            return result
        def create_scheduler(self,num_training_steps,optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer,
                    lambda step:learning_rate_multiplier(step,p['scheduler_horizon'],p['warmup_updates']))
                self._created_lr_scheduler=True
            return self.lr_scheduler
        def _load_from_checkpoint(self,checkpoint,model=None):
            load_checkpoint(checkpoint,spec['resume']['sha256'],model=self.model,contract=contract,
                            trainer=None,spec=spec,guard=guard)
        def _save_checkpoint(self,model,trial):
            step=self.state.global_step;profile.need(self.consumed==32*step,'OPTIMIZER_BOUNDARY_CURSOR')
            cp=Path(self.args.output_dir)/f'checkpoint-{step}';cp.mkdir(parents=True,exist_ok=False)
            profile.need(frozen_fingerprint(model)==frozen,'BACKBONE_AND_FROZEN_TENSORS_UNCHANGED')
            torch.save({n:model.get_parameter(n).detach().cpu()for n in names},cp/'target_parameters.pt')
            self._save_optimizer_and_scheduler(str(cp));self._save_rng_state(str(cp))
            self.state.save_to_json(str(cp/'trainer_state.json'))
            data.processor.save_pretrained(cp/'processor')
            import shutil
            for file in ('statistics.json','DATASET_MANIFEST.json','PROCESSOR_TRAINING_ADMISSION.json','PROCESSOR_VANDA_RECONSTRUCTION.json','PROCESSOR_HOPPER_RECONSTRUCTION.json'):
                if 'RECONSTRUCTION'not in file or(data.root/file).exists():
                    shutil.copyfile(data.root/file,cp/file)
            write(cp/'PARAMETER_CONTRACT.json',contract.describe());write(cp/'FROZEN_PARENT_TENSORS.json',frozen)
            write(cp/'SAMPLER_STATE.json',dict(cursor=self.consumed,sampler=self.sampler.state_dict(self.consumed),
                dataset_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha,recipe=semantic_recipe(p)))
            members={str(f.relative_to(cp)):digest(f)for f in cp.rglob('*')if f.is_file()}
            write(cp/'CHECKPOINT_COMPLETE.json',dict(schema='WM02_TARGET_GR00T_FULL_STATE_DELTA_V1',
                generation_id=spec['generation_id'],global_step=step,cursor=self.consumed,
                parent_manifest_sha256=profile.PARENT,recipe=p,files=members,
                standalone_model=False,full_resume_with_required_parent=True))
            checkpoints.append(str(cp));write(root/'CHECKPOINTS.json',dict(paths=checkpoints))
            if p['mode']=='FORMAL'and step%500==0:
                instance.identity['global_step']=step
                instance.identity['target_checkpoint_sha256']=digest(cp/'CHECKPOINT_COMPLETE.json')
                probe(root,f'at-{step}-TRAIN',instance,data)
                probe(root,f'at-{step}-DEV',instance,dev)
    class Audit(TrainerCallback):
        def on_step_begin(self,args,state,control,**kwargs):
            torch.cuda.synchronize();holder['update_started']=time.monotonic()
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            contract.validate();has_gradient=False;nonzero_gradient=False
            for name,t in model.named_parameters():
                if t.grad is not None:
                    has_gradient=True
                    profile.need(name in names and torch.isfinite(t.grad).all(),'FINITE_DECLARED_GRADIENT:'+name)
                    if not nonzero_gradient:nonzero_gradient=bool(torch.count_nonzero(t.grad))
            profile.need(has_gradient and nonzero_gradient,'ACTUAL_NONZERO_INTERNAL_ACTION_GRADIENT')
        def on_step_end(self,args,state,control,**kwargs):
            trainer=holder['trainer'];step=state.global_step
            torch.cuda.synchronize()
            timing=dict(step=step,epoch=time.time(),optimizer_update_wall_s=time.monotonic()-holder['update_started'],
                consumed_cursor=trainer.consumed,native_loss=trainer.last_native_loss,
                learning_rate=trainer.optimizer.param_groups[0]['lr'])
            # Commit the actual update count before post-update capacity checks.
            # A subsequent rejection must not erase the completed optimizer call.
            write(root/'PROGRESS.json',dict(global_step=step,cursor=trainer.consumed,
                GR00T_updates=step-spec['start_step'],WM_updates=0,Actor_updates=0,
                epoch=time.time(),wall_s=time.monotonic()-start,stop_requested=stop_requested))
            with (root/'OPTIMIZER_STEPS.jsonl').open('a')as handle:
                handle.write(json.dumps(timing,sort_keys=True)+'\n');handle.flush();os.fsync(handle.fileno())
            if step in (1,2,10,p['endpoint']) or step%250==0:measured('UPDATE_'+str(step))
            stopping=bool(stop_requested)or time.time()+spec['closeout_reserve_s']>=spec['deadline_epoch']
            control.should_save=step%250==0 or step==p['endpoint'] or p['mode']=='CAPACITY' or (p['mode']=='RESTORE4'and step==2) or stopping
            control.should_training_stop=stopping
    args=TrainingArguments(output_dir=str(root/'training'),max_steps=p['endpoint'],
        per_device_train_batch_size=p['microbatch'],gradient_accumulation_steps=p['accumulation'],
        learning_rate=1e-4,weight_decay=1e-5,optim='adamw_torch',bf16=True,tf32=True,max_grad_norm=1.,
        dataloader_num_workers=0,remove_unused_columns=False,seed=p['seed'],data_seed=p['seed'],
        eval_strategy='no',save_strategy='no',save_only_model=False,save_total_limit=None,
        gradient_checkpointing=False,report_to=[],disable_tqdm=True,ignore_data_skip=True,
        logging_strategy='steps',logging_steps=1)
    trainer=TargetTrainer(model=model,args=args,train_dataset=data,data_collator=ProcessedCollator(data.processor),
        callbacks=[Audit()],evidence_dir=root/'evidence',recipe=p,smoke=p['mode']in('CAPACITY','RESTORE4'))
    holder['trainer']=trainer
    if backbone_swap is not None:trainer.add_callback(backbone_swap.callback())
    trainer.train(resume_from_checkpoint=spec['resume']['path']if spec.get('resume')else None)
    step=trainer.state.global_step
    instance.identity['global_step']=step
    profile.need(step>spec['start_step']and checkpoints,'NONZERO_ACTUAL_INTERNAL_GR00T_OPTIMIZER_UPDATES')
    after={n:tensor_identity(model.get_parameter(n))for n in names}
    profile.need(after!=before and frozen_fingerprint(model)==frozen,'REAL_ACTION_WEIGHT_CHANGE_AND_FREEZE')
    cp=Path(checkpoints[-1]);saved_opt=tree_identity(trainer.optimizer.state_dict())
    instance.identity['target_checkpoint_sha256']=digest(cp/'CHECKPOINT_COMPLETE.json')
    saved_sched=trainer.lr_scheduler.state_dict();saved_rng=rng_identity()
    after_folder=f'at-{step}-TRAIN'if p['mode']=='FORMAL'and step%500==0 else'after'
    after_probes=(json.loads((root/after_folder/'RESULT.json').read_bytes())['records']
                  if after_folder!='after'else probe(root,'after',instance,data))
    # Reconstruct every learned tensor and complete optimizer/scheduler/RNG/cursor.
    load_checkpoint(cp,digest(cp/'CHECKPOINT_COMPLETE.json'),model=model,contract=contract,
                    trainer=trainer,spec=spec,guard=guard)
    profile.need({n:tensor_identity(model.get_parameter(n))for n in names}==after
        and tree_identity(trainer.optimizer.state_dict())==saved_opt
        and trainer.lr_scheduler.state_dict()==saved_sched and rng_identity()==saved_rng,
        'ACTUAL_COMPLETE_STATE_RESTORE')
    restored_probes=probe(root,'restored',instance,data)
    from .task_learning_probes_v1 import artifact,matched_count
    matched_count(json.loads((root/after_folder/'RESULT.json').read_bytes()),
        json.loads((root/'restored/RESULT.json').read_bytes()))
    for a,b in zip(after_probes,restored_probes):
        with np.load(artifact(root/after_folder,a),allow_pickle=False)as first, np.load(artifact(root/'restored',b),allow_pickle=False)as second:
            profile.need(a['episode']==b['episode']and a['anchor']==b['anchor']and a['noise']==b['noise']
                and np.array_equal(first['physical_H40'],second['physical_H40']),
                'FIXED_INPUT_NOISE_ACTUAL_RELOAD_GENERATION')
    if dev is not None:probe(root,'restored-DEV',instance,dev)
    proof=dict(model_parameters_restored=True,optimizer_restored=True,scheduler_restored=True,
        RNG_restored=True,sampler_cursor=step*32,checkpoint_sha256=digest(cp/'CHECKPOINT_COMPLETE.json'),
        fixed_batch_generation_restored=True,
        next_update_comparison='REQUIRES_INDEPENDENT_CAPACITY_RESUME_SLICE',baseline_changed=False)
    write(root/'RESTORE_ACTUAL.json',proof)
    package=None
    if p['mode']=='FORMAL'and step==p['endpoint']:
        package=_package_and_reconstruct(root,cp,spec,proof,model,trainer,instance,data,guard)
        measured('AFTER_PACKAGE_INDEPENDENT_RECONSTRUCTION')
    return dict(status='CHECKPOINT_SEALED',profile=p['profile'],source_sha=spec['source_sha'],
        generation_id=spec['generation_id'],global_step=step,GR00T_updates=step-spec['start_step'],
        sample_visits=(step-spec['start_step'])*32,WM_updates=0,Actor_updates=0,external_adapter_updates=0,
        actual_VLA_queries=len(list(root.glob('*/journal/Q*.json'))),
        actual_model_loads=len(list(root.glob('*_CONSTRUCTED_ACTUAL.json'))),inference_package=package,
        backbone_frozen=True,real_action_parameter_changed=True,checkpoint=str(cp),restore=proof,
        native_physics=0,Core_DEV='NOT_RUN',CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
