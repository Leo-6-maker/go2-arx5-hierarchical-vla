"""Inspect complete native checkpoint restoration without optimizer updates.

Uses the original model, native Trainer/AdamW, scheduler function and checkpoint
loader. This diagnostic cannot grant full4-vs2+2 or formal SFT admission.
"""
import gc,json,time,weakref,hashlib,pickle,re
from pathlib import Path
from contextlib import closing
from . import target_gr00t_profile_v1 as profile
from .storage import digest

OPERATION='FULL_STATE_RESTORE_MEMORY_ONLY'


def configuration(c):
    need=profile.need
    need(c.get('operation')==OPERATION and c.get('formal_training')is False
        and c.get('original_Trainer_execution')is True and c.get('GR00T_updates')==0
        and c.get('memory_policy')=='CORE_FROZEN_BACKBONE_SWAP_NO_WEIGHT_CACHE_V5'
        and c.get('expected_global_step')==2 and c.get('maximum_model_loads')==1,
        'MEMORY_DIAGNOSTIC_ZERO_UPDATES_ONE_MODEL_COMPLETE_STEP2_ONLY')
    for name in('checkpoint_reference','checkpoint_source_spec','trained_probe_reference'):
        need(isinstance(c.get(name),dict)and set(c[name])=={'path','sha256'}
            and isinstance(c[name]['path'],str)and Path(c[name]['path']).is_absolute()
            and isinstance(c[name]['sha256'],str)and bool(re.fullmatch('[0-9a-f]{64}',c[name]['sha256'])),
            'PINNED_REAL_RESTORE_MEMORY_INPUT_'+name)
    return c


def run(c):
    from .target_gr00t_core_v1 import scope,allocator,trim_unused_logits,verify_memory_predictions
    from .target_gr00t_training_v1 import _load_original_parent,action_contract,load_checkpoint
    from .task_learning_training_admission_v1 import unchanged_training_math
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.numerics import prepare_environment,configure_torch,MODE
    from scripts.pi0294.native_dataset import official_import,ProcessedCollator
    from .target_gr00t_dataset_v1 import TargetNativeDataset
    from .target_gr00t_backbone_swap_v1 import BackboneOptimizerSwap
    from scripts.pi0294.e1.trainer import learning_rate_multiplier
    root,_=scope(configuration(c));stop=[]
    import signal
    signal.signal(signal.SIGTERM,lambda sig,frame:stop.append(sig))
    signal.signal(signal.SIGUSR1,lambda sig,frame:stop.append(sig))
    def guard():
        profile.need(not stop and time.time()+120<c['deadline_epoch']
            and not(root/'STOP.json').exists(),'FINITE_RESTORE_MEMORY_PROFILE_OR_STOP')
    spec_path=Path(c['checkpoint_source_spec']['path'])
    profile.need(digest(spec_path)==c['checkpoint_source_spec']['sha256'],'REAL_COMPLETE_STEP2_SPEC')
    old=json.loads(spec_path.read_bytes());p=profile.validate(old)
    cp=Path(c['checkpoint_reference']['path']);header=json.loads((cp/'CHECKPOINT_COMPLETE.json').read_bytes())
    previous=cp.parent.parent
    previous_lease=json.loads((previous/'control/LEASE.json').read_bytes())
    previous_worker=json.loads((previous/'WORKER_EXIT.private.json').read_bytes())
    previous_after=Path(c['trained_probe_reference']['path'])
    profile.need(old['start_step']==0 and p['mode']=='RESTORE4'and p['variant']=='FULL_ACTION_SIDE'
        and p['endpoint']==header['global_step']==2 and header['cursor']==64
        and p['memory_policy']==c['memory_policy']
        and header['generation_id']==old['generation_id']
        and digest(cp/'CHECKPOINT_COMPLETE.json')==c['checkpoint_reference']['sha256']
        and all(old[k]==c[k]for k in('parent','bindings','GPU_uuid'))
        and p['dataset_sha256']==c['dataset_sha256']and p['statistics_sha256']==c['statistics_sha256'],
        'ACTUAL_NATIVE_TWO_STEP_DIAGNOSTIC_NOT_FORMAL_OR_PARTIAL')
    profile.need(spec_path==previous/'EXECUTION_SPEC.json'
        and previous_lease.get('status')=='CLOSED'
        and previous_lease.get('worker_exit_code')==previous_worker.get('exit_code')==1
        and previous_lease.get('spec_sha256')==c['checkpoint_source_spec']['sha256']
        and previous_lease.get('result_sha256')==digest(previous/'RESULT.json')
        and previous_after==previous/'after/RESULT.json'
        and digest(previous_after)==c['trained_probe_reference']['sha256']
        and all(r['model_global_step']==2 for r in json.loads(previous_after.read_bytes())['records']),
        'PRESERVED_CLOSED_FAILURE_COMPLETE_CHECKPOINT_AND_REAL_TRAINED_PROBES_ONLY')
    unchanged_training_math(old,c)
    with closing(OwnerLock(Path(c['owner_root'])/'NIGHT1_NEURAL_OWNER.lock')):
        guard();profile.need(not(root/'ALLOCATION_STARTED.json').exists(),'RESTORE_MEMORY_NEVER_REPLAY')
        import os
        write(root/'ALLOCATION_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_sha=c['source_sha']))
        for name,pin in c['source_files'].items():profile.need(digest(Path(c['source_root'])/name)==pin,'SEALED_PROFILE_SOURCE_'+name)
        for name,pin in c['parent']['files'].items():profile.need(digest(Path(c['parent']['path'])/name)==pin,'ORIGINAL_PARENT_'+name)
        prepare_environment(MODE);official_import(c['bindings']['source'])
        import torch
        from transformers import TrainingArguments,set_seed,TrainerState
        from scripts.pi0294.native_train import make_trainer_class,tree_identity,tensor_identity,rng_identity
        configure_torch(torch,MODE);torch.set_num_threads(4);set_seed(p['seed'])
        allocator(c,full=True,root=root,name='BEFORE_MODEL',stop=stop)
        from .target_gr00t_training_v1 import gpu_identity
        gpu_identity(torch,dict(c,execution_site='CORELAB_4080S'))
        torch.set_autocast_cache_enabled(False)
        # A reconstruction input is not a spec authorizing training steps.
        s=dict(old,registered_root=str(root),source_root=c['source_root'],source_sha=c['source_sha'],
            source_files=c['source_files'],GR00T_updates=0,kind='COMPLETE_STATE_MEMORY_RECONSTRUCTION_ONLY')
        write(root/'RECONSTRUCTION_INPUT_SPEC.json',s)
        model=_load_original_parent(s,root,'MODEL_LOAD');trim_unused_logits(model)
        contract=action_contract(model,'FULL_ACTION_SIDE');model.to('cuda').eval()
        data=TargetNativeDataset(c['bindings']['dataset'],c['bindings']['source'],c['bindings']['cosmos'],
            for_optimization=True,trace_root=root/'UNUSED_TRAINING_TRACE',optimization_site='CORELAB_4080S')
        data.initialize();write(root/'PARAMETER_WHITELIST.json',contract.describe())
        args=TrainingArguments(output_dir=str(root/'UNUSED_TRAINING_OUTPUT'),max_steps=2,
            per_device_train_batch_size=1,gradient_accumulation_steps=32,learning_rate=1e-4,
            weight_decay=1e-5,optim='adamw_torch',bf16=True,tf32=True,max_grad_norm=1.,
            dataloader_num_workers=0,remove_unused_columns=False,seed=p['seed'],data_seed=p['seed'],
            eval_strategy='no',save_strategy='no',save_only_model=False,save_total_limit=None,
            gradient_checkpointing=False,report_to=[],disable_tqdm=True,ignore_data_skip=True)
        trainer=make_trainer_class()(model=model,args=args,train_dataset=data,
            data_collator=ProcessedCollator(data.processor),evidence_dir=root/'evidence',recipe=p,smoke=True)
        trainer.create_optimizer();contract.validate()
        profile.need({id(t)for g in trainer.optimizer.param_groups for t in g['params']}==
            {id(t)for t in contract.parameters()},'ORIGINAL_NATIVE_ADAMW_EXACT_ACTION_PARAMETERS')
        trainer.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,
            lambda step:learning_rate_multiplier(step,p['scheduler_horizon'],p['warmup_updates']))
        trainer.optimizer=trainer.accelerator.prepare_optimizer(trainer.optimizer)
        swap=BackboneOptimizerSwap(model,None);records=[];old_refs=[]
        def memory(label):
            torch.cuda.synchronize()
            states=[t for state in trainer.optimizer.state.values()for t in state.values()if torch.is_tensor(t)]
            records.append(dict(label=label,epoch=time.time(),allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0],
                optimizer_state_CUDA_bytes=sum(t.numel()*t.element_size()for t in states if t.is_cuda),
                parameter_CUDA_bytes=sum(t.numel()*t.element_size()for t in model.parameters()if t.is_cuda),
                gradient_CUDA_bytes=sum(t.grad.numel()*t.grad.element_size()for t in model.parameters()if t.grad is not None and t.grad.is_cuda),
                old_optimizer_tensors_alive=sum(ref()is not None for ref in old_refs),
                autocast_weight_cache_enabled=torch.is_autocast_cache_enabled()))
            write(root/'RESTORE_MEMORY_ACTUAL.json',dict(records=records,optimizer_calls=0))
        def before_load(optimizer,state):
            memory('AFTER_OLD_STATE_CLEAR_BEFORE_LOAD_CAST')
            gc.collect();torch.cuda.empty_cache();memory('AFTER_GC_BEFORE_LOAD_CAST')
        raw=trainer.optimizer.optimizer
        handle=raw.register_load_state_dict_pre_hook(before_load)
        try:
            memory('BEFORE_FIRST_FRESH_STATE_RESTORE')
            restored=load_checkpoint(cp,c['checkpoint_reference']['sha256'],model=model,
                contract=contract,trainer=trainer,spec=s,guard=guard)
            trainer.state=TrainerState.load_from_json(str(cp/'trainer_state.json'))
            memory('AFTER_FIRST_FRESH_STATE_RESTORE')
            expected=tree_identity(torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False))
            profile.need(tree_identity(trainer.optimizer.state_dict())==expected,'EXACT_ARCHIVED_OPTIMIZER_STATE')
            expected=tree_identity(torch.load(cp/'target_parameters.pt',map_location='cpu',weights_only=True))
            profile.need({n:tensor_identity(model.get_parameter(n))for n in contract.trainable_names}==expected,
                'EXACT_ARCHIVED_MODEL_PARAMETERS')
            expected=torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)
            profile.need(trainer.lr_scheduler.state_dict()==expected and trainer.state.global_step==2,
                'EXACT_ARCHIVED_SCHEDULER_AND_TRAINER_STEP')
            archived=torch.load(cp/'rng_state.pth',map_location='cpu',weights_only=False)
            cuda=archived['cuda']if isinstance(archived['cuda'],list)else[archived['cuda']]
            expected=dict(python=hashlib.sha256(pickle.dumps(archived['python'])).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(archived['numpy'])).hexdigest(),
                torch_cpu=tensor_identity(archived['cpu']),torch_cuda=[tensor_identity(t)for t in cuda])
            profile.need(rng_identity()==expected,'EXACT_ARCHIVED_ALL_RNG_STREAMS')
            write(root/'FIRST_FRESH_FULL_STATE_RESTORE_ACTUAL.json',dict(model_parameters_restored=True,
                optimizer_restored=True,scheduler_restored=True,RNG_restored=True,
                trainer_global_step=2,checkpoint_cursor_verified=64,runtime_sampler_stepped=False,
                optimizer_calls=0,full4_vs_fresh2_plus2_qualification=False))
            old_refs=[weakref.ref(t)for state in trainer.optimizer.state.values()for t in state.values()if torch.is_tensor(t)]
            swap.before_optimizer();memory('FROZEN_BACKBONE_OFFLOADED_BEFORE_SECOND_RESTORE')
            load_checkpoint(cp,c['checkpoint_reference']['sha256'],model=model,
                contract=contract,trainer=trainer,spec=s,guard=guard)
            memory('AFTER_SECOND_RESTORE_BEFORE_BACKBONE_RETURN')
            gc.collect();torch.cuda.empty_cache();memory('AFTER_GC_BEFORE_BACKBONE_RETURN')
            swap.restore();memory('AFTER_SECOND_RESTORE_AND_BACKBONE_RETURN')
            from scripts.pi0294.e1.model import E1ResidentModel
            from .target_gr00t_probes_v1 import measure
            instance=E1ResidentModel.__new__(E1ResidentModel)
            instance.root,instance.bindings,instance.model,instance.processor=root,c['bindings'],model,data.processor
            instance.identity=dict(kind=p['kind'],parent_manifest_sha256=profile.PARENT,
                statistics_sha256=data.statistics_sha,dataset_sha256=data.manifest_sha,
                source_sha=c['source_sha'],generation_id=old['generation_id'],mode=p['mode'],variant=p['variant'],
                source_checkpoint_sha256=c['checkpoint_reference']['sha256'],global_step=restored['global_step'])
            measure(root,'restored',instance,data)
            proof=verify_memory_predictions(c['trained_probe_reference'],root/'restored')
            write(root/'RESTORED_TRAINED_H40_ACTUAL.json',proof)
            return dict(status='RESTORE_MEMORY_PROFILE_AND_TRAINED_H40_COMPLETE_NOT_FULL8_QUALIFICATION',
                optimizer_calls=0,GR00T_updates=0,original_Trainer_constructed=True,train_called=False,
                restored_step=restored['global_step'],verified_checkpoint_cursor=restored['cursor'],
                runtime_sampler_stepped=False,formal_training=False,physics_integrations=0,proof=proof)
        finally:
            handle.remove();memory('FINAL_OBSERVATION_EVEN_ON_FAILURE')
