"""Finite book fitting/learning using the unchanged native loss and Trainer.

The small checkpoint is a parent-dependent training delta, never a standalone
model. Frozen tensors belong to the exact retained full book128 parent, or, only for
the explicit shared run kind, to the original seven-file raw GR00T parent.
"""
import hashlib
from functools import partial
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


def full_plan():
    p=plan();p['diagnostic_windows']=p.pop('windows')
    p.update(schema='BOOK128_BOOK5_ALL3005_TARGET1024_V1',max_steps=1024,TRAIN_visits=32768,
        optimization_population='ALL_3005_WINDOWS_FROM_FIVE_ADMITTED_TRAIN_EPISODES',TRAIN_episodes=5,TRAIN_windows=3005,
        warmup_updates=32,scheduler_horizon_steps=1024,seed=2026103103,
        optimizer_initialization='FRESH_ADAMW_FROM_BOOK128_NOT_TRAIN16_OVERFIT_DELTA',
        purpose='Train on the existing complete five-demo population after the fixed16 fit passed generation gates but failed real DEV. No DEV/TEST optimization, new data, shared-action update or task-success claim.')
    return p


def recovery_plan():
    p=full_plan()
    p.update(schema='BOOK5_EXPERT_RECOVERY3574_TARGET512_V1',
        dataset_manifest_sha256='d3f4999809a9f57288fe5ba62116b70cc8660cff2fb70821d583b81834039d31',
        statistics_sha256='73cc6e2f903f8e56e2de6384bb16d9f4eb9ff98a3a28acd74f1dd0d1b5b80763',
        initial_delta_checkpoint_manifest_sha256='b8327ab9501eac641eae8947a04040fe84844a01e9e5f59cea1c2ffd17ad11f8',
        parent_restore_dataset_manifest_sha256=full_plan()['dataset_manifest_sha256'],
        parent_restore_statistics_sha256=full_plan()['statistics_sha256'],
        max_steps=512,TRAIN_visits=16384,TRAIN_episodes=6,TRAIN_windows=3574,
        pure_full_teacher_episodes=5,mixed_recovery_episodes=1,student_prefix_expert_labels=0,
        optimization_population='ORIGINAL_3005_PLUS_569_EXECUTED_EXPERT_RECOVERY_SUFFIX_WINDOWS',
        diagnostic_windows=[dict(id='BOOK_SOURCE_SEED000',anchor=a,split='TRAIN')for a in(0,48,80,160,240,320,400,600)]
            +[dict(id='BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',anchor=a,split='TRAIN')for a in(0,8,16,32,48,80,160,568)],
        warmup_updates=16,scheduler_horizon_steps=512,seed=2026103104,
        optimizer_initialization='FRESH_ADAMW_FROM_QUALIFIED_BOOK5_TARGET1024',
        statistics_version_changed=True,statistics_refit=False,
        purpose='Exploratory recovery-data successor with separately qualified new TRAIN statistics; not an isolated causal recovery comparison, formal E2 or task-success claim.')
    return p


def validate_manifest(m,p):
    from .shared_training import KIND as SHARED_KIND,validate_manifest as validate_shared
    if m.get('kind')==SHARED_KIND:
        validate_shared(m,p);return
    from .book_m32_learning import is_plan,KIND
    m32=is_plan(p)
    if (not m32 and p not in(plan(),full_plan(),recovery_plan()))or m['checkpoint_manifest_sha256']!=p['checkpoint_manifest_sha256']:
        raise PermissionError('EXACT_FROZEN_BOOK_TRAIN16_FIT_PLAN')
    for key,value in dict(optimizer_updates=p['max_steps'],physics_integrations=0,generation_queries=p['generation_queries'],model_loads=1).items():
        if m['limits'].get(key)!=value:raise PermissionError('BOOK_TRAIN16_FIT_FINITE_'+key)
    expected=KIND if m32 else 'V7_BOOK_TRAIN16_FIT_DIAGNOSTIC'if p==plan()else('V7_BOOK_RECOVERY_FINITE_TARGET_LEARNING'if p==recovery_plan()else'V7_BOOK5_FINITE_TARGET_LEARNING')
    if m.get('kind',expected)!=expected:raise PermissionError('BOOK_EXACT_SEPARATE_SCIENTIFIC_RUN_KIND')
    if (p==recovery_plan()or m32)and m.get('delta_checkpoint_manifest_sha256')!=p['initial_delta_checkpoint_manifest_sha256']:
        raise PermissionError('RECOVERY_LEARNING_EXACT_QUALIFIED1024_INITIAL_WEIGHTS')


def select_rows(rows,fit_recipe=None):
    lookup={(r['id'],a):(r,a)for r,a in rows}
    selected=[]
    p=plan()if fit_recipe is None else fit_recipe
    for row in p.get('diagnostic_windows',p.get('windows',[])):
        value=lookup[row['id'],row['anchor']]
        if value[0]['split']!='TRAIN':raise PermissionError('TRAIN16_FIT_NO_DEV_OR_TEST')
        selected.append(value)
    return selected


def optimization_rows(rows,p):
    from .shared_training import is_recipe as is_shared
    if is_shared(p):
        from .shared_task_sampling import validate_fixed59
        validate_fixed59(rows);return list(rows)
    if p==plan():return select_rows(rows)
    from .book_bc_dagger import SCHEMA as BC_SCHEMA,is_plan as is_bc_plan
    from .book_phase_boundary import PLAN_SCHEMA as PB_SCHEMA,is_plan as is_pb_plan
    if p.get('schema') in (BC_SCHEMA,PB_SCHEMA):
        valid=is_bc_plan(p)if p['schema']==BC_SCHEMA else is_pb_plan(p)
        if not valid:raise PermissionError('EXACT_FROZEN_BC_DAGGER_POPULATION')
        expected={(r['id'],a)for r in p['TRAIN_entries']for a in range(r['windows'])}
        if (len(rows)!=len(expected)or {(r['id'],a)for r,a in rows}!=expected
                or any(r['split']!='TRAIN'or r.get('task')!='BOOK'for r,a in rows)):
            raise PermissionError('ALL_AND_ONLY_DECLARED_BOOK_TRAIN_EXPERT_WINDOWS')
        return list(rows)
    from .book_m32_learning import is_plan
    if is_plan(p):
        from .book_m32_export import COUNTS
        ids=[*[f'BOOK_SOURCE_SEED{i:03}'for i in range(15)],'BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',
             *[f'BOOK_M32_RECOVERY{i:03}_EXPERT_SUFFIX'for i in range(5)]]
        expected={(name,a)for name,count in zip(ids,COUNTS)for a in range(count)}
        if (len(rows)!=len(expected)or {(r['id'],a)for r,a in rows}!=expected
                or any(r['split']!='TRAIN'for r,a in rows)):
            raise PermissionError('EXACT_ADMITTED_M32_TRAIN_POPULATION')
        return list(rows)
    if p not in(full_plan(),recovery_plan()):raise PermissionError('EXACT_FINITE_BOOK_OPTIMIZATION_POPULATION')
    expected={(f'BOOK_SOURCE_SEED{i:03}',a)for i in range(5)for a in range(601)}
    if p==recovery_plan():expected|={('BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',a)for a in range(569)}
    if(len(rows)!=len(expected)or {(r['id'],a)for r,a in rows}!=expected or any(r['split']!='TRAIN'for r,a in rows)):
        raise PermissionError('ALL_AND_ONLY_FIVE_ADMITTED_COMPLETE_TRAIN_POPULATIONS')
    return list(rows)


class FitSampler(CyclicSampler):
    def __init__(self,dataset,*,start=0,stop=None,fit_recipe=None,phase_index=None):
        p=plan()if fit_recipe is None else fit_recipe
        from .book_m32_learning import is_plan
        if not is_plan(p)and p not in(plan(),full_plan(),recovery_plan()):raise ValueError('FIT_EXACT_FROZEN_RECIPE')
        if len(dataset)!=(16 if p==plan()else p['TRAIN_windows']):raise ValueError('FIT_EXACT_DECLARED_TRAIN_POPULATION')
        stop=p['TRAIN_visits']if stop is None else stop
        if stop>p['TRAIN_visits']:raise ValueError('FIT_NO_SAMPLER_EXPOSURE_EXTENSION')
        super().__init__(len(dataset),seed=p['seed'],start=start,stop=stop,positioned=True)
        self.phase_sequence=None
        from .book_phase_sampling import SCHEMA,sequence
        if p['schema']==SCHEMA:
            if phase_index is None:raise PermissionError('PHASE_SAMPLER_REQUIRES_SEALED_TRAIN_INDEX')
            self.phase_sequence=sequence(getattr(dataset,'rows',dataset),phase_index,p)
            self.phase_index_sha256=p['phase_index_sha256']
        from .book_phase_boundary import PLAN_SCHEMA as PB_SCHEMA,recipe_sequence
        self.pb_variant=None
        if p['schema']==PB_SCHEMA:
            if phase_index is None:raise PermissionError('PHASE_SAMPLER_REQUIRES_SEALED_TRAIN_INDEX')
            self.phase_sequence=recipe_sequence(getattr(dataset,'rows',dataset),phase_index,p)
            self.phase_index_sha256=p['phase_index_sha256'];self.pb_variant=p['pb_variant']

    def __iter__(self):
        if self.phase_sequence is None:
            yield from super().__iter__();return
        for cursor in range(self.start,self.stop):yield cursor,int(self.phase_sequence[cursor])

    def state_dict(self,consumed):
        value=super().state_dict(consumed)
        if self.phase_sequence is not None:
            value.update(algorithm='DETERMINISTIC_EPISODE_BALANCED_PHASE_STREAMS_V1',phase_index_sha256=self.phase_index_sha256)
            if self.pb_variant is not None:
                value.update(algorithm='DETERMINISTIC_EPISODE_BALANCED_PHASE_BOUNDARY_V1',pb_variant=self.pb_variant)
        return value

    def index_at(self,cursor):
        import numpy as np
        if not self.start<=cursor<self.stop:raise IndexError('FIT_FINITE_CURSOR')
        if self.phase_sequence is not None:return int(self.phase_sequence[cursor])
        cycle,offset=divmod(cursor,self.size)
        return int(np.random.Generator(np.random.PCG64(self.seed+cycle)).permutation(self.size)[offset])


def validate_delta(delta,parameters,names,*,expected_count=14):
    if set(delta)!=set(names)or len(names)!=expected_count:raise ValueError('EXACT_DECLARED_TRAINABLE_TENSORS_REQUIRED')
    for name in names:
        a,b=delta[name],parameters[name]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('TARGET_DELTA_SHAPE_DTYPE:'+name)


def measure(root,phase,instance,data,trainer,guard,fit_recipe=None):
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
    p=plan()if fit_recipe is None else fit_recipe
    for index,row in enumerate(p.get('diagnostic_windows',p.get('windows',[]))):
        guard();key=row['id']+'-'+str(row['anchor']).zfill(3)
        state,expert,images,text=data.raw(index)
        namespace=p.get('diagnostic_seed_namespace','BOOK128_TRAIN16')
        seed=int.from_bytes(hashlib.sha256(f"{namespace}/{row['id']}/{row['anchor']}".encode()).digest()[:8],'big')%(2**63)
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
    if 'checkpoint_steps'in p:
        result.update(prefix8_XYZ_mean_m=float(np.mean([r['errors']['prefix8']['XYZ_euclidean_m']['mean']for r in records])),
            prefix8_SO3_mean_rad=float(np.mean([r['errors']['prefix8']['SO3_rad']['mean']for r in records])),
            prefix8_gripper_MAE=float(np.mean([r['errors']['prefix8']['gripper_MAE']for r in records])))
    from .shared_training import is_recipe as is_shared
    if is_shared(p):
        from .shared_task_sampling import TASKS
        result['tasks']={task:dict(TRAIN_windows=4,
            native_loss_mean=float(np.mean([r['native_loss']for r in records if r['window']['task']==task])),
            H40_XYZ_mean_m=float(np.mean([r['errors']['H40']['XYZ_euclidean_m']['mean']for r in records if r['window']['task']==task])),
            prefix8_XYZ_mean_m=float(np.mean([r['errors']['prefix8']['XYZ_euclidean_m']['mean']for r in records if r['window']['task']==task])),
            H40_gripper_MAE=float(np.mean([r['errors']['H40']['gripper_MAE']for r in records if r['window']['task']==task])))for task in TASKS}
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
    from .book_m32_learning import is_plan,promote_contract
    from .shared_training import is_recipe as is_shared,validate_data as validate_shared_data
    shared=is_shared(p);m32=is_plan(p)or shared
    steps=p['max_steps'];visits=p['TRAIN_visits'];sampler_class=partial(FitSampler,fit_recipe=p)
    if shared:
        from .shared_task_sampling import TaskBalancedSampler
        sampler_class=partial(TaskBalancedSampler,seed=p['seed'],stop=visits)
    from .book_phase_sampling import SCHEMA as PHASE_SCHEMA,load_index
    from .book_phase_boundary import PLAN_SCHEMA as PB_SCHEMA,validate_population as validate_pb_population
    if p['schema']in(PHASE_SCHEMA,PB_SCHEMA):
        sampler_class=partial(FitSampler,fit_recipe=p,phase_index=load_index(root,p))
    model=instance.model;parameters=dict(model.named_parameters());names=contract.trainable_names
    from .activation_offload import execution,context as activation_context
    activation_storage=execution(root,read(root/'RUN_MANIFEST.json'),model)
    for name in names:parameters[name].requires_grad_(True)
    contract.validate()
    if m32:
        if shared:
            from .book_m32_learning import LimitedContract
            if len(contract.trainable_names)==14:contract=LimitedContract(contract,2)
            if len(contract.trainable_names)!=42 or contract.describe()['trainable_numel']!=p['trainable_numel']:
                raise PermissionError('SHARED_RAW_EXACT_M32_CONTRACT')
        else:contract=promote_contract(contract,p)
        names=contract.trainable_names
    frozen=frozen_fingerprint(model)
    before_parameters={n:tensor_identity(parameters[n])for n in names}
    if shared:
        zero=read(root/'RAW_ZERO_STEP_PARAMETERS.json')
        if zero.get('operation')!='COMPARE_FRESH_LOADED_MODEL'or zero.get('trainable_tensors')!=before_parameters:
            raise PermissionError('SHARED_RAW_FRESH_ZERO_STEP_PARAMETERS_BEFORE_OPTIMIZATION')
    parent_binding=({'required_raw_parent_files_identity':p['raw_parent_files_identity']}if shared
                    else{'required_parent_checkpoint_manifest_sha256':p['checkpoint_manifest_sha256']})
    def restore_base():
        if shared:
            from .shared_checkpoint import raw_parameter_identity
            manifest=read(root/'RUN_MANIFEST.json')
            raw_parameter_identity(model,contract,manifest['raw_parent'],read(root/'RAW_PARENT_AUDIT.json'),guard=guard,restore=True)
        else:load_wrapped(model,Path(read(root/'RUN_MANIFEST.json')['checkpoint']))
    binding=instance.bindings
    data=NativeDataset(binding['dataset'],binding['source'],binding['cosmos'],for_optimization=True,
        raw_bindings=binding.get('raw_bindings'),trace_root=root/'fit_trace')
    probe=NativeDataset(binding['dataset'],binding['source'],binding['cosmos'],for_optimization=False,
        raw_bindings=binding.get('raw_bindings'))
    for ds in(data,probe):
        if ds.manifest_sha!=p['dataset_manifest_sha256']or ds.statistics_sha!=p['statistics_sha256']:raise PermissionError('FIT_ORIGINAL_ADMITTED_TRAIN_DATA_AND_STATS')
        if shared:validate_shared_data(ds,p)
        elif m32:
            if sha(ds.root/'TRAIN_POPULATION.json')!=p['TRAIN_population_sha256']:
                raise PermissionError('M32_OFFICIALLY_ADMITTED_POPULATION')
            from .book_bc_dagger import SCHEMA as BC_SCHEMA,validate_population
            if p['schema']==BC_SCHEMA:
                validate_population(ds.manifest,read(ds.root/'TRAIN_POPULATION.json'),p)
            elif p['schema']==PB_SCHEMA:
                validate_pb_population(ds.manifest,read(ds.root/'TRAIN_POPULATION.json'),p)
            elif ds.manifest['schema']!='BOOK_M32_TRAIN15_SIX_EXPERT_SUFFIXES_V1':
                raise PermissionError('M32_OFFICIALLY_ADMITTED_POPULATION')
        ds.rows=optimization_rows(ds.rows,p)if ds is data else select_rows(ds.rows,p);ds.initialize()
    from .shared_resume import prepare as prepare_resume
    resume=prepare_resume(root,read(root/'RUN_MANIFEST.json'),p,data)if shared else None
    if instance.identity['statistics_sha256']!=data.statistics_sha:raise PermissionError('FIT_GENERATION_AND_TRAINING_SAME_STATISTICS')
    data.make_sampler=sampler_class;set_seed(p['seed'])
    memory=[]
    def record_memory(phase):
        if not m32:return
        torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
        memory.append(dict(phase=phase,free_bytes=free,total_bytes=total,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            maximum_allocated_bytes=torch.cuda.max_memory_allocated(),maximum_reserved_bytes=torch.cuda.max_memory_reserved()))
        write(root/'ACTUAL_M32_MEMORY.json',dict(measurements=memory,observed_epoch=time.time()))
    if m32:
        torch.cuda.reset_peak_memory_stats();record_memory('BEFORE_ACTUAL_ADAMW_ALLOCATION')

    def qualify_intermediate(trainer,cp,step,cursor):
        """Exercise the same full restore at the committed intermediate boundary."""
        complete=read(cp/'CHECKPOINT_COMPLETE.json')
        for name,digest in complete['files'].items():
            if sha(cp/name)!=digest:raise ValueError('CURVE_SAVED_MEMBER_CHANGED:'+name)
        expected_parameters={n:tensor_identity(parameters[n])for n in names}
        expected_optimizer=tree_identity(trainer.optimizer.state_dict())
        expected_scheduler=trainer.lr_scheduler.state_dict();expected_rng=rng_identity()
        restore_base()
        delta=torch.load(cp/'target_parameters.pt',map_location='cpu',weights_only=True)
        validate_delta(delta,parameters,names,expected_count=len(names))
        with torch.no_grad():
            for name in names:parameters[name].copy_(delta[name])
        if {n:tensor_identity(parameters[n])for n in names}!=expected_parameters or frozen_fingerprint(model)!=frozen:
            raise ValueError('CURVE_ACTUAL_PARENT_PLUS_DELTA_RECONSTRUCTION')
        restored=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
        if len(restored['state'])!=len(names)or any(int(x['step'])!=step for x in restored['state'].values()):
            raise ValueError('CURVE_ACTUAL_ADAMW_STATE_STEP')
        trainer.optimizer.state.clear();trainer.optimizer.load_state_dict(restored)
        trainer.lr_scheduler.load_state_dict(torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False))
        trainer._load_rng_state(str(cp))
        saved=read(cp/'SAMPLER_STATE.json');saved_state=TrainerState.load_from_json(str(cp/'trainer_state.json'))
        if (tree_identity(trainer.optimizer.state_dict())!=expected_optimizer
                or trainer.lr_scheduler.state_dict()!=expected_scheduler or rng_identity()!=expected_rng
                or saved_state.global_step!=step or saved['sampler']!=sampler_class(data,start=cursor).state_dict(cursor)):
            raise ValueError('CURVE_EXACT_FULL_STATE_RESTORE')
        write(root/f'ACTUAL_DELTA_CHECKPOINT_RESTORE_{step}.json',dict(checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),
            **parent_binding,model_reconstruction=True,
            optimizer_state_entries=len(names),optimizer_updates=step,scheduler_restored=True,RNG_restored=True,
            sampler_cursor=cursor,frozen_parent_unchanged=True,standalone_model=False))
        intermediate=measure(root,f'after-{step}',instance,probe,trainer,guard,p)
        for original,current in zip(before['records'],intermediate['records']):
            if any(original[k]!=current[k]for k in('window','noise','image_SHA256')):
                raise ValueError('CURVE_MATCHED_INTERMEDIATE_NOISE_AND_INPUTS')
        # Diagnostics must not consume the next optimization draw.
        trainer._load_rng_state(str(cp))
        if rng_identity()!=expected_rng:raise ValueError('CURVE_DIAGNOSTICS_CHANGED_TRAIN_RNG')
        record_memory('AFTER_INTERMEDIATE_FULL_RESTORE_AND_DIAGNOSTICS_'+str(step))

    class FitTrainer(make_trainer_class()):
        def train(self,resume_from_checkpoint=None,**kwargs):
            if resume is not None:
                from gr00t.experiment.trainer import Gr00tTrainer
                if str(resume_from_checkpoint)!=resume['path']:
                    raise PermissionError('SHARED_EXACT_VALIDATED_RESUME_PATH')
                return Gr00tTrainer.train(self,resume_from_checkpoint=resume_from_checkpoint,**kwargs)
            return super().train(resume_from_checkpoint=resume_from_checkpoint,**kwargs)

        def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
            if resume is None:return super()._load_from_checkpoint(resume_from_checkpoint,model=model)
            from .shared_checkpoint import load_delta
            manifest=read(root/'RUN_MANIFEST.json')
            proof=load_delta(self.model,contract,resume_from_checkpoint,manifest_sha256=resume['manifest_sha256'],
                raw_parent=manifest['raw_parent'],audit=read(root/'RAW_PARENT_AUDIT.json'),guard=guard)
            write(root/'RESUME_MODEL_RECONSTRUCTED.json',proof)

        def _load_optimizer_and_scheduler(self,checkpoint):
            result=super()._load_optimizer_and_scheduler(checkpoint)
            if resume is not None:
                state=self.optimizer.state_dict()
                if len(state['state'])!=42 or any(int(v['step'])!=resume['step']for v in state['state'].values()):
                    raise PermissionError('SHARED_ACTUAL_RESUME_ADAMW_STEP_AND_ENTRIES')
                saved=torch.load(Path(checkpoint)/'scheduler.pt',map_location='cpu',weights_only=False)
                if self.lr_scheduler.state_dict()!=saved:raise PermissionError('SHARED_ACTUAL_RESUME_SCHEDULER')
                write(root/'RESUME_OPTIMIZER_SCHEDULER_RESTORED.json',dict(step=resume['step'],optimizer_state_entries=42,
                    scheduler_restored=True,science_run_id=resume['science_run_id']))
            return result

        def _load_rng_state(self,checkpoint):
            result=super()._load_rng_state(checkpoint)
            if resume is not None and Path(checkpoint).resolve()==Path(resume['path']):
                from .shared_resume import rng_identity_from_saved
                saved=torch.load(Path(checkpoint)/'rng_state.pth',map_location='cpu',weights_only=False)
                if rng_identity()!=rng_identity_from_saved(saved):raise PermissionError('SHARED_ACTUAL_RESUME_RNG')
                write(root/'RESUME_RNG_RESTORED.json',dict(step=resume['step'],RNG_restored=True,
                    next_cursor=resume['cursor']))
            return result

        def get_train_dataloader(self):
            self.start_cursor=self.consumed=resume['cursor']if resume else self.state.global_step*32
            self.sampler=sampler_class(self.train_dataset,start=self.consumed)
            return MeasuredLoader(torch.utils.data.DataLoader(self.train_dataset,batch_size=1,sampler=self.sampler,
                collate_fn=self.data_collator,num_workers=0,pin_memory=True,drop_last=True,
                generator=torch.Generator().manual_seed(p['seed'])))

        def create_optimizer(self):
            contract.validate();result=super().create_optimizer()
            if {id(x)for g in self.optimizer.param_groups for x in g['params']}!={id(x)for x in contract.parameters()}:raise ValueError('FIT_EXACT_OPTIMIZER_PARAMETERS')
            return result

        def create_scheduler(self,num_training_steps,optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer,lambda step:learning_rate_multiplier(step,p.get('scheduler_horizon_steps',steps),p['warmup_updates']))
                self._created_lr_scheduler=True
            return self.lr_scheduler

        def training_step(self,model,inputs,num_items_in_batch=None):
            guard();contract.validate()
            with activation_context(activation_storage):
                return super().training_step(model,inputs,num_items_in_batch)

        def _save_checkpoint(self,model,trial):
            guard()
            checkpoint_step=self.state.global_step;checkpoint_cursor=checkpoint_step*32
            if checkpoint_step not in p.get('checkpoint_steps',[steps])or self.consumed!=checkpoint_cursor:
                raise ValueError('FIT_ONLY_DECLARED_COMMITTED_ENDPOINT')
            cp=Path(self.args.output_dir)/f'checkpoint-{checkpoint_step}';cp.mkdir(parents=True,exist_ok=False)
            if frozen_fingerprint(model)!=frozen:raise ValueError('FIT_FROZEN_PARENT_CHANGED')
            torch.save({n:parameters[n].detach().cpu().clone()for n in names},cp/'target_parameters.pt')
            self._save_optimizer_and_scheduler(str(cp));self._save_rng_state(str(cp));self.state.save_to_json(str(cp/'trainer_state.json'))
            self.train_dataset.processor.save_pretrained(cp/'processor')
            for name in('statistics.json','TRAIN_DEV_SEAL.json'):shutil.copyfile(data.root/name,cp/name)
            if m32:shutil.copyfile(data.root/'TRAIN_POPULATION.json',cp/'TRAIN_POPULATION.json')
            if shared:
                from .shared_checkpoint import raw_reference
                write(cp/'RAW_PARENT_REFERENCE.json',raw_reference())
                shutil.copyfile(data.root/'TASK_DEV_POPULATION.json',cp/'TASK_DEV_POPULATION.json')
            if p['schema']in(PHASE_SCHEMA,PB_SCHEMA):shutil.copyfile(root/'TRAIN_PHASE_INDEX.json',cp/'TRAIN_PHASE_INDEX.json')
            if activation_storage is not None:
                shutil.copyfile(root/'M32_MEMORY_EXECUTION.json',cp/'M32_MEMORY_EXECUTION.json')
            write(cp/'ADAPTER_PARAMETER_CONTRACT.json',contract.describe())
            write(cp/'FROZEN_PARENT_TENSORS.json',frozen)
            write(cp/'SAMPLER_STATE.json',dict(recipe=p,sampler=self.sampler.state_dict(self.consumed),
                worker_trace=check_trace(data.trace_root,data,self.start_cursor,checkpoint_cursor),
                total_committed_trace_count=checkpoint_cursor,
                prior_committed_trace=(dict(checkpoint_manifest_sha256=resume['manifest_sha256'],cursor=resume['cursor'],
                    saved_sampler=resume['saved_sampler'])if resume else None),
                dataset_manifest_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha))
            members={str(x.relative_to(cp)):sha(x)for x in sorted(cp.rglob('*'))if x.is_file()}
            required={'target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','statistics.json','TRAIN_DEV_SEAL.json','SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json'}
            if not required<=set(members):raise ValueError('FIT_PARENT_DEPENDENT_FULL_RESUME_MEMBERS')
            from .shared_checkpoint import SCHEMA as SHARED_CHECKPOINT_SCHEMA,validate_header as validate_shared_header
            header=dict(schema=SHARED_CHECKPOINT_SCHEMA if shared else 'BOOK_M32_PARENT_BOUND_TRAINING_DELTA_V1'if m32 else 'BOOK_TRAIN16_PARENT_BOUND_TARGET_DELTA_V1'if p==plan()else'BOOK_PARENT_BOUND_TARGET_DELTA_V2',global_step=checkpoint_step,cursor=checkpoint_cursor,
                full_resume_with_required_parent=True,standalone_model=False,**parent_binding,
                recipe=p,files=members,actual_reload_check='PENDING')
            if shared:validate_shared_header(header)
            write(cp/'CHECKPOINT_COMPLETE.json',header)
            if checkpoint_step!=steps:qualify_intermediate(self,cp,checkpoint_step,checkpoint_cursor)

    args=TrainingArguments(output_dir=str(root/'training'),max_steps=steps,per_device_train_batch_size=1,
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
            if state.global_step in(1,2,10,steps)or state.global_step%32==0:
                record_memory('AFTER_UPDATE_'+str(state.global_step))
                if activation_storage is not None:
                    write(root/'ACTUAL_ACTIVATION_OFFLOAD.json',activation_storage.audit())
            if t.consumed!=state.global_step*32:raise ValueError('FIT_ACTUAL_COMMITTED_CURSOR')
            r=dict(step=state.global_step,cursor=t.consumed,native_loss=t.last_native_loss,learning_rate=t.lr_scheduler.get_last_lr(),epoch=time.time())
            with(root/'OPTIMIZER_STEPS.jsonl').open('a')as f:f.write(json.dumps(r)+'\n');f.flush()
            write(root/'FIT_PROGRESS.json',r)
            if state.global_step in p.get('checkpoint_steps',[steps]):control.should_save=True
            if state.global_step==steps:control.should_training_stop=True
            return control
    trainer=FitTrainer(model=model,args=args,train_dataset=data,data_collator=ProcessedCollator(data.processor),callbacks=[Audit()],evidence_dir=root/'fit_evidence',recipe=p)
    holder['trainer']=trainer
    before=measure(root,'before',instance,probe,trainer,guard,p)
    # A probe uses fork_rng and deterministic preprocessing, but the training
    # seed is reset explicitly; no diagnostic random draw enters optimization.
    set_seed(p['seed']);trainer.train(resume_from_checkpoint=resume['path']if resume else None)
    if trainer.state.global_step!=steps or trainer.consumed!=visits or frozen_fingerprint(model)!=frozen:raise ValueError('FIT_DECLARED_FROZEN_ENDPOINT')
    after_parameters={n:tensor_identity(parameters[n])for n in names}
    changed=[n for n in names if before_parameters[n]!=after_parameters[n]]
    if not changed:raise ValueError('FIT_NO_ACTUAL_PARAMETER_UPDATE')
    cp=root/f'training/checkpoint-{steps}';complete=read(cp/'CHECKPOINT_COMPLETE.json')
    for name,digest in complete['files'].items():
        if sha(cp/name)!=digest:raise ValueError('FIT_SAVED_MEMBER_CHANGED:'+name)
    expected_optimizer=tree_identity(trainer.optimizer.state_dict());expected_scheduler=trainer.lr_scheduler.state_dict();expected_rng=rng_identity()
    # Reconstruct the actual model from the retained full parent plus the delta.
    restore_base()
    delta=torch.load(cp/'target_parameters.pt',map_location='cpu',weights_only=True);validate_delta(delta,parameters,names,expected_count=len(names)if m32 else 14)
    with torch.no_grad():
        for name in names:parameters[name].copy_(delta[name])
    if {n:tensor_identity(parameters[n])for n in names}!=after_parameters or frozen_fingerprint(model)!=frozen:raise ValueError('FIT_ACTUAL_PARENT_PLUS_DELTA_RECONSTRUCTION')
    restored=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    if not restored['state']or len(restored['state'])!=len(names)or any(int(x['step'])!=steps for x in restored['state'].values()):raise ValueError('FIT_ACTUAL_DECLARED_ADAMW_STEPS_REQUIRED')
    trainer.optimizer.state.clear();trainer.optimizer.load_state_dict(restored)
    trainer.lr_scheduler.load_state_dict(torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False));trainer._load_rng_state(str(cp))
    if tree_identity(trainer.optimizer.state_dict())!=expected_optimizer or trainer.lr_scheduler.state_dict()!=expected_scheduler or rng_identity()!=expected_rng:raise ValueError('FIT_EXACT_OPTIMIZER_SCHEDULER_RNG_RESTORE')
    record_memory('AFTER_ACTUAL_MODEL_ADAMW_SCHEDULER_RNG_RESTORE')
    saved=read(cp/'SAMPLER_STATE.json')
    trainer.state=TrainerState.load_from_json(str(cp/'trainer_state.json'))
    restored_sampler=sampler_class(data,start=saved['sampler']['cursor'])
    if trainer.state.global_step!=steps or saved['sampler']!=restored_sampler.state_dict(visits):raise ValueError('FIT_ACTUAL_STEP_AND_SAMPLE_CURSOR_RESTORE')
    write(root/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json',dict(checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),
        **parent_binding,model_reconstruction=True,optimizer_state_entries=len(names),
        optimizer_updates=steps,scheduler_restored=True,RNG_restored=True,sampler_cursor=visits,frozen_parent_unchanged=True,
        changed_trainable_parameters=changed,standalone_model=False))
    after=measure(root,'after',instance,probe,trainer,guard,p)
    if m32:
        record_memory('AFTER_MATCHED_TRAIN_GENERATION')
        write(root/'M32_CAPACITY_RESULT.json',dict(qualification_only=p['capacity_qualification_only'],
            actual_optimizer_updates=steps,saved_and_restored=True,contract=contract.describe(),
            minimum_observed_free_bytes=min(x['free_bytes']for x in memory),
            maximum_allocated_bytes=max(x['maximum_allocated_bytes']for x in memory),
            maximum_reserved_bytes=max(x['maximum_reserved_bytes']for x in memory),
            capacity_pass=min(x['free_bytes']for x in memory)>=2*1024**3,
            measured_points=len(memory),measured_free_is_shared_device_not_exclusive_allocation=True))
    for a,b in zip(before['records'],after['records']):
        if a['window']!=b['window']or a['noise']!=b['noise']or a['image_SHA256']!=b['image_SHA256']:raise ValueError('FIT_MATCHED_BEFORE_AFTER_NOISE_INPUTS')
    trainer=None;gc.collect()
    return dict(status='COMPLETE_SHARED_RAW_FINITE_LEARNING'if shared else'COMPLETE_TRAIN16_FIT256_DIAGNOSTIC'if p==plan()else'COMPLETE_BOOK5_FINITE_TARGET_LEARNING',optimizer_updates=steps,TRAIN_visits=visits,
        checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),checkpoint_requires_retained_full_parent=True,
        generation_queries=p['generation_queries'],native_diagnostic_loss_forwards=p['native_diagnostic_loss_forwards'],physics_integrations=0,DEV_conditions=0,
        before=before,after=after,learned_task_success_not_measured=True)
