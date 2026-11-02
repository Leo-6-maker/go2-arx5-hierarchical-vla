"""Registered raw/shared-checkpoint inference and one source DEV condition.

The existing owner supplies the sealed entry, custody, finite resources and
launch/close lifecycle. This worker is not a scheduler or a target-WBC entry.
"""
import gc,hashlib,json,os,signal,sys,time
from pathlib import Path

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--manifest-sha256',required=True)
    entry=parser.parse_args();body=(Path(entry.root)/'RUN_MANIFEST.json').read_bytes()
    if hashlib.sha256(body).hexdigest()!=entry.manifest_sha256:raise PermissionError('SHARED_POLICY_ENTRY_MANIFEST')
    dependency=json.loads(body)['dependency_code'];sys.path[:0]=[dependency,dependency+'/src']


def qualified_inference_entry(m):
    from .shared_evaluation import KIND,INFERENCE_KIND,LIMITS,INFERENCE_LIMITS
    from .shared_training import is_recipe
    p=m.get('initialization_recipe',{})
    if (not is_recipe(p)or p['max_steps']!=8192 or m.get('kind')not in(KIND,INFERENCE_KIND)
            or m.get('raw_parent_files_identity')!=p['raw_parent_files_identity']
            or type(m.get('raw_zero_step'))is not bool
            or m.get('kind')==KIND and m.get('limits')!=LIMITS
            or m.get('kind')==INFERENCE_KIND and m.get('limits')!=INFERENCE_LIMITS
            or any(m.get(k)is not None for k in('GPU_training_admission','GPU_training_qualification',
                'GPU_capacity_qualification','GPU_inference_admission','checkpoint_manifest_sha256','parent_checkpoint'))):
        raise PermissionError('SHARED_EXACT_FROZEN_RAW_MODEL_AND_INFERENCE_ONLY_SCOPE')
    if m['kind']==INFERENCE_KIND:
        if m.get('shared_inference_admission')is not None:raise PermissionError('ONE_OWN_NATIVE16_INFERENCE_QUALIFICATION')
        return 13*1024**3
    admission=m.get('shared_inference_admission',{})
    required={'RUN_MANIFEST.json','RESULT.json','SHARED_GPU_INFERENCE_CAPACITY.json',
              'ACTUAL_SHARED_INFERENCE_RECONSTRUCTION.json','control/LEASE.json'}
    if admission.get('schema')!='SHARED4_SAME_GPU_NATIVE16_INFERENCE_V1'or set(admission.get('files',{}))!=required:
        raise PermissionError('SHARED_CHECKPOINT_OWN_CLOSED_SAME_GPU_INFERENCE_REQUIRED')
    root=Path(admission['qualification_root']);values={}
    if not root.is_absolute()or root.is_symlink():raise PermissionError('SHARED_INFERENCE_QUALIFICATION_ROOT')
    for name,pin in admission['files'].items():
        f=root/name
        if f.is_symlink()or not f.resolve().is_relative_to(root.resolve()):raise PermissionError('SHARED_CERTIFICATE_PATH')
        body=f.read_bytes()
        if hashlib.sha256(body).hexdigest()!=pin:raise PermissionError('SHARED_INFERENCE_CERTIFICATE_CHANGED:'+name)
        values[name]=json.loads(body)
    q=values['RUN_MANIFEST.json'];result=values['RESULT.json'];capacity=values['SHARED_GPU_INFERENCE_CAPACITY.json']
    reconstruction=values['ACTUAL_SHARED_INFERENCE_RECONSTRUCTION.json'];lease=values['control/LEASE.json']
    if (q.get('kind')!=INFERENCE_KIND
            or any(q.get(k)!=m.get(k)for k in('GPU_uuid','raw_zero_step','initialization_recipe',
                'raw_parent_files_identity','evaluation_checkpoint_manifest_sha256'))
            or result.get('status')!='COMPLETE_SHARED_TRAIN16_INFERENCE_QUALIFICATION'or result.get('error')is not None
            or any(result.get(k)!=v for k,v in dict(model_load_attempts=1,optimizer_updates=0,physics_integrations=0,
                generation_attempts=16,completed_generations=16,formal_DEV_conditions=0).items())
            or capacity.get('capacity_pass')is not True or capacity.get('minimum_free_bytes',0)<1024**3
            or capacity.get('measurements',0)<17
            or reconstruction.get('evaluation_checkpoint_manifest_sha256')!=m['evaluation_checkpoint_manifest_sha256']
            or reconstruction.get('frozen_parent_unchanged')is not True
            or lease.get('status')!='CLOSED'or lease.get('manifest_sha256')!=admission['files']['RUN_MANIFEST.json']
            or lease.get('result_sha256')!=admission['files']['RESULT.json']):
        raise PermissionError('SHARED_ACTUAL_NATIVE16_SAME_MODEL_GPU_AND_ONE_GIB_HEADROOM')
    qualified_inference_entry(q)
    return 13*1024**3


def candidate_custody(root,m):
    from scripts.pi0292.durable import read
    from scripts.pi0294.native_dataset import sha
    from .raw_parent_delivery import PARENT_ID
    from .book_reset import verify_predecessor
    from scripts.pi0294.e1.smoke import node_local
    from .shared_checkpoint import validate_header
    if m['raw_zero_step']:
        if m['evaluation_checkpoint_manifest_sha256']!=PARENT_ID or m.get('delta_checkpoint')is not None:
            raise PermissionError('ORIGINAL_RAW_ZERO_STEP_WITHOUT_ADAPTED_DELTA')
        return None
    producer=node_local(m['delta_producer']['root'])
    verify_predecessor(m['delta_producer'],current_root=root)
    result=read(producer/'RESULT.json')
    if result.get('status')!='COMPLETE_SHARED_RAW_FINITE_LEARNING'or result.get('optimizer_updates')!=8192 or result.get('error')is not None:
        raise PermissionError('COMPLETE_CLOSED_SHARED_PRODUCER_NOT_NATIVE10_OR_LIVE_INTERMEDIATE')
    cp=node_local(m['delta_checkpoint']);complete=validate_header(read(cp/'CHECKPOINT_COMPLETE.json'));step=complete['global_step']
    if cp!=producer/f'training/checkpoint-{step}'or complete['recipe']!=m['initialization_recipe']:
        raise PermissionError('OWN_DECLARED_SHARED_LEARNING_CURVE_POINT_AND_FIXED_DATA')
    name='ACTUAL_DELTA_CHECKPOINT_RESTORE.json'if step==8192 else f'ACTUAL_DELTA_CHECKPOINT_RESTORE_{step}.json'
    proof=read(producer/name)
    if (sha(producer/name)!=m['delta_native_qualification_sha256']
            or proof.get('checkpoint_manifest_sha256')!=m['evaluation_checkpoint_manifest_sha256']
            or proof.get('required_raw_parent_files_identity')!=PARENT_ID
            or any(proof.get(k)!=v for k,v in dict(model_reconstruction=True,optimizer_state_entries=42,
                optimizer_updates=step,scheduler_restored=True,RNG_restored=True,sampler_cursor=step*32,frozen_parent_unchanged=True).items())):
        raise PermissionError('SHARED_CANDIDATE_ACTUAL_FULL_STATE_RESTORE')
    if sha(cp/'CHECKPOINT_COMPLETE.json')!=m['evaluation_checkpoint_manifest_sha256']:
        raise PermissionError('SHARED_CANDIDATE_EXACT_CHECKPOINT_HEADER')
    return cp


def candidate(root,m,model,contract,guard,checkpoint):
    from scripts.pi0292.durable import read
    from .raw_parent_delivery import PARENT_ID
    from .shared_checkpoint import load_delta
    if m['raw_zero_step']:
        return dict(evaluation_checkpoint_manifest_sha256=PARENT_ID,raw_zero_step=True,
            kind='SHARED4_RAW_PARENT_ZERO_STEP',global_step=0,
            full_raw_loading_and_42_parameter_comparison=True,frozen_parent_unchanged=True)
    value=load_delta(model,contract,checkpoint,manifest_sha256=m['evaluation_checkpoint_manifest_sha256'],
        raw_parent=m['raw_parent'],audit=read(root/'RAW_PARENT_AUDIT.json'),guard=guard)
    return dict(value,evaluation_checkpoint_manifest_sha256=m['evaluation_checkpoint_manifest_sha256'],raw_zero_step=False,
        kind='SHARED4_RAW_PARENT_BOUND_DELTA',global_step=read(checkpoint/'CHECKPOINT_COMPLETE.json')['global_step'])


def run(root,manifest_sha256):
    from scripts.pi0292.durable import read,write,OwnerLock
    from scripts.pi0294.e1.smoke import node_local,put
    from scripts.pi0294.native_dataset import NativeDataset,ProcessedCollator,sha,official_import
    from scripts.pi0294.numerics import prepare_environment,configure_torch,MODE
    from .shared_evaluation import KIND,INFERENCE_KIND,validate_manifest,run_condition
    from .shared_training import validate_data
    from .shared_raw_model import load as load_raw
    root=node_local(root);m=read(root/'RUN_MANIFEST.json');bindings=read(root/'NEURAL_BINDINGS.private.json')
    if (sha(root/'RUN_MANIFEST.json')!=manifest_sha256 or m.get('owner_uid')!=os.getuid()
            or os.uname().nodename!='corelab-4080s'):
        raise PermissionError('SHARED_EVAL_REGISTERED_NATIVE_OWNER')
    lease=read(root/'control/LEASE.json')
    if lease.get('status')!='ACTIVE'or lease.get('manifest_sha256')!=manifest_sha256:
        raise PermissionError('SHARED_EVAL_ACTIVE_NATIVE_LEASE')
    if not any(r.get('path')==str(root)and r.get('role')==m['role']for r in read(root/'control/REGISTERED_PATHS.json')['entries']):
        raise PermissionError('SHARED_EVAL_EXISTING_PATH_REGISTRY')
    for name,pin in m['sealed_inputs'].items():
        path=root/name
        if not path.resolve().is_relative_to(root.resolve())or sha(path)!=pin:raise PermissionError('SHARED_EVAL_CHANGED_INPUT:'+name)
    for name in('source','python','cosmos','dataset','cosmos_hub'):node_local(bindings[name])
    for name in('RAW_PARENT_AUDIT.json','NEURAL_BINDINGS.private.json'):
        if m['sealed_inputs'].get(name)!=sha(root/name):raise PermissionError('SHARED_EXACT_SEALED_RAW_AUDIT_AND_BINDINGS')
    node_local(m['raw_parent']);p=m['initialization_recipe'];required=qualified_inference_entry(m)
    checkpoint=candidate_custody(root,m)
    data=NativeDataset(bindings['dataset'],bindings['source'],bindings['cosmos'],for_optimization=True);validate_data(data,p)
    is_qualification=m['kind']==INFERENCE_KIND;condition=None
    if not is_qualification:
        condition=validate_manifest(m,read(root/'SHARED_DEV_PLAN.json'),read(root/'TASK_DEV_POPULATION.json'),read(root/'TASK_NATIVE_QUALIFICATION.json'))
        if sha(root/'TASK_DEV_POPULATION.json')!=p['task_DEV_population_sha256']:raise PermissionError('SHARED_ORIGINAL_TASK_DEV_POPULATION')
        template_info=m['task_templates'][condition['task']];template=node_local(template_info['root'])
        if sha(template/'RUN_MANIFEST.json')!=template_info['manifest_sha256']:raise PermissionError('EXACT_ORIGINAL_SOURCE_TEMPLATE')
        if read(template/'control/NATIVE_SOURCE_LEASE.json')['status']!='CLOSED':raise PermissionError('CLOSED_SOURCE_TEMPLATE')
        assets=read(template/'ASSET_BINDINGS.private.json')
        node_local(assets['robot_root'])
        for pin in assets['files'].values():
            if sha(node_local(pin['path']))!=pin['sha256']:raise PermissionError('ORIGINAL_SHARED_SOURCE_ASSET_CHANGED')
    lock=OwnerLock(root.parent/'global-native-GPU-locks'/(m['GPU_uuid']+'.lock'))
    started=time.monotonic();CPU=time.process_time();loads=0;failure=None;row=diagnostic=None;model=instance=contract=None;torch=None
    capacity=dict(last_sample_monotonic=0.,measurements=0,minimum_free_bytes=2**63,capacity_pass=True)
    def guard():
        if time.time()>=m['deadline_epoch']or(root/'STOP.json').exists():raise PermissionError('SHARED_EVAL_FINITE_OWNER_STOP')
        fs=os.statvfs(root)
        if fs.f_bavail*fs.f_frsize<m['protected_remaining_bytes']+m['free_margin_bytes']:raise PermissionError('SHARED_EVAL_PROTECTED_DISK_MARGIN')
    def sample():
        nonlocal capacity
        from .book_gpu_capacity import runtime_inference_capacity
        capacity=runtime_inference_capacity(torch,capacity,time.monotonic());write(root/'SHARED_GPU_INFERENCE_CAPACITY.json',capacity)
        if not capacity['capacity_pass']:raise PermissionError('SHARED_ACTUAL_INFERENCE_ONE_GIB_HEADROOM')
    def stop(sig,frame):raise TimeoutError('SHARED_EVAL_NATIVE_FINITE_END')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(m['deadline_epoch']-time.time())))
    try:
        guard();prepare_environment(MODE);official_import(bindings['source'])
        import torch as framework
        torch=framework;configure_torch(torch,MODE);torch.set_num_threads(4)
        if not is_qualification:
            from .book_reset import verify_predecessor
            previous=read(root/'RESET_PREDECESSOR.json');closure=verify_predecessor(previous,current_root=root)
            write(root/'RESET_PREDECESSOR_VERIFIED.json',closure)
            sys.path.append(str(node_local(m['physics_packages'])))
            import mujoco
            if mujoco.__version__!='3.2.7':raise PermissionError('SHARED_QUALIFIED_MUJOCO_VERSION')
        count=torch.cuda.device_count();free,total=torch.cuda.mem_get_info()if count==1 else(0,0)
        uuid=str(torch.cuda.get_device_properties(0).uuid)if count==1 else None
        put(root/'GPU_ENTRY_OBSERVATION.json',dict(device_count=count,actual_GPU_uuid=uuid,free_bytes=free,total_bytes=total,required_free_bytes=required))
        if count!=1 or free<required or uuid!=m['GPU_uuid']:raise PermissionError('SHARED_ACTUAL_SAME_DEVICE_INFERENCE_CAPACITY')
        put(root/'MODEL_LOAD_INTENT.json',dict(raw_parent_files_identity=m['raw_parent_files_identity'],epoch=time.time()));loads+=1
        instance,contract=load_raw(root,m,bindings,p,guard);model=instance.model
        reconstruction=candidate(root,m,model,contract,guard,checkpoint);write(root/'ACTUAL_SHARED_INFERENCE_RECONSTRUCTION.json',reconstruction)
        instance.identity=dict(instance.identity,**reconstruction)
        model.eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats();sample()
        original_generate=instance.generate
        def timed_generate(request,seed):
            signal.alarm(max(1,min(60,int(m['deadline_epoch']-time.time()))))
            try:
                value=original_generate(request,seed);sample();return value
            finally:signal.alarm(max(1,int(m['deadline_epoch']-time.time())))
        instance.generate=timed_generate;write(root/'MODEL_LOADED.json',dict(identity=instance.identity,epoch=time.time()))
        if is_qualification:
            from transformers import TrainingArguments
            from scripts.pi0294.native_train import make_trainer_class
            from .book_fit import select_rows,measure
            probe=NativeDataset(bindings['dataset'],bindings['source'],bindings['cosmos'],for_optimization=False)
            probe.rows=select_rows(probe.rows,p);probe.initialize();data.initialize()
            args=TrainingArguments(output_dir=str(root/'no_training'),max_steps=1,per_device_train_batch_size=1,
                gradient_accumulation_steps=32,bf16=True,remove_unused_columns=False,eval_strategy='no',
                save_strategy='no',save_total_limit=None,report_to=[],disable_tqdm=True)
            trainer=make_trainer_class()(model=model,args=args,train_dataset=data,data_collator=ProcessedCollator(data.processor),
                evidence_dir=root/'qualification_loss',recipe=p)
            diagnostic=measure(root,'inference-qualification',instance,probe,trainer,guard,p)
            write(root/'TRAIN16_INFERENCE_RESULT.json',diagnostic)
        else:
            row=run_condition(root,condition,instance,template,guard)
        sample()
    except BaseException as error:
        failure=type(error).__name__+': '+str(error);raise
    finally:
        signal.alarm(0)
        terminals=[read(f)for f in root.glob('conditions/*/TERMINAL.json')]
        source_counts_unknown=not is_qualification and not terminals and any(root.glob('conditions/*/NATIVE_SCENE.private.xml'))
        status=('COMPLETE_SHARED_TRAIN16_INFERENCE_QUALIFICATION'if diagnostic is not None and failure is None else'FAILED_SHARED_TRAIN16_INFERENCE_QUALIFICATION')if is_qualification else(
            'COMPLETE_SHARED_DEV_CONDITION'if row is not None and row['valid_model_condition']and row.get('reset_close_confirmed')else'FAILED_SHARED_DEV_CONDITION')
        write(root/'RESULT.json',dict(status=status,error=failure,scientific_kind=m['kind'],model_load_attempts=loads,
            generation_attempts=len(list(root.glob('inference-qualification/*.INTENT.json')))if is_qualification else len(list(root.glob('conditions/*/predictions/*_intent.json'))),
            completed_generations=len(list(root.glob('inference-qualification/*.RESULT.json')))if is_qualification else len(list(root.glob('conditions/*/predictions/*_prediction.json'))),
            optimizer_updates=0,physics_integrations=None if source_counts_unknown else sum(t['actual_integrations']for t in terminals),
            actual_RGB_pairs=None if source_counts_unknown else sum(t['actual_RGB_pairs']for t in terminals),
            source_counts_unknown=source_counts_unknown,physics_upper_if_unknown=18500 if source_counts_unknown else None,
            formal_DEV_conditions=0 if row is None else int(row['valid_model_condition']),conditions=[]if row is None else[row],
            CPU_s=time.process_time()-CPU,wall_s=time.monotonic()-started,completed_epoch=time.time()))
        data=model=instance=contract=None;gc.collect();lock.close()


if __name__=='__main__':run(entry.root,entry.manifest_sha256)
