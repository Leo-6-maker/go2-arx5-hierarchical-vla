"""Finite independent TASK_ADAPTER training and complete checkpoint recovery."""
import random
import os
import time
from pathlib import Path
import numpy as np
from . import stable_task_adapter_v1 as candidate
from .storage import digest,write_json

PROFILE='wm02-pilot1-neural'
KIND='TASK_ADAPTER'
RECIPE='WM02_STABLE_TASK_ADAPTER_TRAIN_V1'
FROZEN_GR00T='94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'


def contract(spec):
    start,end=spec.get('start_step',0),spec.get('endpoint')
    if not (spec.get('kind')==KIND and spec.get('profile')==PROFILE
            and spec.get('execution_site')=='CORE_PILOT1_NEURAL'
            and spec.get('candidate_recipe')==RECIPE
            and type(start)is int and type(end)is int and 0<=start<end
            and type(spec.get('TASK_ADAPTER_updates'))is int and spec['TASK_ADAPTER_updates']==end-start
            and all(type(spec.get(k))is int and spec[k]==0 for k in ('GR00T_updates','Actor_updates','WM_updates'))
            and not any(spec.get(k)for k in ('fusion_binding','next1_binding','next2_binding','night1_binding'))
            and spec.get('frozen_GR00T_manifest_sha256')==FROZEN_GR00T
            and spec.get('seed')in candidate.SEEDS and spec.get('learning_rate')==.001
            and spec.get('weight_decay')==.001 and spec.get('batch_size')==16
            and spec.get('WBC_changes')is False and spec.get('physics')is False):
        raise PermissionError('REAL_PILOT1_NONZERO_TASK_ADAPTER_CONTRACT_REQUIRED_NO_FUSION_HIDDEN_TRAINING')


def grouped_mse(prediction,target,rows,indices):
    values={}
    for position,index in enumerate(indices):
        values.setdefault(rows[int(index)]['parent_id'],[]).append(float(np.mean((prediction[position]-target[position])**2)))
    return float(np.mean([np.mean(v)for v in values.values()]))


def fit(x,y,rows,train,dev,output,*,seed,endpoint,provenance,device,
        finite=lambda:None,start_step=0,resume=None):
    """Shared numerical engine; site admission and real-data proof live in run."""
    import torch
    output=Path(output);started,cpu=time.monotonic(),time.process_time()
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    mean,scale=candidate.normalizer(x[train]);model=candidate.make_model(mean,scale).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    sampler=candidate.Sampler(len(train),seed)
    curve=[];best=None;best_file=None;best_step=None
    if resume is not None:
        actual=candidate.restore_checkpoint(resume,model,opt,sampler,expected_provenance=provenance,device=device)
        if actual!=start_step:raise ValueError('EXACT_CUMULATIVE_RESUME_STEP_REQUIRED')
        selection=torch.load(resume,map_location='cpu',weights_only=True).get('training_state')
        if not selection:raise ValueError('RESUME_REQUIRES_COMPLETE_MODEL_SELECTION_STATE')
        curve=selection['curve'];best=selection['best_metric'];best_step=selection['best_step']
        best_file=Path(resume)if best_step==start_step and selection['selected_checkpoint_sha256']is None else Path(selection['selected_checkpoint'])
        if selection['selected_checkpoint_sha256']is not None and digest(best_file)!=selection['selected_checkpoint_sha256']:
            raise ValueError('PINNED_PRIOR_SELECTED_CHECKPOINT')
    elif start_step:raise ValueError('NONZERO_START_REQUIRES_COMPLETE_CHECKPOINT')
    tx=torch.from_numpy(x).to(device);ty=torch.from_numpy(y).to(device)
    step=start_step
    for ids in (train,dev):
        if any(not any(rows[int(i)]['task_command']==command for i in ids)for command in candidate.COMMANDS):
            raise ValueError('BOTH_REAL_TASK_CONDITIONS_REQUIRED_IN_EACH_SPLIT')
    # Baseline receives the same declared goal instruction, fitted on TRAIN only.
    means={command:y[[i for i in train if rows[int(i)]['task_command']==command]].mean(0)
           for command in candidate.COMMANDS}
    baseline=np.asarray([means[rows[int(i)]['task_command']]for i in dev])
    baseline_mse=grouped_mse(baseline,y[dev],rows,dev)
    def observe():
        nonlocal best,best_file,best_step
        with torch.no_grad():
            p=model(tx).detach().cpu().numpy()
        row=dict(step=step,TRAIN_parent_mean_command_MSE_m2=grouped_mse(p[train],y[train],rows,train),
                 DEV_parent_mean_command_MSE_m2=grouped_mse(p[dev],y[dev],rows,dev))
        curve.append(row)
        if step>0:
            path=output/f'CHECKPOINT_{step:08d}.pt'
            if best is None or row['DEV_parent_mean_command_MSE_m2']<best:
                best=row['DEV_parent_mean_command_MSE_m2'];best_file=path;best_step=step
            selection=dict(curve=copy_curve(curve),best_metric=best,best_step=best_step,
                selected_checkpoint=str(best_file),selected_checkpoint_sha256=None if best_file==path else digest(best_file))
            candidate.save_checkpoint(path,model,opt,sampler,step=step,provenance=provenance,training_state=selection)
        write_json(output/f'CURVE_{step:08d}.private.json',row)
    if not start_step:observe()
    failure=None
    try:
        while step<endpoint:
            if step%25==0:finite()
            indices=train[sampler.take(16)];opt.zero_grad(set_to_none=True)
            loss=((model(tx[indices])-ty[indices])/candidate.STEP_M).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_CANDIDATE_LOSS')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();step+=1
            if step%250==0 or step==endpoint:observe()
        finite()
        endpoint_file=output/f'CHECKPOINT_{step:08d}.pt'
        actual=candidate.export_arrays(model)
        restored=candidate.make_model(mean,scale).to(device)
        restored_opt=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.001)
        restored_sampler=candidate.Sampler(len(train),0)
        loaded_step=candidate.restore_checkpoint(endpoint_file,restored,restored_opt,restored_sampler,
            expected_provenance=provenance,device=device)
        arrays=candidate.export_arrays(restored)
        error=max(float(np.max(np.abs(actual[k]-arrays[k])))for k in actual if k!='schema')
        if loaded_step!=step or error!=0. or sampler.state()!=restored_sampler.state():
            raise ValueError('EXACT_ENDPOINT_MODEL_SAMPLER_RECOVERY_FAILED')
        # Check optimizer state restoration as well as model/export equality.
        original_state=opt.state_dict();loaded_state=restored_opt.state_dict()
        for key,values in original_state['state'].items():
            for name,value in values.items():
                other=loaded_state['state'][key][name]
                if torch.is_tensor(value):
                    if not torch.equal(value.cpu(),other.cpu()):raise ValueError('OPTIMIZER_RESTORE_MISMATCH')
                elif value!=other:raise ValueError('OPTIMIZER_RESTORE_MISMATCH')
        # Select the earliest checkpoint with minimum grouped DEV command error.
        candidate.restore_checkpoint(best_file,restored,restored_opt,restored_sampler,
            expected_provenance=provenance,device=device)
        export=candidate.export_arrays(restored);np.savez(output/'TASK_ADAPTER_EXPORT.npz',**export)
        with torch.no_grad():reference=restored(tx[dev]).detach().cpu().numpy()
        numpy_error=float(np.max(np.abs(candidate.numpy_predict(export,x[dev])-reference)))
        if numpy_error>1e-7:raise ValueError('DEPLOYABLE_NUMPY_EXPORT_MISMATCH')
        result=dict(schema='WM02_STABLE_TASK_ADAPTER_TRAIN_ACTUAL_V1',status='CANDIDATE_TRAINED_RELOADED_NOT_NATIVE_EVALUATED',
            provenance=provenance,seed=seed,start_step=start_step,endpoint=step,optimizer_updates=step-start_step,
            GR00T_updates=0,Actor_updates=0,WM_updates=0,TASK_ADAPTER_updates=step-start_step,
            TRAIN_episodes=len(train),DEV_episodes=len(dev),curve=curve,
            conditional_TRAIN_mean_baseline_DEV_command_MSE_m2=baseline_mse,
            selected_checkpoint=best_file.name,selected_checkpoint_sha256=digest(best_file),
            final_checkpoint=endpoint_file.name,final_checkpoint_sha256=digest(endpoint_file),
            export_sha256=digest(output/'TASK_ADAPTER_EXPORT.npz'),export_max_error_m=numpy_error,
            endpoint_model_restore_max_error=error,optimizer_restored=True,sampler_restored=True,RNG_restored=True,
            restore_probe_optimizer_updates=0,DEV_native_task_success='NOT_RUN',
            CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-started)
        write_json(output/'RESULT.private.json',result);return result
    except BaseException as e:
        failure=type(e).__name__+': '+str(e)
        write_json(output/'FAILED_TRAINING.private.json',dict(failure=failure,completed_optimizer_updates=step-start_step,
            last_global_step=step,source='INDEPENDENT_TASK_ADAPTER',GR00T_updates=0,provenance=provenance))
        raise


def run(spec,output):
    contract(spec)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from .stable_task_dataset_v1 import load_for_training
    from .pilot1_storage import local_path
    from .pilot1_scope import validate
    x,y,rows,train,dev,dataset=load_for_training(spec['replay'],spec['replay_sha256'])
    for row in rows:
        for descriptor in row['files'].values():local_path(spec,descriptor['path'])
    import torch
    from .runtime import cuda_uuid
    if torch.cuda.device_count()!=1 or cuda_uuid(torch.cuda.get_device_properties(0).uuid)!=spec['GPU_uuid']:
        raise PermissionError('ONE_ACTUAL_DECLARED_LOCAL_GPU')
    props=torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory)
    torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    provenance=dict(dataset_sha256=dataset['sha256'],seed=spec['seed'],recipe=RECIPE,
        adapter_module_sha256=digest(Path(spec['source_root'])/'src/go2arx5_vla/world_model/stable_task_adapter_v1.py'),
        trainer_module_sha256=digest(Path(__file__)),source='REAL_NEW_TARGET_TRAIN_COMMAND_SUPERVISION',
        frozen_GR00T_manifest_sha256=FROZEN_GR00T)
    started=time.monotonic()
    last_validation=[0.]
    def finite():
        if time.time()>=spec['deadline_epoch']or time.monotonic()-started>=spec['wall_limit_s']:
            raise TimeoutError('FINITE_ADMITTED_CANDIDATE_ENDPOINT')
        if any(p.exists()for p in (Path(spec['registered_root'])/'STOP.json',
                Path(spec['campaign_root'])/'STOP.json',Path(spec['pilot1_binding']).parent/'STOP.json')):
            raise InterruptedError('ORIGINAL_STOP')
        if time.monotonic()-last_validation[0]>=5.:
            validate(spec);last_validation[0]=time.monotonic()
    resume=spec.get('resume')
    if resume and digest(resume)!=spec['resume_sha256']:raise ValueError('PINNED_COMPLETE_RESUME_CHECKPOINT')
    result=fit(x,y,rows,train,dev,output,seed=spec['seed'],endpoint=spec['endpoint'],provenance=provenance,
        device='cuda:0',finite=finite,start_step=spec.get('start_step',0),resume=resume)
    torch.cuda.synchronize()
    write_json(Path(output)/'GPU_MEMORY.private.json',dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU_uuid=spec['GPU_uuid']))
    return result


def copy_curve(curve):
    return [dict(row)for row in curve]
