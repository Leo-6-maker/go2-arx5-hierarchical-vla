"""Admitted finite training on fresh, authenticated full-preview TRAIN records."""
import argparse,json,os,time
from pathlib import Path
import numpy as np

from go2arx5_vla.world_model.fusion_program_response_data_v1 import ARMS

KIND='FUSION_LIVE_PROGRAM_RESPONSE_WM'
PROFILE='wm02-fusion-live-program-response-train'
SITE='CORE_FUSION_NEURAL'
RECIPE='PROGRAM_CONDITIONAL_TCP_SO3_FINGERS_H16_V1'


def validate_gpu(spec,actual):
    from go2arx5_vla.world_model.fusion_scope import validate
    from go2arx5_vla.world_model.pilot1_scope import GPU_UUIDS
    if (spec.get('kind')!=KIND or spec.get('profile')!=PROFILE or spec.get('execution_site')!=SITE
            or spec.get('recipe')!=RECIPE or spec.get('physics')is not False
            or spec.get('device')!='cuda' or spec.get('WBC_changes')is not False
            or spec.get('Actor_updates')!=0 or spec.get('GR00T_updates')!=0
            or spec.get('GPU_uuid')not in GPU_UUIDS or actual['uuid']!=spec['GPU_uuid']
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['GPU_uuid']
            or not 0<spec.get('allocator_limit_bytes',0)<=2*1024**3
            or spec['allocator_limit_bytes']+1024**3>=actual['free_bytes']
            or type(spec.get('endpoint'))is not int or spec['endpoint']<2
            or spec.get('arms')!=list(ARMS) or spec.get('hidden')!=32
            or spec.get('batch_size')!=64 or spec.get('learning_rate')!=.001
            or spec.get('weight_decay')!=.0001 or spec.get('numeric_mode')!='FP32_TF32_OFF'
            or spec.get('sampler')!='SEEDED_GLOBAL_SHUFFLE'
            or not isinstance(spec.get('seeds'),list)or not spec['seeds']
            or any(type(s)is not int or s<0 for s in spec['seeds'])
            or len(set(spec['seeds']))!=len(spec['seeds'])
            or spec.get('primary_seed')!=spec['seeds'][0]
            or spec.get('data_roles')!=['TRAIN']):
        raise PermissionError('FRESH_FINITE_PROGRAM_WM_SCOPE_AND_ACTUAL_GPU_HEADROOM_REQUIRED')
    validate(spec)


def _finite(spec):
    if (time.time()>=spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists()
            or(Path(spec['campaign_root'])/'STOP.json').exists()):
        raise TimeoutError('FINITE_PROGRAM_WM_ENDPOINT_OR_ORIGINAL_STOP')


def run(spec,output):
    """Only the original admitted_run/worker may supply this execution context."""
    import resource
    from go2arx5_vla.world_model.fusion_program_source_records_v1 import load_sources
    from go2arx5_vla.world_model import fusion_program_response_data_v1 as contract
    from go2arx5_vla.world_model.storage import write_json,digest
    from go2arx5_vla.world_model.pilot1_storage import local_path
    from scripts.wm02.train_fusion_response import record_status
    output=Path(output);start=time.monotonic();cpu=time.process_time();main_updates=probes=0
    for key in ('source_manifest','source_root','registered_root','data','cache','tmp','logs'):
        local_path(spec,spec[key])
    _finite(spec)
    data,meta=load_sources(spec,required_role='TRAIN');splits=contract.validate(data);train=splits['TRAIN']
    if not len(train):raise ValueError('NO_OBSERVED_FRESH_TRAIN_LABELS_NO_MODEL_CREATED')
    replay=output/'PROGRAM_REPLAY.private.npz'
    with replay.open('xb')as stream:np.savez_compressed(stream,**data);stream.flush();os.fsync(stream.fileno())
    replay_sha=digest(replay);write_json(output/'REPLAY_MANIFEST.private.json',dict(meta,replay_sha256=replay_sha))
    stats=contract.normalizers(data,train)
    baselines=contract.baseline_predictions(data,stats)
    write_json(output/'BASELINES.private.json',{name:contract.metrics(pred,data,train)for name,pred in baselines.items()})
    # CUDA is initialized only after source role, original worker/lease, proposal
    # and complete preview/clock joins have supplied actual TRAIN observations.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import random,torch
    from go2arx5_vla.world_model.fusion_program_response_v1 import ProgramResponse,update,save_checkpoint,load_checkpoint
    from go2arx5_vla.world_model.fusion_program_response_numpy_v1 import predict as numpy_predict
    from go2arx5_vla.world_model.actor_sampling import ExactSampler
    from go2arx5_vla.world_model.neural_training import rng_state,restore_rng,tree_error
    from go2arx5_vla.world_model.runtime import cuda_uuid
    device=torch.device('cuda')
    if torch.cuda.device_count()!=1 or cuda_uuid(torch.cuda.get_device_properties(0).uuid)!=spec['GPU_uuid']:
        raise PermissionError('ONE_ACTUAL_REGISTERED_PROGRAM_TRAINING_GPU')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(len(spec['cpus']))
    x=torch.as_tensor(data['X'],dtype=torch.float32,device=device)
    y=torch.as_tensor(data['Y'],dtype=torch.float32,device=device)
    valid=torch.as_tensor(data['valid'],device=device);results=[]
    try:
        for seed in spec['seeds']:
            for arm in spec['arms']:
                _finite(spec);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                model=ProgramResponse(arm,32).to(device);model.set_normalizers(stats)
                optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
                sampler=ExactSampler(len(train),spec['sampler'],seed);folder=output/(str(seed)+'-'+arm);folder.mkdir()
                provenance=dict(source_sha=spec['source_sha'],source_manifest_sha256=spec['source_manifest_sha256'],
                    replay_sha256=replay_sha,arm=arm,seed=seed,endpoint=spec['endpoint'],recipe=RECIPE,
                    supported_horizons=stats['supported_horizons'].tolist(),numeric_mode=spec['numeric_mode'])
                initial=float(model.loss(x[train],y[train],valid[train]).detach())
                with (folder/'OPTIMIZER_STEPS.private.jsonl').open('x')as log:
                    for step in range(1,spec['endpoint']+1):
                        if step%25==1:_finite(spec)
                        index=torch.as_tensor(train[sampler.take(spec['batch_size'])],device=device)
                        loss,grad=update(model,optimizer,x,y,valid,index);main_updates+=1
                        log.write(json.dumps(dict(update=step,loss=loss,gradient_norm=grad))+'\n')
                        if step in(spec['endpoint']-1,spec['endpoint']):
                            save_checkpoint(folder/('checkpoint-%d.pt'%step),model,optimizer,sampler,step,provenance)
                        if step%100==0:record_status(output,progress=dict(seed=seed,arm=arm,updates=step,total_main_updates=main_updates))
                state=rng_state();order=sampler.state_dict()
                clone,co,cs,_=load_checkpoint(folder/('checkpoint-%d.pt'%(spec['endpoint']-1)),device,provenance)
                index=torch.as_tensor(train[cs.take(spec['batch_size'])],device=device)
                update(clone,co,x,y,valid,index);probes+=1
                restore_error=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(optimizer.state_dict(),co.state_dict()),
                    tree_error(order,cs.state_dict()),tree_error(state,rng_state()))
                endpoint=folder/('checkpoint-%d.pt'%spec['endpoint'])
                reload,ro,rs,_=load_checkpoint(endpoint,device,provenance)
                reload_error=max(tree_error(model.state_dict(),reload.state_dict()),tree_error(optimizer.state_dict(),ro.state_dict()),tree_error(order,rs.state_dict()))
                if restore_error!=0 or reload_error!=0:raise ValueError('PROGRAM_COMPLETE_OPTIMIZER_SAMPLER_RNG_RESTORE_DIFFERS')
                restore_rng(state)
                with torch.no_grad():prediction=model(x).cpu().numpy()
                arrays=model.arrays();portable=folder/'PROGRAM_RESPONSE_NUMPY.npz'
                with portable.open('xb')as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
                port=numpy_predict(arrays,data['X']);error=float(np.max(np.abs(port-prediction)))
                if error>2e-7:raise ValueError('PROGRAM_NUMPY_EXPORT_NUMERICS_DIFFER')
                with (folder/'PREDICTIONS.private.npz').open('xb')as stream:np.savez_compressed(stream,prediction=prediction,truth=data['Y'],valid=data['valid'])
                result=dict(seed=seed,arm=arm,main_WM_updates=spec['endpoint'],restore_probe_optimizer_calls=1,
                    checkpoint_sha256=digest(endpoint),numpy_sha256=digest(portable),full_restore_error=restore_error,
                    endpoint_reload_error=reload_error,numpy_max_absolute_error=error,initial_training_loss=initial,
                    final_training_loss=float(model.loss(x[train],y[train],valid[train]).detach()),
                    TRAIN_metrics=contract.metrics(prediction,data,train),normalizers_fit_only_TRAIN=True,
                    future_H40_removed=arm=='NO_FUTURE_ACTION',same_capacity_and_sampler=True,
                    TRAIN_exposures_min=int(sampler.exposure_counts().min()),TRAIN_exposures_max=int(sampler.exposure_counts().max()))
                results.append(result);write_json(folder/'RESULT.private.json',result);record_status(output,completed=results)
                del model,optimizer,clone,co,reload,ro
        _finite(spec)
        result=dict(schema='WM02_PROGRAM_RESPONSE_TRAINING_RESULT_V1',status='REAL_TRAIN_OPTIMIZED_FULLY_RESTORED_AND_PORTABLE',
            source_sha=spec['source_sha'],source_manifest_sha256=spec['source_manifest_sha256'],replay_sha256=replay_sha,
            data=meta['summary'],results=results,primary_seed=spec['primary_seed'],main_WM_updates=main_updates,
            restore_probe_optimizer_calls=probes,Actor_updates=0,GR00T_updates=0,native_integrations=0,DEV_read=False,TEST_read=False,
            output='LOCAL_TCP_TRANSLATION_ROTATION_VECTOR_AND_TWO_FINGER_DISPLACEMENTS_H16_NOT_COMPLETE_RECURRENT_STATE',
            supported_horizons=stats['supported_horizons'].tolist(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),CPU_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,physical_control_gain=False,task_transfer_gain=False)
        write_json(output/'RESULT.private.json',result);return result
    except BaseException as error:
        write_json(output/'FAILED.private.json',dict(schema='WM02_PROGRAM_RESPONSE_TRAINING_FAILURE_V1',failure=type(error).__name__+': '+str(error),
            actual_main_WM_updates=main_updates,restore_probe_optimizer_calls=probes,completed_models=results,
            source_manifest_sha256=spec['source_manifest_sha256'],replay_sha256=replay_sha,wall_s=time.monotonic()-start))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec,args.output.name,PROFILE,args.output,gpu=True)as spec:
        run(spec,args.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
