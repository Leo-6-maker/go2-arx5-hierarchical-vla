"""Train a finite matched response study, restore exactly and export NumPy models."""
import argparse,json
from pathlib import Path


def record_status(output,progress=None,completed=None):
    """Mutable observer snapshots use atomic replacement, unlike sealed artifacts."""
    from scripts.pi0292.durable import write
    if progress is not None:write(output/'PROGRESS.private.json',progress)
    if completed is not None:write(output/'COMPLETED.private.json',completed)


def run(spec,output):
    import random,time,resource
    import numpy as np,torch
    from go2arx5_vla.world_model.fusion_response import ResponseModel,ARMS,prepare,validate_data,action_effect
    from go2arx5_vla.world_model.fusion_response_numpy import predict as numpy_predict
    from go2arx5_vla.world_model.pilot1_residual import errors,inputs
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.actor_sampling import ExactSampler
    from go2arx5_vla.world_model.neural_training import rng_state,restore_rng,tree_error,memory
    from go2arx5_vla.world_model.runtime import cuda_uuid
    from go2arx5_vla.world_model.storage import digest,write_json
    start=time.monotonic();cpu=time.process_time();device=torch.device(spec['device'])
    if (spec['arms']!=list(ARMS) or spec['seeds']!=[366301,366302,366303]
            or spec['primary_seed']!=366301 or spec['endpoint']!=2000 or spec['batch_size']!=32
            or spec['sampler']!='SEEDED_GLOBAL_SHUFFLE' or spec['hidden']!=32
            or spec['numeric_mode']!='FP32_TF32_OFF' or spec['learning_rate']!=.0003):
        raise ValueError('PREDECLARED_MATCHED_LOCAL_RESPONSE_STUDY_REQUIRED')
    if device.type!='cuda' or torch.cuda.device_count()!=1 or cuda_uuid(torch.cuda.get_device_properties(0).uuid)!=spec['GPU_uuid']:
        raise PermissionError('ONE_REGISTERED_LOCAL_CUDA_DEVICE_REQUIRED')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(len(spec['cpus']))
    for key in ('replay','ARX','teacher_metadata','task'):
        if digest(spec[key])!=spec[key+'_sha256']:raise ValueError('SEALED_TRAINING_INPUT_CHANGED:'+key)
    with np.load(spec['replay'],allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    splits=validate_data(data);train,dev=splits['TRAIN'],splits['DEV'];meta=json.loads(Path(spec['teacher_metadata']).read_text())
    with np.load(spec['ARX'],allow_pickle=False)as z:teacher=CausalARX.from_arrays(meta,z)
    sl=teacher.layout['tcp_position_m'];truth=data['state'][:,9:,sl]
    prepared={arm:prepare(teacher,data,meta,arm) for arm in ARMS};held=inputs(teacher,data,meta,force_hold=True)
    curves=np.stack([(truth[train[data['observed_action'][train]==a]]-data['state'][train[data['observed_action'][train]==a],8:9,sl]).mean(0)for a in range(4)])
    static_path=output/'TRAIN_MEAN_RESPONSE_CURVES.npz';np.savez_compressed(static_path,curves_m=curves,fit_split=np.array('TRAIN'))
    from go2arx5_vla.world_model.pilot2_mean_rule import fit
    records=[dict(parent=str(data['parent'][i]),split='TRAIN',run=str(data['run_id'][i]),
        action=int(data['observed_action'][i]),repeat=int(data['repeat_index'][i]),
        response_xy_m=(truth[i,-1,:2]-data['state'][i,8,sl][:2]).tolist())for i in train]
    static_rule=output/'TRAIN_MEAN_RESPONSE_RULE.json'
    write_json(static_rule,fit(records,sorted(set(data['parent'][train])),spec['task_sha256']))
    baselines=dict(FROZEN_C08_ARX=prepared[ARMS[0]]['base'],TRAIN_MEAN_RESPONSE=data['state'][:,8:9,sl]+curves[data['observed_action']],
        HOLD_CURRENT=np.repeat(data['state'][:,8:9,sl],16,axis=1),
        CONSTANT_VELOCITY=data['state'][:,8:9,sl]+np.arange(1,17)[None,:,None]*(data['state'][:,8:9,sl]-data['state'][:,7:8,sl]))
    baseline_metrics={name:{s:errors(p,truth,ids)for s,ids in splits.items()}for name,p in baselines.items()}
    write_json(output/'BASELINES.private.json',baseline_metrics)
    def tensor(v):return torch.as_tensor(v,dtype=torch.float32,device=device)
    target=tensor(truth);held_t={k:tensor(held[k])for k in ('x','base')};results=[];main_updates=0;probe_updates=0
    for seed in spec['seeds']:
        for arm in ARMS:
            d=output/(str(seed)+'-'+arm);d.mkdir()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            value=prepared[arm];t={k:tensor(value[k])for k in ('x','base')}
            model=ResponseModel(t['x'].shape[1],32).to(device);model.fit_normalizer(t['x'][train])
            opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=.0001)
            sampler=ExactSampler(len(train),spec['sampler'],seed)
            with torch.no_grad():initial=float((model(**t)-t['base']).abs().max())
            if initial!=0:raise ValueError('INITIAL_RESPONSE_MUST_EXACTLY_EQUAL_OWN_ARX_BASE')
            def update(m,o,s):
                ids=torch.as_tensor(train[s.take(spec['batch_size'])],device=device)
                base=t['base'][ids];pred=m(t['x'][ids],base)
                loss=(((pred-target[ids])/.001)**2).mean()+.01*(((pred-base)/.001)**2).mean()
                o.zero_grad(set_to_none=True);loss.backward()
                grad=torch.sqrt(sum(p.grad.square().sum()for p in m.parameters()))
                if not torch.isfinite(loss)or not torch.isfinite(grad):raise FloatingPointError('NONFINITE_RESPONSE_NO_CLIPPING_OR_RETRY')
                o.step();return float(loss.detach()),float(grad.detach())
            def save(step):
                torch.save(dict(schema='WM02_FUSION_RESPONSE_CHECKPOINT_V1',source_sha=spec['source_sha'],seed=seed,arm=arm,
                    model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state_dict(),rng=rng_state(),updates=step,
                    replay_sha256=spec['replay_sha256'],ARX_sha256=spec['ARX_sha256'],metadata=meta,
                    model_config=dict(dimension=t['x'].shape[1],hidden=32),output='WORLD_TCP_XYZ_H16_ONLY'),d/('checkpoint-%d.pt'%step))
            log=d/'OPTIMIZER_STEPS.jsonl'
            with log.open('x')as f:
                for step in range(1,spec['endpoint']+1):
                    if time.time()>=spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('FINITE_RESPONSE_ENDPOINT_OR_STOP')
                    loss,grad=update(model,opt,sampler);main_updates+=1
                    f.write(json.dumps(dict(update=step,loss=loss,gradient_norm=grad))+'\n')
                    if step in (1999,2000):save(step)
                    if step%100==0:record_status(output,progress=dict(seed=seed,arm=arm,updates=step,total_main_updates=main_updates))
            actual_rng=rng_state();actual_sampler=sampler.state_dict()
            clone=ResponseModel(t['x'].shape[1],32).to(device);co=torch.optim.AdamW(clone.parameters(),lr=spec['learning_rate'],weight_decay=.0001)
            cs=ExactSampler(len(train),spec['sampler'],seed);last=torch.load(d/'checkpoint-1999.pt',map_location='cpu',weights_only=True)
            clone.load_state_dict(last['model']);co.load_state_dict(last['optimizer']);cs.load_state_dict(last['sampler']);restore_rng(last['rng'])
            update(clone,co,cs);probe_updates+=1
            restore_error=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()),
                tree_error(actual_sampler,cs.state_dict()),tree_error(actual_rng,rng_state()))
            endpoint=d/'checkpoint-2000.pt';loaded=torch.load(endpoint,map_location='cpu',weights_only=True)
            reload_error=max(tree_error(model.state_dict(),loaded['model']),tree_error(opt.state_dict(),loaded['optimizer']),
                tree_error(actual_sampler,loaded['sampler']))
            if restore_error!=0 or reload_error!=0:raise ValueError('COMPLETE_RESPONSE_RESTORE_DIFFERS')
            with torch.no_grad():prediction=model(**t).cpu().numpy();hold_prediction=model(**held_t).cpu().numpy()
            portable=d/'RESPONSE_NUMPY.npz';np.savez_compressed(portable,**model.arrays())
            with np.load(portable,allow_pickle=False)as arrays:port=numpy_predict(arrays,value['x'],value['base'])
            portability_error=float(np.max(np.abs(port-prediction)))
            if portability_error>2e-7:raise ValueError('NUMPY_NATIVE_EXPORT_DIFFERS')
            np.savez_compressed(d/'PREDICTIONS.private.npz',prediction=prediction,hold_prediction=hold_prediction,truth=truth)
            r=dict(seed=seed,arm=arm,main_WM_updates=2000,restore_probe_optimizer_calls=1,checkpoint_sha256=digest(endpoint),
                numpy_sha256=digest(portable),full_restore_error=restore_error,endpoint_reload_error=reload_error,
                numpy_max_error_m=portability_error,zero_initial_residual_error=initial,
                TRAIN_exposures_min=int(sampler.exposure_counts().min()),TRAIN_exposures_max=int(sampler.exposure_counts().max()),
                normalization_split='OWN_ARM_TRAIN_ONLY',candidate_removed_from_both_features_and_ARX_base=arm==ARMS[1],
                absolute={s:errors(prediction,truth,ids)for s,ids in splits.items()},
                action_effect={s:action_effect(prediction,hold_prediction,data,ids,teacher.layout)for s,ids in splits.items()})
            results.append(r);write_json(d/'RESULT.private.json',r);record_status(output,completed=results)
    result=dict(schema='WM02_FUSION_MATCHED_RESPONSE_STUDY_RESULT_V1',status='TRAINED_EXACTLY_RESTORED_AND_NUMPY_EXPORTED',
        source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],ARX_sha256=spec['ARX_sha256'],results=results,
        primary_seed=spec['primary_seed'],primary_chosen_before_DEV=True,main_WM_updates=main_updates,restore_probe_optimizer_calls=probe_updates,
        Actor_updates=0,GR00T_updates=0,native_integrations=0,TEST_read=False,output='TCP_XYZ_H16_ONLY_NOT_COMPLETE_STATE',
        baseline_metrics=baseline_metrics,TRAIN_mean_curves_sha256=digest(static_path),
        TRAIN_mean_response_rule_sha256=digest(static_rule),
        actual_TRAIN_windows=len(train),actual_DEV_windows=len(dev),GPU_peak=memory(device),
        CPU_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,
        CALIBRATION_CORRECTION_GAIN='REQUIRES_FROZEN_NATIVE_REEVALUATION',TASK_TRANSFER_GAIN=False)
    write_json(output/'RESULT.private.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.output.name,'wm02-fusion-response-train',a.output,gpu=True)as spec:
        print(json.dumps(run(spec,a.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
