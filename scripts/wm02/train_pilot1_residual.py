"""Finite, registered TCP residual experiment; no native physics or Actor."""
import argparse
import json
from pathlib import Path


def run(spec, output):
    import random,time,resource
    import numpy as np
    import torch
    from go2arx5_vla.world_model.pilot1_residual import TCPResidual,inputs,loss,errors,response_errors
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    from go2arx5_vla.world_model.neural_training import rng_state,restore_rng,tree_error,memory
    from go2arx5_vla.world_model.runtime import cuda_uuid
    from go2arx5_vla.world_model.storage import digest,write_json
    started,cpu=time.monotonic(),time.process_time();device=torch.device(spec['device'])
    if device.type=='cuda':
        props=torch.cuda.get_device_properties(0)
        if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('ONE_REGISTERED_GPU_REQUIRED')
        torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory)
        torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True)
    for key in ('replay','ARX'):
        if digest(spec[key])!=spec[key+'_sha256']:raise ValueError('ACTUAL_FROZEN_RESIDUAL_INPUT_CHANGED:'+key)
    if [s['horizon']for s in spec['curriculum']]!=[1,4,8,16]or sum(s['updates']for s in spec['curriculum'])!=spec['endpoint']:
        raise ValueError('PREDECLARED_FREE_RESIDUAL_CURRICULUM_REQUIRED')
    data,meta=load(spec['replay']);train=select_samples(data,meta,'TRAIN');dev=select_samples(data,meta,'DEV')
    with np.load(spec['ARX'],allow_pickle=False)as z:teacher=CausalARX.from_arrays(meta,z)
    prepared={name:inputs(teacher,data,meta,blind_choice=(name=='NO_FUTURE_CHOICE'))for name in spec['arms']}
    held=inputs(teacher,data,meta,force_hold=True);sl=teacher.layout['tcp_position_m'];truth=data['state'][:,9:,sl]
    def tensor(v):return torch.as_tensor(v,dtype=torch.float32,device=device)
    hold_tensor={k:tensor(held[k])for k in ('x','base','observed')};truth_tensor=tensor(truth)
    normalizer_x=tensor(prepared['ACTION_CONDITIONED']['x'][train]);results={};global_updates=0
    baselines=dict(ARX=prepared['ACTION_CONDITIONED']['base'],HOLD_CURRENT=np.repeat(data['state'][:,8:9,sl],16,axis=1),
                   CONSTANT_VELOCITY=data['state'][:,8:9,sl]+np.arange(1,17)[None,:,None]*(data['state'][:,8:9,sl]-data['state'][:,7:8,sl]))
    actual_offsets=data['goal'][:,:3]-data['history_goal'][:,0,:3]
    legal=np.asarray([[0.,0.,0.],[.001,0.,0.],[-.001,0.,0.],[0.,.001,0.]])
    matches=np.all(np.isclose(actual_offsets[:,None],legal[None],atol=1e-8,rtol=0),axis=-1)
    if not np.all(matches.sum(-1)==1):raise ValueError('ONLY_QUALIFIED_FOUR_ACTION_REPLAY_SUPPORTED')
    action_ids=matches.argmax(-1)
    mean_curves=np.stack([(truth[train[action_ids[train]==a]]-data['state'][train[action_ids[train]==a],8:9,sl]).mean(0)for a in range(4)])
    if not np.isfinite(mean_curves).all():raise ValueError('ALL_FOUR_TRAIN_MEAN_RESPONSE_CURVES_REQUIRED')
    baselines['TRAIN_MEAN_DISPLACEMENT']=data['state'][:,8:9,sl]+mean_curves[action_ids]
    write_json(output/'BASELINES.json',{name:{split:errors(pred,truth,ids)for split,ids in [('TRAIN',train),('DEV',dev)]}for name,pred in baselines.items()})
    for arm in spec['arms']:
        directory=output/arm;directory.mkdir();random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed'])
        model=TCPResidual(normalizer_x.shape[1],spec['hidden']).to(device);model.fit_normalizer(normalizer_x)
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=1e-4)
        value=prepared[arm];tensors={k:tensor(value[k])for k in ('x','base','observed')};cursor=0;steps=0;last_loss=None;stages=[]
        with torch.no_grad():initial_error=float(torch.max(torch.abs(model(**tensors)-tensors['base'])))
        if initial_error!=0:raise ValueError('ZERO_RESIDUAL_MUST_EXACTLY_EQUAL_ARX')
        def update(m,opt,at,horizon):
            ids=torch.as_tensor(train[(np.arange(at,at+spec['batch_size'])%len(train))],device=device)
            selected={k:v[ids]for k,v in tensors.items()};pred=m(**selected,horizon=horizon)
            objective=loss(pred,truth_tensor[ids,:horizon],selected['base'][:,:horizon])
            opt.zero_grad(set_to_none=True);objective.backward();grad=torch.sqrt(sum(p.grad.square().sum()for p in m.parameters()))
            if not torch.isfinite(objective)or not torch.isfinite(grad):raise FloatingPointError('NONFINITE_RESIDUAL_NO_CLIPPING_OR_RETRY')
            opt.step();return at+spec['batch_size'],float(objective.detach()),float(grad)
        def save(step,horizon):
            torch.save(dict(schema='WM02_PILOT1_TCP_RESIDUAL_CHECKPOINT_V1',source_sha=spec['source_sha'],arm=arm,
                model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),cursor=cursor,updates=step,
                curriculum_horizon=horizon,replay_sha256=spec['replay_sha256'],ARX_sha256=spec['ARX_sha256'],
                metadata=meta,output='WORLD_TCP_XYZ_ONLY_NOT_FULL_STATE',rotations_predicted=False),directory/f'checkpoint-{step}.pt')
        save(0,1)
        for stage in spec['curriculum']:
            for _ in range(stage['updates']):
                if time.time()>=spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('RESIDUAL_FINITE_WORKER_OR_STOP')
                cursor,last_loss,gradient=update(model,optimizer,cursor,stage['horizon']);steps+=1;global_updates+=1
                with (directory/'OPTIMIZER_STEPS.jsonl').open('a')as f:f.write(json.dumps(dict(update=steps,horizon=stage['horizon'],loss=last_loss,gradient_norm=gradient))+'\n')
                if steps in (spec['endpoint']-1,spec['endpoint']):save(steps,stage['horizon'])
            save(steps,stage['horizon'])if steps!=spec['endpoint']else None
            with torch.no_grad():pred=model(**tensors).cpu().numpy()
            stages.append(dict(updates=steps,horizon=stage['horizon'],TRAIN=errors(pred,truth,train),DEV=errors(pred,truth,dev),max_residual_norm_mm=float(np.max(np.linalg.norm(pred-value['base'],axis=-1)))*1000))
        rng=rng_state();clone=TCPResidual(normalizer_x.shape[1],spec['hidden']).to(device);co=torch.optim.AdamW(clone.parameters(),lr=spec['learning_rate'],weight_decay=1e-4)
        last=torch.load(directory/f"checkpoint-{steps-1}.pt",map_location='cpu',weights_only=True);clone.load_state_dict(last['model']);co.load_state_dict(last['optimizer']);restore_rng(last['rng'])
        recovered_cursor,_,_=update(clone,co,last['cursor'],16)
        restore_error=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(optimizer.state_dict(),co.state_dict()),tree_error(rng,rng_state()))
        endpoint=torch.load(directory/f'checkpoint-{steps}.pt',map_location='cpu',weights_only=True)
        reload_error=max(tree_error(model.state_dict(),endpoint['model']),tree_error(optimizer.state_dict(),endpoint['optimizer']))
        if restore_error!=0 or reload_error!=0 or recovered_cursor!=cursor:raise ValueError('RESIDUAL_COMPLETE_RESTORE_DIFFERS')
        with torch.no_grad():
            prediction=model(**tensors).cpu().numpy();hold_prediction=model(**hold_tensor).cpu().numpy()
            # Finite prefix-input sensitivity. This is neither native Jacobian nor proof of asymptotic stability.
            perturbed=dict(data);perturbed['state']=data['state'].copy();perturbed['state'][:,:9,sl.start]+=.000001
            changed=inputs(teacher,perturbed,meta,blind_choice=(arm=='NO_FUTURE_CHOICE'))
            changed={k:tensor(changed[k])for k in ('x','base','observed')}
            sensitivity=float(torch.max(torch.linalg.vector_norm(model(**changed)[:,-1]-model(**tensors)[:,-1],dim=-1))/.000001)
        if not np.isfinite(prediction).all():raise FloatingPointError('NONFINITE_FREE_ROLLOUT')
        checkpoint=directory/f'checkpoint-{steps}.pt'
        results[arm]=dict(checkpoint_sha256=digest(checkpoint),main_WM_updates=steps,restore_probe_optimizer_calls=1,
            full_restore_error=restore_error,endpoint_reload_error=reload_error,zero_initial_output_error=initial_error,stages=stages,
            absolute={split:errors(prediction,truth,ids)for split,ids in [('TRAIN',train),('DEV',dev)]},
            action_effect={split:response_errors(prediction,hold_prediction,data,meta,ids,teacher.layout)for split,ids in [('TRAIN',train),('DEV',dev)]},
            prefix_TCP_observation_only_probe_1um_amplification=sensitivity,native_physical_gradient_validated=False,
            finite_16_step_only=True,all_outputs_finite=True,no_hard_clipping=True)
        np.savez_compressed(directory/'PREDICTIONS.private.npz',prediction=prediction,hold_prediction=hold_prediction,truth=truth,
                            full_ARX_forecast=value['full_ARX_forecast'])
        write_json(directory/'RESULT.json',results[arm])
    fk_audit=None
    if spec.get('FK_model'):
        import mujoco
        from go2arx5_vla.world_model.native_replay import forbid_integration
        if mujoco.__version__!='3.2.7'or digest(spec['FK_model'])!=spec['FK_model_sha256']:raise ValueError('FROZEN_FK_AUDIT_MODEL_REQUIRED')
        model=mujoco.MjModel.from_binary_path(spec['FK_model']);state=mujoco.MjData(model);site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'x5_ee');fk_audit={}
        with forbid_integration(mujoco)as calls:
            for arm in spec['arms']:
                with np.load(output/arm/'PREDICTIONS.private.npz',allow_pickle=False)as z:
                    raw=z['full_ARX_forecast'];pred=z['prediction'];fk=np.empty_like(pred)
                for i in range(len(raw)):
                    for h in range(16):
                        state.qpos[:]=raw[i,h,teacher.layout['qpos']];mujoco.mj_normalizeQuat(model,state.qpos);mujoco.mj_kinematics(model,state);fk[i,h]=state.site_xpos[site]
                fk_audit[arm]={split:dict(ARX_TCP_vs_ARX_qpos_FK_RMSE_mm=float(np.sqrt(np.mean((raw[ids,:,sl]-fk[ids])**2)))*1000,
                    residual_TCP_vs_ARX_qpos_FK_RMSE_mm=float(np.sqrt(np.mean((pred[ids]-fk[ids])**2)))*1000,
                    qpos_source='UNCHANGED_FROZEN_ARX_AUDIT_ONLY_NOT_RESIDUAL_JOINT_PREDICTIONS')for split,ids in [('TRAIN',train),('DEV',dev)]}
        fk_audit.update(new_integrations=0,forbidden_integration_attempts=len(calls),scene_sha256=spec['FK_model_sha256'])
        write_json(output/'FK_AUDIT.json',fk_audit)
    result=dict(status='LOCAL_TCP_RESIDUAL_TRAINED_AND_RELOADABLE',source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],ARX_sha256=spec['ARX_sha256'],
        output='WORLD_TCP_XYZ_ONLY_NOT_FULL_ROBOT_STATE',TCP_qpos_FK_consistency='REQUIRES_SEPARATE_FK_AUDIT_NOT_ENFORCED_BY_COMPACT_HEAD',
        arms=results,FK_audit=fk_audit,main_WM_updates=global_updates,restore_probe_optimizer_calls=len(results),Actor_updates=0,
        native_integrations=0,contact_positive_training_samples=0,new_TEST_used=False,device=str(device),
        GPU_peak=memory(device)if device.type=='cuda'else None,CPU_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-started)
    write_json(output/'RESULT.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv);raw=json.loads(args.spec.read_text());cpu=raw.get('execution_site')=='LOCAL_PC_PILOT1_CPU'
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec,args.output.name,'wm02-pilot1-offline-cpu'if cpu else'wm02-pilot1-neural',args.output,gpu=not cpu)as spec:
        print(json.dumps(run(spec,args.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
