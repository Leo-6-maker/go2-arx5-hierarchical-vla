"""Finite GPU optimization diagnostic using the exact ridge ARX causal design."""
def run(spec,out):
    from pathlib import Path
    import time,copy,random
    import numpy as np,torch
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.neural_training import rng_state,restore_rng,tree_error,memory
    from go2arx5_vla.world_model.storage import digest,write_json
    from scripts.wm02.audit_next2_models import score
    start=time.monotonic();cpu=time.process_time();out=Path(out);data,meta=load(spec['replay']);ids=select_samples(data,meta,'TRAIN');dev=select_samples(data,meta,'DEV');arx=CausalARX(meta).fit(data,ids)
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('PINNED_REPLAY_CHANGED')
    random.seed(20261105);np.random.seed(20261105);torch.manual_seed(20261105);torch.use_deterministic_algorithms(True);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
    st=data['state'][ids];origin=data['window_start_s'][ids];age=data['source_age'][ids,8];remaining=arx.remaining(data,ids)
    x=np.concatenate([arx.design(arx.feature(st[:,8+j],st[:,7+j],data['goal'][ids],origin,age,remaining,j))for j in range(16)]);y=np.concatenate([(st[:,9+j]-st[:,8+j])/arx.scale for j in range(16)])
    X=torch.tensor(x,device=device,dtype=torch.float64);Y=torch.tensor(y,device=device,dtype=torch.float64);w=torch.nn.Parameter(torch.zeros(x.shape[1],y.shape[1],device=device,dtype=torch.float64));opt=torch.optim.Adam([w],lr=.01);logs=[]
    def objective():return ((X@w-Y)**2).sum()/len(X)+(w[:-1]**2).sum()/len(X)
    def update():
        opt.zero_grad(set_to_none=True);loss=objective();loss.backward();g=float(w.grad.norm());opt.step();return float(loss.detach()),g
    optimum=float(np.mean(np.sum((x@arx.weight-y)**2,axis=1))+(arx.weight[:-1]**2).sum()/len(x));rule=dict(initial=200,conditional_maximum=800,criterion='At200 TRAIN penalized loss exceeds1.1*closed-form optimum and last50 relative loss improvement>10%; no DEV use',TRAIN_only=True,seed=20261105,recipe='ZERO_INITIALIZED_LINEAR_TORCH_EXACT_ARX_FEATURES_FULL_BATCH_ADAM_LR0.01_RIDGE1',output106=True)
    write_json(out/'PREREGISTERED_RECIPE.private.json',rule);endpoints=[];stop=200
    for step in range(1,801):
        loss,g=update();logs.append(dict(step=step,TRAIN_penalized_loss=loss,gradient_norm=g))
        if step not in(200,800):continue
        torch.cuda.synchronize();normalizers=arx.arrays();checkpoint=out/f'checkpoint-{step}.pt';torch.save(dict(schema='WM02_NEXT2_MATCHED_FEATURE_CHECKPOINT_V1',weight=w.detach(),optimizer=opt.state_dict(),rng=rng_state(),cursor=step*len(x),step=step,normalizers={k:torch.tensor(v)for k,v in normalizers.items()if k!='weight'},source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],recipe=rule),checkpoint)
        saved=torch.load(checkpoint,map_location=device,weights_only=True);before=w.detach().clone();reload_error=tree_error(before,saved['weight']);optimizer_before=copy.deepcopy(opt.state_dict());rng_before=rng_state();update();expected=w.detach().clone();expected_opt=copy.deepcopy(opt.state_dict());w.data.copy_(saved['weight']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);update();continuation=max(tree_error(expected,w),tree_error(expected_opt,opt.state_dict()));w.data.copy_(before);opt.load_state_dict(optimizer_before);restore_rng(rng_before)
        if reload_error or continuation:raise ValueError('EXACT_OPTIMIZER_RNG_RESTORE_FAILED')
        endpoints.append(dict(main_updates=step,checkpoint_sha256=digest(checkpoint),reload_error=reload_error,continuation_error=continuation,actual_restore_probe_optimizer_calls=2,TRAIN_objective=float(objective().detach()),closed_form_optimum=optimum))
        if step==200:
            improve=(logs[149]['TRAIN_penalized_loss']-logs[-1]['TRAIN_penalized_loss'])/max(logs[149]['TRAIN_penalized_loss'],1e-30);extend=loss>1.1*optimum and improve>.1;write_json(out/'TRAIN_EXTENSION_DECISION.private.json',dict(extend=extend,relative_improvement_last50=improve,TRAIN_loss=loss,optimum=optimum,DEV_opened=False));stop=800 if extend else 200
        if step>=stop:break
    arx.weight=w.detach().cpu().numpy().copy();evaluations={};parents=np.asarray([meta['episodes'][int(i)]['parent']for i in data['episode_index']])
    for split,ix in [('TRAIN',ids),('DEV_ALREADY_OPENED',dev)]:
        evaluations[split]={}
        for parent in sorted(set(parents[ix])):
            chosen=ix[parents[ix]==parent];evaluations[split][parent]=score(arx.predict(data,chosen),data,meta,chosen)
    all_ids=np.arange(len(data['state']))
    with(out/'PREDICTIONS_FK_AUDIT.npz').open('xb')as f:np.savez_compressed(f,NN=arx.predict(data,all_ids),ARX=CausalARX(meta).fit(data,ids).predict(data,all_ids))
    result=dict(status='MATCHED_CAUSAL_FEATURE_OPTIMIZATION_DIAGNOSTIC_EXECUTED',source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],main_updates=stop,continuation_probe_updates=2*len(endpoints),actual_optimizer_calls=stop+2*len(endpoints),endpoints=endpoints,evaluation=evaluations,main_update_peak=memory(device),GPU_uuid=spec['GPU_uuid'],CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,physics_integrations=0,Actor_updates=0,diagnostic_only_not_new_prospective_candidate=True,architecture='Linear torch neural layer, same ARX features/one-step106D targets and ridge objective; isolates finite optimizer from closed-form solver')
    write_json(out/'TRAIN_LOG.private.json',logs);write_json(out/'RESULT.json',result);return result

def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'wm02-auto1-target-neural',a.output,gpu=True)as s:
        r=run(s,a.output);print(json.dumps({k:r[k]for k in ['status','main_updates','continuation_probe_updates','CPU_s','wall_s']}));return 0
if __name__=='__main__':raise SystemExit(main())
