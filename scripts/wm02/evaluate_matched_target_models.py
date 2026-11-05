"""Frozen matched model/linear-control evaluation, including actual action deltas."""
def delta_metrics(pred,data,meta,split):
    import numpy as np
    from go2arx5_vla.world_model.target_training import slices
    tcp=slices(meta)['tcp_position_m'];rows=[]
    for parent in sorted({e['parent']for e in meta['episodes']if e['split']==split}):
        ids=[i for i,e in enumerate(meta['episodes'])if e['parent']==parent and e['split']==split]
        offset=data['goal'][ids,:3]-data['history_goal'][ids,0,:3]
        holds=[i for i,d in zip(ids,offset)if np.linalg.norm(d)<1e-8]
        if not holds:raise ValueError('ACTUAL_SAME_PARENT_HOLD_REQUIRED')
        for h in [1,8,16]:
            guessed=pred[:,h-1,tcp]-data['state'][:,8,tcp];real=data['state'][:,8+h,tcp]-data['state'][:,8,tcp]
            ph=guessed[holds].mean(0);rh=real[holds].mean(0)
            for i,d in zip(ids,offset):
                if np.linalg.norm(d)<1e-8:continue
                axis=int(np.argmax(np.abs(d)));sign=float(np.sign(d[axis]));p=guessed[i]-ph;t=real[i]-rh
                rows.append(dict(parent=parent,episode=meta['episodes'][i]['episode'],horizon=h,offset_xyz_m=d.tolist(),predicted_delta_vs_HOLD_m=p.tolist(),actual_delta_vs_HOLD_m=t.tolist(),coordinate_squared_error_m2=((p-t)**2).tolist(),expected_direction_correct=bool(p[axis]*sign>0),observed_direction_correct=bool(p[axis]*t[axis]>0)))
    summaries={}
    for h in [1,8,16]:
        selected=[r for r in rows if r['horizon']==h]
        if not selected:continue
        summaries[str(h)]=dict(coordinate_RMSE_m=float(np.sqrt(np.asarray([r['coordinate_squared_error_m2']for r in selected]).mean())),observed_direction_fraction=float(np.mean([r['observed_direction_correct']for r in selected])),expected_direction_fraction=float(np.mean([r['expected_direction_correct']for r in selected])),by_parent={p:dict(coordinate_RMSE_m=float(np.sqrt(np.asarray([r['coordinate_squared_error_m2']for r in selected if r['parent']==p]).mean())),observed_direction_fraction=float(np.mean([r['observed_direction_correct']for r in selected if r['parent']==p])))for p in sorted({r['parent']for r in selected})})
    return dict(summary=summaries,matched_independent_restart_rows=rows,exact_same_state_counterfactual=False)


def run(spec):
    import json,time
    from pathlib import Path
    import numpy as np,torch
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    from go2arx5_vla.world_model.target_training import TargetWM,tensors,prediction_metrics,slices
    from go2arx5_vla.world_model.target_training_v2 import compatible
    from go2arx5_vla.world_model.target_arx import CausalARX
    start=time.monotonic();cpu=time.process_time();torch.set_num_threads(1)
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('FROZEN_EVALUATION_REPLAY_CHANGED')
    data,meta=load(spec['replay']);ids=np.arange(len(data['state']));inp=tensors(data,ids,'cpu');truth={k:torch.as_tensor(data[k],dtype=torch.float64)for k in inp};models={};mean=None;std=None
    def score(pred):
        result={}
        for split in ['TRAIN','DEV']:
            chosen=select_samples(data,meta,split)
            if not len(chosen):continue
            tensor=torch.as_tensor(pred,dtype=torch.float64);b={k:v[chosen]for k,v in truth.items()};by_parent={}
            for parent in sorted({meta['episodes'][i]['parent']for i in chosen}):
                ix=np.asarray([i for i in chosen if meta['episodes'][i]['parent']==parent]);by_parent[parent]=prediction_metrics(tensor[ix],{k:v[ix]for k,v in truth.items()},meta)
            result[split]=dict(windows=len(chosen),parents=len(by_parent),by_parent=by_parent,**prediction_metrics(tensor[chosen],b,meta),action_delta=delta_metrics(pred,data,meta,split))
        return result
    for arm,binding in spec['models'].items():
        if digest(binding['path'])!=binding['sha256']:raise ValueError('FROZEN_MATCHED_CHECKPOINT_CHANGED')
        c=torch.load(binding['path'],map_location='cpu',weights_only=True);compatible(c['metadata'],meta);model=TargetWM(c['metadata'],c['hidden']);model.load_state_dict(c['model']);model.eval()
        if mean is not None and(not torch.equal(mean,model.mean)or not torch.equal(std,model.std)):raise ValueError('MATCHED_MODEL_TRAIN_NORMALIZERS_DIFFER')
        mean=model.mean.clone();std=model.std.clone()
        with torch.no_grad():free=model.predict(**inp).numpy();teacher=model.predict(**inp,teacher_forced=True).numpy()
        goal=inp['goal'].detach().clone().requires_grad_(True);m=dict(inp,goal=goal);prediction=model.predict(**m);sl=slices(meta)['tcp_position_m'];jac=[]
        for h in [1,8,16]:
            entries=[torch.autograd.grad(prediction[:,h-1,sl.start+j].sum(),goal,retain_graph=True)[0][:,:3].detach().numpy()for j in range(3)];jac.append(dict(horizon=h,maximum_absolute_coordinate_m_per_m=float(np.abs(entries).max())))
        models[arm]=dict(checkpoint_sha256=binding['sha256'],FREE=score(free),TEACHER_FORCED_PRIVILEGED_DIAGNOSTIC=score(teacher),future_goal_Jacobian=jac,action_encoding=model.action_encoding)
    with np.load(spec['ARX']['path'],allow_pickle=False)as z:
        if digest(spec['ARX']['path'])!=spec['ARX']['sha256']:raise ValueError('FROZEN_ARX_CHANGED')
        arx=CausalARX.from_arrays(meta,z)
    with np.load(spec['RIDGE']['path'],allow_pickle=False)as z:
        if digest(spec['RIDGE']['path'])!=spec['RIDGE']['sha256']:raise ValueError('FROZEN_RIDGE_CHANGED')
        current=data['state'][:,8].copy();pred=[]
        for _ in range(16):
            feature=np.concatenate(((current-z['mean'])/z['scale'],data['goal'],np.ones((len(current),1))),-1);current=current+feature@z['weight']*z['scale'];pred.append(current.copy())
    controls=dict(ARX=score(arx.predict(data,ids)),RIDGE=score(np.stack(pred,1)))
    no=models['NO_ACTION']['FREE']['DEV']['action_delta']['summary'];gate={}
    for arm in ['RAW','SCALED']:
        g={}
        for h in ['8','16']:
            actual=models[arm]['FREE']['DEV']['action_delta']['summary'][h];base=no[h];g[h]=dict(action_delta_RMSE_m=actual['coordinate_RMSE_m'],NO_ACTION_delta_RMSE_m=base['coordinate_RMSE_m'],relative_improvement=1-actual['coordinate_RMSE_m']/max(base['coordinate_RMSE_m'],1e-15),all_DEV_parents_direction_75pct=all(p['observed_direction_fraction']>=.75 for p in actual['by_parent'].values()))
        absolute={h:dict(model_m=models[arm]['FREE']['DEV']['metrics'][h]['tcp_position_m']['model_RMSE'],ARX_m=controls['ARX']['DEV']['metrics'][h]['tcp_position_m']['model_RMSE'],RIDGE_m=controls['RIDGE']['DEV']['metrics'][h]['tcp_position_m']['model_RMSE'])for h in ['1','8','16']}
        gate[arm]=dict(action_delta_gate=all(v['relative_improvement']>=.2 and v['all_DEV_parents_direction_75pct']for v in g.values()),horizons=g,absolute_TCP=absolute,full_M0_gate='REQUIRES_JOINT_REVIEW_OF_ABSOLUTE_ERRORS_AND_NO_FALSE_EVENTS')
    return dict(schema='WM02_NEXT1_MATCHED_MODEL_EVALUATION_V1',replay_sha256=spec['replay_sha256'],models=models,TRAIN_fitted_frozen_controls=controls,development_gates=gate,teacher_forcing='Privileged future actual states only in separately labelled diagnostic; not planning or FREE inference',DEV_parents=2,no_DEV_normalization_or_recipe_selection=True,event_heads='NOT_TRAINED_NO_CONTACT_POSITIVES',new_optimizer_updates=0,new_integrations=0,Actor_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)


def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=run(json.loads(a.spec.read_text()));write_json(a.output,v);print(json.dumps(v['development_gates']));return 0


if __name__=='__main__':raise SystemExit(main())
