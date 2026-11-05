"""Same observed prefix model interventions; no native counterfactual claim."""
def run(spec):
    import time,json
    from pathlib import Path
    import numpy as np,torch
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_training import TargetWM,tensors,slices
    from go2arx5_vla.world_model.target_training_v2 import compatible
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.target_stability import rotation_wxyz
    from go2arx5_vla.world_model.storage import digest
    start=time.monotonic();cpu=time.process_time();torch.set_num_threads(1)
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('FROZEN_REPLAY_CHANGED')
    data,meta=load(spec['replay']);layout=slices(meta);tcp=layout['tcp_position_m'];q0=layout['qpos'].start;offsets=np.array([[0,0,0],[.001,0,0],[-.001,0,0],[0,.001,0]])
    def relative(state):
        q=state[...,q0+3:q0+7];norm=np.linalg.norm(q,axis=-1,keepdims=True)
        if np.any(norm<1e-8):raise ValueError('INVALID_PREDICTED_BASE_QUATERNION')
        r=rotation_wxyz(q/norm)
        return np.einsum('...ji,...j->...i',r,state[...,tcp]-state[...,q0:q0+3])
    models={}
    for arm,b in spec['models'].items():
        if digest(b['path'])!=b['sha256']:raise ValueError('FROZEN_MODEL_CHANGED')
        c=torch.load(b['path'],map_location='cpu',weights_only=True);compatible(c['metadata'],meta);m=TargetWM(c['metadata'],c['hidden']);m.load_state_dict(c['model']);m.eval();models[arm]=m
    with np.load(spec['ARX']['path'],allow_pickle=False)as z:
        if digest(spec['ARX']['path'])!=spec['ARX']['sha256']:raise ValueError('FROZEN_ARX_CHANGED')
        arx=CausalARX.from_arrays(meta,z)
    rows=[]
    for parent in sorted({e['parent']for e in meta['episodes']}):
        ix=[i for i,e in enumerate(meta['episodes'])if e['parent']==parent];hold=[i for i in ix if np.linalg.norm(data['goal'][i,:3]-data['history_goal'][i,0,:3])<1e-8];i=hold[0]
        shared={k:np.repeat(v[i:i+1],4,axis=0).copy()for k,v in data.items()if k in ['state','goal','source_age','window_start_s','window_start_atom','history_goal','future_prefix_remaining']};shared['goal'][:]=data['history_goal'][i,0];shared['goal'][:,:3]+=offsets
        predictions={}
        inp=tensors(shared,np.arange(4),'cpu')
        with torch.no_grad():
            for arm,m in models.items():predictions[arm]=m.predict(**inp).numpy().astype(float)
        predictions['ARX']=arx.predict(shared,np.arange(4))
        r0=rotation_wxyz(data['state'][i,8,q0+3:q0+7])
        for h in [1,8,16]:
            for action,d in enumerate(offsets[1:],1):
                actual=[j for j in ix if np.allclose(data['goal'][j,:3]-data['history_goal'][j,0,:3],d,atol=1e-8,rtol=0)]
                if len(actual)!=1:raise ValueError('ONE_REAL_MATCHED_ACTION_REQUIRED')
                j=actual[0];world_delta=data['state'][j,8+h,tcp]-data['state'][j,8,tcp]-(data['state'][hold,8+h,tcp]-data['state'][hold,8,tcp]).mean(0)
                rel=relative(data['state'][j,[8,8+h]]);hr=relative(data['state'][hold][:,[8,8+h]]);real_rel=r0@(rel[1]-rel[0]-(hr[:,1]-hr[:,0]).mean(0))
                for arm,p in predictions.items():
                    world=p[action,h-1,tcp]-p[0,h-1,tcp];rr=relative(p[[0,action],h-1]);rel_delta=r0@(rr[1]-rr[0]);rows.append(dict(parent=parent,split=meta['episodes'][i]['split'],model=arm,horizon=h,offset_xyz_m=d.tolist(),same_prefix_model_world_delta_m=world.tolist(),same_prefix_model_relative_delta_m=rel_delta.tolist(),actual_matched_restart_world_delta_m=world_delta.tolist(),actual_matched_restart_relative_delta_m=real_rel.tolist()))
    summaries={}
    for split in ['TRAIN','DEV']:
        summaries[split]={}
        for arm in predictions:
            summaries[split][arm]={}
            for h in [1,8,16]:
                chosen=[r for r in rows if r['split']==split and r['model']==arm and r['horizon']==h];summary={}
                for mode in ['world','relative']:
                    pred=np.array([r['same_prefix_model_'+mode+'_delta_m']for r in chosen]);truth=np.array([r['actual_matched_restart_'+mode+'_delta_m']for r in chosen]);axes=[int(np.argmax(np.abs(r['offset_xyz_m'])))for r in chosen]
                    summary[mode]=dict(coordinate_RMSE_m=float(np.sqrt(np.mean((pred-truth)**2))),mean_prediction_norm_m=float(np.linalg.norm(pred,axis=1).mean()),mean_observed_norm_m=float(np.linalg.norm(truth,axis=1).mean()),direction_fraction=float(np.mean([p[a]*t[a]>0 for p,t,a in zip(pred,truth,axes)])))
                summaries[split][arm][str(h)]=summary
    return dict(schema='WM02_NEXT1_FROZEN_SAME_PREFIX_MODEL_RESPONSE_V1',source_sha=spec['source_sha'],replay_sha256=spec['replay_sha256'],frozen_models=spec['models'],ARX=spec['ARX'],summary=summaries,rows=rows,model_inputs_identical_except_declared_future_goal=True,native_exact_same_state_counterfactual=False,real_comparator='Matched independently restarted episodes; genuine initial scatter retained',predicted_quaternion_normalized_for_relative_diagnostic_only=True,new_optimizer_updates=0,new_integrations=0,Actor_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)


def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=run(json.loads(a.spec.read_text()));write_json(a.output,v);print(json.dumps(v['summary']['DEV']));return 0

if __name__=='__main__':raise SystemExit(main())
