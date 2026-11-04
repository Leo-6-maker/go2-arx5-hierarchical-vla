"""TRAIN-only local linear reference residual baseline; no neural update or physics."""
def main(argv=None):
    import argparse,json,time
    from pathlib import Path
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--eval-replay',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    start=time.monotonic();cpu=time.process_time();data,meta=load(a.replay);ids=select_samples(data,meta,'TRAIN');state=data['state'][ids];goal=data['goal'][ids]
    if not len(ids):raise ValueError('ACTUAL_TRAIN_WINDOWS_REQUIRED')
    mean=state.mean((0,1));scale=np.maximum(state.std((0,1)),.001)
    def feature(current,goal):return np.concatenate(((current-mean)/scale,goal,np.ones((*current.shape[:-1],1))),-1)
    # Fit only the same post-burn phase used for neural prediction labels.
    x=feature(state[:,8:24].reshape(-1,state.shape[-1]),np.repeat(goal,16,axis=0));y=((state[:,9:25]-state[:,8:24])/scale).reshape(-1,state.shape[-1]);weight=np.linalg.solve(x.T@x+np.eye(x.shape[-1]),x.T@y)
    observed,eval_meta=load(a.eval_replay or a.replay)
    if meta['state_fields']!=eval_meta['state_fields']or(meta.get('robot_fingerprint')!=eval_meta.get('robot_fingerprint')if meta.get('schema')in('WM02_TARGET_NATIVE_REPLAY_V2','WM02_TARGET_LATE_DECISION_REPLAY_V3')else meta['model_sha256']!=eval_meta['model_sha256']):raise ValueError('BASELINE_TARGET_LAYOUT_CHANGED')
    pred=[];cur=observed['state'][:,8].copy()
    for i in range(16):cur=cur+feature(cur,observed['goal'])@weight*scale;pred.append(cur.copy())
    pred=np.stack(pred,1);metrics={};offset=0
    for field in meta['state_fields']:
        sl=slice(offset,offset+field['size']);offset=sl.stop;metrics[field['name']]={str(h):float(np.sqrt(np.mean((pred[:,h-1,sl]-observed['state'][:,8+h,sl])**2)))for h in(1,8,16)}
    a.output.mkdir(exist_ok=False)
    with(a.output/'MODEL.npz').open('xb')as f:np.savez_compressed(f,weight=weight,mean=mean,scale=scale)
    partition_metrics={}
    for split in('TRAIN','DEV'):
        chosen=select_samples(observed,eval_meta,split)
        if not len(chosen):continue
        groups={};offset=0
        for field in meta['state_fields']:
            sl=slice(offset,offset+field['size']);offset=sl.stop
            groups[field['name']]={str(h):float(np.sqrt(np.mean((pred[chosen,h-1,sl]-observed['state'][chosen,8+h,sl])**2)))for h in(1,8,16)}
        partition_metrics[split]=groups
    result=dict(status='TARGET_TRAIN_REFERENCE_RESIDUAL_BASELINE_FIT',FIT_replay_sha256=digest(a.replay),evaluation_replay_sha256=digest(a.eval_replay or a.replay),model_sha256=digest(a.output/'MODEL.npz'),lambda_ridge=1.,closed_form_fits=1,WM_optimizer_updates=0,Actor_updates=0,physics_integrations=0,metrics=metrics,partitions=partition_metrics,held_out_parent_generalization=eval_meta['held_out_parent_generalization'],fit_parent_count=len({e['parent']for e in meta['episodes']if e['split']=='TRAIN'}),source_future='Only predeclared fixed goal, no future achieved state or new reference',CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(a.output/'RESULT.json',result);print(json.dumps({k:result[k]for k in('status','model_sha256','closed_form_fits','WM_optimizer_updates')}));return 0
if __name__=='__main__':raise SystemExit(main())
