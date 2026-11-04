"""Fit a stronger TRAIN-only action/time-aware ARX and score sealed target data."""
def main(argv=None):
    import argparse,time,json
    from pathlib import Path
    import numpy as np
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    from go2arx5_vla.world_model.storage import digest,write_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--eval-replay',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);start=time.monotonic();cpu=time.process_time()
    data,meta=load(args.replay);train=select_samples(data,meta,'TRAIN')
    model=CausalARX(meta).fit(data,train)
    observed,eval_meta=load(args.eval_replay or args.replay)
    for key in ('state_fields','schema','action','robot_fingerprint','executor_sha256'):
        if meta.get(key)!=eval_meta.get(key):raise ValueError('CAUSAL_ARX_LAYOUT_OR_EXECUTOR_CHANGED:'+key)
    pred=model.predict(observed,np.arange(len(observed['state'])))
    def metrics(ids):
        result={}
        for name,sl in model.layout.items():
            result[name]={str(h):float(np.sqrt(np.mean((pred[ids,h-1,sl]-observed['state'][ids,8+h,sl])**2)))for h in(1,8,16)}
        return result
    partitions={}
    for split in ('TRAIN','DEV'):
        ids=select_samples(observed,eval_meta,split)
        if not len(ids):continue
        parents={eval_meta['episodes'][int(observed['episode_index'][i])]['parent']for i in ids}
        by_parent={parent:metrics(np.asarray([i for i in ids if eval_meta['episodes'][int(observed['episode_index'][i])]['parent']==parent]))for parent in sorted(parents)}
        partitions[split]=dict(windows=len(ids),parent_count=len(parents),metrics=metrics(ids),by_parent=by_parent)
    args.output.mkdir(exist_ok=False)
    with(args.output/'MODEL.npz').open('xb')as f:np.savez_compressed(f,**model.arrays())
    result=dict(status='TARGET_TRAIN_CAUSAL_ARX_FIT',recipe='TRAIN_STANDARDIZED_ACTION_TIME_ARX1_RECURSIVE_V3',FIT_replay_sha256=digest(args.replay),evaluation_replay_sha256=digest(args.eval_replay or args.replay),model_sha256=digest(args.output/'MODEL.npz'),lambda_ridge=1.,closed_form_fits=1,WM_optimizer_updates=0,Actor_updates=0,physics_integrations=0,partitions=partitions,fit_parent_count=len({e['parent']for e in meta['episodes']if e['split']=='TRAIN'}),no_DEV_hyperparameter_selection=True,future_actual_state_or_reference_input=False,overlapping_windows_independent=False,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    write_json(args.output/'RESULT.json',result);print(json.dumps({k:result[k]for k in('status','model_sha256','fit_parent_count')}));return 0

if __name__=='__main__':raise SystemExit(main())
