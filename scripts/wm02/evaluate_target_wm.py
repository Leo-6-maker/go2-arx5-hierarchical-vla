"""Read-only causal target WM evaluation on a pinned real numerical replay."""
def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.storage import digest,write_json
    spec=json.loads(a.spec.read_text())
    for key in('checkpoint','replay'):
        if digest(spec[key])!=spec[key+'_sha256']:raise ValueError('PINNED_TARGET_EVALUATION_IDENTITY_CHANGED')
    import torch,numpy as np
    from go2arx5_vla.world_model.target_training import restore,evaluate,tensors
    from go2arx5_vla.world_model.target_replay import load
    c=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True);m,opt,c=restore(spec['checkpoint'],'cpu',c['provenance']);data,meta=load(spec['replay'])
    from go2arx5_vla.world_model.target_training_v2 import compatible,partition_metrics
    compatible(c['metadata'],meta,migration=spec.get('schema_migration'))
    if meta['schema']=='WM02_TARGET_NATIVE_REPLAY_V2':
        metrics=dict(partitions=partition_metrics(m,data,meta,'cpu'),windows=len(data['state']),schema_migration=spec.get('schema_migration'),overlapping_windows_independent=False)
    else:metrics=evaluate(m,tensors(data,np.arange(len(data['state'])),'cpu'),meta)
    result=dict(status='ACTUAL_TARGET_FROZEN_WM_EVALUATED',checkpoint_sha256=spec['checkpoint_sha256'],replay_sha256=spec['replay_sha256'],episodes=len(meta['episodes']),physics_integrations=0,WM_updates=0,Actor_updates=0,device='CPU',**metrics)
    write_json(a.output,result);print(json.dumps({k:result[k]for k in('status','episodes','WM_updates','Actor_updates')}));return 0
if __name__=='__main__':raise SystemExit(main())
