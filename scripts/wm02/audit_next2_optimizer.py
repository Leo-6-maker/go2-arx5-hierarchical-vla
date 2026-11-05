"""CPU-only frozen diagnostic checkpoint: local loss versus free recursion."""
def run(spec):
    import numpy as np,torch
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.storage import digest
    if digest(spec['checkpoint'])!=spec['checkpoint_sha256']:raise ValueError('FROZEN_CHECKPOINT_CHANGED')
    data,meta=load(spec['replay']);ids=select_samples(data,meta,'TRAIN');base=CausalARX(meta).fit(data,ids);c=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True);m=CausalARX(meta);m.mean=c['normalizers']['mean'].numpy();m.scale=c['normalizers']['scale'].numpy();m.feature_mean=c['normalizers']['feature_mean'].numpy();m.feature_scale=c['normalizers']['feature_scale'].numpy();m.weight=c['weight'].numpy();models={'ARX':base,'NEURAL_LINEAR_FINITE_ADAM':m};result={}
    for label,model in models.items():
        result[label]={}
        for split in ['TRAIN','DEV']:
            ix=select_samples(data,meta,split);st=data['state'][ix];pred=[]
            for j in range(16):
                x=model.feature(st[:,8+j],st[:,7+j],data['goal'][ix],data['window_start_s'][ix],data['source_age'][ix,8],model.remaining(data,ix),j);pred.append(st[:,8+j]+model.design(x)@model.weight*model.scale)
            pred=np.stack(pred,1);sl=model.layout['tcp_position_m'];err=(pred[:,:,sl]-st[:,9:,sl])**2;result[label][split]=dict(TEACHER_FORCED_TCP_RMSE_m=float(np.sqrt(err.mean())),H16_TEACHER_FORCED_TCP_RMSE_m=float(np.sqrt(err[:,15].mean())),FREE_TCP_RMSE_m={str(h):float(np.sqrt(np.mean((model.predict(data,ix)[:,h-1,sl]-st[:,8+h,sl])**2)))for h in [1,8,16]})
        ix=np.asarray([ids[0]]);same={k:np.repeat(v[ix],4,axis=0)for k,v in data.items()if isinstance(v,np.ndarray)and len(v)==len(data['state'])};same['goal'][:,:3]=same['history_goal'][:,0,:3];same['goal'][:,:2]+=np.asarray([[0,0],[.001,0],[-.001,0],[0,.001]]);p=model.predict(same,np.arange(4));tcp=model.layout['tcp_position_m'];result[label]['same_prefix_future_action_delta_norm_m']={str(h):np.linalg.norm(p[1:,h-1,tcp]-p[0,h-1,tcp],axis=-1).tolist()for h in [1,8,16]}
    return dict(schema='WM02_NEXT2_OPTIMIZER_TEACHER_FORCED_FREE_V1',results=result,checkpoint_sha256=spec['checkpoint_sha256'],model_weight_max_difference=float(np.max(np.abs(base.weight-m.weight))),interpretation='Finite first-order optimizer approximated ridge training objective but its recurrent rollout may be unstable; no DEV-driven endpoint extension, no deployment of this diagnostic',new_integrations=0,new_optimizer_updates=0)

def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=run(json.loads(a.spec.read_text()));write_json(a.output,v);print(json.dumps(v));return 0
if __name__=='__main__':raise SystemExit(main())
