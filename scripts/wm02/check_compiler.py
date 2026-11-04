"""CPU numerical equivalence against native compiler and optional frozen 41-point parser."""
import argparse,json

def check(parser_path=None):
    import numpy as np
    import torch
    from go2arx5_vla.world_model.neural_contract import compile_physical,actor_primitives,VERSION
    from go2arx5_vla.world_model.contracts import exp_rotation
    from go2arx5_vla.world_model.action_compiler import compile_reference,TARGET
    anchor=np.eye(4);anchor[:3,:3]=exp_rotation([.4,-.3,.7]);anchor[:3,3]=[.3,-.1,.6]
    latent=torch.linspace(-3,3,56,dtype=torch.float64).reshape(1,8,7).requires_grad_()
    raw=actor_primitives(latent,target=True);rows,times=compile_physical(raw,torch.tensor(anchor)[None],43.15,target=True)
    original=compile_reference(raw.detach().numpy()[0],anchor,43.15,backend=TARGET,frame='world',tcp='x5_ee')
    error=float(np.max(np.abs(rows.detach().numpy()[0]-original['rows10'])))
    clock_error=float(np.max(np.abs(times.numpy()-original['timestamps_s'])))
    if error>1e-12 or clock_error>1e-12:raise RuntimeError('Physical compiler differs from native')
    rows.square().sum().backward()
    if not torch.isfinite(latent.grad).all()or not latent.grad.norm()>0:raise RuntimeError('Compiler lost input gradient')
    result=dict(version=VERSION,world_H40_max_abs_error=error,absolute_time_max_abs_error=clock_error,
        action_input_gradient_norm=float(latent.grad.norm()),gripper_fraction_min=float(raw[...,6].min().detach()),
        gripper_fraction_max=float(raw[...,6].max().detach()),native_parser_samples=0,physics_integrations=0,
        target_dynamic_qualified=False,gripper_transport='SEPARATE_GRTGRIP1_NOT_EXERCISED')
    if parser_path:
        from scripts.pi0294.dec1.target_time_wire_cpu import load_frozen_parser,PARSER_SHA
        from scripts.pi0294.native_receiver import bind_packet
        from go2arx5_vla.trajectory.so3 import quat_to_matrix
        parser=load_frozen_parser(parser_path);doc=original['wire_document']
        doc['_pi0294']=dict(episode='WM02_COMPILER_V2_CPU',sequence=0,source_mpc_s=43.15,ttl_s=.4,operation='install')
        bound,_,_=bind_packet(doc,episode='WM02_COMPILER_V2_CPU',sequence=-1,now=43.15,
            trajectory_type=parser.TaskTrajectory,ee=parser.Pose.from_state([0,0,0,0,0,0,1]),base=None)
        tt,poses,base=bound.window(43.15,2.,.05);errors=[]
        for i,p in enumerate(poses):
            a=original['rows10'][min(i,39)];rr=a[3:9].reshape(2,3);rot=np.stack((rr[0],rr[1],np.cross(*rr)))
            errors.extend([np.max(np.abs(p.position-a[:3])),np.max(np.abs(quat_to_matrix(p.quaternion_xyzw)-rot))])
        if len(poses)!=41 or base is not None or max(errors)>1e-10 or np.max(np.abs(np.asarray(tt)-(43.15+np.arange(41)*.05)))>1e-10:raise RuntimeError('Frozen parser world/time mapping differs')
        result.update(native_parser_samples=41,parser_sha256=PARSER_SHA,parser_pose_max_abs_error=float(max(errors)))
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parser');a=p.parse_args(argv);print(json.dumps(check(a.parser)));return 0

if __name__=='__main__':raise SystemExit(main())
