"""One source-specific CPU decoder/loss binding check, including out-of-range values."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from pi0260_r2 import ROOT,read,sha
from pi0260_r3_physical import bounds,install,decode_xyz,decode_relative_rotation
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix


def main():
    import pi0260_r2_codec as codec
    import pi0247_u_rollout_overlay as rollout
    import pi0248_rotation_cpu as rotation
    statistics=read(ROOT/'dataset/statistics.json')
    proc=codec.processor(statistics)
    lo,hi=bounds(statistics)
    scale=(hi[:,:3]-lo[:,:3])/2
    offset=(hi[:,:3]+lo[:,:3])/2
    install()
    results=[]
    for seq in ('pi0260-family-020-M1-c0','pi0260-family-026-M3-c0','pi0260-family-120-M1-c0','pi0260-family-120-M3-c0'):
        with np.load(ROOT/'dataset'/(seq+'.npz')) as z:
            state=z['state_v1'][0].copy();packed=z['action'][0].copy()
        for variant in ('original','outside_affine_range'):
            test=packed.copy()
            if variant=='outside_affine_range':test[0,0]=1.25
            stock=codec.decode(proc,test,state)
            actual=decode_xyz(torch.tensor(test[None,:,:3]),torch.tensor(state[None,:9]),scale,offset).detach().numpy()[0]
            position_error=float(np.max(np.linalg.norm(actual-stock[:,:3],axis=1)))
            predicted,_=decode_relative_rotation(torch.tensor(test[None,:,3:9]),torch.tensor(state[None,3:9]),lo[:,3:9],hi[:,3:9])
            target=rot6d_rows_to_matrix(stock[:,3:9])
            rotation_error=float(Rotation.from_matrix(predicted.detach().numpy()[0]@target.transpose(0,2,1)).magnitude().max())
            assert position_error<=1e-6 and rotation_error<=1e-5,(seq,variant,position_error,rotation_error)
            results.append(dict(sequence=seq,variant=variant,max_xyz_m=position_error,max_SO3_rad=rotation_error))
        expert=torch.tensor(packed[None]);mask=torch.zeros_like(expert);mask[:,:,:10]=1
        pos,delta,_=rollout._physical_losses(expert,expert,mask,torch.tensor(state[None,:9]),scale,offset)
        rot,_=rotation.rotation_chordal_loss(expert,expert,mask,torch.tensor(state[None,:9]),lo[:,3:9],hi[:,3:9])
        assert abs(float(pos))<1e-9 and abs(float(delta))<1e-9 and abs(float(rot))<1e-9
    p=ROOT/'R3/physical_binding_cpu.json';assert not p.exists()
    p.write_text(json.dumps(dict(status='PASS_R3_UNCLAMPED_TRAIN_LOSS_INFERENCE_DECODE_BINDING',cases=results,
                                 statistics_sha256=sha(ROOT/'dataset/statistics.json'),
                                 codec_source_sha256=sha(codec.__file__),
                                 model_loads=0,model_queries=0),indent=2)+'\n')
    print(json.dumps(dict(status='PASS_R3_UNCLAMPED_TRAIN_LOSS_INFERENCE_DECODE_BINDING',cases=len(results),sha256=sha(p))))


if __name__=='__main__':main()
