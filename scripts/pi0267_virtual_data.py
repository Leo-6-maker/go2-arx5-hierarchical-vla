"""Fixed 24x2x2x16x7 proposals. All rejections stay in the denominator."""
import isaacgym
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from pi0267_virtual_renderer import BASE,ROOT,Renderer,freeze,collision,visibility,PARTS
sys.path.append(str(BASE/'PI0266/code'))
from pi0267_expert import ArclengthCorrection
from pi0255_motion3 import scene,write,sha
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1/S2')


def raw_gate(p,r,current_p,current_r,anchor,now=0.,source='ARCLENGTH_EXP_CORRECTION_V1',gripper=None):
    t=now+np.arange(40)*.05
    chunk=TrajectoryChunkV1(chunk_id='PI0267-virtual-label',source=source,observation_timestamp=now,
        point_timestamps=t,eef_position=p-anchor,eef_orientation=Rotation.from_matrix(r).as_quat(),source_frame='PI0267_STATIC_POLICY_ANCHOR_WORLD_AXES',
        tcp_semantic_id='PI0267_VIRTUAL_TOOL_TCP_V1',validity_start=now,validity_end=float(t[-1]),gripper=np.full(40,.5) if gripper is None else gripper)
    return TrajectorySafetyGateV1().evaluate(chunk,now=now,achieved_position=current_p-anchor,
        achieved_orientation=Rotation.from_matrix(current_r).as_quat()).as_dict()


def state(p,r,anchor):return np.r_[p-anchor,r[:2,:].reshape(6),.5].astype(np.float32)


def family_run(family_id):
    card=freeze();reg=card['registry'];anchor=np.array(card['policy_anchor_world_m'])
    family=next(f for f in card['families'] if f['family_id']==family_id);assert family['split']=='FEEDBACK_TRAIN'
    out=RAW/family_id;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'first valid production retained'
    write(out/'STARTED.json',dict(source_scene_sha256=sha(ROOT/'design/SCENE_CARD.json'),tool_sha256=sha(ROOT/'R1/design/VIRTUAL_TOOL_CARD.json'),
        producer_sha256=sha(__file__),renderer_sha256=sha(ROOT/'code/pi0267_virtual_renderer.py'),fixed_proposals=448))
    renderer=Renderer(reg,family);data=scene(reg,family);half=np.array(reg['geometry']['obstacle_half_extents_m'])
    cache={};packets=[];rows=[];states=[];actions=[];positions=[];rotations=[];future_p=[];future_r=[]
    try:
        for task in ('M1','M3'):
            for command in (0,1):
                teacher=ArclengthCorrection(reg,family,task,command)
                for i,length in enumerate(np.linspace(0,teacher.ell[-1],16)):
                    bp,br=teacher.at_length(length);bp,br=bp[0],br[0]
                    for variant in range(7):
                        p,r=bp.copy(),br.copy()
                        if variant:
                            j=(variant-1)//2;sign=1 if variant%2 else -1
                            p[j]+=.010*sign;rv=np.eye(3)[j]*.060*sign;r=Rotation.from_rotvec(rv).as_matrix()@r
                        lp,lr,meta=teacher.label(p,r);gate=raw_gate(lp,lr,p,r,anchor)
                        hits=sorted(set(h for q,s in zip(lp,lr) for h in collision(q,s,np.array(data['obstacle']),half)))
                        key=__import__('hashlib').sha256(p.tobytes()+r.tobytes()).hexdigest()
                        if key not in cache:
                            frames,info=renderer.place(p,r,0.)
                            # Camera origin independently checked against the preserved TCP extrinsic.
                            eye=np.linalg.inv(np.array(info['camera_view']['wrist']))[3,:3]
                            expected=p+r@(np.array(reg['renderer']['wrist_attachment'])-np.array([.1424,0,.0001057]))
                            assert np.max(abs(eye-expected))<1e-5,('WRIST_EXTRINSIC',eye,expected)
                            cache[key]=len(packets);packets.append((frames,info))
                        frame_index=cache[key];frames,info=packets[frame_index]
                        check=visibility(frames,renderer.scene_ids,task,command,common_start=i==0)
                        reasons=[]
                        if not meta['valid_label']:reasons.append('UNOBSERVABLE_LABEL')
                        if gate['decision']=='REJECT':reasons.append('GATE_REJECT')
                        if hits:reasons.append('TOOL_COLLISION')
                        if not check['pass_necessary']:reasons.append('VISIBILITY')
                        index=len(rows);rows.append(dict(index=index,task=task,command=command,anchor_index=i,variant=variant,
                            nominal=variant==0,quartile=min(3,int(i/15*4)),terminal=i==15,frame_index=frame_index,frame_key=key,
                            label=meta,gate=gate,collision_hits=hits,visibility=check,precodec_valid=not reasons,reasons=reasons))
                        states.append(state(p,r,anchor));actions.append(np.c_[lp-anchor,lr[:,:2,:].reshape(40,6),np.full(40,.5)].astype(np.float32))
                        positions.append(p);rotations.append(r);future_p.append(lp);future_r.append(lr)
        assert len(rows)==448
        np.savez_compressed(out/'proposals.npz',state_v1=states,raw_action=actions,position_world=positions,rotation_world=rotations,
            label_position_world=future_p,label_rotation_world=future_r)
        np.savez_compressed(out/'policy_frames.npz',body_rgb=np.stack([x[0]['body'] for x in packets]),wrist_rgb=np.stack([x[0]['wrist'] for x in packets]))
        np.savez_compressed(out/'debug_frames.npz',**{k:np.stack([x[0][k] for x in packets]) for k in ('body_segmentation','wrist_segmentation','body_depth','wrist_depth')})
        write(out/'frame_index.json',[dict(frame_key=key,index=index,**packets[index][1]) for key,index in cache.items()])
        write(out/'rows.json',rows)
        write(out/'COMPLETE.json',dict(status='COMPLETE_PRECODEC',family=family_id,proposals=len(rows),unique_frame_pairs=len(packets),
            precodec_valid=sum(r['precodec_valid'] for r in rows),physics_steps=0,hashes={n:sha(out/n) for n in ('proposals.npz','policy_frames.npz','debug_frames.npz','rows.json','frame_index.json')}))
        print(json.dumps(json.loads((out/'COMPLETE.json').read_text())),flush=True)
    finally:renderer.close()


if __name__=='__main__':family_run(sys.argv[1])
