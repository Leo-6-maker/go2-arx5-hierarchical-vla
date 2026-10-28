"""Read-only measured-state and conservative full-H40 local geometry witness."""
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from pi0266_common import ROOT,BASE,REF,DATA,RAW,read
sys.path[:0]=[str(REF/'src'),str(BASE/'PI0258/runtime/collision_deps'),'/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0258/runtime/collision_deps']
from pi0258_geometry import Collision,full_fk,homogeneous
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
from pi0259_kinematic import solve


def main():
    source=read(ROOT/'evidence/SOURCE_IDENTITY.json');path=next(x['path'] for x in source['files'] if x['name']=='native_urdf')
    collision=Collision(path);kin=NativeArmKinematics.from_urdf(path)
    velocity=np.array([float(next(j for j in collision.urdf.findall('joint') if j.attrib['name']==name).find('limit').attrib['velocity']) for name in kin.joint_names])
    import os
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path[:0]=[str(DATA/'runtime'),str(DATA/'qa'),str(BASE.parent/'isaac-gr00t')]
    import pi0260_r2_codec as codec
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    proc=codec.processor(read(DATA/'dataset/statistics.json'))
    print(json.dumps(dict(ready=True,exemptions=collision.exemptions,local_witness='fixed actual root; actual other DOFs; no dynamics guarantee')),flush=True)
    for line in sys.stdin:
        row=json.loads(line);root=np.asarray(row['root']);transform=homogeneous(Rotation.from_quat(root[3:7]).as_matrix(),root[:3]);joints=row['joints']
        def hits(q):
            values=dict(joints);values.update(zip(kin.joint_names,map(float,q)))
            poses=full_fk(collision.urdf,values,transform)
            return [x for x in collision.check(poses,row['obstacle'],row['half_extents']) if 'ground' not in x['pair']]
        q=np.array([joints[name] for name in kin.joint_names]);current=hits(q)
        if row['kind']=='STATE':answer=dict(hits=current)
        else:
            assert row['kind']=='REFERENCE';ledger=[];answer=dict(valid=not current,reason=None if not current else 'CURRENT_FORBIDDEN_COLLISION',rows=ledger)
            if answer['valid']:
                for index,(p,r) in enumerate(zip(row['position'],row['rotation'])):
                    start=q.copy();candidate,e,n,okay,clipped=solve(kin,transform,np.asarray(p),np.asarray(r),q,dict(line_search=[1.,.5,.25,.125,.0625,.03125,.015625,.0078125]))
                    # No failed solution becomes the next start; no random restart or changed target.
                    contact=hits(candidate) if okay else []
                    hard=bool(np.all(candidate>=kin.joint_limits[:,0]-1e-3) and np.all(candidate<=kin.joint_limits[:,1]+1e-3))
                    speed=bool(index==0 or np.all(abs(candidate-start)/.05<=velocity+1e-8))
                    valid=bool(okay and hard and speed and not contact)
                    ledger.append(dict(index=index,start=start.tolist(),candidate=candidate.tolist(),residual=e.tolist(),iterations=n,
                        projected_candidates=clipped,IK_ok=bool(okay),hard_ok=hard,velocity_ok=speed,hits=contact,accepted=valid))
                    if not valid:
                        answer.update(valid=False,reason='IK_RESIDUAL' if not okay else 'LOCAL_FORBIDDEN_COLLISION' if contact else 'LOCAL_HARD_LIMIT' if not hard else 'LOCAL_JOINT_SPEED')
                        break
                    q=candidate
                answer['complete_H40']=len(ledger)==40 and answer['valid']
            state=np.asarray(row['state_v1'],np.float32);label=np.asarray(row['raw_action'],np.float32)
            ps,pa,mask=codec.encode(proc,state,label);decoded=codec.decode(proc,pa,state)
            pe=float(np.linalg.norm(decoded[:,:3]-label[:,:3],axis=-1).max())
            re=float(Rotation.from_matrix(rot6d_rows_to_matrix(decoded[:,3:9])@rot6d_rows_to_matrix(label[:,3:9]).transpose(0,2,1)).magnitude().max())
            finite=bool(np.isfinite(ps).all() and np.isfinite(pa).all() and np.isfinite(decoded).all())
            answer['codec']=dict(valid=finite and pe<=1e-5 and re<=1e-4,position_roundtrip_m=pe,so3_roundtrip_rad=re,finite=finite)
            destination=__import__('pathlib').Path(row['codec_output']).resolve()
            assert str(destination).startswith(str(RAW.resolve())+'/')
            np.savez_compressed(destination,state=ps,action=pa,action_mask=mask,decoded_action=decoded)
            answer['valid']=answer['valid'] and answer['codec']['valid']
        print(json.dumps(answer),flush=True)


if __name__=='__main__':main()
