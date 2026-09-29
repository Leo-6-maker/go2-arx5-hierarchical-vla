"""Finish fixed development geometry for tasks still able to satisfy10/12."""
from pathlib import Path
import json
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from pi0258_kinematic import ROOT,solve,reference
from pi0258_geometry import Collision,full_fk,defaults,homogeneous
from pi0255_motion3 import sha,write,scene
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics


def run():
    reg=json.loads((ROOT/'design/registry.json').read_text())
    seal=json.loads((ROOT/'design/generation_seal.json').read_text())
    extension=json.loads((ROOT/'design/development_geometry_seal.json').read_text())
    assert extension['source_sha256']==sha(__file__) and extension['canonical_comment'] and extension['commit']
    assert seal['registry_sha256']==sha(ROOT/'design/registry.json')
    for name,digest in seal['source_files'].items():assert sha(ROOT/'runtime'/name)==digest,name
    starts=json.loads((ROOT/'evidence/development_start_visibility.json').read_text())
    collision=Collision(reg['urdf']['path']);kin=NativeArmKinematics.from_urdf(reg['urdf']['path'])
    source=json.loads(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0256/evidence/runtime_readback.json').read_text())
    joints_default=defaults(next(x['path'] for x in source['files'] if x['name']=='native_config'))
    limits={j.attrib['name']:float(j.find('limit').attrib['velocity']) for j in collision.urdf.findall('joint') if j.find('limit') is not None}
    velocity=np.array([limits[n] for n in kin.joint_names])
    root=homogeneous(np.eye(3),np.array(reg['root_pose'][:3]));lookup={f['family_id']:f for f in reg['families']}
    summaries=[]
    for row in starts['sequences']:
        seq=row['sequence'];folder=ROOT/'development'/seq['family_id'];case=seq['sequence_id']
        out=folder/(case+'-geometry.json')
        if out.exists():summaries.append(json.loads(out.read_text()));continue
        if starts['totals'][seq['task']]['visible']<10:
            summary=dict(sequence_id=case,task=seq['task'],status='NOT_RUN_TASK_VISIBILITY_CEILING_LT10',
                necessary_visibility_pass=row['start_visible'],start_missing_cues=row['missing_cues'],full_reference_sha256=row['reference_sha256'],
                qualified=False,geometry_samples=0)
            write(out,summary);summaries.append(summary);continue
        family=lookup[seq['family_id']]
        with np.load(folder/row['reference_file']) as a:times=a['synthetic_reference_time'];p=a['desired_position'];quat=a['desired_quaternion']
        assert sha(folder/row['reference_file'])==row['reference_sha256']
        q=np.array(reg['arm_default']);qs=[];errors=[];valid=[];collisions=[];iterations=[];fk=[]
        start=time.time();data=scene(reg,family)
        for i,t in enumerate(times):
            q,e,n,okay=solve(kin,p[i]-root[:3,3],Rotation.from_quat(quat[i]).as_matrix(),q,reg['solver'])
            positions=dict(joints_default);positions.update(zip(kin.joint_names,q))
            poses=full_fk(collision.urdf,positions,root)
            independently=poses['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
            own=root@kin.tcp_pose(q)
            assert np.max(abs(independently-own))<1e-9,'independent FK discrepancy'
            hits=collision.check(poses,data['obstacle'],reg['geometry']['obstacle_half_extents_m'])
            speed_ok=i==0 or np.all(abs(q-qs[-1])/.05<=velocity+1e-10)
            qs.append(q.copy());errors.append(e);iterations.append(n);fk.append(own)
            valid.append(bool(okay and not hits and speed_ok));collisions.append(dict(index=i,contacts=hits,velocity_ok=bool(speed_ok),ik_ok=bool(okay)))
        obs_indices=np.arange(0,len(times),2);active=obs_indices[times[obs_indices]<reg['experts'][seq['task']]['motion_end_s']]
        active_qualified=[]
        for i in active:
            inds=np.minimum(i+np.arange(40),len(times)-1)
            active_qualified.append(bool(np.all(np.asarray(valid)[inds])))
        fraction=float(np.mean(active_qualified));quarters=[]
        for quarter in range(4):
            mask=(times[active]>=quarter*reg['experts'][seq['task']]['motion_end_s']/4)&(times[active]<(quarter+1)*reg['experts'][seq['task']]['motion_end_s']/4)
            quarters.append(bool(np.any(np.asarray(active_qualified)[mask])))
        payload=folder/(case+'-geometry.npz')
        np.savez_compressed(payload,synthetic_reference_time=times,joints=np.asarray(qs),desired_position=p,desired_quaternion=quat,
            FK_world=np.asarray(fk),residual=np.asarray(errors),solver_iterations=np.asarray(iterations),sample_qualified=np.asarray(valid),
            observation_indices=obs_indices,active_indices=active,active_geometry_qualified=np.asarray(active_qualified))
        write(folder/(case+'-contacts.json'),collisions)
        summary=dict(sequence_id=case,task=seq['task'],status='GEOMETRY_COMPLETE_RENDER_AND_CODEC_PENDING',
            geometry_samples=len(times),ik_valid=sum(x['ik_ok'] for x in collisions),collision_free=sum(not x['contacts'] for x in collisions),
            velocity_valid=sum(x['velocity_ok'] for x in collisions),active_windows=len(active),qualified_geometry_fraction=fraction,
            quartile_coverage=quarters,geometry_pass=fraction>=.95 and all(quarters),qualified=False,
            payload_sha256=sha(payload),seconds=time.time()-start,physics_steps=0)
        write(out,summary);summaries.append(summary);print(json.dumps(summary),flush=True)
    write(ROOT/'evidence/development_geometry_summary.json',dict(sequences=summaries,physics_steps=0,GR00T_queries=0,
        note='Geometry is necessary, not sufficient: actual per-window visibility and actual processor/gate checks remain required for training admission.'))


if __name__=='__main__':run()
