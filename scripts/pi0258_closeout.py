"""Independent CPU reconstruction and labelled media from existing PI0258 bytes."""
from pathlib import Path
import json
import sys
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from pi0258_kinematic import ROOT
from pi0258_geometry import Collision,full_fk,defaults,homogeneous
from pi0255_motion3 import sha,write,scene
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1


def replay():
    reg=json.loads((ROOT/'design/registry.json').read_text());summaries=json.loads((ROOT/'evidence/development_geometry_summary.json').read_text())
    collision=Collision(reg['urdf']['path']);source=json.loads(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0256/evidence/runtime_readback.json').read_text())
    default=defaults(next(x['path'] for x in source['files'] if x['name']=='native_config'))
    arm_names=['x5_joint%d'%i for i in range(1,7)];root=homogeneous(np.eye(3),np.array(reg['root_pose'][:3]))
    velocity=np.array([float(next(j for j in collision.urdf.findall('joint') if j.attrib['name']==n).find('limit').attrib['velocity']) for n in arm_names])
    families={f['family_id']:f for f in reg['families']};sequences={s['sequence_id']:s for s in reg['sequences']}
    anchor=np.asarray(reg['geometry']['nominal_tcp_world']);replayed=[];gate=TrajectorySafetyGateV1()
    for summary in summaries['sequences']:
        if not summary.get('geometry_samples'):continue
        case=summary['sequence_id'];sequence=sequences[case];folder=ROOT/'development'/sequence['family_id'];family=families[sequence['family_id']]
        path=folder/(case+'-geometry.npz');assert sha(path)==summary['payload_sha256']
        with np.load(path) as a:d={k:a[k].copy() for k in a.files}
        contacts=json.loads((folder/(case+'-contacts.json')).read_text());poses=[];flags=[];max_error=0.;max_residual_difference=0.
        for i,q in enumerate(d['joints']):
            joints=dict(default);joints.update(zip(arm_names,q));fk=full_fk(collision.urdf,joints,root)
            tcp=fk['x5_link6']@homogeneous(np.eye(3),np.array([.1424,0,.0001057]));poses.append(tcp)
            max_error=max(max_error,float(np.max(abs(tcp-d['FK_world'][i]))))
            p_error=d['desired_position'][i]-tcp[:3,3]
            rotation=(Rotation.from_quat(d['desired_quaternion'][i])*Rotation.from_matrix(tcp[:3,:3]).inv()).as_rotvec()
            max_residual_difference=max(max_residual_difference,float(np.max(abs(np.r_[p_error,rotation]-d['residual'][i]))))
            ik=np.linalg.norm(p_error)<=.002 and np.linalg.norm(rotation)<=.01
            hits=collision.check(fk,scene(reg,family)['obstacle'],reg['geometry']['obstacle_half_extents_m'])
            speed=i==0 or np.all(abs(q-d['joints'][i-1])/.05<=velocity+1e-10)
            assert ik==contacts[i]['ik_ok'] and speed==contacts[i]['velocity_ok'] and bool(hits)==bool(contacts[i]['contacts'])
            flags.append(bool(ik and not hits and speed))
        assert max_error<1e-9 and max_residual_difference<1e-9
        assert np.array_equal(flags,d['sample_qualified'])
        windows=[all(flags[min(int(i)+j,len(flags)-1)] for j in range(40)) for i in d['active_indices']]
        assert np.array_equal(windows,d['active_geometry_qualified'])
        assert float(np.mean(windows))==summary['qualified_geometry_fraction']
        rendered=folder/(case+'-render.npz');raw_gate_replayed=0;state_error=None
        if rendered.exists():
            packet=json.loads((folder/(case+'-render.json')).read_text());assert sha(rendered)==packet['render_sha256']
            with np.load(rendered) as a:images={k:a[k].copy() for k in a.files}
            independent=np.asarray([np.r_[poses[i][:3,3]-anchor,poses[i][:2,:3].reshape(-1),.5] for i in d['observation_indices']])
            state_error=float(np.max(abs(independent-images['state_v1'])));assert state_error<1e-9
            assert np.array_equal(images['synthetic_reference_time'],d['synthetic_reference_time'][d['observation_indices']])
            recorded=json.loads((folder/(case+'-raw-gate.json')).read_text())
            for entry in recorded:
                i=entry['index'];t=float(d['synthetic_reference_time'][i]);ii=np.minimum(i+np.arange(40),len(flags)-1)
                chunk=TrajectoryChunkV1(case+'-'+str(i),'KINEMATIC_RENDERED_REFERENCE',t,t+np.arange(40)*.05,
                    d['desired_position'][ii]-anchor,d['desired_quaternion'][ii],'NATIVE_WBC_RELATIVE_REFERENCE',
                    'x5_link6_gripper_center',t,t+1.95,gripper=np.full(40,.5))
                verdict=gate.evaluate(chunk,now=t,achieved_position=poses[i][:3,3]-anchor,achieved_orientation=Rotation.from_matrix(poses[i][:3,:3]).as_quat()).as_dict()
                assert verdict['decision']==entry['verdict']['decision']
                raw_gate_replayed+=1
        replayed.append(dict(sequence=case,samples=len(flags),geometry_qualified_fraction=float(np.mean(windows)),
            independent_full_URDF_FK_max_abs=max_error,residual_max_abs=max_residual_difference,
            framed_state_max_abs=state_error,raw_gates_replayed=raw_gate_replayed))
    projections=[]
    for family in reg['families'][:6]:
        info=json.loads((ROOT/'development'/family['family_id']/'render_readback.json').read_text());data=scene(reg,family)
        for name,p in [('target0',data['targets'][0]),('target1',data['targets'][1]),('goal',data['goal'])]:
            row=dict(family=family['family_id'],cue=name,cameras={})
            for camera in ('body','wrist'):
                clip=np.r_[p,1.]@np.asarray(info['camera_view'][camera])@np.asarray(info['camera_projection'][camera]);ndc=clip[:2]/clip[3]
                row['cameras'][camera]=dict(center_pixel_xy_bottom_origin=((ndc+1)*48).tolist(),clip_w=float(clip[3]),
                    center_in_frustum=bool(clip[3]>0 and np.all(abs(ndc)<=1)),recorded_actual_visible_pixels=info['visible_pixels'][name][camera])
            projections.append(row)
    receipt=dict(status='PASS_INDEPENDENT_CPU_FK_RESIDUAL_VELOCITY_COLLISION_WINDOW_STATE_AND_RAW_GATE_REPLAY',
        sequences=replayed,camera_projection=projections,raw_gates=sum(x['raw_gates_replayed'] for x in replayed),
        limitations=['Actual segmentation counts are producer observations, not independently reconstructed segmentation masks.',
            'Full-path endpoint visibility fractions are descriptive; the generation card required cues at the shared start. They do not create a new acceptance threshold.',
            'No C_KIN_MT codec or real model recovery evidence exists because no task cleared development admission.'],
        physics_steps=0,model_queries=0,optimizer_updates=0)
    write(ROOT/'evidence/independent_cpu_replay.json',receipt)
    print(json.dumps(dict(status=receipt['status'],sequences=len(replayed),raw_gates=receipt['raw_gates'])),flush=True)


def media():
    import cv2
    from eef_video_workflow import encoder,decode,text
    out=ROOT/'delivery';out.mkdir(exist_ok=True);family='pi0258-family-001';folder=ROOT/'development'/family
    packets=[];curves=[]
    for command in (0,1):
        case=family+'-M3-c'+str(command)
        with np.load(folder/(case+'-render.npz')) as a:packets.append({k:a[k].copy() for k in a.files})
        with np.load(folder/(case+'-geometry.npz')) as a:curves.append({k:a[k].copy() for k in a.files})
    assert np.array_equal(packets[0]['body_rgb'][0],packets[1]['body_rgb'][0])
    assert np.array_equal(packets[0]['wrist_rgb'][0],packets[1]['wrist_rgb'][0])
    path=out/'M3_fixed_pair_kinematic_reference.mp4';process=encoder(path,fps=10);mapping=[]
    for frame,t in enumerate(packets[0]['synthetic_reference_time']):
        canvas=np.full((720,1280,3),(25,22,18),dtype=np.uint8)
        text(canvas,'KINEMATIC REFERENCE / NO WBC EXECUTION',28,42,.95,(70,225,255))
        text(canvas,'PI0258 M3-v2 | family001 | shared nominal start | no trained model',28,78,.65)
        text(canvas,'DATA ADMISSION FAILED: fixed M3 geometry 8/12 (required10/12)',28,110,.65,(120,150,255))
        for command,offset in [(0,0),(1,640)]:
            label='BLACK -> WHITE / COUNTERCLOCKWISE' if command==0 else 'WHITE -> BLACK / CLOCKWISE'
            text(canvas,label,offset+28,142,.55)
            for name,x in [('body_rgb',28),('wrist_rgb',320)]:
                tile=cv2.cvtColor(packets[command][name][frame],cv2.COLOR_RGB2BGR)
                canvas[160:416,offset+x:offset+x+256]=cv2.resize(tile,(256,256),interpolation=cv2.INTER_NEAREST)
                text(canvas,name.replace('_rgb','')+' original96x96',offset+x,438,.5)
            curve=curves[command];i=frame*2
            def pixel(p):return (offset+int(70+(p[0]-.16)*1550),int(610-p[1]*800))
            desired=[pixel(p) for p in curve['desired_position']];actual=[pixel(p) for p in curve['FK_world'][:,:3,3]]
            cv2.polylines(canvas,[np.array(desired,np.int32)],False,(180,180,180),1)
            cv2.polylines(canvas,[np.array(actual[:i+1],np.int32)],False,(30,185,250),2)
            cv2.circle(canvas,pixel(curve['FK_world'][i,:3,3]),4,(80,240,90),-1)
            error=float(np.linalg.norm(curve['desired_position'][i]-curve['FK_world'][i,:3,3]))
            text(canvas,'XY plot: grey desired / orange IK-FK (not achieved dynamics)',offset+28,468,.43)
            text(canvas,'FK-reference residual %.3f mm'%(1000*error),offset+28,675,.5)
        text(canvas,'synthetic_reference_time %.2f s | playback1x | selection/deployment NONE'%t,28,706,.52)
        process.stdin.write(canvas.tobytes());mapping.append(dict(frame=frame,synthetic_reference_time=float(t),source_index=frame))
        if frame in (0,90,180):cv2.imwrite(str(out/('review_frame_%03d.png'%frame)),canvas)
    process.stdin.close();assert process.wait()==0
    qa=decode(path);assert qa['decoded_frames']==181 and qa['shape']==[720,1280,3]
    write(out/'video_frame_mapping.json',mapping)
    write(out/'video_qa.json',dict(video=qa,source_files=[dict(path=str(folder/(family+'-M3-c'+str(c)+'-render.npz')),sha256=sha(folder/(family+'-M3-c'+str(c)+'-render.npz'))) for c in (0,1)],
        evidence_class='KINEMATIC_RENDERED_REFERENCE',physics_steps=0,model_queries=0,
        selection='lowest fixed family ID with two complete geometry-qualified render streams, before any model exists; geometry is not dataset admission'))
    print(json.dumps(dict(video=str(path),frames=qa['decoded_frames'],bytes=qa['bytes'])),flush=True)


if __name__=='__main__':
    if '--replay' in sys.argv:replay()
    if '--media' in sys.argv:media()
