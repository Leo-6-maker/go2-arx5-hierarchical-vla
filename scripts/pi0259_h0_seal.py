"""Seal existing PI0259 H0 bytes and raw gate records for independent Agent V."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from pi0259_kinematic import ROOT,OLD,AUTHORITY,source_geometry
from pi0255_motion3 import sha,write,scene
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1


def seal(commit):
    target=ROOT/'handoff/H0_v1';assert not target.exists(),'sealed H0 cannot be overwritten'
    camera=json.loads((ROOT/'evidence/camera_calibration_cpu_corrected.json').read_text())
    reg=json.loads((ROOT/'design/registry.json').read_text());geometry=json.loads((ROOT/'evidence/calibration_geometry.json').read_text())
    kin,collision,base,root,velocity=source_geometry(reg);anchor=np.asarray(reg['initial_TCP_world'])[:3,3]
    gate=TrajectorySafetyGateV1();raw_gate=[]
    for summary in geometry:
        seq=summary['sequence'];path=ROOT/'calibration'/seq['sequence_id']/'geometry.npz'
        assert sha(path)==summary['payload_sha256']
        with np.load(path) as a:data={k:a[k].copy() for k in a.files}
        rows=[]
        for index in range(0,len(data['joints']),2):
            if not data['sample_qualified'][index]:
                rows.append(dict(index=index,status='NO_ACCEPTED_CURRENT_FK'));continue
            t=float(data['synthetic_reference_time'][index]);ii=np.minimum(index+np.arange(40),len(data['joints'])-1)
            desired=data['desired_position'][ii]-anchor;quats=Rotation.from_matrix(data['desired_rotation'][ii]).as_quat()
            chunk=TrajectoryChunkV1(seq['sequence_id']+'-'+str(index),'KINEMATIC_RENDERED_REFERENCE',t,t+np.arange(40)*.05,
                desired,quats,'NATIVE_WBC_RELATIVE_REFERENCE','x5_link6_gripper_center',t,t+1.95,gripper=np.full(40,.5))
            verdict=gate.evaluate(chunk,now=t,achieved_position=data['FK_world'][index,:3,3]-anchor,
                achieved_orientation=Rotation.from_matrix(data['FK_world'][index,:3,:3]).as_quat()).as_dict()
            rows.append(dict(index=index,synthetic_reference_time=t,status='RAW_GATE_RECORDED',verdict=verdict,
                phase='ACTIVE' if t<reg['experts'][seq['task']]['motion_end_s'] else 'TERMINAL_HOLD_AUDIT_ONLY'))
        write(path.parent/'raw_gate.json',rows);raw_gate.append(dict(sequence=seq['sequence_id'],recorded=sum(x['status']=='RAW_GATE_RECORDED' for x in rows)))
    cues=[]
    for family in reg['calibration_families']:
        data=scene(reg,family);g=reg['geometry'];values={};r0=np.asarray(reg['initial_TCP_world'])[:3,:3]
        colors=dict(red=[.9,.03,.03],blue=[.03,.15,.9],green=[.03,.85,.1])
        for i,p in enumerate(data['targets']):
            values['target'+str(i)]=dict(position=p.tolist(),radius=g['target_radius_m'],color=colors[family['factors']['colors'][i]])
            rotation=r0@Rotation.from_euler('z',g['target_yaw_rad'][i]).as_matrix()
            for a in range(3):values['axis%d_%d'%(i,a)]=dict(position=(p+rotation[:,a]*g['axes_length_m']).tolist(),radius=g['axes_marker_radius_m'],color=np.eye(3)[a].tolist())
        values['obstacle']=dict(position=data['obstacle'].tolist(),half_extents=g['obstacle_half_extents_m'],color=[.45,.45,.45])
        values['goal']=dict(position=data['goal'].tolist(),radius=g['target_radius_m'],color=[1.,.35,.03])
        for i,a in enumerate(np.linspace(-3*np.pi/4,3*np.pi/4,g['contour_points'])):
            values['contour'+str(i)]=dict(position=(data['center']+[data['radius']*np.cos(a),data['radius']*np.sin(a),0]).tolist(),radius=g['contour_marker_radius_m'],color=[.95,.8,.05],sector=i//2)
        for name,a,color in [('black_A',-3*np.pi/4,[.03,.03,.03]),('white_B',3*np.pi/4,[1.,1.,1.])]:
            values[name]=dict(position=(data['center']+[data['radius']*np.cos(a),data['radius']*np.sin(a),0]).tolist(),radius=g['start_marker_radius_m'],color=color)
        cues.append(dict(family_id=family['family_id'],cues=values))
    write(ROOT/'design/cue_geometry.json',cues)
    write(ROOT/'evidence/raw_gate_summary.json',raw_gate)
    source_binding=dict(urdf=reg['urdf'],private_not_for_public_archive=True,full_fk_source=dict(path=str(ROOT/'runtime/pi0258_geometry.py'),sha256=sha(ROOT/'runtime/pi0258_geometry.py')),
        native_kinematics=dict(path=str((ROOT/'src/go2arx5_vla/wbc/native/kinematics.py').resolve()),sha256=sha(ROOT/'src/go2arx5_vla/wbc/native/kinematics.py')),
        joint_names=kin.joint_names,velocity_limits=velocity.tolist(),tcp_offset_xyz=kin.tcp_offset_xyz.tolist(),root_transform=root.tolist(),default_joints=base,
        collision_exemptions=collision.exemptions,collision_tolerance_m=1e-6)
    write(ROOT/'design/FK_source_binding.json',source_binding)
    files=[]
    for name in ('design','calibration','calibration_rigs','evidence'):
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file())
    files.extend((ROOT/'runtime').glob('pi0259_*.py'))
    files=sorted(set(files));entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    assert camera['image_pairs']<=512 and len(geometry)==36
    target.mkdir(parents=True)
    manifest=dict(schema='PI0259_H0_V1',stage='H0',producer_role='M',code_commit=commit,protocol_id=AUTHORITY,
        root=str(ROOT),entries=entries,identities={name:sha(ROOT/'design'/name) for name in ('registry.json','cue_geometry.json','FK_source_binding.json')},
        outcome_index=geometry,camera_outcome=camera,cohort='OLD_EXPOSED_QUARANTINED_CALIBRATION_ONLY',
        state_origin='KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT',evidence_class='KINEMATIC_RENDERED_REFERENCE',
        missing_fields=['No TRAIN processor/codec/model or fresh development before H0 correctness closure.',
            'Failed geometry suffixes have NaN joints/FK, never rendered as a valid continuation. Full desired references and every attempted solver step remain.'],
        counters=dict(calibration_sequences=36,calibration_image_pairs=camera['image_pairs'],physics_steps=0,WBC_actor_calls=0,model_queries=0,optimizer_updates=0),
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',review_state='PENDING_AGENT_V')
    write(target/'manifest.json',manifest)
    print(json.dumps(dict(manifest=str(target/'manifest.json'),sha256=sha(target/'manifest.json'),files=len(entries),camera_status=camera['status'])),flush=True)


if __name__=='__main__':seal(sys.argv[1])
