"""Seal and read-only replay the finite DEV attempt set. No IK or renderer execution."""
from pathlib import Path
import collections
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).parent/'visualization'))
from pi0259_h0 import raster
from pi0259_review import require,semantic_command,pose
from pi0259_kinematic import source_geometry,full_fk,homogeneous
from go2arx5_vla.wbc.native.kinematics import rpy_to_matrix
from pi0255_motion3 import sha,write,scene,ordering
from pi0260_design import ROOT,AUTHORITY,key
from pi0260_route import WORKFLOW,OWNER
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1


def seal(commit):
    path=ROOT/'handoff/DEV_v1/manifest.json';require(not path.exists(),'already sealed')
    selection=json.loads((ROOT/'state/selection_DEVELOPMENT.json').read_text())
    files=[]
    for folder in ['design','proposals']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    files += [ROOT/'state'/s for s in ['H0_ACCEPTED.json','design_freeze.json','selection_DEVELOPMENT.json','DEV_launch.json']]
    files += [ROOT/'runtime'/s for s in ['pi0260_data.py','pi0260_render_family.py','pi0259_kinematic.py','pi0258_geometry.py','pi0259_observation.py']]
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    write(path,dict(schema='PI0260_DEV_V1',science=AUTHORITY,workflow=WORKFLOW,owner=OWNER,root=str(ROOT),code_commit=commit,
        entries=entries,selection=selection,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(manifest=str(path),sha256=sha(path),files=len(entries),bytes=sum(e['bytes'] for e in entries))))


def eligible(complete,fraction,quarters,required):
    return complete and fraction>=.95 and all(quarters) and required


def reconstruct(reg,fam,task,command,times):
    # Deliberately reconstruct the mathematical reference; do not call producer.reference.
    initial=np.array(reg['initial_TCP_world']);d=scene(reg,fam)
    q=lambda u:10*np.clip(u,0,1)**3-15*np.clip(u,0,1)**4+6*np.clip(u,0,1)**5
    if task=='M1':
        u=q(times/6);p=initial[:3,3]+u[:,None]*(d['targets'][command]-initial[:3,3])
        yaw=reg['geometry']['target_yaw_rad'][command]
        r=initial[:3,:3]@Rotation.from_rotvec(np.c_[np.zeros(len(u)),np.zeros(len(u)),u*yaw]).as_matrix()
    else:
        start=(-1 if command==0 else 1)*3*np.pi/4;direction=1 if command==0 else -1
        entry=d['center']+[d['radius']*np.cos(start),d['radius']*np.sin(start),0]
        p=initial[:3,3]+q(times/4)[:,None]*(entry-initial[:3,3]);arc=times>=4
        a=start+direction*1.5*np.pi*q((times[arc]-4)/12)
        p[arc]=d['center']+np.c_[d['radius']*np.cos(a),d['radius']*np.sin(a),np.zeros(len(a))]
        r=np.repeat(initial[None,:3,:3],len(times),axis=0)
    return p,r


def replay(digest):
    manifest=ROOT/'handoff/DEV_v1/manifest.json';require(sha(manifest)==digest,'MANIFEST_CHANGED')
    m=json.loads(manifest.read_text());files={r['path']:r for r in m['entries']}
    def file(name):
        require(name in files,'UNSEALED_INPUT '+name);p=ROOT/name;row=files[name]
        require(p.resolve().is_relative_to(ROOT) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'CHANGED_INPUT '+name)
        return p
    for name in files:file(name)
    out=ROOT.parent/'PI0260_review/DEV_v1';require(not out.exists(),'replay output exists');out.mkdir(parents=True)
    reg=json.loads(file('design/registry.json').read_text());selection=json.loads(file('state/selection_DEVELOPMENT.json').read_text())
    frozen=json.loads(file('state/design_freeze.json').read_text());require(frozen['registry_sha256']==sha(file('design/registry.json')),'design changed')
    families=reg['families'];require([f['split'] for f in families[:160]]==['DEVELOPMENT']*20+['TRAIN']*100+['KIN-VAL']*40,'STRATA_CHANGED')
    require([f['geometry_key'] for f in families]==sorted([f['geometry_key'] for f in families],key=lambda f:ordering(f,'PI0260_GEOMETRY_PROPOSALS:26001')),'ORDER_CHANGED')
    exclusions=json.loads(file('design/exclusions.json').read_text());excluded={tuple(x['geometry_key']) for x in exclusions['sources']}
    require(len({key(f,reg) for f in families})==len(families) and not {key(f,reg) for f in families}&excluded,'GEOMETRY_OVERLAP')
    kin,collision,defaults,root,velocity=source_geometry(reg);gate=TrajectorySafetyGateV1();anchor=np.array(reg['initial_TCP_world'])[:3,3]
    results=[];frame_cache={};collision_calls=0;raw_calls=0;max_fk_delta=0.;max_reference_delta=0.;max_state_delta=0.
    for family in families[:20]:
        outcomes=[];d=scene(reg,family)
        for seq in [s for s in reg['sequences'] if s['family_id']==family['family_id']]:
            base='proposals/'+family['family_id']+'/'+seq['sequence_id']+'/'
            require(semantic_command(seq['task'],seq['instruction'],family['factors']['colors'])==seq['command'],'INSTRUCTION_SWAPPED')
            with np.load(file(base+'geometry.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            times=a['synthetic_reference_time'];expected=np.arange(round(reg['experts'][seq['task']]['nominal_end_s']/.05)+1)*.05
            require(np.array_equal(times,expected),'TIME_GRID_CHANGED')
            desired_p,desired_r=reconstruct(reg,family,seq['task'],seq['command'],times)
            delta=max(float(np.max(abs(desired_p-a['desired_position']))),float(np.max(abs(desired_r-a['desired_rotation']))));max_reference_delta=max(max_reference_delta,delta)
            require(delta<1e-12,'REFERENCE_SEMANTICS_CHANGED')
            finite=np.isfinite(a['joints']).all(axis=1);require(np.array_equal(finite,np.arange(len(finite))<finite.sum()),'NONCONTIGUOUS_ACCEPTED_PREFIX')
            actual=np.array([root@kin.tcp_pose(q) for q in a['joints'][finite]])
            delta=float(np.max(abs(actual-a['FK_world'][finite])));max_fk_delta=max(max_fk_delta,delta);require(delta<1e-9,'FK_CHANGED')
            pos=np.linalg.norm(actual[:,:3,3]-desired_p[finite],axis=1)
            angle=Rotation.from_matrix(desired_r[finite]@actual[:,:3,:3].transpose(0,2,1)).magnitude()
            require(np.all(pos<=.002+1e-12) and np.all(angle<=.01+1e-12),'ACCEPTED_POSE_RESIDUAL')
            require(np.all(a['joints'][finite]>=kin.joint_limits[:,0]) and np.all(a['joints'][finite]<=kin.joint_limits[:,1]),'HARD_LIMIT')
            require(np.all(abs(np.diff(a['joints'][finite],axis=0))/.05<=velocity+1e-8),'VELOCITY_LIMIT')
            contacts=[]
            for i,qj in enumerate(a['joints'][finite]):
                joints=dict(defaults);joints.update(zip(kin.joint_names,qj));poses=full_fk(collision.urdf,joints,root)
                hits=collision.check(poses,d['obstacle'],reg['geometry']['obstacle_half_extents_m']);collision_calls+=1
                if hits:contacts.append(dict(index=i,hits=hits))
            complete=bool(finite.all() and not contacts)
            original=json.loads(file(base+'geometry_result.json').read_text())
            require(complete==(original['status']=='COMPLETE_GEOMETRY_PASS'),'PRODUCER_GEOMETRY_FLAG_CONTRADICTED')
            row=dict(sequence=seq['sequence_id'],task=seq['task'],command=seq['command'],complete_geometry=complete,
                finite_prefix=int(finite.sum()),requested_samples=len(times),collision_replayed_states=int(finite.sum()),contacts=contacts,
                position_residual_max=float(pos.max()),rotation_residual_max=float(angle.max()),raw_gate='NOT_RUN',observation='NOT_RUN')
            if base+'raw_gate.json' in files:
                raw=json.loads(file(base+'raw_gate.json').read_text());decisions=[]
                for saved in raw['rows']:
                    i=saved['index'];t=float(times[i]);ii=np.minimum(i+np.arange(40),len(times)-1)
                    chunk=TrajectoryChunkV1(seq['sequence_id']+'-'+str(i),'KINEMATIC_RENDERED_REFERENCE',t,t+np.arange(40)*.05,
                        desired_p[ii]-anchor,Rotation.from_matrix(desired_r[ii]).as_quat(),'NATIVE_WBC_RELATIVE_REFERENCE','x5_link6_gripper_center',t,t+1.95,gripper=np.full(40,.5))
                    verdict=gate.evaluate(chunk,now=t,achieved_position=actual[i,:3,3]-anchor,achieved_orientation=Rotation.from_matrix(actual[i,:3,:3]).as_quat()).as_dict();raw_calls+=1
                    require(verdict['decision']==saved['verdict']['decision'] and verdict['issues']==saved['verdict']['issues'],'RAW_GATE_REPLAY_MISMATCH')
                    decisions.append(dict(time=t,good=verdict['decision']!='REJECT',active=saved['active']))
                active=[r for r in decisions if r['active']];end=reg['experts'][seq['task']]['motion_end_s']
                require([r['time'] for r in active]==times[a['active_indices']].tolist(),'MISSING_RAW_ACTIVE_WINDOWS')
                required=[r for r in decisions if any(abs(r['time']-t)<1e-8 for t in reg['renderer']['calibration_times'][seq['task']])]
                q=[any(r['good'] and k*end/4<=r['time']<(k+1)*end/4 for r in active) for k in range(4)]
                row['raw_gate']='PASS' if eligible(complete,sum(r['good'] for r in active)/len(active),q,all(r['good'] for r in required)) else 'FAIL'
            if base+'observations.json' in files:
                observation=json.loads(file(base+'observations.json').read_text());checks=[]
                indices=sorted(set(a['active_indices'])|{round(t/.05) for t in reg['renderer']['calibration_times'][seq['task']]})
                require([r['index'] for r in observation['rows']]==indices,'OMITTED_OBSERVATION_TIME')
                for obs in observation['rows']:
                    fbase='proposals/'+family['family_id']+'/frames/'+obs['frame'];i=obs['index'];t=float(times[i])
                    if fbase not in frame_cache:
                        with np.load(file(fbase+'.npz'),allow_pickle=False) as z:frame={k:z[k].copy() for k in z.files}
                        meta=json.loads(file(fbase+'.json').read_text());require(sha(file(fbase+'.npz'))==meta['array_sha256'],'FRAME_HASH_MISMATCH')
                        frame_cache[fbase]=(frame,meta)
                        for s in ('body','wrist'):
                            require(frame[s+'_rgb'].shape==(96,96,3) and frame[s+'_rgb'].dtype==np.uint8,'RGB_ABI')
                            require(frame[s+'_depth'].shape==frame[s+'_segmentation'].shape==(96,96),'DIAGNOSTIC_ABI')
                        fk=root@kin.tcp_pose(frame['arm_joints']);native=np.linalg.inv(pose(frame['T_world_native']))@fk
                        state=np.r_[native[:3,3],native[:2,:3].reshape(-1),.5]
                        delta=float(np.max(abs(state-frame['state_v1'])));max_state_delta=max(max_state_delta,delta);require(delta<1e-9,'STATE_FRAME_MISMATCH')
                        require(np.max(abs(fk-frame['FK_world']))<1e-9 and meta['sim_time']==0 and meta['physics_steps']==0,'IMAGE_FK_OR_PHYSICS')
                        joints=dict(defaults);joints.update(zip(kin.joint_names,frame['arm_joints']));poses=full_fk(collision.urdf,joints,root)
                        require(all(np.max(abs(poses[n]-v))<1e-9 for n,v in meta['link_transforms_world'].items()),'RENDER_LINK_PLACEMENT')
                        cue_names=['target0','axis0_0','axis0_1','axis0_2','target1','axis1_0','axis1_1','axis1_2','obstacle','goal']+[f'contour{k}' for k in range(24)]+['black_A','white_B']
                        require(meta['scene_ids']==dict(zip(cue_names,range(100,136))),'CUE_ID_ALTERED')
                        for stream in ('body','wrist'):
                            camera=pose(np.linalg.inv(np.asarray(meta['camera_view'][stream]).T));proj=np.array(meta['camera_projection'][stream])
                            if stream=='body':
                                position=(root@np.r_[reg['renderer']['rigs_base_positions'][frozen['selected_rig']],1])[:3]
                                axis=np.array(reg['renderer']['look_at_world'])-position;axis/=np.linalg.norm(axis)
                                right=np.cross(axis,[0.,0.,1.]);right/=np.linalg.norm(right)
                                expected=homogeneous(np.column_stack((right,np.cross(right,axis),-axis)),position)
                                require(abs(2*np.arctan(1/proj[0,0])*180/np.pi-70)<1e-4,'BODY_FOV')
                            else:
                                expected=poses['x5_link6']@homogeneous(rpy_to_matrix(np.array([0,3*np.pi/4,0])),np.array([.28,-.10,.12]))
                                expected[:3,:3]=expected[:3,:3]@np.array([[0,0,-1],[-1,0,0],[0,1,0]])
                            require(np.allclose(camera,expected,atol=1e-6,rtol=0),'CAMERA_EXTRINSIC_CHANGED')
                            for sid in meta['scene_ids'].values():
                                mask=frame[stream+'_segmentation']==sid
                                require(np.isfinite(frame[stream+'_depth'][mask]).all() and np.all(frame[stream+'_depth'][mask]<0),'CUE_DEPTH_INVALID')
                    frame,meta=frame_cache[fbase]
                    require(np.array_equal(frame['arm_joints'],a['joints'][i]) and abs(obs['time']-t)<1e-9,'OBSERVATION_STATE_PAIRING')
                    check=raster(frame,meta['scene_ids'],seq['task'],seq['command'],t==0)
                    checks.append(dict(index=i,time=t,good=check['necessary_pass'],failures=check['failures'],counts=check['counts'],frame=fbase))
                active=[r for r in checks if r['index'] in set(a['active_indices'])];end=reg['experts'][seq['task']]['motion_end_s']
                q=[any(r['good'] and k*end/4<=r['time']<(k+1)*end/4 for r in active) for k in range(4)]
                required=[r for r in checks if any(abs(r['time']-t)<1e-8 for t in reg['renderer']['calibration_times'][seq['task']])]
                fraction=sum(r['good'] for r in active)/len(active)
                row.update(observation='PASS' if eligible(complete,fraction,q,all(r['good'] for r in required)) else 'FAIL',
                    visibility_fraction=fraction,visibility_quartiles=q,failed_cues=[r for r in checks if not r['good']],common_frame=checks[0]['frame'])
            outcomes.append(row)
        family_ok=all(r['complete_geometry'] and r['raw_gate']=='PASS' and r['observation']=='PASS' for r in outcomes)
        require(family_ok==(family['family_id'] in selection['selected']),'PRODUCER_FAMILY_FLAG_CONTRADICTED')
        starts={r['common_frame'] for r in outcomes if 'common_frame' in r};require(len(starts)<=1,'PAIRED_START_INPUT_MISMATCH')
        result=dict(family=family['family_id'],eligible=family_ok,geometry_key=family['geometry_key'],sequences=outcomes)
        results.append(result);write(out/(family['family_id']+'.json'),result)
        print(json.dumps(dict(family=family['family_id'],eligible=family_ok,collision_calls=collision_calls)),flush=True)
    selected=[r['family'] for r in results if r['eligible']]
    require(selected==selection['selected'],'FIRST_VALID_ORDER_CHANGED')
    tests=dict(producer_PASS_cannot_override_arrays=not eligible(True,.94,[True]*4,True),
        missing_first_quartile_rejected=not eligible(True,1,[False,True,True,True],True),
        invalid_geometry_rejected=not eligible(False,1,[True]*4,True),mandatory_state_rejected=not eligible(True,1,[True]*4,False))
    require(all(tests.values()),'NEGATIVE_TEST_FAILED')
    receipt=dict(stage='DEV',review_mode='SAME_OWNER_READONLY_REPLAY',execution_owner=OWNER,science=AUTHORITY,workflow=WORKFLOW,
        producer_manifest_sha256=digest,verifier_sha256=sha(__file__),files_checked=len(files),selected=selected,
        status='PASS_DEV_READONLY_REPLAY' if len(selected)==6 else 'PI0260_ELIGIBLE_SCENE_SHORTFALL',required_families=6,evaluated_families=len(results),
        geometry_complete_sequences=sum(s['complete_geometry'] for r in results for s in r['sequences']),
        behavior_complete_geometry={t:sum(s['complete_geometry'] for r in results for s in r['sequences'] if s['task']==t) for t in ('M1','M3')},
        full_family_geometry=sum(all(s['complete_geometry'] for s in r['sequences']) for r in results),
        saved_frames_checked=len(frame_cache),collision_replayed_states=collision_calls,raw_gate_recomputed_windows=raw_calls,
        maximum_fk_delta=max_fk_delta,maximum_reference_delta=max_reference_delta,maximum_state_delta=max_state_delta,
        negative_tests=tests,shared_primitives='Original full URDF FK/FCL, gate; reference algebra reconstructed; V raster rules; producer eligibility flags compared only after replay',
        cumulative_counters=selection['counters'],new_model_queries=0,new_optimizer_updates=0,new_IK_calls=0,physics_steps=0,
        next_stage='TRAIN_PROPOSALS' if len(selected)==6 else 'STOP_FIXED_DEV_STRATUM_EXHAUSTED',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    write(out/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='--seal':seal(sys.argv[2])
    else:replay(sys.argv[1])
