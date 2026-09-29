"""PI0259 fixed FK-start references and bounded residual-Jacobian continuation."""
from pathlib import Path
import copy
import itertools
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from pi0255_motion3 import sha, write, scene, ordering
from pi0258_kinematic import instructions
from pi0258_geometry import Collision, full_fk, defaults, homogeneous
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics

ROOT=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0259')
OLD=ROOT.parent/'PI0258'
AUTHORITY='240dedfd3954dd4cbbb48645a2aad5b19738e2c4'


def quintic(u):
    u=np.clip(np.asarray(u,dtype=float),0,1)
    return 10*u**3-15*u**4+6*u**5


def reference(reg,family,task,command,times):
    assert not family['split'].startswith('TEST'), 'TEST remains metadata only'
    t=np.asarray(times,dtype=float);initial=np.asarray(reg['initial_TCP_world'])
    p0=initial[:3,3];r0=initial[:3,:3];data=scene(reg,family)
    if task=='M1':
        u=quintic(t/6);p=p0+(data['targets'][command]-p0)*u[:,None]
        yaw=reg['geometry']['target_yaw_rad'][command]
        r=r0@Rotation.from_rotvec(np.column_stack((np.zeros(len(t)),np.zeros(len(t)),u*yaw))).as_matrix()
    elif task=='M2':
        side=1 if command==0 else -1;g=reg['geometry'];c=data['center']
        points=[p0,c+[g['route_waypoint_x_offsets_m'][0],side*g['route_side_offset_m'],0],
                c+[g['route_waypoint_x_offsets_m'][1],side*g['route_side_offset_m'],0],data['goal']]
        ends=np.array([0,3,6,10]);segments=np.clip(np.searchsorted(ends,t,side='right')-1,0,2)
        p=np.asarray([points[s]+quintic((v-ends[s])/(ends[s+1]-ends[s]))*(points[s+1]-points[s]) for v,s in zip(t,segments)])
        r=np.repeat(r0[None],len(t),axis=0)
    else:
        assert task=='M3';a=(-3 if command==0 else 3)*np.pi/4
        entry=data['center']+[data['radius']*np.cos(a),data['radius']*np.sin(a),0]
        p=p0+(entry-p0)*quintic(t/4)[:,None];active=t>=4
        angle=a+(1 if command==0 else -1)*1.5*np.pi*quintic((t[active]-4)/12)
        p[active]=data['center']+np.column_stack((data['radius']*np.cos(angle),data['radius']*np.sin(angle),np.zeros(len(angle))))
        r=np.repeat(r0[None],len(t),axis=0)
    return p,r


def residual(kin,root,q,p,r):
    actual=root@kin.tcp_pose(q)
    return np.r_[p-actual[:3,3],Rotation.from_matrix(r@actual[:3,:3].T).as_rotvec()]


def residual_jacobian(kin,root,q,p,r,h=1e-6):
    eye=np.eye(len(q))*h
    return np.column_stack([(residual(kin,root,q+v,p,r)-residual(kin,root,q-v,p,r))/(2*h) for v in eye])


def solve(kin,root,p,r,start,settings):
    q=np.asarray(start).copy();weight=np.array([1,1,1,.2,.2,.2]);projections=0
    for iteration in range(201):
        e=residual(kin,root,q,p,r)
        okay=np.linalg.norm(e[:3])<=.002 and np.linalg.norm(e[3:])<=.01
        if okay or iteration==200:return q,e,iteration,bool(okay),projections
        j=weight[:,None]*residual_jacobian(kin,root,q,p,r)
        delta=-j.T@np.linalg.solve(j@j.T+1e-6*np.eye(6),weight*e)
        delta*=min(1.,.1/max(np.max(abs(delta)),1e-12));cost=np.linalg.norm(weight*e)
        for scale in settings['line_search']:
            proposed=q+scale*delta;candidate=np.clip(proposed,kin.joint_limits[:,0],kin.joint_limits[:,1])
            projections+=int(not np.array_equal(proposed,candidate))
            if np.linalg.norm(weight*residual(kin,root,candidate,p,r))<cost:
                q=candidate;break
        else:return q,e,iteration+1,False,projections


def continuation(target,solver,validate,t0,q0,t1,ledger,depth=0):
    p0,r0=target(t0);p,r=target(t1)
    exceeds=np.linalg.norm(p-p0)>.005 or np.linalg.norm(Rotation.from_matrix(r@r0.T).as_rotvec())>.02
    if not exceeds or depth==4:
        q,e,n,okay,clipped=solver(p,r,q0)
        checks=validate(q0,q,t1-t0) if okay else dict(valid=False,reason='IK_RESIDUAL')
        accepted=bool(okay and checks['valid'])
        ledger.append(dict(t0=t0,t1=t1,depth=depth,start=q0.tolist(),candidate=q.tolist(),residual=e.tolist(),
            iterations=n,projected_candidates=clipped,ik_ok=okay,accepted=accepted,checks=checks))
        if accepted:return True,t1,q
    if depth==4:return False,t0,q0.copy()
    mid=(t0+t1)/2
    ok,accepted_t,accepted_q=continuation(target,solver,validate,t0,q0,mid,ledger,depth+1)
    if not ok:return False,accepted_t,accepted_q
    return continuation(target,solver,validate,mid,accepted_q,t1,ledger,depth+1)


def source_geometry(reg):
    kin=NativeArmKinematics.from_urdf(reg['urdf']['path']);collision=Collision(reg['urdf']['path'])
    source=json.loads(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0256/evidence/runtime_readback.json').read_text())
    positions=defaults(next(x['path'] for x in source['files'] if x['name']=='native_config'))
    root=homogeneous(Rotation.from_quat(reg['root_pose'][3:]).as_matrix(),reg['root_pose'][:3])
    velocity=np.array([float(next(j for j in collision.urdf.findall('joint') if j.attrib['name']==n).find('limit').attrib['velocity']) for n in kin.joint_names])
    return kin,collision,positions,root,velocity


def freeze():
    path=ROOT/'design/registry.json';assert not path.exists()
    reg=copy.deepcopy(json.loads((OLD/'design/registry.json').read_text()))
    assert sha(reg['urdf']['path'])==reg['urdf']['sha256']
    kin,collision,positions,root,velocity=source_geometry(reg)
    q=np.array(reg['arm_default']);initial=root@kin.tcp_pose(q)
    independent=full_fk(collision.urdf,positions,root)['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
    assert np.max(abs(independent-initial))<1e-10
    reg.update(schema='PI0259_KIN_MT_V2',authority=AUTHORITY,initial_TCP_world=initial.tolist(),
        source_default_joints=positions,source_registry_sha256=sha(OLD/'design/registry.json'),
        old_initial_difference=dict(position_m=float(np.linalg.norm(initial[:3,3]-reg['geometry']['nominal_tcp_world'])),
            identity_orientation_rad=float(np.linalg.norm(Rotation.from_matrix(initial[:3,:3]).as_rotvec()))))
    reg['geometry']['nominal_tcp_world']=initial[:3,3].tolist()
    reg['solver'].update(method='ACTUAL_WORLD_RESIDUAL_CENTRAL_JACOBIAN_NEGATIVE_DLS',warm_start='last ACCEPTED only',
        continuation_max_depth=4,subdivide_position_m=.005,subdivide_rotation_rad=.02,failed_interval='stop; retain accepted prefix and every attempt')
    calibrations=[]
    for i,f in enumerate(reg['families'][:6]):
        f=copy.deepcopy(f);f.update(family_id='pi0259-cal-%03d'%i,split='CALIBRATION',old_family_id=f['family_id']);calibrations.append(f)
    keys=('x_shift','y_shift','z_shift','radius')
    exposed={tuple(f['factors'][k] for k in keys) for f in calibrations}
    pool=[dict(zip(keys,v)) for v in itertools.product([-.04,-.02,0,.02,.04],[-.04,-.02,0,.02,.04],[0,.015,.03],[.035,.045,.055])]
    excluded=[f for f in pool if tuple(f[k] for k in keys) in exposed]
    pool=[f for f in pool if tuple(f[k] for k in keys) not in exposed]
    pool.sort(key=lambda f:ordering(f,'PI0259_GEOMETRY_SPLIT:25901'))
    train=pool[6:46]
    assert all(all(any(f[k]==x[k] for x in train) for k in keys) for f in pool[56:76]),'composition factor constraint'
    rng=np.random.default_rng(25902);colors=[['red','blue'],['blue','green'],['green','red']]
    color_order=np.tile(np.arange(3),26)[:76];rng.shuffle(color_order)
    families=[]
    for i,factors in enumerate(pool[:76]):
        f=dict(factors,colors=colors[int(color_order[i])]);split='DEVELOPMENT' if i<6 else 'TRAIN' if i<46 else 'KIN-VAL' if i<56 else 'TEST_COMPOSITION'
        families.append(dict(family_id='pi0259-family-%03d'%i,split=split,factors=f,
            center_world=(np.array(reg['geometry']['nominal_center_world'])+[f[k] for k in keys[:3]]).tolist(),
            geometry_key={k:f[k] for k in keys},state_origin='KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT'))
    for i,f in enumerate(reg['families'][-20:]):
        f=copy.deepcopy(f);f.update(family_id='pi0259-geometry-test-%03d'%i,split='TEST_GEOMETRY',manifest_only=True);families.append(f)
    rows=[];template_order=np.tile([0,1],51);rng.shuffle(template_order)
    for f in calibrations+families:
        for task in ('M1','M2','M3'):
            for command in (0,1):
                texts=instructions(task,command,f['factors']['colors']);index=(int(template_order[len(rows)//6])+command)%2
                if f['split']=='KIN-VAL' or f['split'].startswith('TEST'):index+=2
                rows.append(dict(sequence_id=f['family_id']+'-'+task+'-c'+str(command),family_id=f['family_id'],task=task,command=command,
                    split=f['split'],instruction=texts[index],paraphrases=texts,execute=not f['split'].startswith('TEST')))
    reg.update(families=families,calibration_families=calibrations,sequences=rows,candidate_factor_order=pool,
        duplicate_log=excluded,family_order='PI0259_GEOMETRY_SPLIT:25901',language_order='25902 balanced colors; balanced templates0/1 or2/3',
        reference_identity='PI0259_KIN_REF_V2',time_law='quintic10u3-15u4+6u5',camera_identity='PI0259_KIN_OBS_V2')
    # Envelope is frozen from non-TEST metadata, never from a successful trajectory.
    envelope=[initial[:3,3]]
    for f in calibrations+[x for x in families if not x['split'].startswith('TEST')]:
        data=scene(reg,f);g=reg['geometry'];c=data['center'];radius=data['radius']
        for p in data['targets']:
            envelope.extend([p-np.ones(3)*g['target_radius_m'],p+np.ones(3)*g['target_radius_m']])
        for i,p in enumerate(data['targets']):
            r=initial[:3,:3]@Rotation.from_euler('z',g['target_yaw_rad'][i]).as_matrix()
            envelope.extend(p+r[:,a]*g['axes_length_m'] for a in range(3))
        envelope.extend([data['obstacle']-g['obstacle_half_extents_m'],data['obstacle']+g['obstacle_half_extents_m'],data['goal'],c+[-radius,-radius,0],c+[radius,radius,0]])
        for x in g['route_waypoint_x_offsets_m']:
            envelope.extend([c+[x,g['route_side_offset_m'],0],c+[x,-g['route_side_offset_m'],0]])
    lower=np.min(envelope,axis=0);upper=np.max(envelope,axis=0)
    reg['renderer']=dict(rigs_base_positions=[[1.15,-1.15,.95],[.8,.8,.8],[.4,0,1.15]],horizontal_fov_deg=70,
        look_at_world=((lower+upper)/2).tolist(),envelope_min=lower.tolist(),envelope_max=upper.tolist(),
        resolution=[96,96],wrist_attachment=[.28,-.10,.12],wrist_local_y_rotation_deg=135,
        calibration_times={'M1':[0,1.5,3,4.5,6],'M2':[0,3,6,7.5,10],'M3':[0,4,8,12,16]},
        calibration_common_start_shared=True,max_planned_pairs_per_rig=150,selected_rig=None)
    reg['budgets']=dict(calibration_sequences=36,calibration_image_pairs=512,fresh_development=36,conditional_train_val=300,
        sequence_attempts=396,active_windows=60000,image_pairs=120000,physics_steps=0,optimizer_calls=12096)
    write(path,reg);print(json.dumps({'registry':sha(path),'initial_TCP_world':initial.tolist(),'old_start_difference':reg['old_initial_difference']}),flush=True)


def geometry(reg,seq):
    family=next(f for f in reg['calibration_families']+reg['families'] if f['family_id']==seq['family_id'])
    assert seq['split']=='CALIBRATION', 'fresh data needs sealed H0 review'
    folder=ROOT/'calibration'/seq['sequence_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    assert not (folder/'started.json').exists(),'incomplete attempt requires explicit recovery accounting'
    write(folder/'started.json',dict(sequence=seq,source_sha256=sha(__file__),registry_sha256=sha(ROOT/'design/registry.json')))
    kin,collision,base,root,velocity=source_geometry(reg);q=np.array(reg['arm_default']);data=scene(reg,family)
    times=np.arange(round(reg['experts'][seq['task']]['nominal_end_s']/.05)+1)*.05
    desired_p,desired_r=reference(reg,family,seq['task'],seq['command'],times)
    def target(t):
        p,r=reference(reg,family,seq['task'],seq['command'],[t]);return p[0],r[0]
    def validate(q0,q,dt):
        joints=dict(base);joints.update(zip(kin.joint_names,q));poses=full_fk(collision.urdf,joints,root)
        independent=poses['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
        assert np.max(abs(independent-root@kin.tcp_pose(q)))<1e-9
        hits=collision.check(poses,data['obstacle'],reg['geometry']['obstacle_half_extents_m'])
        hard=bool(np.all(q>=kin.joint_limits[:,0]) and np.all(q<=kin.joint_limits[:,1]))
        speed=bool(np.all(abs(q-q0)/dt<=velocity+1e-10));finite=bool(np.isfinite(independent).all())
        return dict(valid=hard and speed and finite and not hits,hard_ok=hard,velocity_ok=speed,finite=finite,contacts=hits)
    start=validate(q,q,.05);ledger=[];qs=[q.copy()];valid=[start['valid']];prefix_time=0.;prefix_q=q.copy()
    if start['valid']:
        for i in range(1,len(times)):
            okay,t,q=continuation(target,lambda p,r,q:solve(kin,root,p,r,q,reg['solver']),validate,float(times[i-1]),q,float(times[i]),ledger)
            prefix_time=t;prefix_q=q.copy()
            if not okay:break
            qs.append(q.copy());valid.append(True)
    completed=len(qs);all_q=np.full((len(times),6),np.nan);all_q[:completed]=qs
    fk=np.full((len(times),4,4),np.nan);fk[:completed]=[root@kin.tcp_pose(x) for x in qs]
    qualified=(np.arange(len(times))<completed)&bool(start['valid'])
    indices=np.arange(0,len(times),2);active=indices[times[indices]<reg['experts'][seq['task']]['motion_end_s']]
    windows=np.array([bool(np.all(qualified[np.minimum(i+np.arange(40),len(times)-1)])) for i in active])
    end=reg['experts'][seq['task']]['motion_end_s']
    quarters=[bool(np.any(windows[(times[active]>=k*end/4)&(times[active]<(k+1)*end/4)])) for k in range(4)]
    np.savez_compressed(folder/'geometry.npz',synthetic_reference_time=times,joints=all_q,FK_world=fk,
        desired_position=desired_p,desired_rotation=desired_r,sample_qualified=qualified,active_indices=active,
        active_geometry_qualified=windows,initial_TCP_world=np.array(reg['initial_TCP_world']),root_transform=root,
        last_accepted_internal_time=prefix_time,last_accepted_internal_joints=prefix_q)
    write(folder/'solver_ledger.json',dict(start_checks=start,attempts=ledger))
    result=dict(sequence=seq,completed_samples=completed,total_samples=len(times),last_accepted_time=prefix_time,
        status='GEOMETRY_PASS' if np.mean(windows)>=.95 and all(quarters) else 'VALID_GEOMETRY_FAIL',
        window_fraction=float(np.mean(windows)),quartiles=quarters,internal_solves=len(ledger),payload_sha256=sha(folder/'geometry.npz'),physics_steps=0)
    write(folder/'result.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    if '--freeze' in sys.argv:freeze()
    if '--calibrate' in sys.argv:
        reg=json.loads((ROOT/'design/registry.json').read_text())
        assert json.loads((ROOT/'evidence/cpu_fixtures.json').read_text())['status']=='PASS'
        outcomes=[geometry(reg,s) for s in reg['sequences'] if s['split']=='CALIBRATION']
        write(ROOT/'evidence/calibration_geometry.json',outcomes)
