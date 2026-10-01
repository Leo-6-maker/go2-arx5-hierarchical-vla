"""Bounded actual expert-WBC comparison; no GR00T or training entry point."""
import sys
if '--collision-worker' not in sys.argv:
    import isaacgym
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import textwrap
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from pi0265_source import ROOT,BASE,sha,write
from pi0265_metrics import score,check_clock,so3

REF=BASE/'PI0250/runtime/reference-317a'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0265')
sys.path[:0]=[str(REF/'src'),str(REF/'scripts'),str(BASE/'PI0256/runtime')]
from pi0255_collect import append,array_sha,WBC
from pi0255_motion3 import scene


def forbidden_hits(hits):
    return [x for x in hits if 'ground' not in x['pair']]


def source_reference(reg,family,task,command,times):
    # Reuse the exact authorized source functions without importing its IK pipeline.
    path=ROOT/'code/pi0259_kinematic.py';text=path.read_text();nodes=ast.parse(text)
    selected=[n for n in nodes.body if isinstance(n,ast.FunctionDef) and n.name in ('quintic','reference')]
    scope=dict(np=np,Rotation=Rotation,scene=scene)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    return scope['reference'](reg,family,task,command,times)


def collision_worker():
    from pi0258_geometry import Collision,full_fk,homogeneous
    source=json.loads((ROOT/'evidence/SOURCE_IDENTITY.json').read_text())
    collision=Collision(next(x['path'] for x in source['files'] if x['name']=='native_urdf'))
    print(json.dumps(dict(ready=True,exemptions=collision.exemptions)),flush=True)
    for line in sys.stdin:
        row=json.loads(line);root=homogeneous(Rotation.from_quat(row['root'][3:7]).as_matrix(),np.array(row['root'][:3]))
        poses=full_fk(collision.urdf,row['joints'],root)
        hits=collision.check(poses,row.get('obstacle'),row.get('half_extents'))
        # Ground support is handled by unchanged native termination; this check covers forbidden self/obstacle pairs.
        hits=forbidden_hits(hits)
        print(json.dumps(dict(hits=hits)),flush=True)


def scene_class(base,reg,family,gymapi):
    data=scene(reg,family);geo=reg['geometry']
    class ActualScene(base):
        def _create_envs(self):
            base.__mro__[1]._create_envs(self)
            self.pi0265_axes=[]
            def marker(p,size,color,name,box=False,collision=False):
                options=gymapi.AssetOptions();options.fix_base_link=True;options.disable_gravity=True
                asset=self.gym.create_box(self.sim,*size,options) if box else self.gym.create_sphere(self.sim,float(size),options)
                pose=gymapi.Transform();pose.p=gymapi.Vec3(*np.asarray(p).tolist())
                actor=self.gym.create_actor(self.envs[0],asset,pose,name,0 if collision else 1,0,100+len(self.pi0265_axes))
                self.gym.set_rigid_body_color(self.envs[0],actor,0,gymapi.MESH_VISUAL,gymapi.Vec3(*color))
                return actor
            colors=dict(red=(.9,.03,.03),green=(.03,.85,.1),blue=(.03,.15,.9))
            r0=np.array(reg['initial_TCP_world'])[:3,:3]
            for i,p in enumerate(data['targets']):
                marker(p,geo['target_radius_m'],colors[family['factors']['colors'][i]],'target%d'%i)
                r=r0@Rotation.from_euler('z',geo['target_yaw_rad'][i]).as_matrix()
                for a,color in enumerate(((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))):
                    handle=marker(p+r[:,a]*geo['axes_length_m'],geo['axes_marker_radius_m'],color,'axis%d_%d'%(i,a))
                    self.pi0265_axes.append((handle,i,a))
            self.pi0265_obstacle=marker(data['obstacle'],2*np.asarray(geo['obstacle_half_extents_m']),(.45,.45,.45),'obstacle',True,True)
            self.stage21_target_actor=marker(data['goal'],geo['target_radius_m'],(1.,.35,.03),'goal')
            self.stage21_target_root_index=int(self.gym.get_actor_index(self.envs[0],self.stage21_target_actor,gymapi.DOMAIN_SIM))
            for i,a in enumerate(np.linspace(-3*np.pi/4,3*np.pi/4,geo['contour_points'])):
                marker(data['center']+[data['radius']*np.cos(a),data['radius']*np.sin(a),0],geo['contour_marker_radius_m'],(.95,.8,.05),'contour%d'%i)
            for name,a,color in [('black_A',-3*np.pi/4,(.03,.03,.03)),('white_B',3*np.pi/4,(1.,1.,1.))]:
                marker(data['center']+[data['radius']*np.cos(a),data['radius']*np.sin(a),0],geo['start_marker_radius_m'],color,name)
            props=gymapi.CameraProperties();props.width=props.height=96;props.horizontal_fov=reg['renderer']['horizontal_fov_deg']
            body=self.gym.create_camera_sensor(self.envs[0],props)
            root=np.array(reg['root_pose']);position=Rotation.from_quat(root[3:]).apply(reg['renderer']['rigs_base_positions'][reg['renderer']['selected_rig']])+root[:3]
            self.gym.set_camera_location(body,self.envs[0],gymapi.Vec3(*position),gymapi.Vec3(*reg['renderer']['look_at_world']))
            wrist_props=gymapi.CameraProperties();wrist_props.width=wrist_props.height=96
            wrist=self.gym.create_camera_sensor(self.envs[0],wrist_props)
            attachment=gymapi.Transform();attachment.p=gymapi.Vec3(*reg['renderer']['wrist_attachment'])
            attachment.r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0),np.deg2rad(reg['renderer']['wrist_local_y_rotation_deg']))
            link=self.gym.find_actor_rigid_body_handle(self.envs[0],self.actor_handles[0],self.body_names[self.ee_idx])
            self.gym.attach_camera_to_body(wrist,self.envs[0],link,attachment,gymapi.FOLLOW_TRANSFORM)
            self.stage21_camera_handles=dict(body=body,wrist=wrist)
    return ActualScene


def run(case_id):
    import torch
    import pytorch3d.transforms as transforms
    import stage21_native_receding_horizon as api
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
    from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
    from modules.trajectory import to_canonical,quat_to_mat
    seal=json.loads((ROOT/'design/EXECUTION_SEAL.json').read_text())
    for path,digest in seal['files'].items():assert sha(ROOT/path)==digest,path
    card=json.loads((ROOT/'design/SCENE_CARD.json').read_text());reg=card['registry']
    cases=card['G1_initial_cases']+json.loads((ROOT/'design/CONFIRM_CARD.json').read_text())['conditional_cases']
    case=next(x for x in cases if x['case_id']==case_id)
    source=json.loads((ROOT/'evidence/SOURCE_IDENTITY.json').read_text())
    paths={x['name']:x['path'] for x in source['files']}
    for row in source['files']:assert sha(row['path'])==row['sha256']
    if case['version']=='ROW6D':transforms.matrix_to_rotation_6d=lambda m:m[..., :2,:].clone().reshape(m.shape[:-2]+(6,))
    case_root=RAW/'controls'/case_id;case_root.mkdir(parents=True,exist_ok=True)
    prior=list(case_root.glob('attempt-*/result.json'))
    assert not any(json.loads(p.read_text())['validity']=='VALID' and not (p.parent/'INVALIDATION.json').exists() for p in prior),'first valid retained'
    assert len(list(case_root.glob('attempt-*')))==len(prior),'live/unresolved attempt'
    out=case_root/('attempt-%02d'%len(prior))
    out.mkdir(parents=True);(out/'frames').mkdir();(out/'actor_inputs').mkdir()
    result=dict(case=case,validity='ENGINEERING_INCOMPLETE',success=False,selected_checkpoint='NONE',GR00T_queries=0,
        execution_seal_sha256=sha(ROOT/'design/EXECUTION_SEAL.json'))
    native=None;worker=None;records=[];actor_calls=[];camera_checks=[]
    kin=NativeArmKinematics.from_urdf(paths['native_urdf']);urdf=ET.parse(paths['native_urdf']).getroot()
    bounds={j.attrib['name']:dict(lower=float(j.find('limit').attrib['lower']),upper=float(j.find('limit').attrib['upper']),
        unit='m' if j.attrib['type']=='prismatic' else 'rad') for j in urdf.findall('joint') if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
    gate=TrajectorySafetyGateV1();clock=lambda n:float(n.gym.get_sim_time(n.sim))
    def fk(n):
        root=n.root_states[0].detach().cpu().numpy();q=np.array([n.dof_pos[0,n.dof_names.index(j)].item() for j in kin.joint_names])
        pose=kin.tcp_pose(q);r=Rotation.from_quat(root[3:7]).as_matrix();return r@pose[:3,3]+root[:3],r@pose[:3,:3]
    def chunk(p,q,t):
        return TrajectoryChunkV1(chunk_id=case_id,source='PI0265_ACTUAL_EXPERT',observation_timestamp=float(t[0]),
            point_timestamps=t,eef_position=p,eef_orientation=q,source_frame=api.SOURCE_FRAME,tcp_semantic_id=api.TCP_SEMANTIC_ID,
            validity_start=float(t[0]),validity_end=float(t[-1]),gripper=np.full(len(t),.5))
    def snapshot(n):
        p,r=fk(n);native_p,native_q=api._current_eef(n)
        s=n.traj_batch.s_ref(n.traj_sim_time);desired=n.traj_batch.p_at(s)[0].detach().cpu().numpy()
        desired_r=Rotation.from_quat(n.traj_batch.quat_at(s)[0].detach().cpu().numpy()).as_matrix()
        return dict(time=clock(n),root=n.root_states[0].detach().cpu().tolist(),DOF=n.dof_state.detach().cpu().tolist(),
            TCP=p.tolist(),rotation=r.tolist(),native_TCP=native_p.tolist(),native_rotation=Rotation.from_quat(native_q).as_matrix().tolist(),
            native_FK_xyz_m=float(np.linalg.norm(p-native_p)),native_FK_SO3_rad=float(so3(r,Rotation.from_quat(native_q).as_matrix())),
            installed=desired.tolist(),installed_rotation=desired_r.tolist(),cursor=n.traj_sim_time.detach().cpu().tolist(),
            telemetry=api._native_runtime_telemetry(n))
    def immutable(n,env=None):
        values={k:api._tensor_hash(getattr(n,k)) for k in ('root_states','dof_state','rigid_body_state','end_effector_state','episode_length_buf','reset_buf','traj_sim_time','traj_s') if hasattr(n,k)}
        values['time']=clock(n);values['python_rng']=repr(random.getstate());state=np.random.get_state()
        values['numpy_rng']=[state[0],array_sha(state[1]),*state[2:]]
        values['torch_rng']=api._tensor_hash(torch.get_rng_state());values['cuda_rng']=[api._tensor_hash(v) for v in torch.cuda.get_rng_state_all()]
        if env is not None:values['history']={k:api._tensor_hash(v) for k,v in vars(env).items() if 'history' in k and hasattr(v,'detach')}
        return values
    original=api._Stage21NativeEnvFactory.make
    def factory(WBCEnv,initial,torch,gymapi,gymtorch):
        base=original(WBCEnv,initial,torch,gymapi,gymtorch)
        cls=base if case['task']=='C0' else scene_class(base,reg,case['family'],gymapi)
        class Audited(cls):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                cached=self.end_effector_state.clone();raw=api._tensor_hash(self.rigid_body_state)
                self.end_effector_state=cached
                assert raw==api._tensor_hash(self.rigid_body_state)
                write(out/'cache_isolation.json',dict(clone_once_before_implicit_step=True,raw_unchanged=True,gripper_drives='ACTIVE_FIXED_ZERO'))
            def _load_trajectory_for(self,ids):
                if hasattr(self,'pi0265_reference'):return super()._load_trajectory_for(ids)
                p0,r0=fk(self)
                proxy=SimpleNamespace(_shoulder_frame=self._shoulder_frame,_traj_anchor_offset=self._traj_anchor_offset,
                    base_quat=self.base_quat,reach_table=self.reach_table,cfg=self.cfg,device=self.device,
                    traj_batch=SimpleNamespace(gamma_p=torch.zeros_like(self.traj_batch.gamma_p)))
                base._anchor_external_reference(proxy,ids);anchor=proxy.traj_batch.gamma_p[0,0].detach().cpu().numpy().copy()
                t=np.arange(round(case['max_task_s']/.05)+1)*.05
                if case['task']=='C0':
                    actual=dict(np.load(BASE/'PI0252/pre_full_sim/controls/A_attempt2/actual_expert_reference.npz',allow_pickle=False))
                    c0=json.loads((ROOT/'evidence/C0_SOURCE_INDEX.json').read_text())['control']
                    old_anchor=np.array(c0['anchor_world_m'])
                    ts=actual['timestamp'];p=np.column_stack([np.interp(t,ts,actual['position'][:,a]) for a in range(3)])+old_anchor
                    q=np.repeat(actual['quaternion'][:1],len(t),axis=0)
                    assert np.max(abs(actual['quaternion']-actual['quaternion'][0]))<1e-6
                else:
                    task_reg=copy.deepcopy(reg);pose=np.eye(4);pose[:3,3]=p0;pose[:3,:3]=r0;task_reg['initial_TCP_world']=pose.tolist()
                    p,r=source_reference(task_reg,case['family'],case['task'],case['command'],t);q=Rotation.from_matrix(r).as_quat()
                    # Bind only visual target-axis actors to the same frozen measured-reset R0 before physics.
                    data=scene(reg,case['family']);indices=[]
                    for handle,i,a in self.pi0265_axes:
                        index=int(self.gym.get_actor_index(self.envs[0],handle,gymapi.DOMAIN_SIM));indices.append(index)
                        target_r=r0@Rotation.from_euler('z',reg['geometry']['target_yaw_rad'][i]).as_matrix()
                        self.root_states[index,:3]=self.root_states.new_tensor(data['targets'][i]+target_r[:,a]*reg['geometry']['axes_length_m'])
                    tensor=torch.tensor(indices,device=self.device,dtype=torch.int32)
                    self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(tensor),len(indices))
                    self.pi0265_task_reg=task_reg
                c=chunk(p-anchor,q,t)
                startup=initial if case['task']=='C0' else c
                verdict=gate.evaluate(startup,now=0.,achieved_position=p0-anchor,achieved_orientation=Rotation.from_matrix(r0).as_quat())
                write(out/'construction_gate.json',verdict.as_dict())
                self.pi0265_gate_reject=verdict.decision is GateDecision.REJECT
                if self.pi0265_gate_reject:raise ValueError('INITIAL_REFERENCE_GATE_REJECT_NO_PHYSICS')
                self._stage21_initial_chunk=startup;self.pi0265_reference=dict(timestamp=t,position=p,quaternion=q,anchor=anchor)
                np.savez_compressed(out/'reference.npz',**self.pi0265_reference,reset_TCP=p0,reset_rotation=r0)
                return super()._load_trajectory_for(ids)
            def step(self,*args,**kwargs):
                import fcntl
                before=snapshot(self)
                with (ROOT/'physics.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);ledger=ROOT/'evidence/RAW_BUDGET.json'
                    budget=json.loads(ledger.read_text()) if ledger.exists() else dict(starts=0,control_ticks=0,simulate_calls=0,actor_calls=0,simulated_seconds=0.)
                    if not (out/'physics_started.json').exists():
                        assert budget['starts']<352;budget['starts']+=1
                        write(out/'physics_started.json',dict(case_id=case_id,ordinal=budget['starts'],pid=os.getpid(),epoch=time.time(),state=before))
                        np.savez_compressed(out/'pre_first_step.npz',root=self.root_states.detach().cpu().numpy(),dof=self.dof_state.detach().cpu().numpy(),
                            torch_rng=torch.get_rng_state().numpy(),cuda_rng=torch.cuda.get_rng_state().cpu().numpy())
                    assert budget['control_ticks']<369600 and budget['simulated_seconds']+float(self.dt)<=7392+1e-6
                    budget['control_ticks']+=1;budget['simulate_calls']+=int(self.cfg.control.decimation)
                    budget['simulated_seconds']+=float(self.dt);budget['actor_calls']+=2 if self.stage21_wbc_forward_count else 0
                    write(ledger,budget)
                returned=super().step(*args,**kwargs);after=snapshot(self)
                event=getattr(self,'_stage16_pending_termination',None)
                append(out/'physics_steps.jsonl',dict(before=before,after=after,pre_autoreset=event),sync=False)
                return returned
        return Audited
    api._Stage21NativeEnvFactory.make=staticmethod(factory)
    try:
        initial=dict(np.load('/home/liuyu-nfs/pi0223_goal_b_20260918/refs_r00/train-a_c0_continuous_reference.npz',allow_pickle=False))
        args=SimpleNamespace(source_root=WBC,sim_device='cuda:0',seed=case['reset_seed'],mode='gt_receding',
            arm=Path(paths['arm_checkpoint']),dog=Path(paths['dog_checkpoint']))
        native,env,actors,loads,cfg=api._build_native(args,initial);result['WBC_loads']=loads
        assert abs(native.dt-.02)<1e-8
        origin=clock(native);result['initialization_sim_s']=origin;assert origin<=1+1e-6
        initial_event=native.pop_stage16_termination()
        if initial_event or bool(native.reset_buf[0].item()):
            result.update(validity='VALID',failure='NATIVE_INITIALIZATION_TERMINATION',safety_failure=True,pre_autoreset=initial_event);return result
        ref=native.pi0265_reference;anchor=native.stage21_anchor_world.detach().cpu().numpy().copy();assert np.linalg.norm(anchor-ref['anchor'])<1e-5
        c=chunk(ref['position']-anchor,ref['quaternion'],origin+ref['timestamp'])
        p,r=fk(native);verdict=gate.evaluate(c,now=origin,achieved_position=p-anchor,achieved_orientation=Rotation.from_matrix(r).as_quat())
        write(out/'provider_gate.json',verdict.as_dict())
        if verdict.decision is GateDecision.REJECT:
            result.update(validity='VALID',failure='GATE_REJECT_BEFORE_INSTALL',safety_failure=True);return result
        before=immutable(native,env)
        if case['route']=='DIRECT_NATIVE':
            gamma,law=to_canonical(torch.tensor(c.eef_position,dtype=torch.float32),quat_to_mat(torch.tensor(c.eef_orientation,dtype=torch.float32)),.05)
            ids=torch.tensor([0],device=native.device);native.traj_batch.load(ids,[gamma],[law]);native.traj_batch.gamma_p[ids]+=native.stage21_anchor_world.view(1,1,3)
            native._stage21_reset_reference_progress(ids);native._stage21_current_chunk=c
            event=dict(route='DIRECT_NATIVE_CANONICAL_TABLE')
        else:event=native.initial_bind_reference(c)
        after=immutable(native,env)
        for key in ('root_states','dof_state','rigid_body_state','end_effector_state','episode_length_buf','reset_buf','history','time','python_rng','numpy_rng','torch_rng','cuda_rng'):assert before[key]==after[key],key
        native._update_ee_task_space_error();assert before['history']==immutable(native,env)['history']
        write(out/'reference_only_install.json',dict(event=event,unchanged_keys=list(before),native_cache_sync=True))
        arrays={k:v.detach().cpu().numpy() for k,v in vars(native.traj_batch).items() if hasattr(v,'detach')}
        np.savez_compressed(out/'installed_table.npz',**arrays)
        worker_env=dict(os.environ,PYTHONPATH=':'.join(map(str,[ROOT/'code',REF/'src',BASE/'PI0256/runtime'])))
        for name in ('LD_LIBRARY_PATH','LD_PRELOAD'):worker_env.pop(name,None)
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(ROOT/'code/pi0265_control.py'),'--collision-worker'],
            env=worker_env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        ready=json.loads(worker.stdout.readline());assert ready['ready'];write(out/'collision_exclusions.json',ready)
        original_actors=actors;tick=[0]
        class RecordActor:
            def __init__(self,actor,name):self.actor,self.name=actor,name
            def __call__(self,value):
                array=value.detach().cpu().numpy().copy();answer=self.actor(value)
                np.savez_compressed(out/'actor_inputs'/('%04d-%s.npz'%(tick[0],self.name)),history=array,output=answer.detach().cpu().numpy())
                actor_calls.append(dict(tick=tick[0],actor=self.name,time=clock(native),shape=list(array.shape),input_sha256=array_sha(array)))
                assert np.isfinite(array).all() and torch.isfinite(answer).all();return answer
        actors=(RecordActor(actors[0],'arm'),RecordActor(actors[1],'dog'),actors[2],actors[3])
        limits=cfg.asset.terminate_after_contacts_on
        ground_links=[name for name in native.body_names if any(item in name for item in limits)]
        family=case.get('family');data=scene(reg,family) if family else None
        if case['task']=='C0':spec=dict(task='M1',goal=json.loads((ROOT/'evidence/C0_SOURCE_INDEX.json').read_text())['control']['target_world_m'],
            goal_rotation=Rotation.from_quat(ref['quaternion'][-1]).as_matrix().tolist())
        else:spec=dict(task=case['task'],command=case['command'],center=data['center'].tolist(),radius=data['radius'],
            goal=ref['position'][-1].tolist(),goal_rotation=Rotation.from_quat(ref['quaternion'][-1]).as_matrix().tolist())
        write(out/'task_spec.json',spec);safety=False;next_frame=origin
        while clock(native)-origin<case['max_task_s']-native.dt/2:
            now=clock(native);pre=snapshot(native)
            assert pre['native_FK_xyz_m']<=1e-5 and pre['native_FK_SO3_rad']<=1e-4
            if now>=next_frame-1e-6:
                a=immutable(native,env);body,wrist=api._capture_scene(native);b=immutable(native,env);assert a==b,'CAMERA_MUTATED_CONTROL_OR_RNG'
                camera_checks.append(dict(time=now,exact_unchanged=True));np.savez_compressed(out/'frames'/('%04d.npz'%tick[0]),body_rgb=body,wrist_rgb=wrist,time=now)
                next_frame+=.1
            penetrations={name:max(0.,bounds[name]['lower']-native.dof_pos[0,i].item(),native.dof_pos[0,i].item()-bounds[name]['upper'])
                for i,name in enumerate(native.dof_names) if name in bounds}
            unsafe=[name for name,v in penetrations.items() if v>(1e-5 if bounds[name]['unit']=='m' else 1e-3)]
            worker.stdin.write(json.dumps(dict(root=pre['root'],joints={name:native.dof_pos[0,i].item() for i,name in enumerate(native.dof_names)},
                obstacle=data['obstacle'].tolist() if data else None,half_extents=reg['geometry']['obstacle_half_extents_m'],native_ground_termination_links=ground_links))+'\n');worker.stdin.flush()
            collision=json.loads(worker.stdout.readline())
            telemetry=pre['telemetry'];safety=bool(unsafe or collision['hits'] or telemetry.get('invalid_state') or telemetry.get('fall_flag') or telemetry.get('joint_limit_termination'))
            records.append(dict(pre,elapsed=now-origin,hard_penetration=penetrations,hard_unsafe=unsafe,collision=collision))
            if safety:result['failure']='PRE_STEP_SAFETY';break
            with torch.no_grad():done,policy=api._policy_step(native,env,actors)
            event=native.pop_stage16_termination();post=snapshot(native)
            check_clock(post['time'],now,native.dt,post['time'])
            if done or event or bool(native.reset_buf[0].item()):
                result['failure']='NATIVE_TERMINATION';result['pre_autoreset']=event;safety=True;break
            tick[0]+=1
        if not safety:
            final=snapshot(native);records.append(dict(final,elapsed=final['time']-origin))
        if records:
            t=np.array([x['elapsed'] for x in records]);p=np.array([x['TCP'] for x in records]);r=np.array([x['rotation'] for x in records])
            installed=np.array([x['installed'] for x in records]);ir=np.array([x['installed_rotation'] for x in records])
            metrics=score(t,p,r,installed,ir,spec,safety);result.update(metrics)
            np.savez_compressed(out/'world_tcp.npz',elapsed=t,position=p,rotation=r,installed_position=installed,installed_rotation=ir)
        result.update(validity='VALID',safety_failure=safety,WBC_actor_calls=len(actor_calls),control_ticks=tick[0],
            camera_checks=camera_checks,hard_limits=bounds,native_dt=float(native.dt))
    except Exception as exc:
        result['error']=repr(exc)
        if 'INITIAL_REFERENCE_GATE_REJECT_NO_PHYSICS' in str(exc):result.update(validity='VALID',failure='CONSTRUCTION_GATE_REJECT',safety_failure=True)
        else:raise
    finally:
        if worker is not None:
            worker.stdin.close();worker.wait(timeout=30)
        append(out/'actor_calls.jsonl',dict(calls=actor_calls),sync=True)
        write(out/'rows.json',records);write(out/'result.json',result)
        write(ROOT/'evidence'/('result-'+case_id+'.json'),result)
        if native is not None:native.gym.destroy_sim(native.sim)
    print(json.dumps(dict(case=case_id,validity=result['validity'],success=result.get('success'),failure=result.get('failure'))),flush=True)
    return result


if __name__=='__main__':
    if '--collision-worker' in sys.argv:collision_worker()
    else:
        parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);args=parser.parse_args();run(args.case)
