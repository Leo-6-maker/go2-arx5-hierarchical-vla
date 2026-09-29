"""Finite preparation-only A/B/C over the actual, unchanged native actors."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
if '--case' in sys.argv or '--native-cpu-check' in sys.argv:
    # Native binary import precedes scientific libraries, as required by IsaacGym.
    import isaacgym
import numpy as np
import pi0256_preparation as prep
import pi0255_collect as old
import pi0255_run as runner
from pi0255_motion3 import sha,write

ROOT=prep.ROOT
REF=old.REF
AUTH=prep.AUTH


def run(case_id):
    contract=json.loads((ROOT/'design/PREPARATION_V2_CONTRACT.json').read_text())
    seal=json.loads((ROOT/'runtime/execution_seal.json').read_text())
    assert seal['authority']==AUTH and seal['canonical_comment'] and seal['commit']
    for name,digest in seal['files'].items():assert sha(ROOT/name)==digest,name
    for row in json.loads((ROOT/'evidence/runtime_readback.json').read_text())['files']:
        assert row['matches'] and sha(row['path'])==row['sha256']
    case=next(x for x in contract['cases'] if x['case_id']==case_id)
    if case['group']=='C':
        for c in contract['cases']:
            if c['group']=='B':assert json.loads((ROOT/'episodes'/c['case_id']/'result.json').read_text())['success']
    out=ROOT/'episodes'/case_id
    assert not out.exists(),'first attempt exists; inspect rather than repeat'
    out.mkdir(parents=True)
    for d in ('references','actor_inputs','frames'):(out/d).mkdir()
    result=dict(schema='PI0256_PREPARATION_RESULT_V1',case=case,validity='ENGINEERING_INCOMPLETE',success=False,
                execution_seal=seal,task_ticks=0,GR00T_queries=0,optimizer_updates=0,selected_checkpoint='NONE')
    native=env=None
    try:
        sys.path[:0]=[str(REF/'scripts'),str(REF/'src')]
        import stage21_native_receding_horizon as api
        from go2arx5_vla.contracts import TrajectoryChunkV1
        from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
        from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
        from go2arx5_vla.trajectory.so3 import quat_to_matrix,matrix_to_quat,geodesic_angle
        source=json.loads((ROOT.parent/'PI0255/evidence/platform_source_readback.json').read_text())
        registry=json.loads((ROOT.parent/'PI0255/design/task_registry_v1.json').read_text())
        kin=NativeArmKinematics.from_urdf(source['components']['native_urdf']['path'])
        import xml.etree.ElementTree as ET
        joint_limits={j.attrib['name']:[float(j.find('limit').attrib['lower']),float(j.find('limit').attrib['upper'])]
                      for j in ET.parse(source['components']['native_urdf']['path']).getroot().findall('joint')
                      if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
        write(out/'hard_URDF_joint_limits.json',joint_limits)
        template=dict(np.load(ROOT/'evidence/evidenced_C0_template.npz',allow_pickle=False))
        gate=TrajectorySafetyGateV1()
        def clock(n):return float(n.gym.get_sim_time(n.sim))
        def measured_fk(n):
            root=n.root_states[0].detach().cpu().numpy()
            q=[float(n.dof_pos[0,n.dof_names.index(name)].detach().cpu().item()) for name in kin.joint_names]
            pose=kin.tcp_pose(np.asarray(q));rotation=quat_to_matrix(root[3:7])
            return rotation@pose[:3,3]+root[:3],matrix_to_quat(rotation@pose[:3,:3])
        def snapshot(n):
            row=dict(time_s=clock(n),root=n.root_states.detach().cpu().tolist(),dof=n.dof_state.detach().cpu().tolist(),
                     episode_length=n.episode_length_buf.detach().cpu().tolist(),reset=n.reset_buf.detach().cpu().tolist(),
                     telemetry=api._native_runtime_telemetry(n))
            p,q=api._current_eef(n);f,k=measured_fk(n)
            row.update(TCP_world=p.tolist(),quaternion_xyzw=q.tolist(),FK_TCP_world=f.tolist(),FK_quaternion_xyzw=k.tolist())
            positions=n.dof_pos[0].detach().cpu().numpy()
            row['hard_URDF_joint_margins']={name:float(min(positions[i]-joint_limits[name][0],joint_limits[name][1]-positions[i]))
                for i,name in enumerate(n.dof_names) if name in joint_limits}
            if hasattr(n,'traj_batch'):
                s=n.traj_batch.s_ref(n.traj_sim_time)
                row.update(desired_world=n.traj_batch.p_at(s)[0].detach().cpu().tolist(),
                           desired_quaternion=n.traj_batch.quat_at(s)[0].detach().cpu().tolist(),
                           native_reference_time=n.traj_sim_time.detach().cpu().tolist())
            return row
        def immutable(n):
            return dict(root=api._tensor_hash(n.root_states),dof=api._tensor_hash(n.dof_state),time_s=clock(n),
                        episode=api._tensor_hash(n.episode_length_buf),reset=api._tensor_hash(n.reset_buf))
        def make_chunk(position,quaternion,timestamps,name):
            return TrajectoryChunkV1(chunk_id=name,source='PI0256_NATIVE_EXPERT_PREPARATION',observation_timestamp=float(timestamps[0]),
                point_timestamps=timestamps,eef_position=position,eef_orientation=quaternion,source_frame=api.SOURCE_FRAME,
                tcp_semantic_id=api.TCP_SEMANTIC_ID,validity_start=float(timestamps[0]),validity_end=float(timestamps[-1]),
                gripper=np.full(len(timestamps),.5),provenance=dict(preparation_only=True,policy_H40_unchanged=True))
        original=api._Stage21NativeEnvFactory.make
        def factory(WBCEnv,initial,torch,gymapi,gymtorch):
            cls=old.scene_class(original(WBCEnv,initial,torch,gymapi,gymtorch),registry,case['family'],gymapi)
            class Instrumented(cls):
                def _load_trajectory_for(self,ids):
                    if case['group']!='A' and not hasattr(self,'pi0256_reference'):
                        before=immutable(self);p,q=measured_fk(self)
                        ref=prep.entry_reference(template,p,q)
                        self.pi0256_anchor=p.copy();self.pi0256_reference=ref
                        self._stage21_initial_chunk=make_chunk(ref['position']-p,ref['quaternion'],ref['timestamp'],'pi0256-full-native-prepare')
                        verdict=gate.evaluate(self._stage21_initial_chunk,now=0.,achieved_position=np.zeros(3),achieved_orientation=q)
                        write(out/'construction_gate.json',verdict.as_dict())
                        if verdict.decision is GateDecision.REJECT:raise ValueError('SEALED_PREPARATION_REFERENCE_GATE_REJECT')
                        np.savez_compressed(out/'exact_reference.npz',**ref,T_world_native=np.block([[np.eye(3),p[:,None]],[np.zeros((1,3)),np.ones((1,1))]]))
                        prep.unchanged(before,immutable(self))
                    return super()._load_trajectory_for(ids)
                def _anchor_external_reference(self,ids):
                    if case['group']=='A':return super()._anchor_external_reference(ids)
                    self.traj_batch.gamma_p[ids]+=self.traj_batch.gamma_p.new_tensor(self.pi0256_anchor).view(1,1,3)
                def _stage21_load_chunk(self,c,ids,initial):
                    before=immutable(self)
                    event=super()._stage21_load_chunk(c,ids,initial=initial)
                    prep.unchanged(before,immutable(self))
                    arrays={k:v.detach().cpu().numpy().copy() for k,v in vars(self.traj_batch).items() if hasattr(v,'detach')}
                    arrays.update(raw_position=c.eef_position,raw_quaternion=c.eef_orientation,timestamps=c.point_timestamps)
                    name='references/load-%03d.npz'%(len(self._stage21_load_events)-1)
                    np.savez_compressed(out/name,**arrays)
                    old.append(out/'reference_loads.jsonl',dict(event=event,file=name,sha256=sha(out/name)),sync=False)
                    return event
                def step(self,*args,**kwargs):
                    import fcntl
                    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<128*1024**2,'case storage bound reached'
                    before=snapshot(self)
                    with (ROOT/'physical_budget.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        budget_path=ROOT/'physical_budget.json'
                        budget=json.loads(budget_path.read_text()) if budget_path.exists() else dict(starts=0,steps=0,preparation_starts=0)
                        assert budget['steps']<480000
                        if not (out/'physics_started.json').exists():
                            assert budget['starts']<384 and budget['preparation_starts']<24
                            budget['starts']+=1;budget['preparation_starts']+=1
                            row=dict(case=case_id,pid=os.getpid(),time_s=clock(self),epoch=time.time(),ordinal=budget['starts'])
                            write(out/'physics_started.json',row);old.append(ROOT/'physical_starts.jsonl',row)
                            rng=np.random.get_state()
                            np.savez_compressed(out/'pre_first_step_identity.npz',root=self.root_states.detach().cpu().numpy(),
                                dof=self.dof_state.detach().cpu().numpy(),numpy_rng=rng[1],torch_rng=torch.get_rng_state().numpy(),
                                cuda_rng=torch.cuda.get_rng_state().cpu().numpy())
                            write(out/'pre_first_step.json',dict(state=before,python_rng=repr(random.getstate()),numpy_cursor=rng[2],
                                root_sha=api._tensor_hash(self.root_states),dof_sha=api._tensor_hash(self.dof_state),
                                tensor_file='pre_first_step_identity.npz'))
                        budget['steps']+=1;write(budget_path,budget)
                    returned=super().step(*args,**kwargs)
                    after=snapshot(self)
                    old.append(out/'physics_steps.jsonl',dict(before=before,after=after,
                        pre_autoreset=getattr(self,'_stage16_pending_termination',None),step_ordinal=budget['steps']),sync=False)
                    return returned
            return Instrumented
        api._Stage21NativeEnvFactory.make=staticmethod(factory)
        initial=dict(timestamp=np.array([0.,2.]),position=np.asarray(case['family']['center_world'])+[[0.,0.,0.],[.005,0.,0.]],
                     quaternion=np.tile([0.,0.,0.,1.],(2,1)))
        args=SimpleNamespace(source_root=old.WBC,sim_device='cuda:0',seed=case['seed'],mode='gt_receding',
            arm=Path(source['components']['arm_checkpoint']['path']),dog=Path(source['components']['dog_checkpoint']['path']))
        native,env,actors,loads,cfg=api._build_native(args,initial)
        import torch
        assert abs(native.dt-.02)<1e-9
        result['WBC_loads']=loads
        anchor=native.stage21_anchor_world.detach().cpu().numpy().reshape(3).copy()
        result['T_world_native']=np.block([[np.eye(3),anchor[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist()
        write(out/'post_reset.json',snapshot(native))
        body,wrist=api._capture_scene(native);np.savez_compressed(out/'reset_images.npz',body_rgb=body,wrist_rgb=wrist)
        result['reset_image_hashes']=[old.array_sha(body),old.array_sha(wrist)]
        if case['group']!='A':
            before=immutable(native);history={k:api._tensor_hash(v) for k,v in vars(env).items() if hasattr(v,'detach') and 'history' in k}
            write(out/'reference_cache_before.json',dict(ee_pos_err=native.ee_pos_err.detach().cpu().tolist(),state=snapshot(native)))
            native._update_ee_task_space_error()
            prep.unchanged(before,immutable(native))
            assert history=={k:api._tensor_hash(v) for k,v in vars(env).items() if hasattr(v,'detach') and 'history' in k}
            write(out/'reference_cache_after.json',dict(ee_pos_err=native.ee_pos_err.detach().cpu().tolist(),history_hashes=history,state=snapshot(native)))
        actor_tick=[0]
        for name in ('arm','dog'):
            original_getter=getattr(env,'get_'+name+'_observations')
            def getter(original_getter=original_getter,name=name):
                values=original_getter()
                arrays={k:v.detach().cpu().numpy().copy() for k,v in values.items() if hasattr(v,'detach')}
                np.savez_compressed(out/'actor_inputs'/('%04d-%s-observation.npz'%(actor_tick[0],name)),**arrays)
                return values
            setattr(env,'get_'+name+'_observations',getter)
        class ActorRecord:
            def __init__(self,actor,name):self.actor,self.name=actor,name
            def __call__(self,value):
                payload=value.detach().cpu().numpy().copy()
                path=out/'actor_inputs'/('%04d-%s.npz'%(actor_tick[0],self.name))
                np.savez_compressed(path,history=payload)
                answer=self.actor(value)
                old.append(out/'actor_calls.jsonl',dict(tick=actor_tick[0],actor=self.name,time_s=clock(native),input_file=str(path.relative_to(out)),
                    array_sha256=old.array_sha(payload),file_sha256=sha(path),shape=list(payload.shape),
                    reconstructable_head=payload.reshape(-1)[:16].tolist(),output=answer.detach().cpu().tolist()),sync=False)
                return answer
        actors=(ActorRecord(actors[0],'arm'),ActorRecord(actors[1],'dog'),actors[2],actors[3])
        from isaacgym import gymtorch
        contacts=gymtorch.wrap_tensor(native.gym.acquire_net_contact_force_tensor(native.sim))
        p0,q0=api._current_eef(native);previous=p0.copy();previous_q=q0.copy();previous_time=clock(native)
        origin=clock(native);start=json.loads((out/'physics_started.json').read_text())['time_s']
        state='PREPARE';stable=None;last_handoff=-math.inf;last_capture=-math.inf;rows=[]
        while state=='PREPARE':
            now=clock(native)
            if case['group']=='A' and now-last_handoff>=.20-1e-8:
                t=now-origin+np.arange(40)*.05
                c=make_chunk(p0+np.clip(t/5.,0,1)[:,None]*[.005,0,0]-anchor,np.tile(q0,(40,1)),now+np.arange(40)*.05,
                             'pi0256-legacy-%04d'%actor_tick[0])
                p,q=api._current_eef(native)
                verdict=gate.evaluate(c,now=now,achieved_position=p-anchor,achieved_orientation=q)
                old.append(out/'handoffs.jsonl',dict(time_s=now,gate=verdict.as_dict()),sync=False)
                if verdict.decision is GateDecision.REJECT:
                    state='TERMINAL';result['failure']='LEGACY_GATE_REJECT';break
                native.initial_bind_reference(c) if native.stage21_wbc_forward_count==0 else native.load_reference_chunk(c)
                last_handoff=now
            if now-last_capture>=.10-1e-8:
                body,wrist=api._capture_scene(native)
                np.savez_compressed(out/'frames'/('%04d.npz'%actor_tick[0]),body_rgb=body,wrist_rgb=wrist,time_s=np.array(now))
                last_capture=now
            pre=snapshot(native)
            with torch.no_grad():done,policy=api._policy_step(native,env,actors)
            post=snapshot(native);event=native.pop_stage16_termination()
            native.gym.refresh_net_contact_force_tensor(native.sim)
            force=float(contacts[native.pi0255_obstacle_body].abs().sum().cpu().item())
            telemetry=post['telemetry'];clean=not(done or event or bool(native.reset_buf[0].item()) or telemetry['invalid_state'] or
                telemetry.get('fall_flag',False) or telemetry.get('joint_limit_termination',False) or force>registry['geometry']['contact_force_threshold_n'])
            p=np.asarray(post['TCP_world']);q=np.asarray(post['quaternion_xyzw']);dt=post['time_s']-previous_time
            row=dict(time_s=post['time_s'],elapsed_s=post['time_s']-start,dt_s=dt,fresh=abs(dt-.02)<1e-6,
                clean=clean,joint_limit_violation=telemetry.get('joint_limit_violation',True),
                speed_mps=float(np.linalg.norm(p-previous)/dt),angular_speed_radps=float(geodesic_angle(quat_to_matrix(q),quat_to_matrix(previous_q))/dt),
                position_error_m=float(np.linalg.norm(p-post['desired_world'])),rotation_error_rad=float(geodesic_angle(quat_to_matrix(q),quat_to_matrix(post['desired_quaternion']))),
                pre_step=pre,post_step=post,pre_autoreset_event=event,obstacle_contact_force_n=force,policy=policy)
            row['failure_pose_world']=prep.failure_pose(post,event)
            row['post_step_is_failure_pose']=event is None
            state,stable=prep.advance(state,row,start,stable);row.update(state=state,stable_since=stable)
            old.append(out/'raw_ticks.jsonl',row,sync=False);rows.append(row)
            previous,previous_q,previous_time=p,q,post['time_s'];actor_tick[0]+=1
            # Legacy A retains its absolute-clock timeout; the new predicate is separately observed.
            if case['group']=='A' and clock(native)>=5.-1e-9 and state=='PREPARE':state='TERMINAL'
        result.update(validity='VALID',success=state=='SETTLED',state=state,rows=len(rows),stable_since=stable,
            failure=None if state=='SETTLED' else result.get('failure','NATIVE_TERMINATION' if rows and not rows[-1]['clean'] else 'PREPARATION_TIMEOUT'))
    except Exception as e:
        result['engineering_error']=dict(type=type(e).__name__,message=str(e))
        import traceback
        traceback.print_exc()
    finally:
        if native is not None:result['end_time_s']=float(native.gym.get_sim_time(native.sim))
        if env is not None:env.close()
        result['physics_started']=(out/'physics_started.json').exists()
        result['files']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file()}
        write(out/'result.json',result);old.append(ROOT/'attempt_results.jsonl',result)
    print(json.dumps(dict(case=case_id,validity=result['validity'],success=result['success'],failure=result.get('failure'),error=result.get('engineering_error'))),flush=True)
    return 0 if result['validity']=='VALID' else 3


def cohort():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    runner.ROOT=ROOT
    py,env=runner.environment()
    script=Path(__file__).resolve()
    subprocess.run([str(py),str(script)],env=env,check=True)
    cases=json.loads((ROOT/'design/PREPARATION_V2_CONTRACT.json').read_text())['cases']
    for case in cases:
        if case['group']=='C' and not all(json.loads((ROOT/'episodes'/c['case_id']/'result.json').read_text())['success'] for c in cases if c['group']=='B'):
            print('C_NOT_RELEASED_B_NOT_6_OF_6',flush=True);break
        result_path=ROOT/'episodes'/case['case_id']/'result.json'
        if result_path.exists():
            assert json.loads(result_path.read_text())['validity']=='VALID','engineering attempt requires same-contract repair'
            continue
        with (ROOT/'logs'/(case['case_id']+'.log')).open('wb') as output:
            status=subprocess.run([str(py),str(script),'--case',case['case_id'],'--authorization',AUTH],env=env,stdout=output,stderr=subprocess.STDOUT)
        if status.returncode:raise SystemExit(status.returncode)
        d=json.loads(result_path.read_text());print(json.dumps(dict(case=case['case_id'],success=d['success'],failure=d.get('failure'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--cohort',action='store_true');p.add_argument('--authorization');p.add_argument('--native-cpu-check',action='store_true');a=p.parse_args()
    if a.native_cpu_check:
        import torch
        from modules.trajectory import to_canonical,quat_to_mat
        from go2arx5_vla.contracts import TrajectoryChunkV1
        from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
        template=dict(np.load(ROOT/'evidence/evidenced_C0_template.npz',allow_pickle=False))
        ref=prep.entry_reference(template,np.array([.2,-.1,.6]),np.array([0.,0.,np.sin(.3),np.cos(.3)]))
        c=TrajectoryChunkV1('cpu-native-full-reference','CPU',0.,ref['timestamp'],ref['position']-ref['position'][0],ref['quaternion'],
                            'NATIVE_WBC_RELATIVE_REFERENCE','x5_link6_gripper_center',0.,5.)
        verdict=TrajectorySafetyGateV1().evaluate(c,now=0.,achieved_position=np.zeros(3),achieved_orientation=c.eef_orientation[0])
        assert verdict.decision is not GateDecision.REJECT,verdict.as_dict()
        gamma,law=to_canonical(torch.as_tensor(c.eef_position,dtype=torch.float32),quat_to_mat(torch.as_tensor(c.eef_orientation,dtype=torch.float32)),.05)
        for obj in (gamma,law):
            for value in vars(obj).values():
                if hasattr(value,'isfinite'):assert torch.isfinite(value).all()
        write(ROOT/'design/native_reference_cpu_receipt.json',dict(status='PASS_ACTUAL_NATIVE_CANONICAL_FULL_PREPARATION',gate=verdict.as_dict(),
            points=c.point_count,physics=0,GR00T=0,source_sha256=sha(__file__)))
        print('PASS_ACTUAL_NATIVE_CANONICAL_FULL_PREPARATION',flush=True)
        raise SystemExit(0)
    if a.case or a.cohort:
        assert a.authorization==AUTH
        if a.case:raise SystemExit(run(a.case))
        cohort()
    else:p.print_help()
