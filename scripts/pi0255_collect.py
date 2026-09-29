"""Bounded expert-only PI0255 collector over the unchanged native WBC.

Native imports are delayed until a sealed --episode execution. No VLA loader.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pi0255_motion3 as mt

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT=BASE/'PI0255'
REF=BASE/'PI0250/runtime/reference-317a'
WBC=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare')
AUTH='PI0255_MULTITASK_GENERALIZATION_READINESS_GO_V1'


def append(path,row):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def physics_audit(cls,episode,out,ledger):
    """Audit the real first step, including the step called by env.reset()."""
    class Audited(cls):
        def step(self,*args,**kwargs):
            if not (out/'physics_started.json').exists():
                import fcntl
                with (ledger.parent/'physical_budget.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    prior=ledger.read_text().splitlines() if ledger.exists() else []
                    assert len(prior)<360,'expert physical budget exhausted'
                    row=dict(episode=episode,attempt=str(out),ordinal=len(prior)+1,epoch_s=time.time(),pid=os.getpid(),
                             first_call='ACTUAL_WBC_STEP_INCLUDING_ENV_RESET',sim_time_before=float(self.gym.get_sim_time(self.sim)))
                    append(ledger,row);mt.write(out/'physics_started.json',row)
            return super().step(*args,**kwargs)
    return Audited


def scene_class(base,registry,family,gymapi):
    data=mt.scene(registry,family);geo=registry['geometry']
    class MotionScene(base):
        def _create_envs(self):
            # Keep original robot creation; replace only the project-owned Stage21 scene.
            base.__mro__[1]._create_envs(self)
            assert hasattr(self,'rendering_camera')
            def actor(position,dimensions,color,name,collidable=False,box=False):
                options=gymapi.AssetOptions();options.fix_base_link=True;options.disable_gravity=True
                options.collapse_fixed_joints=True
                asset=self.gym.create_box(self.sim,*dimensions,options) if box else self.gym.create_sphere(self.sim,float(dimensions),options)
                pose=gymapi.Transform();pose.p=gymapi.Vec3(*np.asarray(position,float).tolist())
                handle=self.gym.create_actor(self.envs[0],asset,pose,name,0 if collidable else 1,0,0)
                self.gym.set_rigid_body_color(self.envs[0],handle,0,gymapi.MESH_VISUAL,gymapi.Vec3(*color))
                return handle
            colors=dict(red=(.9,.03,.03),blue=(.03,.15,.9),green=(.03,.85,.1))
            for index,position in enumerate(data['targets']):
                name=family['factors']['colors'][index]
                actor(position,geo['target_radius_m'],colors[name],'pi0255-'+name+'-pose')
                angle=geo['target_yaw_rad'][index];c,s=math.cos(angle),math.sin(angle)
                rotation=np.asarray([[c,-s,0],[s,c,0],[0,0,1.]])
                for axis,color in enumerate(((1.,0,0),(0,1.,0),(0,0,1.))):
                    actor(position+rotation[:,axis]*geo['axes_length_m'],geo['axes_marker_radius_m'],color,'pi0255-pose-axis-%d-%d'%(index,axis))
            obstacle=actor(data['obstacle'],2*np.asarray(geo['obstacle_half_extents_m']),(.45,.45,.45),'pi0255-obstacle',True,True)
            self.pi0255_obstacle_body=int(self.gym.get_actor_rigid_body_index(self.envs[0],obstacle,0,gymapi.DOMAIN_SIM))
            self.stage21_target_actor=actor(data['goal'],geo['target_radius_m'],(1.,.35,.03),'pi0255-orange-goal')
            self.stage21_target_root_index=int(self.gym.get_actor_index(self.envs[0],self.stage21_target_actor,gymapi.DOMAIN_SIM))
            for index in range(geo['contour_points']):
                angle=2*math.pi*index/geo['contour_points']
                position=data['center']+[data['radius']*math.cos(angle),data['radius']*math.sin(angle),0]
                actor(position,geo['contour_marker_radius_m'],(.95,.8,.05),'pi0255-contour-%d'%index)
            actor(data['center']+[-data['radius'],0,0],geo['start_marker_radius_m'],(.03,.03,.03),'pi0255-black-start')
            body=self.gym.find_actor_rigid_body_handle(self.envs[0],self.actor_handles[0],self.body_names[0])
            wrist=self.gym.find_actor_rigid_body_handle(self.envs[0],self.actor_handles[0],self.body_names[self.ee_idx])
            self.stage21_camera_handles={'body':self.rendering_camera}
            wrist_props=gymapi.CameraProperties();wrist_props.width=int(self.camera_props.width);wrist_props.height=int(self.camera_props.height)
            self.stage21_camera_handles['wrist']=self.gym.create_camera_sensor(self.envs[0],wrist_props)
            for name,handle,translation in [('body',body,(1.15,-1.15,.95)),('wrist',wrist,(.28,-.10,.12))]:
                pose=gymapi.Transform();pose.p=gymapi.Vec3(*translation)
                pose.r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0),math.radians(135.))
                self.gym.attach_camera_to_body(self.stage21_camera_handles[name],self.envs[0],handle,pose,gymapi.FOLLOW_TRANSFORM)
    return MotionScene


def input_from_snapshot(path):
    packet=json.loads(Path(path).read_text());input_row=packet['input'];assert set(input_row)=={'frames','state_v1','language'}
    with np.load(Path(path).parent/input_row['frames'],allow_pickle=False) as data:
        example=dict(body_rgb=data['body_rgb'].copy(),wrist_rgb=data['wrist_rgb'].copy(),state_v1=np.asarray(input_row['state_v1']),language=input_row['language'])
    return mt.policy_input(example)


def run_episode(episode_id):
    registry=json.loads((ROOT/'design/task_registry_v1.json').read_text())
    seal=json.loads((ROOT/'design/execution_seal.json').read_text())
    assert seal['authority']==AUTH and seal['registry_sha256']==mt.sha(ROOT/'design/task_registry_v1.json')
    for name,digest in seal['runtime_files'].items():assert mt.sha(ROOT/'runtime'/name)==digest,name
    assert seal['pre_execution_commit'] and seal['canonical_comment']
    episode=next(row for row in registry['episodes'] if row['episode_id']==episode_id);assert episode['execute']
    family=next(row for row in registry['families'] if row['family_id']==episode['family_id'])
    episode_root=ROOT/'episodes'/episode_id;episode_root.mkdir(parents=True,exist_ok=True)
    completed=[json.loads(p.read_text()) for p in episode_root.glob('attempt-*/result.json')]
    assert not any(r['validity']=='VALID' for r in completed),'first valid episode already exists'
    assert not any(p.name.startswith('attempt-') and not (p/'result.json').exists() for p in episode_root.iterdir()),'live or unresolved attempt exists'
    out=episode_root/('attempt-%02d'%len(completed));out.mkdir();(out/'snapshots').mkdir();(out/'settling_frames').mkdir()
    source=json.loads((ROOT/'evidence/platform_source_readback.json').read_text())
    for row in source['components'].values():assert mt.sha(row['path'])==row['sha256']
    for row in source['native_helpers'].values():assert mt.sha(row['path'])==row['sha256']
    native=env=None;records=[];snapshots=[];handoffs=[];settling=[]
    result=dict(schema='PI0255_EXPERT_EPISODE_V1',episode=episode,attempt=str(out),validity='ENGINEERING_INCOMPLETE',success=False,
                new_GR00T_loads=0,new_GR00T_queries=0,new_GR00T_backward=0,new_GR00T_optimizer=0,learned_policy_episodes=0,
                registry_sha256=seal['registry_sha256'],runtime_files=seal['runtime_files'],selected_checkpoint='NONE')
    try:
        sys.path[:0]=[str(REF/'scripts'),str(REF/'src')]
        import stage21_native_receding_horizon as api
        from go2arx5_vla.contracts import TrajectoryChunkV1
        from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
        original=api._Stage21NativeEnvFactory.make
        def factory(WBCEnv,initial_chunk,torch,gymapi,gymtorch):
            cls=original(WBCEnv,initial_chunk,torch,gymapi,gymtorch)
            return physics_audit(scene_class(cls,registry,family,gymapi),episode_id,out,ROOT/'physics_attempts.jsonl')
        api._Stage21NativeEnvFactory.make=staticmethod(factory)
        initial=dict(timestamp=np.array([0.,2.]),position=np.asarray(family['center_world'])+np.asarray([[0.,0.,0.],[.005,0.,0.]]),quaternion=np.tile([0.,0.,0.,1.],(2,1)))
        args=SimpleNamespace(source_root=WBC,sim_device='cuda:0',seed=episode['reset_seed'],mode='gt_receding',
            arm=Path(source['components']['arm_checkpoint']['path']),dog=Path(source['components']['dog_checkpoint']['path']))
        native,env,actors,loads,cfg=api._build_native(args,initial)
        result['WBC_actor_loads']=loads;result['control_dt_s']=float(native.dt)
        assert abs(native.dt-.02)<1e-9
        gym=native.gym;clock=lambda:float(gym.get_sim_time(native.sim))
        anchor=native.stage21_anchor_world.detach().cpu().numpy().reshape(3).copy();transform=np.eye(4);transform[:3,3]=anchor
        result['T_world_native']=transform.tolist();result['tcp_offset_link6_m']=source['tcp_offset_link6_m']
        import random
        import torch
        rng=np.random.get_state()
        np.savez_compressed(out/'reset_state.npz',root=native.root_states.detach().cpu().numpy(),dof=native.dof_state.detach().cpu().numpy(),
            numpy_rng_keys=rng[1],torch_cpu_rng=torch.get_rng_state().numpy(),torch_cuda_rng=torch.cuda.get_rng_state().cpu().numpy())
        result['initial_reset']=dict(root_sha256=api._tensor_hash(native.root_states),dof_sha256=api._tensor_hash(native.dof_state),actual_sim_time_s=clock(),
            python_rng_state=repr(random.getstate()),numpy_rng_metadata=[rng[0],int(rng[2]),int(rng[3]),float(rng[4])],
            tensor_rng_payload='reset_state.npz',tensor_rng_sha256=mt.sha(out/'reset_state.npz'),physics_seed=episode['reset_seed'])
        initial_body,initial_wrist=api._capture_scene(native)
        result['initial_reset'].update(body_rgb_sha256=array_sha(initial_body),wrist_rgb_sha256=array_sha(initial_wrist))
        np.savez_compressed(out/'reset_images.npz',body_rgb=initial_body,wrist_rgb=initial_wrist)
        result['camera_projection']={name:np.asarray(gym.get_camera_proj_matrix(native.sim,native.envs[0],handle)).tolist() for name,handle in native.stage21_camera_handles.items()}
        result['initial_camera_view']={name:np.asarray(gym.get_camera_view_matrix(native.sim,native.envs[0],handle)).tolist() for name,handle in native.stage21_camera_handles.items()}
        result['camera_intrinsics_source']='ACTUAL_NATIVE_CAMERA_PROJECTION_MATRICES'
        from isaacgym import gymtorch
        contact_tensor=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(native.sim))
        gate=TrajectorySafetyGateV1()
        def chunk(world,quaternion,now):
            return TrajectoryChunkV1(chunk_id=episode_id+'-%06d'%len(handoffs),source='PI0255_REGISTERED_DESIRED_EXPERT',
                observation_timestamp=now,point_timestamps=now+np.arange(40)*.05,eef_position=world-anchor,eef_orientation=quaternion,
                source_frame=api.SOURCE_FRAME,tcp_semantic_id=api.TCP_SEMANTIC_ID,validity_start=now,validity_end=now+1.95,
                gripper=np.full(40,.5),provenance=dict(registry_sha256=seal['registry_sha256'],privileged_geometry_expert_only=True))
        def handoff(c):
            position,quaternion=api._current_eef(native)
            verdict=gate.evaluate(c,now=clock(),achieved_position=position-anchor,achieved_orientation=quaternion)
            row=dict(actual_sim_time_s=clock(),chunk_id=c.chunk_id,gate=verdict.as_dict())
            if verdict.decision is GateDecision.REJECT:
                handoffs.append(row);append(out/'handoffs.jsonl',row);return False
            load=native.initial_bind_reference(c) if native.stage21_wbc_forward_count==0 else native.load_reference_chunk(c)
            assert load['physics_root_state_unchanged'] and load['physics_dof_state_unchanged'] and load['episode_length_unchanged']
            row['load']=load;handoffs.append(row);append(out/'handoffs.jsonl',row);return True
        def tick():
            before=clock()
            with torch.no_grad():done,policy=api._policy_step(native,env,actors)
            now=clock();position,quaternion=api._current_eef(native);event=native.pop_stage16_termination()
            telemetry=api._native_runtime_telemetry(native)
            clean=not(done or bool(native.reset_buf[0].item()) or event or telemetry.get('invalid_state',False) or telemetry.get('fall_flag',False) or telemetry.get('joint_limit_termination',False))
            gym.refresh_net_contact_force_tensor(native.sim)
            force=float(contact_tensor[native.pi0255_obstacle_body].abs().sum().cpu().item())
            clean=clean and force<=registry['geometry']['contact_force_threshold_n']
            s=native.traj_batch.s_ref(native.traj_sim_time);desired=native.traj_batch.p_at(s)[0].detach().cpu().numpy()
            desired_q=native.traj_batch.quat_at(s)[0].detach().cpu().numpy()
            row=dict(actual_sim_time_s=now,actual_dt_s=now-before,achieved_world=position.tolist(),achieved_quaternion=quaternion.tolist(),
                desired_world=desired.tolist(),desired_quaternion=desired_q.tolist(),native_reference_time_s=float(native.traj_sim_time[0].cpu().item()),
                base_position=native.base_pos[0].detach().cpu().tolist(),base_quaternion=native.base_quat[0].detach().cpu().tolist(),
                obstacle_contact_force_n=force,physical_clean=clean,done=done,reset_buf=bool(native.reset_buf[0].item()),termination_event=event,
                telemetry=telemetry,WBC_policy=policy,
                gripper_joint_measurements={name:dict(position=float(native.dof_pos[0,i].cpu().item()),velocity=float(native.dof_vel[0,i].cpu().item()))
                    for i,name in enumerate(native.dof_names) if name in ('x5_gripper_joint','x5_joint8')},parallel_gripper_executor='UNVERIFIED_NOT_COMMANDED')
            append(out/'raw_ticks.jsonl',row);return row
        p0,q0=api._current_eef(native);settle_origin=clock();previous=p0.copy();previous_time=clock();stable_start=None;last_handoff=-math.inf;last_settle_capture=-math.inf
        while clock()<5.-1e-9:
            now=clock()
            if now-last_handoff>=.20-1e-8:
                t=now-settle_origin+np.arange(40)*.05;positions=p0+np.clip(t/5.,0,1)[:,None]*np.asarray([.005,0,0])
                if not handoff(chunk(positions,np.tile(q0,(40,1)),now)):
                    result.update(validity='VALID',failure='SETTLING_EXPERT_GATE_REJECT');break
                last_handoff=now
            if now-last_settle_capture>=.10-1e-8:
                body,wrist=api._capture_scene(native)
                np.savez_compressed(out/'settling_frames'/('%06d.npz'%len(list((out/'settling_frames').glob('*.npz')))),body_rgb=body,wrist_rgb=wrist,actual_sim_time_s=np.array(now))
                last_settle_capture=now
            row=tick();speed=float(np.linalg.norm(np.asarray(row['achieved_world'])-previous)/(row['actual_sim_time_s']-previous_time))
            row['achieved_speed_mps']=speed;settling.append(row);previous=np.asarray(row['achieved_world']);previous_time=row['actual_sim_time_s']
            if not row['physical_clean']:
                body,wrist=api._capture_scene(native);np.savez_compressed(out/'terminal_images.npz',body_rgb=body,wrist_rgb=wrist,actual_sim_time_s=np.array(clock()))
                result.update(validity='VALID',failure='SETTLING_PHYSICAL_TERMINATION');break
            stable_start=row['actual_sim_time_s'] if speed<=.02 and stable_start is None else stable_start if speed<=.02 else None
            if stable_start is not None and clock()-stable_start>=.5-1e-8:break
        if result['validity']=='VALID':pass
        elif stable_start is None or clock()-stable_start<.5-1e-8:
            result.update(validity='VALID',failure='SETTLING_TIMEOUT')
        else:
            p0,q0=api._current_eef(native);start=clock();result['task_start_actual_sim_time_s']=start
            result['settled_pose']=dict(position_world=p0.tolist(),quaternion_xyzw=q0.tolist(),stable_since_actual_sim_time_s=stable_start)
            task,command=episode['task'],episode['command'];profile=registry['experts'][task]
            last_handoff=-math.inf;last_capture=-math.inf
            while clock()-start<profile['nominal_end_s']-1e-9:
                now=clock();elapsed=now-start
                if elapsed<profile['motion_end_s']-1e-8 and now-last_handoff>=.20-1e-8:
                    positions,quats=mt.expert_at(registry,family,task,command,p0,q0,elapsed+np.arange(40)*.05)
                    if not handoff(chunk(positions,quats,now)):
                        result['failure']='TASK_EXPERT_GATE_REJECT';break
                    last_handoff=now
                if now-last_capture>=.10-1e-8:
                    body,wrist=api._capture_scene(native);position,quaternion=api._current_eef(native)
                    state=api._current_state_v1(native).as_array().tolist()
                    future_p,future_q=mt.expert_at(registry,family,task,command,p0,q0,elapsed+np.arange(40)*.05)
                    directory=out/'snapshots';stem='%06d'%len(snapshots)
                    np.savez_compressed(directory/(stem+'.npz'),body_rgb=body,wrist_rgb=wrist,future_world=future_p,
                        future_native=future_p-anchor,future_quaternion=future_q,future_times=now+np.arange(40)*.05,
                        future_gripper=np.full((40,1),.5),terminal_padding=elapsed+np.arange(40)*.05>profile['motion_end_s'],
                        state_v1=np.asarray(state),body_camera_timestamp=np.array(now),wrist_camera_timestamp=np.array(now),state_timestamp=np.array(now))
                    packet=dict(input=dict(frames=stem+'.npz',state_v1=state,language=episode['instruction']),
                        audit=dict(episode=episode_id,actual_sim_time_s=now,task_elapsed_s=elapsed,frame_sha256=mt.sha(directory/(stem+'.npz')),
                            target_geometry_excluded=True,gripper='UNEXECUTED_CONSTANT_NEUTRAL_0.5',label='DESIRED_NOT_ACHIEVED',
                            T_world_native=transform.tolist(),camera_view_matrices={name:np.asarray(gym.get_camera_view_matrix(native.sim,native.envs[0],handle)).tolist()
                                for name,handle in native.stage21_camera_handles.items()}))
                    mt.write(directory/(stem+'.json'),packet);input_from_snapshot(directory/(stem+'.json'));snapshots.append(packet)
                    last_capture=now
                row=tick();row['task_elapsed_s']=clock()-start;records.append(row)
                if not row['physical_clean']:
                    result['failure']='TASK_PHYSICAL_TERMINATION';break
            if len(records)>=2:
                metric=mt.score(registry,family,task,command,[r['task_elapsed_s'] for r in records],[r['achieved_world'] for r in records],
                    [r['achieved_quaternion'] for r in records],[r['obstacle_contact_force_n']>registry['geometry']['contact_force_threshold_n'] for r in records],
                    all(r['physical_clean'] for r in records) and not result.get('failure'),[r['desired_world'] for r in records])
            else:metric=dict(success=False,reason='NO_COMPLETE_TASK_TRACE')
            if records:
                rotations=2*np.arccos(np.clip(np.abs(np.sum(np.asarray([r['achieved_quaternion'] for r in records])*np.asarray([r['desired_quaternion'] for r in records]),axis=1)),0,1))
                metric['same_tick_orientation_rms_rad']=float(np.sqrt(np.mean(rotations**2)))
                metric['same_tick_orientation_p95_rad']=float(np.quantile(rotations,.95))
            result.update(validity='VALID',success=metric['success'],task_metrics=metric)
        result['settling_ticks']=len(settling);result['task_ticks']=len(records);result['snapshots']=len(snapshots)
    except Exception as error:
        result['engineering_error']=dict(type=type(error).__name__,message=str(error))
    finally:
        if native is not None:
            result['end_actual_sim_time_s']=float(native.gym.get_sim_time(native.sim));result['reference_load_events']=native._stage21_load_events
        if env is not None:env.close()
        result['physics_started']=(out/'physics_started.json').exists()
        result['raw_ticks_sha256']=mt.sha(out/'raw_ticks.jsonl') if (out/'raw_ticks.jsonl').exists() else None
        mt.write(out/'result.json',result);append(ROOT/'attempt_results.jsonl',result)
    print(json.dumps(dict(episode=episode_id,validity=result['validity'],success=result['success'],failure=result.get('failure'),error=result.get('engineering_error'))),flush=True)
    return 0 if result['validity']=='VALID' else 3


def cpu_check(output):
    import tempfile
    with tempfile.TemporaryDirectory(dir=str(output)) as temporary:
        root=Path(temporary);ledger=root/'physics.jsonl';out=root/'attempt';out.mkdir()
        class Fake:
            def __init__(self):self.gym=SimpleNamespace(get_sim_time=lambda sim:0.);self.sim=None;self.calls=[]
            def step(self,*args,**kwargs):self.calls.append((args,kwargs));return 'unchanged'
        model=physics_audit(Fake,'CPU_FIXTURE_NOT_AN_EXPERT_EPISODE',out,ledger)()
        assert model.step(1,value=2)=='unchanged' and model.step(3)=='unchanged' and len(model.calls)==2
        assert len(ledger.read_text().splitlines())==1
        np.savez_compressed(root/'input.npz',body_rgb=np.zeros((96,96,3),np.uint8),wrist_rgb=np.zeros((96,96,3),np.uint8))
        packet=dict(input=dict(frames='input.npz',state_v1=[0.]*10,language='CPU'),audit=dict(target_world=[1,2,3]))
        mt.write(root/'input.json',packet);assert input_from_snapshot(root/'input.json')['language']=='CPU'
        packet['input']['target_world']=[1,2,3];mt.write(root/'bad.json',packet)
        try:input_from_snapshot(root/'bad.json')
        except AssertionError:pass
        else:raise AssertionError('actual dataset loader accepted privilege')
    receipt=dict(status='PASS_REAL_COLLECTOR_STEP_AUDIT_AND_DATASET_LOADER_CPU',collector_sha256=mt.sha(__file__),
        tests=['same_step_arguments_and_return','first_actual_step_once','actual_npz_JSON_loader','privilege_negative_through_actual_loader'],
        GR00T_loads=0,real_physics_steps=0)
    mt.write(Path(output)/'collector_cpu_receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--episode');parser.add_argument('--authorization');parser.add_argument('--cpu-check',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'design');args=parser.parse_args()
    if args.cpu_check:args.output.mkdir(parents=True,exist_ok=True);cpu_check(args.output)
    elif args.episode:
        assert args.authorization==AUTH,'explicit registered expert authorization required'
        raise SystemExit(run_episode(args.episode))
    else:parser.print_help()
