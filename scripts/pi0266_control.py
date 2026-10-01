"""Actual fixed teacher diagnostics / TRAIN window collection; no GR00T rescue."""
import isaacgym
import copy
import json
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from pi0266_common import ROOT,BASE,OLD,RAW,REF,read,sha,write,verify_sources
sys.path[:0]=[str(REF/'src'),str(REF/'scripts'),str(BASE/'PI0256/runtime')]
import stage21_native_receding_horizon as api
from pi0265_control import scene_class
from pi0266_reset import chunk as neutral_chunk
from pi0266_expert import ObservableCorrection,reference
from pi0266_metrics import score,so3
from pi0255_collect import WBC,array_sha,append
from pi0255_motion3 import scene
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics


def run(case_id):
    verify_sources();seal=read(ROOT/'design/TEACHER_EXECUTION_SEAL.json')
    for rel,digest in seal['files'].items():assert sha(ROOT/rel)==digest,rel
    card=read(ROOT/'design/SCENE_CARD.json');cases=read(ROOT/'design/TEACHER_CARD.json')['diagnostics']+read(ROOT/'design/TEACHER_CARD.json')['collection']
    case=next(x for x in cases if x['case_id']==case_id);family=case['family'];reg=copy.deepcopy(card['registry'])
    source=read(ROOT/'evidence/SOURCE_IDENTITY.json');paths={x['name']:x['path'] for x in source['files']}
    case_root=RAW/('diagnostics' if case['stage']=='DIAGNOSTIC' else 'collection')/case_id;case_root.mkdir(parents=True,exist_ok=True)
    prior=list(case_root.glob('attempt-*'));assert all((p/'result.json').is_file() for p in prior),'unresolved/live attempt'
    assert not any(read(p/'result.json')['validity']=='VALID' for p in prior),'first valid retained'
    out=case_root/('attempt-%02d'%len(prior));out.mkdir()
    for d in ('frames','references','actor_inputs','windows'):(out/d).mkdir()
    result=dict(case=case,validity='ENGINEERING_INCOMPLETE',success=False,GR00T_queries=0,optimizer_updates=0,selected_checkpoint='NONE',
        deployment_selected_checkpoint='NONE',execution_seal_sha256=sha(ROOT/'design/TEACHER_EXECUTION_SEAL.json'),simulation_mode='NON_REALTIME_LOCKSTEP')
    kin=NativeArmKinematics.from_urdf(paths['native_urdf']);gate=TrajectorySafetyGateV1();native=None;worker=None
    rows=[];windows=[];actor_calls=[];tick=[0];camera_checks=[];clock=lambda n:float(n.gym.get_sim_time(n.sim))
    bounds={j.attrib['name']:dict(lower=float(j.find('limit').attrib['lower']),upper=float(j.find('limit').attrib['upper']),unit='m' if j.attrib['type']=='prismatic' else 'rad')
        for j in ET.parse(paths['native_urdf']).getroot().findall('joint') if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
    def fk(n):
        root=n.root_states[0].detach().cpu().numpy();q=np.array([n.dof_pos[0,n.dof_names.index(j)].item() for j in kin.joint_names]);pose=kin.tcp_pose(q)
        r=Rotation.from_quat(root[3:7]).as_matrix();return r@pose[:3,3]+root[:3],r@pose[:3,:3]
    def chunk(p,r,anchor,now):
        t=now+np.arange(len(p))*.05
        return TrajectoryChunkV1(chunk_id=case_id+'-%04d'%tick[0],source='PI0266_'+case['expert'],observation_timestamp=now,
            point_timestamps=t,eef_position=p-anchor,eef_orientation=Rotation.from_matrix(r).as_quat(),source_frame=api.SOURCE_FRAME,
            tcp_semantic_id=api.TCP_SEMANTIC_ID,validity_start=now,validity_end=float(t[-1]),gripper=np.full(len(t),.5))
    def immutable(n,env):
        values={k:api._tensor_hash(getattr(n,k)) for k in ('root_states','dof_state','rigid_body_state','end_effector_state','episode_length_buf','reset_buf','traj_sim_time','traj_s') if hasattr(n,k)}
        state=np.random.get_state();values.update(time=clock(n),python_rng=repr(random.getstate()),numpy_rng=[state[0],array_sha(state[1]),*state[2:]],
            torch_rng=api._tensor_hash(torch.get_rng_state()),cuda_rng=[api._tensor_hash(v) for v in torch.cuda.get_rng_state_all()],
            history={k:api._tensor_hash(v) for k,v in vars(env).items() if 'history' in k and hasattr(v,'detach')})
        return values
    def snapshot(n):
        p,r=fk(n);np_p,np_q=api._current_eef(n);s=n.traj_batch.s_ref(n.traj_sim_time)
        desired=n.traj_batch.p_at(s)[0].detach().cpu().numpy();dr=Rotation.from_quat(n.traj_batch.quat_at(s)[0].detach().cpu().numpy()).as_matrix()
        assert np.linalg.norm(p-np_p)<=1e-5 and so3(r,Rotation.from_quat(np_q).as_matrix())<=1e-4
        return dict(time=clock(n),root=n.root_states[0].detach().cpu().tolist(),DOF=n.dof_state.detach().cpu().tolist(),TCP=p.tolist(),rotation=r.tolist(),
            installed=desired.tolist(),installed_rotation=dr.tolist(),cursor=n.traj_sim_time.detach().cpu().tolist(),telemetry=api._native_runtime_telemetry(n))
    import pytorch3d.transforms as transforms
    transforms.matrix_to_rotation_6d=lambda m:m[..., :2,:].clone().reshape(m.shape[:-2]+(6,))
    original=api._Stage21NativeEnvFactory.make
    def factory(WBCEnv,initial,torch,gymapi,gymtorch):
        base=original(WBCEnv,initial,torch,gymapi,gymtorch);cls=scene_class(base,reg,family,gymapi)
        class Audited(cls):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);before=api._tensor_hash(self.rigid_body_state);values=api._tensor_hash(self.end_effector_state)
                self.end_effector_state=self.end_effector_state.clone()
                assert before==api._tensor_hash(self.rigid_body_state) and values==api._tensor_hash(self.end_effector_state)
            def _load_trajectory_for(self,ids):
                if not hasattr(self,'pi0266_neutral'):
                    p,r=fk(self);proxy=SimpleNamespace(_shoulder_frame=self._shoulder_frame,_traj_anchor_offset=self._traj_anchor_offset,
                        base_quat=self.base_quat,reach_table=self.reach_table,cfg=self.cfg,device=self.device,traj_batch=SimpleNamespace(gamma_p=torch.zeros_like(self.traj_batch.gamma_p)))
                    base._anchor_external_reference(proxy,ids);anchor=proxy.traj_batch.gamma_p[0,0].detach().cpu().numpy().copy()
                    c=neutral_chunk(p,r,anchor);verdict=gate.evaluate(c,now=0.,achieved_position=p-anchor,achieved_orientation=Rotation.from_matrix(r).as_quat())
                    write(out/'neutral_construction_gate.json',verdict.as_dict());assert verdict.decision is not GateDecision.REJECT
                    self._stage21_initial_chunk=c;self.pi0266_neutral=True
                return super()._load_trajectory_for(ids)
            def step(self,*args,**kwargs):
                import fcntl
                before=snapshot(self)
                with (ROOT/'physics.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);ledger=ROOT/'evidence/RAW_BUDGET.json';budget=read(ledger);caps=read(ROOT/'design/BUDGET_CARD.json')
                    if not (out/'physics_started.json').exists():
                        assert budget['starts']<caps['hard_starts'];budget['starts']+=1
                        write(out/'physics_started.json',dict(case=case_id,pid=os.getpid(),epoch=time.time(),ordinal=budget['starts'],state=before))
                        np.savez_compressed(out/'pre_first_step.npz',root=self.root_states.detach().cpu().numpy(),dof=self.dof_state.detach().cpu().numpy(),
                            torch_rng=torch.get_rng_state().numpy(),cuda_rng=torch.cuda.get_rng_state().cpu().numpy())
                    assert budget['control_ticks']<caps['hard_control_ticks'] and budget['simulated_seconds']+self.dt<=caps['hard_simulated_seconds']+1e-6
                    forwards=2 if self.stage21_wbc_forward_count else 0;assert budget['actor_calls']+forwards<=caps['hard_actor_calls']
                    budget['control_ticks']+=1;budget['simulate_calls']+=int(self.cfg.control.decimation);budget['simulated_seconds']+=float(self.dt);budget['actor_calls']+=forwards
                    write(ledger,budget)
                returned=super().step(*args,**kwargs)
                append(out/'physics_steps.jsonl',dict(before=before,after=snapshot(self),pre_autoreset=getattr(self,'_stage16_pending_termination',None)),sync=False)
                return returned
        return Audited
    api._Stage21NativeEnvFactory.make=staticmethod(factory)
    try:
        initial=dict(np.load('/home/liuyu-nfs/pi0223_goal_b_20260918/refs_r00/train-a_c0_continuous_reference.npz',allow_pickle=False))
        args=SimpleNamespace(source_root=WBC,sim_device='cuda:0',seed=case['reset_seed'],mode='gt_receding',arm=Path(paths['arm_checkpoint']),dog=Path(paths['dog_checkpoint']))
        native,env,actors,loads,cfg=api._build_native(args,initial);result['WBC_loads']=loads;origin=clock(native)
        assert origin<=1+1e-6 and abs(native.dt-read(ROOT/'design/BUDGET_CARD.json')['control_dt'])<1e-9
        event=native.pop_stage16_termination();result['initialization_sim_s']=origin
        if event or native.reset_buf[0].item():result.update(validity='VALID',failure='NATIVE_INITIALIZATION_TERMINATION',safety_failure=True);return result
        p0,r0=fk(native);task_reg=copy.deepcopy(reg);pose=np.eye(4);pose[:3,3]=p0;pose[:3,:3]=r0;task_reg['initial_TCP_world']=pose.tolist()
        anchor=native.stage21_anchor_world.detach().cpu().numpy().copy();policy_anchor=np.asarray(card['policy_anchor_world_m'])
        from isaacgym import gymapi,gymtorch
        physical_before=api._tensor_hash(native.root_states[0]);dof_before=api._tensor_hash(native.dof_state);ids=[];data=scene(reg,family)
        for handle,i,a in native.pi0265_axes:
            index=int(native.gym.get_actor_index(native.envs[0],handle,gymapi.DOMAIN_SIM));ids.append(index)
            target_r=r0@Rotation.from_euler('z',reg['geometry']['target_yaw_rad'][i]).as_matrix()
            native.root_states[index,:3]=native.root_states.new_tensor(data['targets'][i]+target_r[:,a]*reg['geometry']['axes_length_m'])
        ids=torch.tensor(ids,device=native.device,dtype=torch.int32)
        native.gym.set_actor_root_state_tensor_indexed(native.sim,gymtorch.unwrap_tensor(native.root_states),gymtorch.unwrap_tensor(ids),len(ids))
        assert physical_before==api._tensor_hash(native.root_states[0]) and dof_before==api._tensor_hash(native.dof_state)
        pp,rr=reference(task_reg,family,case['task'],case['command'],np.arange(401)*.05)
        expert=ObservableCorrection(task_reg,family,case['task'],case['command'])
        spec=dict(task=case['task'],command=case['command'],goal=pp[-1].tolist(),goal_rotation=rr[-1].tolist(),center=data['center'].tolist(),radius=data['radius'])
        write(out/'task_spec.json',spec);write(out/'task_binding.json',dict(post_initialization_pose=pose.tolist(),normal_seed=case['reset_seed'],native_anchor=anchor.tolist(),policy_anchor=policy_anchor.tolist(),robot_state_unchanged=True))
        worker_env=dict(os.environ,PYTHONPATH=':'.join(map(str,[ROOT/'code',REF/'src',BASE/'PI0256/runtime'])))
        for name in ('LD_LIBRARY_PATH','LD_PRELOAD'):worker_env.pop(name,None)
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(ROOT/'code/pi0266_geometry_worker.py')],env=worker_env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        ready=json.loads(worker.stdout.readline());assert ready['ready'];write(out/'collision_exclusions.json',ready)
        def query(kind,extra=None):
            row=dict(kind=kind,root=native.root_states[0].detach().cpu().tolist(),joints={name:native.dof_pos[0,i].item() for i,name in enumerate(native.dof_names)},
                obstacle=data['obstacle'].tolist(),half_extents=reg['geometry']['obstacle_half_extents_m'])
            if extra:row.update(extra)
            worker.stdin.write(json.dumps(row)+'\n');worker.stdin.flush();answer=worker.stdout.readline();assert answer,'geometry worker exited';return json.loads(answer)
        class Actor:
            def __init__(self,actor,name):self.actor,self.name=actor,name
            def __call__(self,value):
                x=value.detach().cpu().numpy().copy();answer=self.actor(value);assert np.isfinite(x).all() and torch.isfinite(answer).all()
                np.savez_compressed(out/'actor_inputs'/('%04d-%s.npz'%(tick[0],self.name)),history=x,output=answer.detach().cpu().numpy())
                actor_calls.append(dict(tick=tick[0],actor=self.name,time=clock(native),input_sha256=array_sha(x)))
                return answer
        actors=(Actor(actors[0],'arm'),Actor(actors[1],'dog'),actors[2],actors[3])
        next_plan=origin;installed_clock=False;safety=False;failure=None
        while clock(native)-origin<40-native.dt/2:
            now=clock(native);pre=snapshot(native);pre['elapsed']=now-origin
            pen={name:max(0.,bounds[name]['lower']-native.dof_pos[0,i].item(),native.dof_pos[0,i].item()-bounds[name]['upper']) for i,name in enumerate(native.dof_names) if name in bounds}
            unsafe=[name for name,v in pen.items() if v>(1e-5 if bounds[name]['unit']=='m' else 1e-3)];collision=query('STATE')
            pre.update(hard_penetration=pen,hard_unsafe=unsafe,collision=collision);rows.append(pre)
            safety=bool(unsafe or collision['hits'] or pre['telemetry'].get('invalid_state') or pre['telemetry'].get('fall_flag') or pre['telemetry'].get('joint_limit_termination'))
            if safety:failure='PRE_STEP_SAFETY';break
            if now>=next_plan-1e-6:
                p,r=fk(native);before=immutable(native,env);body,wrist=api._capture_scene(native);assert before==immutable(native,env),'CAMERA_MUTATED_CONTROL_OR_RNG'
                camera_checks.append(dict(time=now,exact_unchanged=True));state=np.r_[p-policy_anchor,r[:2,:].reshape(6),.5].astype(np.float32)
                frame=out/'frames'/('%04d.npz'%tick[0]);np.savez_compressed(frame,body_rgb=body,wrist_rgb=wrist,state_v1=state,time=now,root=native.root_states[0].detach().cpu().numpy())
                if case['expert']=='CLOCK_DRIVEN':
                    if not installed_clock:
                        c=chunk(pp,rr,anchor,now);verdict=gate.evaluate(c,now=now,achieved_position=p-anchor,achieved_orientation=Rotation.from_matrix(r).as_quat())
                        write(out/'clock_original_gate.json',verdict.as_dict())
                        if verdict.decision is GateDecision.REJECT:failure='CLOCK_GATED_REFERENCE_REJECT';break
                        physical=immutable(native,env);event=native.initial_bind_reference(c);native._update_ee_task_space_error();after=immutable(native,env)
                        for key in ('root_states','dof_state','rigid_body_state','end_effector_state','episode_length_buf','reset_buf','time','history','torch_rng','cuda_rng','numpy_rng','python_rng'):assert physical[key]==after[key],key
                        installed_clock=True;write(out/'clock_reference_only_install.json',event)
                        np.savez_compressed(out/'references/clock_full20s.npz',position=pp,rotation=rr,point_timestamps=c.point_timestamps)
                        current=snapshot(native)
                        pre.update(installed_before_handoff=pre['installed'],installed=current['installed'],installed_rotation=current['installed_rotation'],cursor=current['cursor'])
                else:
                    lp,lr,metadata=expert.label(p,r);label=np.c_[lp-policy_anchor,lr[:,:2,:].reshape(40,6),np.full(40,.5)].astype(np.float32)
                    folder=out/'windows'/('%04d'%tick[0]);folder.mkdir();proposal=folder/'proposal.npz';np.savez_compressed(proposal,state_v1=state,raw_action=label,position_world=lp,rotation_world=lr,time=now)
                    c=chunk(lp,lr,anchor,now);verdict=gate.evaluate(c,now=now,achieved_position=p-anchor,achieved_orientation=Rotation.from_matrix(r).as_quat())
                    checks=None;reason=None
                    if not metadata['valid_label']:reason='NO_LABEL_PHASE_ALIAS'
                    elif verdict.decision is GateDecision.REJECT:reason='ORIGINAL_RAW_GATE_REJECT'
                    else:
                        checks=query('REFERENCE',dict(position=lp.tolist(),rotation=lr.tolist(),state_v1=state.tolist(),raw_action=label.tolist(),codec_output=str(folder/'codec.npz')))
                        if not checks['valid']:reason=checks.get('reason') or 'CODEC_INVALID'
                    window=dict(id=case_id+'-%04d'%tick[0],family=family['family_id'],task=case['task'],command=case['command'],timestamp=now,elapsed=now-origin,
                        frame=str(frame),frame_sha256=sha(frame),proposal=str(proposal),proposal_sha256=sha(proposal),gate=verdict.as_dict(),metadata=metadata,checks=checks,
                        admitted=reason is None,label_class='WINDOW_VALID_REFERENCE' if reason is None else reason,parent_result_pending=True)
                    windows.append(window);write(folder/'window.json',window)
                    if reason:failure=reason;break
                    physical=immutable(native,env)
                    event=native.initial_bind_reference(c) if native.stage21_wbc_forward_count==0 else native.load_reference_chunk(c)
                    native._update_ee_task_space_error();after=immutable(native,env)
                    for key in ('root_states','dof_state','rigid_body_state','end_effector_state','episode_length_buf','reset_buf','time','history','torch_rng','cuda_rng','numpy_rng','python_rng'):assert physical[key]==after[key],key
                    np.savez_compressed(folder/'installed_table.npz',**{k:v.detach().cpu().numpy() for k,v in vars(native.traj_batch).items() if hasattr(v,'detach')})
                    write(folder/'reference_only_install.json',event)
                    current=snapshot(native)
                    pre.update(installed_before_handoff=pre['installed'],installed=current['installed'],installed_rotation=current['installed_rotation'],cursor=current['cursor'])
                next_plan+=.4
            with torch.no_grad():done,policy=api._policy_step(native,env,actors)
            event=native.pop_stage16_termination();append(out/'policy_actions.jsonl',dict(tick=tick[0],time=now,policy=policy),sync=False)
            tick[0]+=1
            if done or event or native.reset_buf[0].item():failure='NATIVE_TERMINATION';safety=True;result['pre_autoreset']=event;break
        if not safety:
            final=snapshot(native);final['elapsed']=final['time']-origin
            if not rows or final['time']>rows[-1]['time']+1e-9:rows.append(final)
        t=np.array([x['elapsed'] for x in rows]);p=np.array([x['TCP'] for x in rows]);r=np.array([x['rotation'] for x in rows]);ip=np.array([x['installed'] for x in rows]);ir=np.array([x['installed_rotation'] for x in rows])
        metrics=score(t,p,r,ip,ir,spec,safety)
        if failure in ('ORIGINAL_RAW_GATE_REJECT','CLOCK_GATED_REFERENCE_REJECT'):metrics['success']=False
        result.update(validity='VALID',failure=failure or (None if metrics['success'] else 'TASK_CRITERIA_FAIL'),safety_failure=safety,metrics=metrics,success=metrics['success'],
            task_ticks=tick[0],WBC_actor_calls=len(actor_calls),proposed_windows=len(windows),valid_windows=sum(w['admitted'] for w in windows),
            camera_checks=camera_checks,dof_names=native.dof_names,actual_task_s=float(t[-1]),window_manifest='windows.json')
        np.savez_compressed(out/'world_tcp.npz',elapsed=t,position=p,rotation=r,installed_position=ip,installed_rotation=ir)
        for window in windows:window.update(parent_result_pending=False,parent_task_success=result['success'],parent_failure=result['failure'],parent_safety_failure=safety)
        write(out/'windows.json',windows)
    finally:
        if worker is not None:worker.stdin.close();worker.wait(timeout=30)
        write(out/'rows.json',rows);write(out/'actor_calls.json',actor_calls);write(out/'result.json',result)
        write(ROOT/'evidence'/('result-'+case_id+'.json'),result)
        if native is not None:native.gym.destroy_sim(native.sim)
    print(json.dumps(dict(case=case_id,validity=result['validity'],success=result['success'],failure=result.get('failure'),valid_windows=result.get('valid_windows'))),flush=True)
    return result


if __name__=='__main__':run(sys.argv[sys.argv.index('--case')+1])
