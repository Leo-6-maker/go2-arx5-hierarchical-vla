"""Four fresh-process, neutral-reference normal resets. No task or model rollout."""
import isaacgym
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
from pi0265_control import scene_class,forbidden_hits
from pi0255_collect import WBC,array_sha,append
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
from pi0265_metrics import so3


def chunk(p,r,anchor):
    t=np.arange(40)*.05
    return TrajectoryChunkV1(chunk_id='PI0266_NEUTRAL_RESET',source='PI0266_NEUTRAL_CURRENT_POSE',observation_timestamp=0.,
        point_timestamps=t,eef_position=np.repeat((p-anchor)[None],40,axis=0),eef_orientation=np.repeat(Rotation.from_matrix(r).as_quat()[None],40,axis=0),
        source_frame=api.SOURCE_FRAME,tcp_semantic_id=api.TCP_SEMANTIC_ID,validity_start=0.,validity_end=1.95,gripper=np.full(40,.5))


def native_cpu():
    from modules.trajectory import to_canonical,quat_to_mat,TrajectoryBatch
    c=chunk(np.array([.3,0,.6]),Rotation.from_euler('xyz',[.31,-.47,.83]).as_matrix(),np.array([.1,.02,.5]))
    gate=TrajectorySafetyGateV1().evaluate(c,now=0.,achieved_position=c.eef_position[0],achieved_orientation=c.eef_orientation[0])
    assert gate.decision is not GateDecision.REJECT,gate.as_dict()
    gamma,law=to_canonical(torch.tensor(c.eef_position,dtype=torch.float32),quat_to_mat(torch.tensor(c.eef_orientation,dtype=torch.float32)),.05)
    batch=TrajectoryBatch(1,1536,device='cpu',max_tl_points=512);batch.load([0],[gamma],[law])
    s=batch.s_ref(torch.tensor([0.]));p=batch.p_at(s);q=batch.quat_at(s)
    assert torch.isfinite(p).all() and torch.isfinite(q).all()
    assert np.linalg.norm(p.numpy()[0]-c.eef_position[0])<=1e-5
    assert so3(Rotation.from_quat(q.numpy()[0]).as_matrix(),Rotation.from_quat(c.eef_orientation[0]).as_matrix())<=1e-4
    write(ROOT/'evidence/NEUTRAL_NATIVE_CPU.json',dict(status='PASS_ORIGINAL_GATE_CANONICAL_NEUTRAL_CURRENT_POSE',gate=gate.as_dict(),physics=0,model_queries=0))
    print('PASS_NATIVE_NEUTRAL_NO_PHYSICS',flush=True)


def run(case_id):
    verify_sources();seal=read(ROOT/'design/RESET_EXECUTION_SEAL.json')
    for rel,digest in seal['files'].items():assert sha(ROOT/rel)==digest,rel
    card=read(ROOT/'design/SCENE_CARD.json');case=next(x for x in card['reset_cases'] if x['case_id']==case_id)
    source=read(ROOT/'evidence/SOURCE_IDENTITY.json');paths={x['name']:x['path'] for x in source['files']}
    root=RAW/'resets'/case_id;root.mkdir(parents=True,exist_ok=True);prior=list(root.glob('attempt-*'))
    assert all((p/'result.json').is_file() for p in prior),'unresolved/live reset'
    assert not any(read(p/'result.json')['validity']=='VALID' for p in prior),'first valid retained'
    out=root/('attempt-%02d'%len(prior));out.mkdir();native=None;worker=None
    result=dict(case=case,validity='ENGINEERING_INCOMPLETE',safe=False,GR00T_queries=0,WBC_actor_calls=0,optimizer_updates=0,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',reset_execution_seal_sha256=sha(ROOT/'design/RESET_EXECUTION_SEAL.json'))
    kin=NativeArmKinematics.from_urdf(paths['native_urdf']);gate=TrajectorySafetyGateV1();clock=lambda n:float(n.gym.get_sim_time(n.sim))
    bounds={j.attrib['name']:dict(lower=float(j.find('limit').attrib['lower']),upper=float(j.find('limit').attrib['upper']),unit='m' if j.attrib['type']=='prismatic' else 'rad')
        for j in ET.parse(paths['native_urdf']).getroot().findall('joint') if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
    def fk(n):
        root=n.root_states[0].detach().cpu().numpy();q=np.array([n.dof_pos[0,n.dof_names.index(j)].item() for j in kin.joint_names])
        pose=kin.tcp_pose(q);r=Rotation.from_quat(root[3:7]).as_matrix();return r@pose[:3,3]+root[:3],r@pose[:3,:3]
    def immutable(n,env):
        values={k:api._tensor_hash(getattr(n,k)) for k in ('root_states','dof_state','rigid_body_state','end_effector_state','episode_length_buf','reset_buf','traj_sim_time','traj_s') if hasattr(n,k)}
        state=np.random.get_state();values.update(time=clock(n),python_rng=repr(random.getstate()),numpy_rng=[state[0],array_sha(state[1]),*state[2:]],
            torch_rng=api._tensor_hash(torch.get_rng_state()),cuda_rng=[api._tensor_hash(v) for v in torch.cuda.get_rng_state_all()],
            history={k:api._tensor_hash(v) for k,v in vars(env).items() if 'history' in k and hasattr(v,'detach')})
        return values
    import pytorch3d.transforms as transforms
    transforms.matrix_to_rotation_6d=lambda m:m[..., :2,:].clone().reshape(m.shape[:-2]+(6,))
    original=api._Stage21NativeEnvFactory.make
    def factory(WBCEnv,initial,torch,gymapi,gymtorch):
        base=original(WBCEnv,initial,torch,gymapi,gymtorch);scene=scene_class(base,card['registry'],case['family'],gymapi)
        class Audited(scene):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                raw=api._tensor_hash(self.rigid_body_state);before=api._tensor_hash(self.end_effector_state)
                self.end_effector_state=self.end_effector_state.clone()
                assert raw==api._tensor_hash(self.rigid_body_state) and before==api._tensor_hash(self.end_effector_state)
            def _load_trajectory_for(self,ids):
                if not hasattr(self,'pi0266_neutral'):
                    p,r=fk(self);proxy=SimpleNamespace(_shoulder_frame=self._shoulder_frame,_traj_anchor_offset=self._traj_anchor_offset,
                        base_quat=self.base_quat,reach_table=self.reach_table,cfg=self.cfg,device=self.device,traj_batch=SimpleNamespace(gamma_p=torch.zeros_like(self.traj_batch.gamma_p)))
                    base._anchor_external_reference(proxy,ids);anchor=proxy.traj_batch.gamma_p[0,0].detach().cpu().numpy().copy()
                    c=chunk(p,r,anchor);verdict=gate.evaluate(c,now=0.,achieved_position=p-anchor,achieved_orientation=Rotation.from_matrix(r).as_quat())
                    write(out/'construction_gate.json',verdict.as_dict());assert verdict.decision is not GateDecision.REJECT
                    self._stage21_initial_chunk=c;self.pi0266_neutral=True
                    np.savez_compressed(out/'neutral_reference.npz',position=c.eef_position,quaternion=c.eef_orientation,anchor=anchor)
                return super()._load_trajectory_for(ids)
            def step(self,*args,**kwargs):
                import fcntl
                with (ROOT/'physics.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);ledger=ROOT/'evidence/RAW_BUDGET.json'
                    budget=read(ledger) if ledger.exists() else dict(starts=0,control_ticks=0,simulate_calls=0,actor_calls=0,simulated_seconds=0.)
                    caps=read(ROOT/'design/BUDGET_CARD.json')
                    if not (out/'physics_started.json').exists():
                        assert budget['starts']<caps['hard_starts'];budget['starts']+=1
                        write(out/'physics_started.json',dict(pid=os.getpid(),epoch=time.time(),ordinal=budget['starts'],clock_before_s=clock(self)))
                        np.savez_compressed(out/'pre_first_step.npz',root=self.root_states.detach().cpu().numpy(),dof=self.dof_state.detach().cpu().numpy(),
                            torch_rng=torch.get_rng_state().numpy(),cuda_rng=torch.cuda.get_rng_state().cpu().numpy())
                    assert budget['control_ticks']<caps['hard_control_ticks'] and budget['simulated_seconds']+self.dt<=caps['hard_simulated_seconds']+1e-6
                    budget['control_ticks']+=1;budget['simulate_calls']+=int(self.cfg.control.decimation);budget['simulated_seconds']+=float(self.dt)
                    write(ledger,budget)
                return super().step(*args,**kwargs)
        return Audited
    api._Stage21NativeEnvFactory.make=staticmethod(factory)
    try:
        initial=dict(np.load('/home/liuyu-nfs/pi0223_goal_b_20260918/refs_r00/train-a_c0_continuous_reference.npz',allow_pickle=False))
        args=SimpleNamespace(source_root=WBC,sim_device='cuda:0',seed=case['reset_seed'],mode='reset_only',arm=None,dog=None)
        native,env,actors,loads,cfg=api._build_native(args,initial)
        caps=read(ROOT/'design/BUDGET_CARD.json');assert abs(native.dt-caps['control_dt'])<=1e-9
        assert int(cfg.control.decimation)==caps['decimation'] and int(cfg.sim.substeps)==caps['sim_substeps']
        p,r=fk(native);np_p,np_q=api._current_eef(native)
        assert np.linalg.norm(p-np_p)<=1e-5 and so3(r,Rotation.from_quat(np_q).as_matrix())<=1e-4
        event=native.pop_stage16_termination();telemetry=api._native_runtime_telemetry(native)
        pen={name:max(0.,bounds[name]['lower']-native.dof_pos[0,i].item(),native.dof_pos[0,i].item()-bounds[name]['upper']) for i,name in enumerate(native.dof_names) if name in bounds}
        unsafe=[name for name,v in pen.items() if v>(1e-5 if bounds[name]['unit']=='m' else 1e-3)]
        from pi0255_motion3 import scene
        data=scene(card['registry'],case['family'])
        worker_env=dict(os.environ,PYTHONPATH=':'.join(map(str,[OLD/'code',REF/'src',BASE/'PI0256/runtime'])))
        for name in ('LD_LIBRARY_PATH','LD_PRELOAD'):worker_env.pop(name,None)
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(OLD/'code/pi0265_control.py'),'--collision-worker'],env=worker_env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        ready=json.loads(worker.stdout.readline());assert ready['ready'];write(out/'collision_exclusions.json',ready)
        worker.stdin.write(json.dumps(dict(root=native.root_states[0].detach().cpu().tolist(),joints={name:native.dof_pos[0,i].item() for i,name in enumerate(native.dof_names)},
            obstacle=data['obstacle'].tolist(),half_extents=card['registry']['geometry']['obstacle_half_extents_m']))+'\n');worker.stdin.flush()
        collision=json.loads(worker.stdout.readline());safe=not (event or native.reset_buf[0].item() or unsafe or collision['hits'] or telemetry.get('invalid_state') or telemetry.get('fall_flag') or telemetry.get('joint_limit_termination'))
        before=immutable(native,env);body,wrist=api._capture_scene(native);after=immutable(native,env)
        assert before==after,'CAMERA_MUTATED_CONTROL_OR_RNG'
        now=clock(native);assert now<=1+1e-6
        history={k:v.detach().cpu().numpy() for k,v in vars(env).items() if 'history' in k and hasattr(v,'detach')}
        np.savez_compressed(out/'reset_state.npz',root=native.root_states[0].detach().cpu().numpy(),dof=native.dof_state.detach().cpu().numpy(),TCP=p,rotation=r,**history)
        np.savez_compressed(out/'reset_images.npz',body_rgb=body,wrist_rgb=wrist,time=now)
        result.update(validity='VALID',safe=bool(safe),failure=None if safe else 'FIXED_RESET_UNSAFE',collision=collision,hard_unsafe=unsafe,
            hard_penetration=pen,telemetry=telemetry,native_termination=event,initialization_sim_s=now,camera_exact_noninterference=True,
            state_sha256=sha(out/'reset_state.npz'),images_sha256=sha(out/'reset_images.npz'),dof_names=native.dof_names,history_fields=list(history),
            native_FK_xyz_m=float(np.linalg.norm(p-np_p)),native_FK_so3_rad=float(so3(r,Rotation.from_quat(np_q).as_matrix())),normal_reset_only=True)
    finally:
        if worker is not None:worker.stdin.close();worker.wait(timeout=30)
        if native is not None:native.gym.destroy_sim(native.sim)
        write(out/'result.json',result);write(ROOT/'evidence'/('result-'+case_id+'.json'),result)
    print(json.dumps(dict(case=case_id,validity=result['validity'],safe=result['safe'],failure=result.get('failure'))),flush=True)


if __name__=='__main__':
    if '--native-cpu' in sys.argv:native_cpu()
    else:run(sys.argv[sys.argv.index('--case')+1])
