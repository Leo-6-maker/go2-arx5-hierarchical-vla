"""Delivered frozen WBC binding and the two authorized native fixtures."""
import isaacgym  # Native library must precede torch.
import copy,hashlib,json,os,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import stage21_native_receding_horizon as api
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280')
WBC=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare')
REF=ROOT.parent/'PI0250/runtime/reference-317a'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def append(p,value,sync=True):
    with Path(p).open('a') as f:
        f.write(json.dumps(value)+'\n');f.flush()
        if sync:os.fsync(f.fileno())

def configure(cfg):
    # The old evaluation helper disables body hard stops. Preserve delivered values.
    keep=[{k:copy.deepcopy(v) for k,v in vars(g).items() if 'terminal' in k} for g in (cfg.rewards,cfg.wbc.rewards)]
    api._configure_stage21_original(cfg)
    for group,values in zip((cfg.rewards,cfg.wbc.rewards),keep):
        for k,v in values.items():setattr(group,k,v)

api._configure_stage21_original=api._configure_stage21
api._configure_stage21=configure

def chunk(name,p,q,t):
    return TrajectoryChunkV1(chunk_id=name,source='PI0280_NATIVE_FIXTURE',observation_timestamp=float(t[0]),
        point_timestamps=t,eef_position=p,eef_orientation=q,source_frame=api.SOURCE_FRAME,tcp_semantic_id=api.TCP_SEMANTIC_ID,
        validity_start=float(t[0]),validity_end=float(t[-1]),gripper=np.full(len(t),.044),
        provenance={'gripper_unit':'m_per_prismatic_joint','positive':'open','both_fingers_commanded':True})

def bind():
    from go1_gym.envs.config import RoboDuetRuntimeOptions,build_roboduet_config,cfg_to_dict
    from native_wbc_benchmark import actor_model_classes,safe_load_actor
    from modules.trajectory import quat_to_mat
    from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d,quat_to_matrix
    import pytorch3d.transforms as transforms
    previous=json.loads((ROOT.parent/'PI0265/evidence/SOURCE_IDENTITY.json').read_text())
    files=[]
    for row in previous['files']:
        digest=sha(row['path']);assert digest==row['sha256'],row['name']
        files.append(dict(name=row['name'],path=row['path'],sha256=digest,bytes=Path(row['path']).stat().st_size))
    paths={x['name']:x['path'] for x in files}
    cfg=build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs=1,robot='go2_x5',use_rot6d=True,dyna_gait=True,
        goal_reaching=True,traj_tracking=True,arm_action_mode='end_to_end',reach_table=True),debug=True)
    original=cfg_to_dict(cfg);configure(cfg);resolved=cfg_to_dict(cfg)
    for group in ('rewards',):
        for k,v in original[group].items():
            if 'terminal' in k:assert resolved[group][k]==v,k
    for k,v in original['wbc']['rewards'].items():
        if 'terminal' in k:assert resolved['wbc']['rewards'][k]==v,k
    assert cfg.control.control_type=='M' and abs(cfg.sim.dt*cfg.control.decimation-.02)<1e-8
    arm,dog=actor_model_classes();loads=[safe_load_actor('arm',Path(paths['arm_checkpoint']),arm(),torch),safe_load_actor('dog',Path(paths['dog_checkpoint']),dog(),torch)]
    matrix=Rotation.from_euler('xyz',[[.31,-.47,.83],[-.7,.4,.2]]).as_matrix();q=Rotation.from_matrix(matrix).as_quat();t=np.array([0.,.05])
    c=TrajectoryChunkV1.from_rot6d(chunk_id='cpu',source='PI0280_CPU',observation_timestamp=0.,point_timestamps=t,eef_position=np.zeros((2,3)),
        eef_rot6d=rotation_matrix_to_rot6d(matrix),source_frame=api.SOURCE_FRAME,tcp_semantic_id=api.TCP_SEMANTIC_ID,validity_start=0.,validity_end=.05)
    assert np.max(abs(quat_to_matrix(c.eef_orientation)-matrix))<1e-6
    assert np.max(abs(quat_to_mat(torch.tensor(q,dtype=torch.float32)).numpy()-matrix))<1e-6
    packed=transforms.matrix_to_rotation_6d(torch.tensor(matrix,dtype=torch.float32)).numpy()
    assert np.max(abs(packed-matrix[:,:,:2].reshape(2,6)))<1e-6
    urdf=ET.parse(paths['native_urdf']).getroot();joints={j.attrib['name']:ET.tostring(j,encoding='unicode') for j in urdf.findall('joint')}
    for name in ('x5_gripper_joint','x5_joint8'):
        j=next(j for j in urdf.findall('joint') if j.attrib['name']==name)
        assert j.attrib['type']=='prismatic' and float(j.find('limit').attrib['upper'])==.044
    # Bind all executable WBC configuration and referenced geometry, including collisions.
    extra=set(WBC.joinpath('go1_gym/envs/config').rglob('*.py'))
    extra.update((REF/'src/go2arx5_vla/contracts').glob('*.py'));extra.update((REF/'src/go2arx5_vla/safety').glob('*.py'))
    extra.update((WBC/'resources/robots/go2_x5_v3').rglob('*'))
    for p in sorted(extra):
        if p.is_file():files.append(dict(name=str(p.relative_to(WBC)) if WBC in p.parents else str(p),path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    write(ROOT/'design/WBC_CONFIG.json',dict(original=original,resolved=resolved))
    receipt=dict(status='DELIVERED_OLD_WBC_BOUND',files=files,actor_loads=loads,configuration_sha256=sha(ROOT/'design/WBC_CONFIG.json'),
        runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,isaacgym_file=isaacgym.__file__,native_dt=cfg.sim.dt*cfg.control.decimation,sim_dt=cfg.sim.dt,decimation=cfg.control.decimation),
        rotation_check='PASS_NONIDENTITY_CANONICAL_QUATERNION_AND_CONVENIENCE_COLUMN6D',native_shim='DELIVERED_COLUMN_PACKING_UNCHANGED',
        gripper_joints={k:joints[k] for k in ('x5_gripper_joint','x5_joint8','gripper_center_fixed')},
        fixtures=dict(seed=4242,duration_s=10,hold='measured reset TCP',vertical='5cm smooth world-Z translate-return',open_joint_target_m=.044),
        new_model_queries=0,actor_forwards=0,physics_steps=0,training_updates=0)
    write(ROOT/'evidence/WBC_BINDING.json',receipt);print(json.dumps(dict(status=receipt['status'],files=len(files))))

def fixture(name):
    binding=json.loads((ROOT/'evidence/WBC_BINDING.json').read_text())
    for row in binding['files']:assert sha(row['path'])==row['sha256'],row['name']
    paths={x['name']:x['path'] for x in binding['files']};kin=NativeArmKinematics.from_urdf(paths['native_urdf'])
    out=ROOT/'wbc'/name;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'START.json').exists(),'Adopt existing attempt; never repeat a valid fixture'
    write(out/'START.json',dict(pid=os.getpid(),time=time.time(),implementation=sha(__file__)))
    gate=TrajectorySafetyGateV1();records=[];frames=[];native=None;result=dict(status='ENGINEERING_INCOMPLETE',fixture=name)
    ledger=ROOT/'evidence/wbc_budget.jsonl';bounds={}
    for j in ET.parse(paths['native_urdf']).getroot().findall('joint'):
        limit=j.find('limit')
        if limit is not None and 'lower' in limit.attrib:bounds[j.attrib['name']]=[float(limit.attrib['lower']),float(limit.attrib['upper'])]
    counts={}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            k=json.loads(line)['kind'];counts[k]=counts.get(k,0)+1
    def budget(kind):
        cap={'physical_start':22,'control_tick':30000,'actor_forward':120000,'integration':250000,'gripper_dispatch':60000,'RGB_pair':30000}[kind]
        assert counts.get(kind,0)<cap,(kind,counts)
        counts[kind]=counts.get(kind,0)+1
        append(ledger,dict(kind=kind,fixture=name,wall=time.time()),sync=kind in ('physical_start','control_tick'))
    def fk(n):
        root=n.root_states[0].detach().cpu().numpy();q=[n.dof_pos[0,n.dof_names.index(j)].item() for j in kin.joint_names]
        pose=kin.tcp_pose(q);rotation=Rotation.from_quat(root[3:7]).as_matrix()
        return rotation@pose[:3,3]+root[:3],rotation@pose[:3,:3]
    def snapshot(n):
        p,r=fk(n);s=n.traj_batch.s_ref(n.traj_sim_time)
        q=n.dof_pos[0].detach().cpu().numpy();native_p,native_q=api._current_eef(n)
        return dict(time=float(n.gym.get_sim_time(n.sim)),TCP=p.tolist(),rotation=r.tolist(),native_TCP=native_p.tolist(),
            installed=n.traj_batch.p_at(s)[0].detach().cpu().tolist(),root=n.root_states[0].detach().cpu().tolist(),q=q.tolist(),
            hard_penetration={joint:max(0.,lo-q[i],q[i]-hi) for i,joint in enumerate(n.dof_names) if joint in bounds for lo,hi in [bounds[joint]]},
            FK_cache_error_m=float(np.linalg.norm(p-native_p)),telemetry=api._native_runtime_telemetry(n))
    old=api._Stage21NativeEnvFactory.make
    def factory(WBCEnv,initial,torch,gymapi,gymtorch):
        base=old(WBCEnv,initial,torch,gymapi,gymtorch)
        class Fixture(base):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                self.end_effector_state=self.end_effector_state.clone()
                self.pi0280_fingers=[self.dof_names.index(k) for k in ('x5_gripper_joint','x5_joint8')]
            def _load_trajectory_for(self,ids):
                if not hasattr(self,'pi0280_reference'):
                    p,r=fk(self)
                    proxy=SimpleNamespace(_shoulder_frame=self._shoulder_frame,_traj_anchor_offset=self._traj_anchor_offset,base_quat=self.base_quat,reach_table=self.reach_table,cfg=self.cfg,device=self.device,traj_batch=SimpleNamespace(gamma_p=torch.zeros_like(self.traj_batch.gamma_p)))
                    base._anchor_external_reference(proxy,ids);anchor=proxy.traj_batch.gamma_p[0,0].detach().cpu().numpy().copy()
                    t=np.arange(201)*.05;position=np.repeat(p[None],len(t),axis=0)
                    if name=='vertical':position[:,2]+=.025*(1-np.cos(2*np.pi*t/10))
                    q=np.repeat(Rotation.from_matrix(r).as_quat()[None],len(t),axis=0)
                    c=chunk(name,position-anchor,q,t);verdict=gate.evaluate(c,now=0.,achieved_position=p-anchor,achieved_orientation=q[0]);write(out/'construction_gate.json',verdict.as_dict())
                    if verdict.decision is GateDecision.REJECT:raise ValueError('GATE_REJECT_NO_INSTALL')
                    self._stage21_initial_chunk=c;self.pi0280_reference=(position,q,t,anchor)
                    np.savez_compressed(out/'reference.npz',position=position,quaternion=q,timestamp=t,anchor=anchor)
                return super()._load_trajectory_for(ids)
            def _compute_torques(self,actions):
                command=super()._compute_torques(actions)
                # Separate gripper actuator; leg/body/arm outputs and gains untouched.
                c=self.stage21_current_chunk;elapsed=float(self.traj_sim_time[0].item())
                i=min(len(c.gripper)-1,max(0,int(np.searchsorted(c.point_timestamps-c.point_timestamps[0],elapsed,side='right')-1)))
                opening=float(c.gripper[i]);assert 0<=opening<=.044
                for k in self.pi0280_fingers:command[:,k]=opening
                budget('gripper_dispatch');budget('integration')
                return command
            def step(self,*a,**kw):
                budget('control_tick');answer=super().step(*a,**kw);records.append(snapshot(self));return answer
        return Fixture
    api._Stage21NativeEnvFactory.make=staticmethod(factory)
    try:
        trajectory=dict(timestamp=np.arange(40)*.05,position=np.zeros((40,3)),quaternion=np.tile([0.,0.,0.,1.],(40,1)))
        args=SimpleNamespace(source_root=WBC,sim_device='cuda:0',seed=4242,mode='gt_receding',arm=Path(paths['arm_checkpoint']),dog=Path(paths['dog_checkpoint']))
        budget('physical_start');native,env,actors,loads,cfg=api._build_native(args,trajectory)
        origin=float(native.gym.get_sim_time(native.sim));result['loads']=loads;result['initialization_s']=origin
        event=native.pop_stage16_termination()
        if event or bool(native.reset_buf[0]):result.update(status='BLOCKED_EXECUTOR_NATIVE_INITIALIZATION_TERMINATION',event=event);return
        p,q,t,anchor=native.pi0280_reference;c=chunk(name,p-anchor,q,t+origin)
        current,r=fk(native);verdict=gate.evaluate(c,now=origin,achieved_position=current-anchor,achieved_orientation=Rotation.from_matrix(r).as_quat());write(out/'install_gate.json',verdict.as_dict())
        if verdict.decision is GateDecision.REJECT:result.update(status='VALID_GATE_REJECT');return
        write(out/'installation.json',native.initial_bind_reference(c));native._update_ee_task_space_error()
        class Actor:
            def __init__(self,value):self.value=value
            def __call__(self,x):budget('actor_forward');return self.value(x)
        actors=(Actor(actors[0]),Actor(actors[1]),actors[2],actors[3]);tick=0
        while float(native.gym.get_sim_time(native.sim))-origin<10-native.dt/2:
            if tick%5==0:
                budget('RGB_pair');body,wrist=api._capture_scene(native);frames.append((body,wrist,float(native.gym.get_sim_time(native.sim))))
            with torch.inference_mode():done,policy=api._policy_step(native,env,actors)
            append(out/'actor_outputs.jsonl',dict(tick=tick,outputs=policy));tick+=1;event=native.pop_stage16_termination()
            if done or event or bool(native.reset_buf[0]):result.update(status='VALID_NATIVE_TERMINATION',event=event);break
        else:result.update(status='WBC_FIXTURE_EXECUTED')
        result.update(controlled_ticks=tick,actual_elapsed_s=float(native.gym.get_sim_time(native.sim))-origin,
            gripper_actual_q=native.dof_pos[0,native.pi0280_fingers].detach().cpu().tolist(),task_success='NOT_AN_OBJECT_TASK')
    except Exception as exc:
        result['error']=repr(exc)
        if str(exc)=='GATE_REJECT_NO_INSTALL':result['status']='VALID_GATE_REJECT'
        else:raise
    finally:
        if ledger.exists():
            with ledger.open('a') as f:f.flush();os.fsync(f.fileno())
        write(out/'rows.json',records);write(out/'RESULT.json',result)
        if frames:np.savez_compressed(out/'frames.npz',body=np.array([f[0] for f in frames]),wrist=np.array([f[1] for f in frames]),time=np.array([f[2] for f in frames]))
        if native is not None:native.gym.destroy_sim(native.sim)
        print(json.dumps(result),flush=True)

if __name__=='__main__':
    phase=sys.argv[1]
    if phase=='bind':bind()
    else:assert phase in ('hold','vertical');fixture(phase)
