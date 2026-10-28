"""PI0281 native/one-repair fixture entry. Publication and owner approval required."""
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
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281')
OLD=ROOT.parent/'PI0280'
from budget import guard,reserve,actual
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

_journals={}
_journal_lines={}

def append(p,value,sync=True):
    # IO-only revision: payload/ordering unchanged; finalization fsyncs once.
    p=Path(p)
    if p not in _journals:
        _journals[p]=p.open('a',buffering=8<<20)
        _journal_lines[p]=0
    _journals[p].write(json.dumps(value)+'\n')
    _journal_lines[p]+=1

def flush_journals():
    receipt={}
    for p,f in _journals.items():
        try:f.flush();os.fsync(f.fileno())
        finally:f.close()
        h=hashlib.sha256();lines=0;size=0
        with p.open('rb') as source:
            for b in iter(lambda:source.read(8<<20),b''):
                h.update(b);lines+=b.count(b'\n');size+=len(b)
        assert lines==_journal_lines[p],('JOURNAL_INCOMPLETE',str(p),lines,_journal_lines[p])
        receipt[p.name]=dict(lines=lines,expected_lines=_journal_lines[p],bytes=size,sha256=h.hexdigest())
    return dict(status='PASS_BUFFERED_JOURNALS_LINES_SHA256',files=receipt)

def configure(cfg):
    # The old evaluation helper disables body hard stops. Preserve delivered values.
    keep=[{k:copy.deepcopy(v) for k,v in vars(g).items() if 'terminal' in k} for g in (cfg.rewards,cfg.wbc.rewards)]
    api._configure_stage21_original(cfg)
    for group,values in zip((cfg.rewards,cfg.wbc.rewards),keep):
        for k,v in values.items():setattr(group,k,v)

api._configure_stage21_original=api._configure_stage21
api._configure_stage21=configure

def chunk(name,p,q,t):
    return TrajectoryChunkV1(chunk_id=name,source='PI0281_NATIVE_FIXTURE',observation_timestamp=float(t[0]),
        point_timestamps=t,eef_position=p,eef_orientation=q,source_frame=api.SOURCE_FRAME,tcp_semantic_id=api.TCP_SEMANTIC_ID,
        validity_start=float(t[0]),validity_end=float(t[-1]),gripper=np.full(len(t),.044),
        provenance={'gripper_unit':'m_per_prismatic_joint','positive':'open','both_fingers_commanded':True})

def fixture(version,name,technical=False):
    guard()
    seal=json.loads((ROOT/'design/IMPLEMENTATION.json').read_text())
    assert seal['files'].get('wbc.py')==sha(__file__),'WBC_NOT_IN_FROZEN_PUBLICATION'
    if version=='repair':assert seal['files'].get('wbc_repair.py')==sha(ROOT/'code/wbc_repair.py'),'REPAIR_NOT_IN_FROZEN_PUBLICATION'
    assert version in ('native','repair') and name in ('hold','monotonic','return')
    if version=='repair':
        from wbc_repair import install
        install()
    # Admission checks do not load another neural model.
    import subprocess
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
    assert free>=12288, ('GPU_HEADROOM_BLOCKER',free)
    binding=json.loads((OLD/'evidence/WBC_BINDING.json').read_text())
    for row in binding['files']:assert sha(row['path'])==row['sha256'],row['name']
    paths={x['name']:x['path'] for x in binding['files']};kin=NativeArmKinematics.from_urdf(paths['native_urdf'])
    if technical:
        assert version=='repair' and (ROOT/'wbc/REPAIR_INSTALLATION_DEFECT.json').exists()
    uid=version+'_'+name+('-engineering-001' if technical else '')
    out=ROOT/'wbc'/uid;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'START.json').exists(),'Adopt existing attempt; never repeat a valid fixture'
    write(out/'START.json',dict(pid=os.getpid(),time=time.time(),implementation=sha(__file__)))
    gate=TrajectorySafetyGateV1();records=[];frames=[];native=None;result=dict(status='ENGINEERING_INCOMPLETE',fixture=name)
    ledger=out/'events.jsonl';bounds={}
    for j in ET.parse(paths['native_urdf']).getroot().findall('joint'):
        limit=j.find('limit')
        if limit is not None and 'lower' in limit.attrib:bounds[j.attrib['name']]=[float(limit.attrib['lower']),float(limit.attrib['upper'])]
    caps=dict(physical_starts=1,wbc_controls=501,wbc_actor_forwards=1000,integrations=2004,dispatches=2004,RGB_pairs=100)
    if technical:caps['technical_replacements']=1
    counts={k:0 for k in caps}
    reserve('WBC',uid,caps)
    def budget(kind):
        key={'physical_start':'physical_starts','control_tick':'wbc_controls','actor_forward':'wbc_actor_forwards',
             'integration':'integrations','gripper_dispatch':'dispatches','RGB_pair':'RGB_pairs'}[kind]
        assert counts[key]<caps[key],(key,counts)
        counts[key]+=1
        if technical and kind=='physical_start':counts['technical_replacements']=1
        append(ledger,dict(kind=kind,attempt=uid,wall=time.time()),sync=False)
    def fk(n):
        root=n.root_states[0].detach().cpu().numpy();q=[n.dof_pos[0,n.dof_names.index(j)].item() for j in kin.joint_names]
        pose=kin.tcp_pose(q);rotation=Rotation.from_quat(root[3:7]).as_matrix()
        return rotation@pose[:3,3]+root[:3],rotation@pose[:3,:3]
    def snapshot(n):
        p,r=fk(n);s=n.traj_batch.s_ref(n.traj_sim_time)
        q=n.dof_pos[0].detach().cpu().numpy();native_p,native_q=api._current_eef(n)
        return dict(time=float(n.gym.get_sim_time(n.sim)),TCP=p.tolist(),rotation=r.tolist(),native_TCP=native_p.tolist(),
            native_quaternion=native_q.tolist(), actual_link6_quaternion=n.rigid_body_state.view(n.num_envs,n.num_bodies,13)[0,n.ee_idx,3:7].detach().cpu().tolist(),
            installed=n.traj_batch.p_at(s)[0].detach().cpu().tolist(),installed_quaternion=n.traj_batch.quat_at(s)[0].detach().cpu().tolist(),root=n.root_states[0].detach().cpu().tolist(),q=q.tolist(),
            hard_penetration={joint:max(0.,lo-q[i],q[i]-hi) for i,joint in enumerate(n.dof_names) if joint in bounds for lo,hi in [bounds[joint]]},
            FK_cache_error_m=float(np.linalg.norm(p-native_p)),
            progress=float(n.traj_s[0]),s_ref=float(s[0]),timing_err=float(n.traj_timing_err[0]),
            base_quaternion=n.base_quat[0].detach().cpu().tolist(),heading_quaternion=n._heading_quat()[0].detach().cpu().tolist(),
            target_body_quaternion=n.arm_target_quat_body[0].detach().cpu().tolist(),
            orientation_error_proxy=n.ee_rot_err_axis_angle[0].detach().cpu().tolist(),
            telemetry=api._native_runtime_telemetry(n))
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
                    if name=='return':position[:,2]+=.025*(1-np.cos(2*np.pi*t/10))
                    if name=='monotonic':position[:,2]+=.025*(1-np.cos(np.pi*np.minimum(t,5)/5))
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
            def reset_idx(self,ids):
                if getattr(self,'pi0280_initialized',False) and len(ids):
                    # Keep the native stop and achieved state; never auto-reset a scientific trial.
                    event=getattr(self,'_stage16_pending_termination',None)
                    write(out/'NATIVE_STOP_BEFORE_AUTORESET.json',dict(event=event,state=snapshot(self),reset_ids=ids.detach().cpu().tolist()))
                    return
                return super().reset_idx(ids)
            def step(self,*a,**kw):
                budget('control_tick');answer=super().step(*a,**kw);records.append(snapshot(self));return answer
        return Fixture
    api._Stage21NativeEnvFactory.make=staticmethod(factory)
    try:
        trajectory=dict(timestamp=np.arange(40)*.05,position=np.zeros((40,3)),quaternion=np.tile([0.,0.,0.,1.],(40,1)))
        args=SimpleNamespace(source_root=WBC,sim_device='cuda:0',seed=4242,mode='gt_receding',arm=Path(paths['arm_checkpoint']),dog=Path(paths['dog_checkpoint']))
        budget('physical_start');native,env,actors,loads,cfg=api._build_native(args,trajectory)
        native.pi0280_initialized=True
        origin=float(native.gym.get_sim_time(native.sim));result['loads']=loads;result['initialization_s']=origin
        event=native.pop_stage16_termination()
        if event or bool(native.reset_buf[0]):result.update(status='BLOCKED_EXECUTOR_NATIVE_INITIALIZATION_TERMINATION',event=event);return
        p,q,t,anchor=native.pi0280_reference;c=chunk(name,p-anchor,q,t+origin)
        current,r=fk(native);verdict=gate.evaluate(c,now=origin,achieved_position=current-anchor,achieved_orientation=Rotation.from_matrix(r).as_quat());write(out/'install_gate.json',verdict.as_dict())
        if verdict.decision is GateDecision.REJECT:result.update(status='VALID_GATE_REJECT');return
        write(out/'installation.json',native.initial_bind_reference(c));native._update_ee_task_space_error()
        write(out/'observation_layout.json',dict(parts=__import__('go1_gym.envs.config.core',fromlist=['arm_obs_dim_parts']).arm_obs_dim_parts(cfg),history=cfg.arm.arm_num_obs_history,configuration=binding['configuration_sha256']))
        class Actor:
            def __init__(self,value,kind):self.value=value;self.kind=kind
            def __call__(self,x):
                budget('actor_forward')
                if self.kind=='arm':
                    sk,pk,qk,vk=native.traj_batch.sample_preview(native.traj_s,float(cfg.wbc.goal_reaching.trajectory.preview_horizon),int(cfg.wbc.goal_reaching.trajectory.preview_points))
                    append(out/'actor_inputs.jsonl',dict(tick=tick,obs_history=x[0].detach().cpu().tolist(),
                        state=snapshot(native),preview_s=sk[0].detach().cpu().tolist(),preview_position=pk[0].detach().cpu().tolist(),
                        preview_quaternion=qk[0].detach().cpu().tolist()),sync=False)
                return self.value(x)
        actors=(Actor(actors[0],'arm'),Actor(actors[1],'dog'),actors[2],actors[3]);tick=0
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
        write(out/'IO_COMPLETENESS.json',flush_journals())
        if version=='repair':
            from wbc_repair import runtime_receipt
            write(out/'REPAIR_RUNTIME_BINDING.json',runtime_receipt())
        result.update(version=version,counts=counts,allocation=caps)
        write(out/'rows.json',records);write(out/'RESULT.json',result)
        actual('WBC',uid,counts,result['status'])
        if frames:np.savez_compressed(out/'frames.npz',body=np.array([f[0] for f in frames]),wrist=np.array([f[1] for f in frames]),time=np.array([f[2] for f in frames]))
        if native is not None:native.gym.destroy_sim(native.sim)
        print(json.dumps(result),flush=True)

if __name__=='__main__':
    assert len(sys.argv) in (3,4)
    technical=len(sys.argv)==4
    if technical:assert sys.argv[3]=='--technical-replacement'
    fixture(sys.argv[1],sys.argv[2],technical)
