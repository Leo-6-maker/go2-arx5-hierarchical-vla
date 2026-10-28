"""Actual-state RPC, Cartesian resampling and native-clone feasibility checks."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .mp_runtime import FOLDER,reserve_request
from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix,quat_slerp,geodesic_angle


def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)


def root_pose(q):
    p=np.eye(4);p[:3,3]=q[:3];p[:3,:3]=quat_to_matrix(np.r_[q[4:7],q[3]]);return p


def check_coverage(model,binding):
    if binding['leg_joint_count']!=12 or binding['native_robot_bodies']!=model.nbody-2:
        raise ValueError('FULL_NATIVE_BODY_LEG_COLLISION_COVERAGE_REQUIRED')
    expected=set(np.flatnonzero((model.geom_contype|model.geom_conaffinity)&(model.geom_bodyid!=0)&(model.geom_bodyid!=model.nbody-1)))
    if {r['native_geom'] for r in binding['geometry_map']}!=expected:
        raise ValueError('NATIVE_COLLISION_GEOMETRY_MISSING')


def contact_permitted(names,distance,stage):
    from .task_contacts import allowed
    permitted=allowed(*names)
    task_pad='task_cube_geom' in names and any(n in ('task_left_pad','task_right_pad') for n in names)
    if task_pad and stage in ('approach','align'):return False
    return bool(permitted and not (task_pad and distance<-.004))


class Client:
    def __init__(self,root,engine):
        import mujoco
        self.root=Path(root);self.series=self.root/FOLDER;self.engine=engine;self.model=engine.model
        self.assets=self.series/'planning-assets';self.binding=read(self.assets/'ASSET_BINDING.private.json')
        check_coverage(self.model,self.binding)
        self.arm_j=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'x5_joint'+str(i)) for i in range(1,7)])
        self.arm_q=self.model.jnt_qposadr[self.arm_j];self.arm_v=self.model.jnt_dofadr[self.arm_j]
        self.native_data=mujoco.MjData(self.model);self.site=engine.tcp
        self.pads=[next(x['link'] for x in self.binding['geometry_map'] if x['native_geom']==g) for g in engine.pad]

    def payload(self,state,target,stage,seed,*,attached=False,inspect=False,hold=False):
        import mujoco
        m,d=self.model,self.engine.data;objects=[];pairs=[]
        if np.asarray(target).shape!=(4,4) or not np.isfinite(target).all():raise ValueError('FINITE_TARGET_REQUIRED')
        if attached and not state['native_grasp']:raise ValueError('PHYSICAL_GRASP_REQUIRED_FOR_ATTACHED_COLLISION')
        for g in range(m.ngeom):
            b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))
            if (m.geom_bodyid[g]!=0 and b!='task_cube') or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g);p=np.eye(4);p[:3,3]=d.geom_xpos[g];p[:3,:3]=d.geom_xmat[g].reshape(3,3)
            typ=int(m.geom_type[g])
            if typ==int(mujoco.mjtGeom.mjGEOM_PLANE):kind='halfspace';size=[]
            elif typ==int(mujoco.mjtGeom.mjGEOM_BOX):kind='box';size=(2*m.geom_size[g]).tolist()
            else:raise ValueError('WORLD_COLLISION_SHAPE_NOT_DROPPED')
            objects.append(dict(name=name,kind=kind,size=size,pose=p.tolist()))
        for g in self.binding['geometry_map']:
            if g['native_name'] in ('FL','FR','RL','RR'):pairs.append([g['link'],'floor'])
        if stage not in ('approach','align'):
            pairs += [[p,'task_cube_geom'] for p in self.pads]
        pairs.append(['task_cube_geom','task_table'])
        packet_hash=hashlib.sha256(np.asarray(state['qpos'],dtype='<f8').tobytes()+np.asarray(state['qvel'],dtype='<f8').tobytes()+np.asarray(state['ctrl'],dtype='<f8').tobytes()).hexdigest()
        return dict(schema='MANISKILL_CPU_PLAN_REQUEST_V1',binding=str(self.assets/'ASSET_BINDING.private.json'),
            seed=int(seed),joint_qpos=np.asarray(state['qpos'])[self.binding['joint_qpos_addresses']].tolist(),
            root_pose=root_pose(state['qpos']).tolist(),target=np.asarray(target).tolist(),world_objects=objects,
            allowed_world_pairs=pairs,pad_links=self.pads,inspect_only=inspect,hold_only=hold,stage=stage,
            actual_packet_arrays_sha256=packet_hash,attached_cube=(np.linalg.inv(state['tcp'])@state['object_pose']).tolist() if attached else None)

    def call(self,state,target,stage,condition,index,*,cpu=False,inspect=False,attached=False,hold=False):
        from scripts.pi0292.durable import write
        folder=self.series/'requests'/condition;folder.mkdir(parents=True,exist_ok=True)
        stem=('inspect-' if inspect else '')+f'{index:04d}';req=folder/(stem+'-input.private.json');out=folder/(stem+'-output.private.json')
        cached=req.exists() and out.exists()
        if cached:
            seed=read(req)['seed']
        else:seed=0 if inspect else reserve_request(self.root,condition,index,cpu=cpu)
        payload=self.payload(state,target,stage,seed,attached=attached,inspect=inspect,hold=hold)
        if cached:
            if payload!=read(req):raise ValueError('CACHED_PLANNING_INPUT_DIFFERS_NO_REQUERY')
        else:write(req,payload)
        capsule=Path(__file__).resolve().parents[2]
        env=dict(os.environ,PYTHONPATH=str(capsule)+':'+str(capsule/'src'),LD_LIBRARY_PATH='',LD_PRELOAD='',CUDA_VISIBLE_DEVICES='',
            MPLCONFIGDIR=str(self.series/'tmp/mpl-config'),XDG_CACHE_HOME=str(self.series/'cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.series.joinpath('tmp').mkdir(exist_ok=True)
        if not cached:
            with (folder/(stem+'-backend.private.log')).open('x') as log:
                subprocess.run([str(self.series/'planner-env/bin/python'),'-m','scripts.pi0294.mp_backend',
                    '--request',str(req),'--output',str(out)],check=True,timeout=30,stdout=log,stderr=subprocess.STDOUT,env=env)
        result=read(out)
        if result['status']=='ENGINEERING_BACKEND_EXCEPTION':raise RuntimeError(result['error']+'\n'+result['trace'])
        if result['input_sha256']!=sha(req):raise ValueError('CROSS_PROCESS_INPUT_IDENTITY')
        fk=np.asarray(result.get('initial_fk_world',state['tcp']))
        difference=[float(np.linalg.norm(fk[:3,3]-state['tcp'][:3,3])),float(geodesic_angle(fk[:3,:3],state['tcp'][:3,:3]))]
        if difference[0]>1e-5 or difference[1]>1e-4:raise ValueError('NATIVE_AND_PLANNING_FK_MISMATCH:'+str(difference))
        result.update(input_path=str(req),output_path=str(out),request_sha256=sha(req),response_sha256=sha(out),FK_error=difference,
            reused_saved_computation=cached,new_IK_or_RRT_on_reuse=False)
        return result

    def fk_path(self,state,arm_path):
        import mujoco
        m,d=self.model,self.native_data;d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.ctrl[:]=state['ctrl']
        poses=[]
        for q in arm_path:
            d.qpos[self.arm_q]=q;mujoco.mj_forward(m,d);p=np.eye(4);p[:3,3]=d.site_xpos[self.site];p[:3,:3]=d.site_xmat[self.site].reshape(3,3);poses.append(p)
        return np.asarray(poses)

    def cartesian(self,state,response,*,speed=.06,attached=False,stage='approach'):
        if response['status']!='Success':raise PlannerFailure(response['status'],response)
        t=np.asarray(response['time']);q=np.asarray(response['position']);raw=self.fk_path(state,q)
        scale=1.
        for i,dt in enumerate(np.diff(t)):
            if dt<=0:raise ValueError('PLANNED_TIME_GRID')
            scale=max(scale,float(np.linalg.norm(raw[i+1,:3,3]-raw[i,:3,3])/(dt*speed)),float(geodesic_angle(raw[i,:3,:3],raw[i+1,:3,:3])/(dt*.3)))
        for attempt in range(3):
            sample=np.arange(40)*.05;at=t*scale
            qp=np.stack([np.interp(sample,at,q[:,j]) for j in range(6)],axis=1);poses=self.fk_path(state,qp)
            v=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)/.05
            w=np.array([geodesic_angle(a,b) for a,b in zip(poses[:-1,:3,:3],poses[1:,:3,:3])])/.05
            ratio=max(float(v.max()/speed),float(w.max()/.3))
            if ratio<=1+1e-10:break
            scale*=ratio*(1+1e-6)
        else:raise PlannerFailure('RATE_CERTIFICATION_FAILED',response)
        dense=self.dense_check(state,poses,qp,stage=stage,attached=attached)
        return poses,dict(duration_s=float(t[-1]*scale),uniform_time_scale=scale,max_linear_speed=float(v.max()),max_angular_speed=float(w.max()),dense=dense,
            input_sha256=response['request_sha256'],response_sha256=response['response_sha256'],FK_error=response['FK_error'],IK_calls=response.get('IK_calls',0),RRT_calls=response.get('RRT_calls',0))

    def dense_check(self,state,poses,arm_samples,*,stage,attached):
        """Deterministic interpolation feasibility, never a second target planner.

        Each of the <=1001 declared window checks uses continuation from the
        existing planned path; no random restart or alternate goal is generated.
        """
        import mujoco
        from .task_contacts import allowed
        m,d=self.model,self.native_data;d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.ctrl[:]=state['ctrl']
        poses=np.concatenate([poses,poses[-1:]]);path=np.vstack([arm_samples,arm_samples[-1]])
        jacp=np.zeros((3,m.nv));jacr=np.zeros_like(jacp);relative=np.linalg.inv(state['tcp'])@state['object_pose']
        cubej=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'task_cube_free');cqa=int(m.jnt_qposadr[cubej])
        max_pos=max_rot=0.;total=0
        for time_s in np.arange(1001)*.002:
            i=min(int(time_s/.05),39);fraction=(time_s-i*.05)/.05
            target=np.eye(4);target[:3,3]=(1-fraction)*poses[i,:3,3]+fraction*poses[i+1,:3,3]
            target[:3,:3]=quat_to_matrix(quat_slerp(matrix_to_quat(poses[i,:3,:3]),matrix_to_quat(poses[i+1,:3,:3]),fraction))
            d.qpos[self.arm_q]=(1-fraction)*path[i]+fraction*path[i+1]
            for _ in range(5):
                mujoco.mj_forward(m,d);current=d.site_xmat[self.site].reshape(3,3)
                ep=target[:3,3]-d.site_xpos[self.site];delta=matrix_to_quat(target[:3,:3]@current.T)
                if delta[3]<0:delta=-delta
                length=np.linalg.norm(delta[:3]);er=delta[:3]*(2*np.arctan2(length,delta[3])/max(length,1e-15))
                if np.linalg.norm(ep)<=1e-5 and np.linalg.norm(er)<=1e-4:break
                mujoco.mj_jacSite(m,d,jacp,jacr,self.site);j=np.vstack([jacp[:,self.arm_v],jacr[:,self.arm_v]])
                step=np.linalg.lstsq(j,np.r_[ep,er],rcond=1e-10)[0]
                if not np.isfinite(step).all():raise PlannerFailure('DENSE_NONFINITE_IK',{})
                d.qpos[self.arm_q]+=step
            else:raise PlannerFailure('CARTESIAN_INTERPOLATION_NOT_NATIVE_IK_FEASIBLE',{})
            if np.any(d.qpos[self.arm_q]<m.jnt_range[self.arm_j,0]) or np.any(d.qpos[self.arm_q]>m.jnt_range[self.arm_j,1]):
                raise PlannerFailure('DENSE_NATIVE_JOINT_LIMIT',{})
            max_pos=max(max_pos,float(np.linalg.norm(ep)));max_rot=max(max_rot,float(np.linalg.norm(er)))
            if attached:
                obj=target@relative;d.qpos[cqa:cqa+3]=obj[:3,3];quat=matrix_to_quat(obj[:3,:3]);d.qpos[cqa+3:cqa+7]=np.r_[quat[3],quat[:3]]
            mujoco.mj_forward(m,d)
            for c in d.contact:
                names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (c.geom1,c.geom2)]
                if not contact_permitted(names,c.dist,stage):raise PlannerFailure('DENSE_NATIVE_COLLISION:'+str(names),{})
            total+=1
        return dict(points=total,max_native_IK_position_residual_m=max_pos,max_native_IK_rotation_residual_rad=max_rot,
            physics_steps=0,actual_state_modified=False,scope='CURRENT_ROOT_LEGS_SNAPSHOT_NOT_FUTURE_DYNAMICS')


class PlannerFailure(RuntimeError):
    def __init__(self,reason,evidence):super().__init__(reason);self.evidence=evidence
