"""MPlib 0.2.1 CPU process, isolated from native NumPy/ROS. Never executes physics."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import mplib
from mplib import Pose
from mplib.collision_detection import fcl


class AuditedKinematics:
    def __init__(self,wrapped):self.wrapped=wrapped;self.calls=[]
    def __getattr__(self,key):return getattr(self.wrapped,key)
    def compute_IK_CLIK(self,*args,**kw):
        result=self.wrapped.compute_IK_CLIK(*args,**kw)
        self.calls.append(dict(start=np.asarray(args[2]).tolist(),solution=np.asarray(result[0]).tolist(),
            success=bool(result[1]),residual=np.asarray(result[2]).tolist()))
        return result


def transform(value):
    p=np.asarray(value,dtype=float)
    if p.shape!=(4,4) or not np.isfinite(p).all():raise ValueError('FINITE_POSE')
    from transforms3d.quaternions import mat2quat
    return Pose(p[:3,3],mat2quat(p[:3,:3]))


def collision_rows(rows):
    return [dict(a=r.link_name1,b=r.link_name2,object1=r.object_name1,object2=r.object_name2) for r in rows]


def solve(request):
    started=time.monotonic();binding=json.loads(Path(request['binding']).read_text())
    folder=Path(request['binding']).parent
    for name,key in (('robot.urdf','urdf_sha256'),('robot.srdf','srdf_sha256')):
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=binding[key]:raise ValueError('PLANNING_ASSET_CHANGED')
    seed=int(request['seed']);np.random.seed(seed);mplib.set_global_seed(seed)
    planner=mplib.Planner(folder/'robot.urdf','x5_ee',srdf=folder/'robot.srdf',
        user_joint_names=binding['joint_names'],joint_vel_limits=np.full(6,.7),joint_acc_limits=np.ones(6))
    q=np.asarray(request['joint_qpos'],dtype=float)
    if q.shape!=(len(binding['joint_names']),) or not np.isfinite(q).all():raise ValueError('FULL_ACTUAL_JOINTS')
    if np.any(q<planner.joint_limits[:,0]) or np.any(q>planner.joint_limits[:,1]):raise ValueError('ACTUAL_OUTSIDE_LIMIT_NO_CLIP')
    original=q.copy();root=transform(request['root_pose']);planner.set_base_pose(root);planner.robot.set_qpos(q,True)
    for item in request['world_objects']:
        shape={'box':lambda:fcl.Box(item['size']),'halfspace':lambda:fcl.Halfspace(np.array([0.,0.,1.]),0.)}[item['kind']]()
        planner.planning_world.add_object(item['name'],fcl.CollisionObject(shape,transform(item['pose'])))
    for a,b in request['allowed_world_pairs']:planner.acm.set_entry(a,b,True)
    if request.get('attached_cube'):
        planner.planning_world.attach_object('task_cube_geom',planner.robot.get_name(),planner.move_group_link_id,
            transform(request['attached_cube']),request['pad_links'])
    start_collisions=collision_rows(planner.planning_world.check_collision())
    planner.pinocchio_model.compute_forward_kinematics(q)
    fk=(root*planner.pinocchio_model.get_link_pose(planner.move_group_link_id)).to_transformation_matrix()
    result=dict(schema='MANISKILL_CPU_PLAN_RESPONSE_V1',input_sha256=request['input_sha256'],seed=seed,
        actual_unchanged=bool(np.array_equal(q,original)),initial_fk_world=np.asarray(fk).tolist(),
        start_collisions=start_collisions,joint_names=planner.user_joint_names,move_group_joint_indices=planner.move_group_joint_indices,
        internal_IK=[],physics_steps=0,joint_command_emitted=False)
    if request.get('inspect_only'):
        result.update(status='FK_AND_COLLISION_INSPECT_ONLY',service_s=time.monotonic()-started);return result
    if start_collisions:
        result.update(status='INVALID_ACTUAL_START_COLLISION',service_s=time.monotonic()-started);return result
    if request.get('hold_only'):
        result.update(status='Success',time=[0.,2.],position=[q[planner.move_group_joint_indices].tolist()]*2,
            IK_calls=0,RRT_calls=0,service_s=time.monotonic()-started);return result
    proxy=AuditedKinematics(planner.pinocchio_model);planner.pinocchio_model=proxy
    mask=np.ones(len(q),dtype=bool);mask[planner.move_group_joint_indices]=False
    target=root.inv()*transform(request['target'])
    status,goals=planner.IK(target,q.copy(),mask,n_init_qpos=8,threshold=1e-3,return_closest=False)
    result.update(internal_IK=proxy.calls,IK_calls=1,RRT_calls=0)
    if status!='Success':
        result.update(status=status,service_s=time.monotonic()-started);return result
    candidates=[(float(np.linalg.norm(np.asarray(g)[planner.move_group_joint_indices]-q[planner.move_group_joint_indices])),i,np.asarray(g)) for i,g in enumerate(goals)]
    candidates=[x for x in candidates if np.all(x[2]>=planner.joint_limits[:,0]) and np.all(x[2]<=planner.joint_limits[:,1])]
    if not candidates:
        result.update(status='IK_CANDIDATES_OUTSIDE_NATIVE_LIMITS',service_s=time.monotonic()-started);return result
    _,chosen,goal=min(candidates,key=lambda x:(x[0],x[1]));result['chosen_IK_index']=chosen
    planned=planner.plan_qpos([goal],q.copy(),time_step=.05,rrt_range=.1,planning_time=1.,fix_joint_limits=False,simplify=False)
    result.update(RRT_calls=1,status=planned['status'],service_s=time.monotonic()-started)
    if planned['status']=='Success':
        result.update(time=np.asarray(planned['time']).tolist(),position=np.asarray(planned['position']).tolist(),duration=float(planned['duration']))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    request=json.loads(Path(a.request).read_text());raw=Path(a.request).read_bytes()
    request['input_sha256']=hashlib.sha256(raw).hexdigest()
    try:result=solve(request)
    except Exception as exc:
        import traceback
        result=dict(status='ENGINEERING_BACKEND_EXCEPTION',error=type(exc).__name__+': '+str(exc),trace=traceback.format_exc(),input_sha256=request['input_sha256'],physics_steps=0)
    out=Path(a.output)
    with out.open('x') as f:json.dump(result,f,allow_nan=False)
    print('PLANNER_RESULT='+result['status'])


if __name__=='__main__':main()
