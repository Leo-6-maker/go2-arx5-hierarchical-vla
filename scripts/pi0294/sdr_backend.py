"""Isolated fixed MPlib search. Every library call has a durable named slot."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import mplib
from mplib.collision_detection import fcl
from .mp_backend import transform,AuditedKinematics,collision_rows
from .sdr_runtime import reserve_internal,finish_internal
from scripts.pi0292.durable import write

def planner_for(request,stage,closed=False):
    binding=json.loads(Path(request['binding']).read_text());folder=Path(request['binding']).parent
    for name,key in [('robot.urdf','urdf_sha256'),('robot.srdf','srdf_sha256')]:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=binding[key]:raise ValueError('SDR_ASSET_CHANGED')
    planner=mplib.Planner(folder/'robot.urdf','x5_ee',srdf=folder/'robot.srdf',user_joint_names=binding['joint_names'],
        joint_vel_limits=np.full(6,.7),joint_acc_limits=np.ones(6))
    q=np.asarray(request['joint_qpos']).copy()
    if q.shape!=(len(binding['joint_names']),) or not np.isfinite(q).all():raise ValueError('SDR_FULL_FINITE_ACTUAL_Q_REQUIRED')
    if np.any(q<planner.joint_limits[:,0]) or np.any(q>planner.joint_limits[:,1]):raise ValueError('SDR_ACTUAL_OUTSIDE_LIMIT_NO_CLIP')
    if closed:
        for i,name in enumerate(binding['joint_names']):
            if name in ('task_left_slide','task_right_slide'):q[i]=.002
    planner.set_base_pose(transform(request['root_pose']));planner.robot.set_qpos(q,True)
    for item in request['world_objects']:
        shape=fcl.Box(item['size']) if item['kind']=='box' else fcl.Halfspace(np.array([0.,0.,1.]),0.)
        planner.planning_world.add_object(item['name'],fcl.CollisionObject(shape,transform(item['pose'])))
    pairs=[p for p in request['allowed_world_pairs'] if not ('task_cube_geom' in p and any(x in request['pad_links'] for x in p))]
    if stage not in ('approach','align'):pairs += [[p,'task_cube_geom'] for p in request['pad_links']]
    for a,b in pairs:planner.acm.set_entry(a,b,True)
    if request.get('attached_cube'):
        planner.planning_world.attach_object('task_cube_geom',planner.robot.get_name(),planner.move_group_link_id,
            transform(request['attached_cube']),request['pad_links'])
    return planner,q

def inverse(request,folder):
    rows=[];root=request['run_root'];key=request['bundle_key']
    for slot,item in enumerate(request['goals']):
        path=folder/f'IK-{slot:02d}.private.json'
        if path.exists():
            row=json.loads(path.read_text());assert row['input_sha256']==request['input_sha256'];rows.append(row);continue
        try:reserve_internal(root,key,'IK',slot)
        except PermissionError as e:
            if str(e)=='SDR_FIRST_INTERNAL_SLOT_PRESERVED':
                rows.append(dict(slot=slot,status='INCOMPLETE_INTERNAL_SLOT_NO_REDRAW',legal_goals=[]));continue
            raise
        seed=request['streams']['IK'][slot];np.random.seed(seed);mplib.set_global_seed(seed)
        planner,q=planner_for(request,item['stage'],closed=item.get('closed',False))
        proxy=AuditedKinematics(planner.pinocchio_model);planner.pinocchio_model=proxy
        mask=np.ones(len(q),dtype=bool);mask[planner.move_group_joint_indices]=False
        target=transform(request['root_pose']).inv()*transform(item['pose'])
        started=time.monotonic()
        status,goals=planner.IK(target,q.copy(),mask,n_init_qpos=16,threshold=1e-3,return_closest=False)
        legal=[]
        if status=='Success':
            for index,goal in enumerate(goals):
                goal=np.asarray(goal)
                if np.all(goal>=planner.joint_limits[:,0]) and np.all(goal<=planner.joint_limits[:,1]):
                    planner.robot.set_qpos(goal,True)
                    if not planner.planning_world.check_collision():legal.append(dict(index=index,q=goal.tolist()))
        row=dict(slot=slot,status=status,seed=seed,input_sha256=request['input_sha256'],package=item['package'],
            endpoint=item['endpoint'],internal_IK=proxy.calls,legal_goals=legal,service_s=time.monotonic()-started)
        write(path,row);finish_internal(root,key,'IK',slot,dict(actual_CLIK_initializations=len(proxy.calls),legal_goals=len(legal)))
        rows.append(row)
    base=np.asarray(request['joint_qpos']);planner,_=planner_for(request,request['stage']);indices=planner.move_group_joint_indices
    pairs=[]
    for package in sorted({g['package'] for g in request['goals']}):
        goal_slots=[i for i,g in enumerate(request['goals']) if g['package']==package]
        if len(goal_slots)==2:
            for a in rows[goal_slots[0]]['legal_goals']:
                for c in rows[goal_slots[1]]['legal_goals']:
                    qa,qc=np.asarray(a['q']),np.asarray(c['q'])
                    pairs.append(dict(package=package,approach_slot=goal_slots[0],contact_slot=goal_slots[1],
                        a_index=a['index'],c_index=c['index'],qa=a['q'],qc=c['q'],
                        cost=float(np.linalg.norm(qa[indices]-base[indices])+np.linalg.norm(qc[indices]-qa[indices]))))
        else:
            for c in rows[goal_slots[0]]['legal_goals']:
                qc=np.asarray(c['q']);pairs.append(dict(package=package,contact_slot=goal_slots[0],a_index=-1,c_index=c['index'],
                    qa=None,qc=c['q'],cost=float(np.linalg.norm(qc[indices]-base[indices]))))
    # Each package contributes its best pair, then the first two packages.
    best=[]
    for package in sorted({p['package'] for p in pairs}):
        best.append(min((p for p in pairs if p['package']==package),key=lambda p:(p['cost'],p['a_index'],p['c_index'])))
    best=sorted(best,key=lambda p:(p['cost'],p['package'],p['a_index'],p['c_index']))[:2]
    planner.pinocchio_model.compute_forward_kinematics(base)
    fk=(transform(request['root_pose'])*planner.pinocchio_model.get_link_pose(planner.move_group_link_id)).to_transformation_matrix()
    return dict(status='IK_PAIRS_READY' if best else 'NO_LEGAL_REGISTERED_PAIR',input_sha256=request['input_sha256'],
        initial_fk_world=np.asarray(fk).tolist(),goals=rows,pairs=best,physics_steps=0,joint_command_emitted=False)

def route(request,folder,index):
    ik=json.loads((folder/'IK_RESULT.private.json').read_text());pair_index=index//2;stream=index%2
    if pair_index>=len(ik['pairs']):return dict(status='UNUSED_NO_PAIR',input_sha256=request['input_sha256'])
    pair=ik['pairs'][pair_index];links=[]
    goal_slots=([pair['approach_slot'],pair['contact_slot']] if pair['qa'] is not None else [pair['contact_slot']])
    for link,goal_slot in enumerate(goal_slots):
        slot=index*2+link;path=folder/f'RRT-{slot:02d}.private.json'
        if path.exists():
            row=json.loads(path.read_text());assert row['input_sha256']==request['input_sha256']
        else:
            try:reserve_internal(request['run_root'],request['bundle_key'],'RRT',slot)
            except PermissionError as e:
                if str(e)=='SDR_FIRST_INTERNAL_SLOT_PRESERVED':return dict(status='INCOMPLETE_INTERNAL_SLOT_NO_REDRAW',input_sha256=request['input_sha256'])
                raise
            goal=request['goals'][goal_slot];planner,q=planner_for(request,goal['stage'],closed=goal.get('closed',False))
            if link==1:
                arm=planner.move_group_joint_indices;q[arm]=np.asarray(pair['qa'])[arm]
            endpoint=np.asarray(pair['qa'] if link==0 and pair['qa'] is not None else pair['qc'])
            seed=request['streams']['RRT'][slot];np.random.seed(seed);mplib.set_global_seed(seed)
            started=time.monotonic()
            planned=planner.plan_qpos([endpoint],q.copy(),time_step=.05,rrt_range=.1,planning_time=1.,fix_joint_limits=False,simplify=False)
            row=dict(status=planned['status'],seed=seed,input_sha256=request['input_sha256'],route=index,link=link,service_s=time.monotonic()-started)
            if planned['status']=='Success':
                position=np.asarray(planned['position'])
                if len(position)>1024:row['status']='ROUTE_OUTPUT_ROW_CAP'
                else:row.update(time=np.asarray(planned['time']).tolist(),position=position.tolist(),duration=float(planned['duration']))
            write(path,row);finish_internal(request['run_root'],request['bundle_key'],'RRT',slot,dict(status=row['status']))
        links.append(row)
        if row['status']!='Success':return dict(status=row['status'],links=links,input_sha256=request['input_sha256'],package=pair['package'])
    # The leading certified window comes from the first link. The second is a
    # required existence/geometry certificate, not an expert future installed now.
    return dict(links[0],status='Success',links=links,package=pair['package'],route=index,physics_steps=0,joint_command_emitted=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--action',choices=['IK','ROUTE'],required=True);p.add_argument('--route',type=int,default=0);a=p.parse_args()
    path=Path(a.request);request=json.loads(path.read_text());request['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();folder=path.parent
    from .task_capture_runtime import validate_entry
    from .sdr_runtime import require_adoption,FOLDER
    root=Path(request['run_root']);manifest,_=validate_entry(root,read_only=True);require_adoption(root,manifest)
    if not path.resolve().is_relative_to((root/FOLDER/'requests').resolve()) or request['implementation_sha']!=manifest['code']['vla_commit']:
        raise PermissionError('SDR_REGISTERED_SOURCE_AND_REQUEST_ROOT_REQUIRED')
    target=folder/('IK_RESULT.private.json' if a.action=='IK' else f'ROUTE-{a.route:02d}.private.json')
    if target.exists():
        assert json.loads(target.read_text())['input_sha256']==request['input_sha256'];return
    result=inverse(request,folder) if a.action=='IK' else route(request,folder,a.route)
    write(target,result);print(json.dumps(dict(status=result['status'],action=a.action,route=a.route)))

if __name__=='__main__':main()
