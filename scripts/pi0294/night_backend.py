"""Fixed MPlib calls, old-goal references and first-source ordered routes."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import mplib
from .sdr_backend import planner_for
from .mp_backend import transform,AuditedKinematics
from .night_runtime import reserve_internal,finish_internal,require_adoption,FOLDER,goal_consumer as consumer
from scripts.pi0292.durable import write

def inverse(request,folder):
    rows=[];root=Path(request['run_root']);key=request['bundle_key'];cpu=request['condition_kind']=='SAVED_INPUT_REGISTERED_SCENE_PREVIEW'
    for slot,goal in enumerate(request['goals']):
        file=folder/f'IK-{slot:02d}.private.json'
        if file.exists():
            row=json.loads(file.read_text());assert row['input_sha256']==request['input_sha256'];rows.append(row);continue
        if goal.get('reuse_v2_slot') is not None:
            old=root/'source-demo-retime-join-v2/requests/SDR2_CPU_T2/0000';old_input=json.loads((old/'INPUT.private.json').read_text());old_goal=old_input['goals'][goal['reuse_v2_slot']]
            if consumer(request,goal)!=consumer(old_input,old_goal):raise ValueError('SDR3_REUSED_GOAL_CONSUMER_NOT_EQUIVALENT')
            original=old/f"IK-{goal['reuse_v2_slot']:02d}.private.json";digest=hashlib.sha256(original.read_bytes()).hexdigest()
            if digest!=goal['old_response_sha256']:raise ValueError('SDR3_REUSED_GOAL_RESPONSE_CHANGED')
            previous=json.loads(original.read_text());assert previous['input_sha256']==hashlib.sha256((old/'INPUT.private.json').read_bytes()).hexdigest()
            row=dict(slot=slot,status=previous['status'],legal_goals=previous['legal_goals'],new_CLIK_initializations=0,
                reused_original_input_sha256=previous['input_sha256'],reused_original_response_sha256=digest,
                consumer_equivalence=True,input_sha256=request['input_sha256'],endpoint=goal['endpoint'],package=goal['package'])
        else:
            reserve_internal(root,key,'IK',slot);seed=goal['IK_seed'];np.random.seed(seed);mplib.set_global_seed(seed)
            planner,q=planner_for(request,goal['stage'],closed=goal.get('closed',False));proxy=AuditedKinematics(planner.pinocchio_model);planner.pinocchio_model=proxy
            mask=np.ones(len(q),dtype=bool);mask[planner.move_group_joint_indices]=False;target=transform(request['root_pose']).inv()*transform(goal['pose']);started=time.monotonic()
            status,goals=planner.IK(target,q.copy(),mask,n_init_qpos=16,threshold=1e-3,return_closest=False);legal=[]
            if status=='Success':
                for i,value in enumerate(goals):
                    value=np.asarray(value)
                    if np.all(value>=planner.joint_limits[:,0]) and np.all(value<=planner.joint_limits[:,1]):
                        planner.robot.set_qpos(value,True)
                        if not planner.planning_world.check_collision():legal.append(dict(index=i,q=value.tolist()))
            row=dict(slot=slot,status=status,legal_goals=legal,internal_IK=proxy.calls,seed=seed,input_sha256=request['input_sha256'],
                endpoint=goal['endpoint'],package=goal['package'],service_s=time.monotonic()-started)
            finish_internal(root,key,'IK',slot,dict(actual_CLIK_initializations=len(proxy.calls),legal_goals=len(legal)))
        write(file,row);rows.append(row)
    planner,base=planner_for(request,request['stage']);indices=planner.move_group_joint_indices;pairs=[]
    for package in sorted({g['package'] for g in request['goals']}):
        slots=[i for i,g in enumerate(request['goals']) if g['package']==package];choices=[]
        if len(slots)==2:
            for a in rows[slots[0]]['legal_goals']:
                for b in rows[slots[1]]['legal_goals']:
                    qa,qc=np.asarray(a['q']),np.asarray(b['q']);choices.append(dict(package=package,goal_slots=slots,qa=a['q'],qc=b['q'],
                        a_index=a['index'],c_index=b['index'],cost=float(np.linalg.norm(qa[indices]-base[indices])+np.linalg.norm(qc[indices]-qa[indices]))))
        else:
            for b in rows[slots[0]]['legal_goals']:
                qc=np.asarray(b['q']);choices.append(dict(package=package,goal_slots=slots,qa=None,qc=b['q'],a_index=-1,c_index=b['index'],cost=float(np.linalg.norm(qc[indices]-base[indices]))))
        for rank,choice in enumerate(sorted(choices,key=lambda p:(p['cost'],p['a_index'],p['c_index']))[:1 if cpu else 2]):
            pairs.append(dict(choice,branch_rank=rank))
    planner.pinocchio_model.compute_forward_kinematics(base);fk=(transform(request['root_pose'])*planner.pinocchio_model.get_link_pose(planner.move_group_link_id)).to_transformation_matrix()
    return dict(status='IK_PAIRS_READY' if pairs else 'NO_LEGAL_REGISTERED_PAIR',input_sha256=request['input_sha256'],goals=rows,pairs=pairs,
        source_order_before_cost=True,initial_fk_world=np.asarray(fk).tolist(),physics_steps=0)

def route(request,folder,index):
    from .task_capture_runtime import read
    ik=read(folder/'IK_RESULT.private.json');pair=ik['pairs'][index//2];stream=index%2;links=[]
    for link,goal_slot in enumerate(pair['goal_slots']):
        goal=request['goals'][goal_slot];ledger=read(Path(request['run_root'])/FOLDER/'SEARCH_LEDGER.json');slots=ledger['bundles'][request['bundle_key']]['slots'];slot=sum(k.startswith('RRT/') for k in slots)
        planner,q=planner_for(request,goal['stage'],closed=goal.get('closed',False))
        if link==1:q[planner.move_group_joint_indices]=np.asarray(pair['qa'])[planner.move_group_joint_indices]
        endpoint=np.asarray(pair['qa'] if link==0 and pair['qa'] is not None else pair['qc'])
        if np.array_equal(q,endpoint):
            arm=q[planner.move_group_joint_indices].tolist()
            row=dict(status='Success',time=[0.,.05],position=[arm,arm],duration=.05,package=pair['package'],input_sha256=request['input_sha256'],exact_zero_connector=True,new_RRT_calls=0)
            write(folder/f'EXACT-ZERO-{index:02d}-{link:02d}.private.json',row);links.append(row);continue
        try:reserve_internal(request['run_root'],request['bundle_key'],'RRT',slot)
        except PermissionError as e:return dict(status=str(e),input_sha256=request['input_sha256'],links=links,package=pair['package'])
        seed=goal['RRT_seed_table'][pair['branch_rank']*2+stream][link];np.random.seed(seed);mplib.set_global_seed(seed);started=time.monotonic()
        planned=planner.plan_qpos([endpoint],q.copy(),time_step=.05,rrt_range=.1,planning_time=1.,fix_joint_limits=False,simplify=False)
        row=dict(status=planned['status'],seed=seed,route=index,link=link,shared_slot=slot,package=pair['package'],input_sha256=request['input_sha256'],service_s=time.monotonic()-started)
        if planned['status']=='Success':
            if len(planned['position'])>1024:row['status']='ROUTE_OUTPUT_ROW_CAP'
            else:row.update(time=np.asarray(planned['time']).tolist(),position=np.asarray(planned['position']).tolist(),duration=float(planned['duration']))
        write(folder/f'RRT-{slot:02d}.private.json',row);finish_internal(request['run_root'],request['bundle_key'],'RRT',slot,dict(status=row['status']));links.append(row)
        if row['status']!='Success':return dict(status=row['status'],links=links,input_sha256=request['input_sha256'],package=pair['package'])
    times=[];positions=[];offset=0.
    for segment in links:
        t=np.asarray(segment['time']);q=np.asarray(segment['position'])
        times.extend((offset+t if not times else offset+t[1:]).tolist());positions.extend(q.tolist() if not positions else q[1:].tolist());offset+=float(t[-1])
    return dict(status='Success',time=times,position=positions,duration=offset,links=links,package=pair['package'],route=index,input_sha256=request['input_sha256'],physics_steps=0)

def main():
    from .task_capture_runtime import validate_entry
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--action',choices=['IK','ROUTE'],required=True);p.add_argument('--route',type=int,default=0);a=p.parse_args();file=Path(a.request);request=json.loads(file.read_text());request['input_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();root=Path(request['run_root']);m,_=validate_entry(root,read_only=True);require_adoption(root,m)
    assert file.resolve().is_relative_to((root/FOLDER/'requests').resolve()) and request['implementation_sha']==m['code']['vla_commit']
    target=file.parent/('IK_RESULT.private.json' if a.action=='IK' else f'ROUTE-{a.route:02d}.private.json')
    if target.exists():assert json.loads(target.read_text())['input_sha256']==request['input_sha256'];return
    value=inverse(request,file.parent) if a.action=='IK' else route(request,file.parent,a.route);write(target,value)

if __name__=='__main__':main()
