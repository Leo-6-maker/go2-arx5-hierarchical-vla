"""Changed-contract CPU evidence; synthetic fixtures never count as task success."""
import copy
import json
import struct
from pathlib import Path
import numpy as np
from .night_runtime import read,sha,CUTOFF,DEADLINE
from scripts.pi0292.durable import write


def run(root,card):
    import mujoco
    from unittest.mock import patch
    from .night_runtime import validate_entry
    from .night_geometry import xml_scene,initial_data,initial_audit
    from .night_safety import JournalRangeGuard
    from .night_score import bind,score
    from .night_source import local_schedule
    from .score import score_push
    from go2arx5_vla.target_sim.scorer import score_episode,BOOL_FIELDS
    root=Path(root);checks=[];folder=root/'evidence/NIGHT_CHANGED_CONTRACTS';folder.mkdir(exist_ok=False)
    def put(name,**e):checks.append(dict(name=name,status='PASS',evidence=e))
    def reject(fn,kind=PermissionError):
        try:fn()
        except kind:return True
        raise AssertionError('NIGHT_NEGATIVE_DID_NOT_REJECT')
    m,_=validate_entry(root,read_only=True)
    with patch('scripts.pi0294.night_runtime.time.time',return_value=CUTOFF):assert reject(lambda:validate_entry(root))
    with patch('scripts.pi0294.night_runtime.time.time',return_value=DEADLINE):assert reject(lambda:validate_entry(root,read_only=True))
    put('INDEPENDENT_AUTHORITY_AND_HARD_CUTOFF',old_scope_not_reopened=True)
    model=mujoco.MjModel.from_xml_string(xml_scene(card,read(root/'INPUT_BINDINGS.private.json')['asset_root']))
    d=initial_data(model);initial=initial_audit(model,d,card)
    prior=Path(m['dependencies']['prior_run']);baseline=mujoco.MjModel.from_binary_path(str(prior/'source-demo-access-v3/episodes/SDR3_SEED_W0_T1/task-scene.private.mjb'))
    for name in ['jnt_range','jnt_type','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','geom_size','geom_margin','geom_friction','geom_contype','geom_conaffinity','body_mass','body_inertia','dof_damping']:
        assert np.array_equal(getattr(model,name),getattr(baseline,name)),name
    assert model.opt.timestep==baseline.opt.timestep==.002
    put('FROZEN_FULLBODY_DYNAMICS_AND_SAFETY',initial=initial)
    def packet(q,step):
        raw=struct.pack('<8sQdIII',b'GRTSTATE',step,step*.002,34,32,20)+np.r_[q,np.zeros(52)].astype('<f8').tobytes()+struct.pack('<I',0)
        return struct.pack('<I',len(raw))+raw
    q=d.qpos.copy();path=folder/'CPU_FIXTURE_NOT_PHYSICS.journal';path.write_bytes(packet(q,1));guard=JournalRangeGuard(model,path);assert guard.check()==1
    q[26]=.0440001
    with path.open('ab') as f:f.write(packet(q,2))
    assert reject(guard.check,ValueError);failure=read(folder/'HIGH_RATE_RANGE_FAILURE.json');assert failure['step']==2 and failure['joint']=='task_right_slide'
    put('ALL_NATIVE_LIMITED_JOINTS_HIGH_RATE_NO_TOLERANCE_RELAXATION',fixture_only=True,raw_excess_rejected=True)
    def fixture(T,task):
        n=20*T+1;t=np.arange(n)/20;obj=np.tile(np.eye(4),(n,1,1));tcp=obj.copy();obj[:,:3,3]=[.45,-.08,card['table_top_m']+.02];tcp[:,:3,3]=[.35,0,.75]
        f=np.clip((t-.5)/1.5,0,1);obj[:,1,3]+=.16*f
        data=dict(time_s=t,object_pose=obj,tcp_world=tcp,object_linear_velocity=np.zeros((n,3)),object_angular_velocity=np.zeros((n,3)))
        data.update({k:np.full(n,k in ['object_present','contact_valid','gripper_state_valid','safety_valid'],dtype=bool) for k in BOOL_FIELDS})
        grasp=(t>=.5)&(t<=2.) if task=='T1' else np.zeros(n,dtype=bool)
        for k in ['native_grasp','left_object_contact','right_object_contact']:data[k]=grasp.copy()
        if task=='T1':data['object_pose'][grasp,2,3]+=.03
        contact=(t>=.5)&(t<=2.);data['robot_object_contact']=contact
        tc=np.arange(1,500*T+1)*.002;micro=np.zeros((len(tc),5));micro[:,0]=(tc>=.5)&(tc<=2.);micro[:,3]=card['table_top_m']+.02
        data.update(contact_time_s=tc,contact_audit=micro)
        return data
    for task in ['T1','T2']:
        for T in [32,64,96,128]:
            data=fixture(T,task);c=dict(card,task=task,duration_s=T)
            value=score(data,c,complete=True,high_rate_ranges_PASS=True)
            assert value['persistent_success'] and value['hold_observed']==41 and value['hold_pass']==41,(task,T,value)
            raw=bind(score_episode if task=='T1' else score_push,T,card['table_top_m'])
            if T==32:
                if task=='T1':assert raw(data,goal_center=[*card['goal_xy'],card['table_top_m']+.02],initial_object_pose=data['object_pose'][0],table_height_m=card['table_top_m'])==score_episode(data,goal_center=[*card['goal_xy'],card['table_top_m']+.02],initial_object_pose=data['object_pose'][0],table_height_m=card['table_top_m'])
                else:
                    original_card=dict(c,table_top_m=.44);assert raw(data,original_card)==score_push(data,original_card)
            late={k:v.copy() for k,v in data.items()};late['native_abort'][-1]=True
            assert not score(late,c,complete=True,high_rate_ranges_PASS=True)['persistent_success']
            assert not score(data,c,complete=True,high_rate_ranges_PASS=False)['persistent_success']
            if task=='T1':
                for k in ['native_grasp','left_object_contact','right_object_contact']:data[k][:]=False
            else:data['robot_object_contact'][:]=False;data['contact_audit'][:,0]=0
            assert not score(data,c,complete=True,high_rate_ranges_PASS=True)['persistent_success']
    put('T32_IDENTICAL_AND_SLOW_WINDOWS_41_POINTS_LATE_FAILURE_FALSE_CONTACT',tasks=2,durations=[32,64,96,128],fixtures_not_physics=True)
    poses=np.tile(np.eye(4),(6,1,1));q=np.zeros((6,6));q[:,0]=[0,.001,.002,.02,.021,.022]
    original=np.arange(6)*.05;t,proof=local_schedule(poses,q,original)
    assert proof['changed_intervals']>0 and np.all(np.diff(t)>=.05-1e-12) and not proof['spatial_knots_changed']
    put('LOCAL_INTERVAL_RETIME_NO_POSE_OR_GRIPPER_SMOOTHING',proof=proof)
    from .mp_window import SavedWindow
    action=np.arange(520.).reshape(40,13);whole=SavedWindow(action).take(8);split=SavedWindow(action);left=split.take(4);right=SavedWindow.restore(json.loads(json.dumps(split.snapshot()))).take(4)
    assert np.array_equal(whole,np.vstack([left,right]))
    put('ORIGINAL_PREFIX8_RESTORE_IDENTITY',new_physics=0)
    result=dict(status='PASS_NIGHT_CHANGED_CPU_CONTRACTS',source=m['code']['vla_commit'],checks=checks,checks_passed=len(checks),new_physics=0,new_neural_queries=0,fixtures_are_not_success_seeds=True)
    write(folder/'RECEIPT.json',result);return result
