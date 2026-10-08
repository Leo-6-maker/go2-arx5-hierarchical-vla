"""Independent task scoring on real timestamped arrays, including every hold tick."""
import itertools,json
import numpy as np
from scipy.spatial.transform import Rotation


def sustained(times,flags,duration=.30):
    start=None
    for t,flag in zip(times,flags):
        if not flag:start=None
        elif start is None:start=t
        elif t-start>=duration-1e-8:return True
    return False


def predicates(data,zone,initial_red,initial_green):
    pose=data['cubeA_pose'];red=pose[:,:3]
    rot=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
    offsets=np.asarray(list(itertools.product((-.02,.02),repeat=3)))
    corners=np.einsum('nij,kj->nki',rot,offsets)+red[:,None,:]
    grasp=np.asarray(data['is_cubeA_grasped'],dtype=bool)
    linear=np.linalg.norm(data['cubeA_velocity'],axis=1)
    angular=np.linalg.norm(data['cubeA_angular'],axis=1)
    static=(linear<=.01)&(angular<=.5)
    separation=np.linalg.norm(data['tcp'][:,:3,3]-red,axis=1)>=.08
    zone_ok=(abs(corners[:,:,:2]-np.asarray(zone))<=.07+1e-8).all(axis=(1,2))
    floor=(abs(corners[:,:,2].min(axis=1))<=.005)
    offset=red-data['cubeB_pose'][:,:3]
    on_top=(np.linalg.norm(offset[:,:2],axis=1)<=np.linalg.norm([.02,.02])+.005)&(abs(offset[:,2]-.04)<=.005)
    original_stack=on_top&static&~grasp
    return dict(grasp=grasp,lift=red[:,2]>=initial_red[2]+.02-1e-8,
                horizontal=np.linalg.norm(red[:,:2]-initial_red[:2],axis=1)>=.1,
                zone=zone_ok,table=floor,static=static,not_grasped=~grasp,TCP_clear=separation,
                green_unmoved=np.linalg.norm(data['cubeB_pose'][:,:3]-initial_green,axis=1)<=.01,
                original_stack=original_stack,
                T1_hold=zone_ok&floor&static&~grasp&separation,
                T2_hold=original_stack&separation)


def score(data,task,zone,initial_red,initial_green,status,controls,neutral_controls=20):
    assert task in ('T1','T2')
    times=np.asarray(data['time_s'],dtype=float)-neutral_controls*.05
    masks=predicates(data,zone,np.asarray(initial_red),np.asarray(initial_green))
    tasktime=times>=-1e-8
    history=dict(grasp=sustained(times[tasktime],masks['grasp'][tasktime]),lift=sustained(times[tasktime],masks['lift'][tasktime]),horizontal_transfer=bool(masks['horizontal'][tasktime].any()))
    # Select original 20Hz samples from the physics stream, inclusive 28..30.
    idx=np.arange(len(times))[::5];hold=idx[(times[idx]>=28-1e-8)&(times[idx]<=30+1e-8)]
    full=bool(status=='VALID_SOURCE_COMPLETE' and controls==neutral_controls+640 and times[-1]>=32-1e-8)
    passed=int(masks[task+'_hold'][hold].sum())
    green=bool(masks['green_unmoved'][-1])
    required=history['grasp'] and history['lift'] and (history['horizontal_transfer'] if task=='T1' else True)
    success=bool(full and required and len(hold)==41 and passed==41 and (green if task=='T1' else True))
    def events(value):
        ids=np.flatnonzero(np.diff(value.astype(int))!=0)+1
        return [dict(time_s=float(times[i]),value=bool(value[i])) for i in ids if times[i]>=0]
    report=dict(task=task,complete_32s=full,history=history,hold_observed=len(hold),hold_pass=passed,green_final_unmoved=green,persistent_success=success,
                original_stack_hold_pass=int(masks['original_stack'][hold].sum()),subpredicates={key:dict(hold_pass=int(value[hold].sum()),first_true_s=float(times[np.flatnonzero(value&tasktime)[0]]) if (value&tasktime).any() else None,changes=events(value)) for key,value in masks.items()},neutral_controls=neutral_controls,status=status)
    return report


def fixture():
    t=np.arange(3301)*.01;red=np.tile([.16,0,.02,1,0,0,0],(len(t),1));red[:100,:3]=[-.10,0,.02]
    red[100:400,2]=.07
    grasp=(t>=1)&(t<=1.6)
    tcp=np.tile(np.eye(4),(len(t),1,1));tcp[:,:3,3]=[.16,0,.15]
    return dict(time_s=t,cubeA_pose=red,cubeB_pose=np.tile([-.1,.15,.02,1,0,0,0],(len(t),1)),cubeA_velocity=np.zeros((len(t),3)),cubeA_angular=np.zeros((len(t),3)),tcp=tcp,is_cubeA_grasped=grasp,is_cubeA_on_cubeB=np.zeros(len(t),bool))


def self_check():
    def result(d):return score(d,'T1',[.16,0],[-.10,0,.02],[-.10,.15,.02],'VALID_SOURCE_COMPLETE',660)
    d=fixture();assert result(d)['persistent_success']
    d=fixture();d['is_cubeA_grasped'][:]=False;assert not result(d)['persistent_success'] # push
    d=fixture();d['is_cubeA_grasped'][-600:]=True;assert not result(d)['persistent_success'] # held
    d=fixture();d['cubeA_pose'][-600:,0]=-.10;assert not result(d)['persistent_success'] # wrong target
    d=fixture();d['cubeA_pose'][-600:,0]=-.10;d['cubeA_pose'][2900,0]=.16;assert not result(d)['persistent_success'] # one tick
    d=fixture();d['cubeA_pose'][3000,0]=-.10;assert not result(d)['persistent_success'] # target loss
    d=fixture();d['tcp'][-600:,:3,3]=[.16,0,.04];assert not result(d)['persistent_success'] # no withdraw
    d=fixture();d['cubeA_angular'][3000,0]=.51;assert not result(d)['persistent_success']
    assert sustained(np.asarray([0,.1,.2,.3]),[True]*4) and not sustained(np.asarray([0,.1,.2]),[True]*3)
    d=fixture();d['cubeA_pose'][400:,2]=.06;d['cubeB_pose'][:,:3]=[.16,0,.02]
    assert score(d,'T2',[.16,0],[-.10,0,.02],[.16,0,.02],'VALID_SOURCE_COMPLETE',660)['persistent_success']
    d['cubeA_pose'][3000,2]=.10
    assert not score(d,'T2',[.16,0],[-.10,0,.02],[.16,0,.02],'VALID_SOURCE_COMPLETE',660)['persistent_success']
    print('PASS_CPU_TASK_SUCCESS_AND_SEVEN_NEGATIVE_FIXTURES')

if __name__=='__main__':self_check()
