"""New final-five-second predicate; never changes the original V3 result."""
import numpy as np

def reliable(time,position,rotation,goal,goal_rotation,original,*,gate_refusals,collisions,execution_failure):
    t,p,r=map(np.asarray,(time,position,rotation))
    assert len(t)>0 and t.ndim==1 and p.shape==(len(t),3) and r.shape==(len(t),3,3)
    assert np.isfinite(p).all() and np.isfinite(r).all() and np.all(np.diff(t)>0)
    xyz=np.linalg.norm(p-np.asarray(goal),axis=-1)
    from scipy.spatial.transform import Rotation
    angle=Rotation.from_matrix(r@np.asarray(goal_rotation).T).magnitude()
    in_pose=(xyz<=.035)&(angle<=.10)
    hit=original['time_to_success_s'];done=original['success'] and hit is not None and hit<=35.
    full=abs(t[-1]-40.)<=1e-6
    window=t>=35.-1e-9
    covered=full and int(window.sum())==251 and abs(t[window][0]-35.)<=1e-6 and np.allclose(np.diff(t),.02,rtol=0,atol=1e-6)
    stable=covered and bool(in_pose[window].all())
    safe=not (gate_refusals or collisions or execution_failure or original['safety_failure'])
    exits=np.flatnonzero((t>hit)&~in_pose) if hit is not None else []
    return dict(schema='PI0268_R1_FINAL5S_RELIABILITY_V1',reliable=bool(done and stable and safe),
        original_success=bool(original['success']),original_completion_s=hit,original_completed_by35=bool(done),
        full40s=bool(full),final5s_complete_sampling=bool(covered),final5s_every_tick_in_pose=bool(stable),
        final5s_ticks=int(window.sum()),final5s_outside_ticks=int((~in_pose[window]).sum()),
        final5s_max_xyz_m=float(xyz[window].max()) if window.any() else None,
        final5s_max_so3_rad=float(angle[window].max()) if window.any() else None,
        first_pose_exit_after_success_s=float(t[exits[0]]) if len(exits) else None,
        full_episode_safe=bool(safe),gate_refusals=int(gate_refusals),dangerous_collisions=int(collisions),
        execution_failure=bool(execution_failure),original_V3_unchanged=True)

def self_check():
    t=np.arange(2001)*.02;p=np.zeros((2001,3));r=np.repeat(np.eye(3)[None],2001,axis=0)
    original=dict(success=True,time_to_success_s=34.,safety_failure=False)
    def evaluate(pp=p,tt=t,rr=r,orig=original,**kw):
        return reliable(tt,pp,rr,np.zeros(3),np.eye(3),orig,gate_refusals=kw.get('gate_refusals',0),collisions=kw.get('collisions',0),execution_failure=kw.get('execution_failure',False))
    assert evaluate()['reliable']
    changed=p.copy();changed[1750,0]=.03500001;assert not evaluate(changed)['reliable']
    changed[1750,0]=.035;assert evaluate(changed)['reliable']
    for key in ('gate_refusals','collisions','execution_failure'):assert not evaluate(**{key:1})['reliable']
    assert not evaluate(orig=dict(original,time_to_success_s=35.02))['reliable']
    assert evaluate(orig=dict(original,time_to_success_s=35.))['reliable']
    assert not evaluate(orig=dict(original,success=False,time_to_success_s=None))['reliable']
    assert not evaluate(p[:-1],t[:-1],r[:-1])['reliable']
    keep=np.arange(2001)!=1800;assert not evaluate(p[keep],t[keep],r[keep])['reliable']
    from scipy.spatial.transform import Rotation
    changed_r=r.copy();changed_r[1900]=Rotation.from_rotvec([.100001,0,0]).as_matrix();assert not evaluate(rr=changed_r)['reliable']
    assert original==dict(success=True,time_to_success_s=34.,safety_failure=False)
    print('PASS_CPU_RELIABILITY_BOUNDARIES_NO_ORIGINAL_SCORE_MUTATION',flush=True)

if __name__=='__main__':self_check()
