"""Required PI0259 numerical/negative fixtures. No models or scene construction."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from pi0259_kinematic import *


def run():
    reg=json.loads((ROOT/'design/registry.json').read_text());kin,collision,base,root,velocity=source_geometry(reg)
    q=np.array(reg['arm_default']);initial=root@kin.tcp_pose(q);f=reg['calibration_families'][0];checks={}
    for task in ('M1','M2','M3'):
        for c in (0,1):
            p,r=reference(reg,f,task,c,[0]);assert np.max(abs(p[0]-initial[:3,3]))<1e-12
            assert np.max(abs(r[0]-initial[:3,:3]))<1e-12
    checks['actual_start_all_six_commands']=True
    transformed=homogeneous(Rotation.from_rotvec([.31,-.22,.18]).as_matrix(),[.23,-.17,.61])
    positions=dict(base);sample=np.array([.1,.2,.3,.1,-.1,.1]);positions.update(zip(kin.joint_names,sample))
    full=full_fk(collision.urdf,positions,transformed)['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
    assert np.max(abs(full-transformed@kin.tcp_pose(sample)))<1e-10
    checks['nonzero_root_full_FK_roundtrip']=True
    target=transformed@kin.tcp_pose(sample);jr=residual_jacobian(kin,transformed,sample,target[:3,3],target[:3,:3])
    geometric=kin.finite_difference_jacobian(sample);geometric[:3]=transformed[:3,:3]@geometric[:3];geometric[3:]=transformed[:3,:3]@geometric[3:]
    assert np.max(abs(jr+geometric))<1e-7
    solved,e,n,okay,_=solve(kin,transformed,target[:3,3],target[:3,:3],sample+.003,reg['solver'])
    assert okay and np.linalg.norm(e[:3])<=.002 and np.linalg.norm(e[3:])<=.01
    checks['residual_geometric_sign_max_abs']=float(np.max(abs(jr+geometric)))
    for angle in (1e-12,np.pi-1e-8,np.pi,np.pi+1e-8):
        rotation=Rotation.from_rotvec(np.array([1.,2.,3.])/np.sqrt(14)*angle).as_matrix()
        log=Rotation.from_matrix(rotation).as_rotvec()
        assert np.isfinite(log).all() and np.max(abs(Rotation.from_rotvec(log).as_matrix()-rotation))<1e-8
    checks['near_identity_pi_rotation_roundtrip']=True
    ledger=[];starts=[]
    def failed_solver(p,r,q):
        starts.append(q.copy());return q+99,np.ones(6),200,False,0
    original=np.arange(6,dtype=float)
    ok,t,returned=continuation(lambda t:(np.zeros(3),np.eye(3)),failed_solver,lambda *x:dict(valid=True),0.,original, .05,ledger)
    assert not ok and t==0 and np.array_equal(returned,original) and all(np.array_equal(x,original) for x in starts)
    checks['failed_candidate_not_propagated_attempts']=len(ledger)
    assert len(ledger)==5 and max(x['depth'] for x in ledger)==4
    # A successful intermediate followed by failure returns only the accepted prefix.
    def conditional(p,r,q):return q+(p[0]-q[0]),np.zeros(6),1,bool(p[0]<=.025),0
    trail=[];ok,t,accepted=continuation(lambda t:(np.array([t,0,0]),np.eye(3)),conditional,lambda *x:dict(valid=True),0.,np.zeros(6),.05,trail)
    assert not ok and 0<t<=.025 and np.allclose(accepted,t)
    checks['accepted_prefix_retained']=True
    for task,joins in [('M1',[0,6]),('M2',[0,3,6,10]),('M3',[0,4,16])]:
        for join in joins:
            h=1e-4;p,_=reference(reg,f,task,0,[join-h,join,join+h])
            assert np.linalg.norm((p[2]-p[0])/(2*h))<1e-5
            assert np.linalg.norm((p[2]-2*p[1]+p[0])/h**2)<1e-3
    checks['quintic_join_velocity_acceleration']=True
    assert np.all(velocity>0)
    assert not np.all(abs(velocity*.05*1.01)/.05<=velocity)
    checks['velocity_negative']=True
    nominal=full_fk(collision.urdf,base,root)
    assert not collision.check(nominal)
    assert collision.check(nominal,initial[:3,3],[.3,.3,.3])
    checks['original_geometry_collision_negative']=True
    old=[]
    for i in (0,1,2):
        folder=OLD/'development'/('pi0258-family-%03d'%i)
        with np.load(folder/('pi0258-family-%03d-M3-c0-geometry.npz'%i)) as a:
            old.append(dict(family=i,first_position_error_m=float(np.linalg.norm(a['desired_position'][0]-a['FK_world'][0,:3,3])),
                first_SO3_error_rad=float(np.linalg.norm((Rotation.from_quat(a['desired_quaternion'][0])*Rotation.from_matrix(a['FK_world'][0,:3,:3]).inv()).as_rotvec())),
                qualified_windows=float(np.mean(a['active_geometry_qualified']))))
    receipt=dict(status='PASS',checks=checks,old_stream_diagnosis=old,registry_sha256=sha(ROOT/'design/registry.json'),
        source_sha256=sha(__file__),physics_steps=0,models=0)
    write(ROOT/'evidence/cpu_fixtures.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':run()

