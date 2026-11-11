"""Explicit offline minimum-change H40 fit, not the original policy output.

No object truth, expert labels, model calls, retiming or physics enter the fit.
Endpoints and rotations stay fixed. The original full reference certificate is
still required; passing this numerical fit does not authorize execution.
"""
import time

import numpy as np

from . import fusion_trajectory_adapter as limits

VERSION='H40_ENDPOINT_FIXED_TRANSLATION_GRIP_CONSTRAINED_L2_DIAGNOSTIC_V3'


def difference_operators():
    velocity=np.diff(np.eye(40),axis=0)
    acceleration=np.diff(np.vstack((np.zeros((1,40)),velocity,np.zeros((1,40)))),axis=0)
    return velocity,acceleration


def fit(raw):
    """Return a separately labelled fit or an explicit rejection, never fallback."""
    from scipy.optimize import Bounds,LinearConstraint,minimize
    started=time.monotonic();raw=np.asarray(raw,float)
    if raw.shape!=(40,10)or not np.isfinite(raw).all():raise ValueError('Finite original full H40 required')
    velocity,acceleration=difference_operators();operator=np.vstack((velocity,acceleration))
    # Millimetres and dimensionless constraints avoid metre-squared conditioning.
    origin=raw[0,:3].copy();target=(raw[:,:3]-origin)*1000
    bound=np.r_[np.full(39,limits.MAX_TRANSLATION_SPEED_M_S*limits.ATOM_DT_S*1000),
        np.full(40,limits.MAX_TRANSLATION_ACCELERATION_M_S2*limits.ATOM_DT_S**2*1000)]
    # Solver tolerance is kept inside, never outside, the original mechanical gate.
    bound*=1.-1e-7
    def objective(x):return .5*np.square(x.reshape(40,3)-target).sum()
    def objective_jac(x):return (x.reshape(40,3)-target).reshape(-1)
    def constraints(x):return 1.-np.square((operator@x.reshape(40,3))/bound[:,None]).sum(axis=1)
    def constraints_jac(x):
        value=(operator@x.reshape(40,3))/np.square(bound[:,None])
        return (-2*operator[:,:,None]*value[:,None,:]).reshape(79,120)
    fixed=np.zeros((6,120));fixed[:3,:3]=np.eye(3);fixed[3:,-3:]=np.eye(3)
    endpoints=np.r_[target[0],target[-1]]
    u=np.linspace(0.,1.,40);weight=u**3*(10+u*(-15+6*u))
    initial=target[0]+weight[:,None]*(target[-1]-target[0])
    xyz=minimize(objective,initial.reshape(-1),jac=objective_jac,method='SLSQP',
        constraints=[dict(type='ineq',fun=constraints,jac=constraints_jac),LinearConstraint(fixed,endpoints,endpoints)],
        options=dict(ftol=1e-11,maxiter=300))
    # The nominal working grip is a separate convex problem. No np.clip step.
    grip_target=raw[:,9];grip_limit=limits.MAX_FINGER_SPEED_M_S*limits.ATOM_DT_S/(
        limits.WORKING_OPEN_M-limits.WORKING_CLOSED_M)*(1.-1e-7)
    edge=np.zeros((2,40));edge[0,0]=edge[1,-1]=1.
    grip=minimize(lambda x:.5*np.square(x-grip_target).sum(),np.linspace(grip_target[0],grip_target[-1],40),
        jac=lambda x:x-grip_target,method='SLSQP',bounds=Bounds(0.,1.),
        constraints=[LinearConstraint(velocity,-grip_limit,grip_limit),
            LinearConstraint(edge,grip_target[[0,-1]],grip_target[[0,-1]])],
        options=dict(ftol=1e-12,maxiter=300))
    result=raw.copy();result[:,:3]=origin+xyz.x.reshape(40,3)/1000;result[:,9]=grip.x
    # Preserve the exact original endpoint bytes; constraints are rechecked below.
    result[[0,-1]]=raw[[0,-1]]
    feasible=bool(xyz.success and grip.success and constraints((result[:,:3]-origin).reshape(-1)*1000).min()>=-1e-8
        and np.all((result[:,9]>=0)&(result[:,9]<=1))and np.max(np.abs(velocity@result[:,9]))<=grip_limit*(1+1e-8))
    return result,dict(version=VERSION,solver_success=feasible,
        translation_solver_status=int(xyz.status),translation_solver_message=str(xyz.message),translation_iterations=int(xyz.nit),
        gripper_solver_status=int(grip.status),gripper_solver_message=str(grip.message),gripper_iterations=int(grip.nit),
        mean_XYZ_change_m=float(np.linalg.norm(result[:,:3]-raw[:,:3],axis=1).mean()),
        maximum_XYZ_change_m=float(np.linalg.norm(result[:,:3]-raw[:,:3],axis=1).max()),
        endpoint_XYZ_drift_m=float(np.linalg.norm(result[-1,:3]-raw[-1,:3])),
        first_XYZ_drift_m=float(np.linalg.norm(result[0,:3]-raw[0,:3])),
        maximum_gripper_change=float(np.abs(result[:,9]-raw[:,9]).max()),
        rotations_exact_unchanged=bool(np.array_equal(result[:,3:9],raw[:,3:9])),
        endpoints_exact_unchanged=bool(np.array_equal(result[[0,-1]],raw[[0,-1]])),
        object_truth_or_expert_inputs_used=False,native_reference_certificate='NOT_MEASURED',
        physics_integrations=0,model_queries=0,optimizer_updates=0,wall_s=time.monotonic()-started)


def summarize_change(raw,decoded,expert):
    """Expert is used only after fitting for diagnostics, never to choose output."""
    def crossing(values,threshold):
        index=np.flatnonzero(values<=threshold)
        return int(index[0])if len(index)else None
    r=raw[-1,:3]-raw[0,:3];d=decoded[-1,:3]-decoded[0,:3];e=expert[-1,:3]-expert[0,:3]
    angle=lambda a,b:float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))if np.linalg.norm(a)*np.linalg.norm(b)>1e-15 else None
    return dict(raw_expert_mean_XYZ_m=float(np.linalg.norm(raw[:,:3]-expert[:,:3],axis=1).mean()),
        decoded_expert_mean_XYZ_m=float(np.linalg.norm(decoded[:,:3]-expert[:,:3],axis=1).mean()),
        endpoint_direction_cosine_raw_vs_decoded=angle(r,d),endpoint_direction_cosine_decoded_vs_expert=angle(d,e),
        endpoint_motion_m=float(np.linalg.norm(d)),expert_endpoint_motion_m=float(np.linalg.norm(e)),
        first_close_knots={str(t):dict(raw=crossing(raw[:,9],t),decoded=crossing(decoded[:,9],t),
            expert=crossing(expert[:,9],t))for t in(.5,.1,.01)},
        numerical_feasibility_is_task_success=False)
