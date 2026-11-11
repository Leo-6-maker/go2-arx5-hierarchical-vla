"""Separate offline conic trajectory fit; no raw policy output is overwritten.

Same endpoint/rotation/physical constraints and separable L2 objective as the
SLSQP diagnosis. A convex SOCP/QP replaces the generic nonlinear solver. Only
Clarabel Solved with independently verified constraints is admissible here.
https://clarabel.org/stable/python/getting_started_py/
No object truth, expert labels, retiming, clipping or physical commands.
"""
import time
import numpy as np
from .task_learning_feasible_decoder_v3 import difference_operators
from . import fusion_trajectory_adapter as limits

VERSION='H40_ENDPOINT_FIXED_CONIC_L2_OFFLINE_V3_01'
SOLVER_VERSION='0.11.1'
MARGIN=1.-1e-7


def _solve(target,A,b,cones):
    import clarabel
    from scipy import sparse
    if clarabel.__version__!=SOLVER_VERSION:raise ValueError('Explicit pinned solver version required')
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.max_iter=200;settings.time_limit=2.
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-11
    solver=clarabel.DefaultSolver(sparse.eye(len(target),format='csc'),-target,
        sparse.csc_matrix(A),np.asarray(b),cones,settings)
    s=solver.solve()
    values=dict(status=str(s.status),iterations=s.iterations,solve_time_s=s.solve_time,
        primal_residual=s.r_prim,dual_residual=s.r_dual,
        objective_primal=s.obj_val,objective_dual=s.obj_val_dual,
        objective_gap=abs(s.obj_val-s.obj_val_dual))
    # Infeasibility certificates can have undefined/infinite objective values.
    # Retain that solver status with explicit missing metrics, so a valid policy
    # rejection is not converted into a JSON/serving technical failure.
    values={k:None if isinstance(v,float)and not np.isfinite(v)else v for k,v in values.items()}
    return np.asarray(s.x)if str(s.status)=='Solved'else None,values


def fit(raw):
    import clarabel
    from scipy import sparse
    started=time.monotonic();raw=np.asarray(raw,float)
    if raw.shape!=(40,10)or not np.isfinite(raw).all():raise ValueError('Finite original full H40 required')
    velocity,acceleration=difference_operators();operator=np.vstack((velocity,acceleration))
    origin=raw[0,:3].copy();target=(raw[:,:3]-origin)*1000
    bound=MARGIN*np.r_[np.full(39,limits.MAX_TRANSLATION_SPEED_M_S*limits.ATOM_DT_S*1000),
        np.full(40,limits.MAX_TRANSLATION_ACCELERATION_M_S2*limits.ATOM_DT_S**2*1000)]
    fixed=np.zeros((6,120));fixed[:3,:3]=np.eye(3);fixed[3:,-3:]=np.eye(3)
    blocks=[sparse.csc_matrix(fixed)];rhs=[np.r_[target[0],target[-1]]];cones=[clarabel.ZeroConeT(6)]
    for row,limit in zip(operator,bound):
        blocks.append(sparse.vstack([sparse.csc_matrix((1,120)),-sparse.kron(row[None,:],np.eye(3))]))
        rhs.append(np.r_[limit,np.zeros(3)]);cones.append(clarabel.SecondOrderConeT(4))
    xyz,xyz_audit=_solve(target.reshape(-1),sparse.vstack(blocks),np.concatenate(rhs),cones)
    grip_limit=MARGIN*limits.MAX_FINGER_SPEED_M_S*limits.ATOM_DT_S/(
        limits.WORKING_OPEN_M-limits.WORKING_CLOSED_M)
    edge=np.zeros((2,40));edge[0,0]=edge[1,-1]=1.
    grip,grip_audit=_solve(raw[:,9],np.vstack((edge,np.eye(40),-np.eye(40),velocity,-velocity)),
        np.r_[raw[[0,-1],9],np.ones(40),np.zeros(40),np.full(78,grip_limit)],
        [clarabel.ZeroConeT(2),clarabel.NonnegativeConeT(158)])
    audit=dict(version=VERSION,solver='CLARABEL_'+SOLVER_VERSION,translation=xyz_audit,gripper=grip_audit,
        solver_success=False,object_truth_or_expert_inputs_used=False,native_reference_certificate='NOT_MEASURED',
        physics_integrations=0,model_queries=0,optimizer_updates=0)
    if xyz is None or grip is None:
        audit.update(rejection='SOLVER_NOT_SOLVED_NO_FALLBACK',wall_s=time.monotonic()-started)
        return None,audit
    result=raw.copy();result[:,:3]=origin+xyz.reshape(40,3)/1000;result[:,9]=grip
    result[[0,-1]]=raw[[0,-1]]
    relative=(result[:,:3]-origin)*1000
    ratios=np.linalg.norm(operator@relative,axis=1)/bound
    # No tolerance outside the actual original rate/acceleration/gripper bounds.
    verified=bool(np.isfinite(result).all()and ratios.max()<=1.+1e-8
        and np.all((result[:,9]>=0)&(result[:,9]<=1))
        and np.max(np.abs(velocity@result[:,9]))<=grip_limit*(1.+1e-8))
    audit.update(solver_success=verified,maximum_translation_constraint_ratio=float(ratios.max()),
        mean_XYZ_change_m=float(np.linalg.norm(result[:,:3]-raw[:,:3],axis=1).mean()),
        maximum_XYZ_change_m=float(np.linalg.norm(result[:,:3]-raw[:,:3],axis=1).max()),
        maximum_gripper_change=float(np.abs(result[:,9]-raw[:,9]).max()),
        rotations_exact_unchanged=bool(np.array_equal(result[:,3:9],raw[:,3:9])),
        endpoints_exact_unchanged=bool(np.array_equal(result[[0,-1]],raw[[0,-1]])),
        wall_s=time.monotonic()-started)
    if not verified:
        audit['rejection']='INDEPENDENT_CONSTRAINT_CHECK_FAILED_NO_FALLBACK'
        return None,audit
    return result,audit
