"""Saved-state source-controller replay, without advancing physics.

Reconstructed commands are diagnostics, never recorded expert labels. The
original sealed controller prefix is used rather than a second controller.
"""
import ast
import copy
import hashlib

import numpy as np


def control_prefix(source, expected_sha256, namespace):
    """Compile Engine.step only up to its single top-level mj_step call."""
    if hashlib.sha256(source).hexdigest()!=expected_sha256:
        raise PermissionError('EXACT_CLOSED_ACTOR_CONTROLLER_SOURCE')
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Engine')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='step')
    def integration(node):
        return (isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name)and node.func.value.id=='mujoco'
            and node.func.attr in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta'))
    calls=[n for n in ast.walk(method)if integration(n)]
    if len(calls)!=1 or calls[0].func.attr!='mj_step':
        raise ValueError('ONE_EXPLICIT_NATIVE_INTEGRATION_SEAM')
    positions=[i for i,n in enumerate(method.body)if isinstance(n,ast.Expr)and n.value is calls[0]]
    if len(positions)!=1 or positions[0]==0:
        raise ValueError('INTEGRATION_MUST_BE_TOP_LEVEL_AFTER_CONTROL')
    prefix=copy.deepcopy(method)
    prefix.name='saved_state_control_without_integration'
    prefix.body=prefix.body[:positions[0]]
    if any(integration(n)for n in ast.walk(prefix)):
        raise PermissionError('REPLAY_MUST_NOT_INTEGRATE')
    module=ast.fix_missing_locations(ast.Module(body=[prefix],type_ignores=[]))
    scope=dict(namespace)
    exec(compile(module,'<sealed-controller-prefix-no-integration>','exec'),scope)
    return scope[prefix.name]


def restore_observed_state(engine, state, mujoco):
    """Use actual saved q/qdot; this does not predict or relabel trajectories."""
    q=np.asarray(state['qpos'],dtype=float);v=np.asarray(state['qvel'],dtype=float)
    if (q.shape!=(engine.model.nq,)or v.shape!=(engine.model.nv,)
            or not np.isfinite(q).all()or not np.isfinite(v).all()):
        raise ValueError('EXACT_FINITE_SAVED_PHYSICAL_STATE')
    if engine.model.na!=0:
        raise PermissionError('UNRECORDED_DYNAMIC_ACTUATOR_STATE')
    engine.data.qpos[:]=q;engine.data.qvel[:]=v;engine.data.time=float(state['time_s'])
    mujoco.mj_forward(engine.model,engine.data)


def replay_tick(engine, prefix, target, opening, before, after, mujoco):
    """Reconstruct one control update, checked against its recorded force.

    Re-evaluate stateless actuation at the observed post-state, as the native
    recorder did. No forward dynamics or mj_step is run. A force discrepancy
    is evidence that reconstruction is incomplete, not an ignorable residual.
    """
    restore_observed_state(engine,before,mujoco)
    prefix(engine,np.asarray(target,dtype=float),float(opening))
    targets=engine.arm_targets.copy();ik=engine._cached_ik_solution.copy()
    controls=engine.data.ctrl.copy();finger_targets=engine.finger_targets.copy()
    restore_observed_state(engine,after,mujoco)
    expected=engine.data.actuator_force.copy();actual=np.asarray(after['actuator_force'],dtype=float)
    if expected.shape!=actual.shape or not np.isfinite(actual).all():
        raise ValueError('EXACT_RECORDED_ACTUATOR_FORCE')
    q=np.asarray(after['arm_q'],dtype=float);bounds=engine.arm_bounds
    return dict(commanded_arm_targets=targets,IK_solution=ik,controls=controls,
        commanded_finger_targets=finger_targets,recorded_actuator_force=actual,
        reconstructed_actuator_force=expected,force_max_abs_error=float(np.max(abs(expected-actual))),
        commanded_margin=np.minimum(targets-bounds[:,0],bounds[:,1]-targets),
        IK_margin=np.minimum(ik-bounds[:,0],bounds[:,1]-ik),
        actual_margin=np.minimum(q-bounds[:,0],bounds[:,1]-q),
        actual_minus_command=q-targets,
        IK_calls=engine.ik_solve_calls,IK_cache_hits=engine.ik_cache_hits)
