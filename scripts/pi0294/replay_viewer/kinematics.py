"""Optional local skeleton reconstruction, reusing the recorded-state mj_forward rule.

Caller supplies a legally available, identity-checked model. This module neither
loads/downloads assets nor creates an Engine, controller, renderer or policy.
"""
def recorded_skeleton(model, state, *, mujoco_module=None):
    if mujoco_module is None:
        import mujoco as mujoco_module
    import numpy as np
    q, v = np.asarray(state['qpos'], dtype=float), np.asarray(state['qvel'], dtype=float)
    if q.shape != (model.nq,) or v.shape != (model.nv,) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('EXACT_RECORDED_STATE_LAYOUT_REQUIRED')
    d = mujoco_module.MjData(model)
    d.qpos[:] = q
    d.qvel[:] = v
    d.time = float(state['state_time_s'])
    if not np.isfinite(d.time):
        raise ValueError('RECORDED_STATE_CLOCK_REQUIRED')
    if state.get('ctrl') is not None:
        ctrl = np.asarray(state['ctrl'], dtype=float)
        if ctrl.shape != (model.nu,) or not np.isfinite(ctrl).all():
            raise ValueError('RECORDED_CTRL_LAYOUT_REQUIRED')
        d.ctrl[:] = ctrl
    mujoco_module.mj_forward(model, d)
    return {'points': d.xpos.tolist(), 'edges': [[int(model.body_parentid[i]), i] for i in range(1, model.nbody)],
            'state_time_s': float(d.time), 'method': 'mj_forward_only', 'new_integrations': 0}
