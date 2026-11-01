"""Non-mutating source handover evidence; no reset/restore/stepping API."""
from copy import deepcopy
import hashlib
import json
import numpy as np


def capture_handover(engine,actual_state):
    import mujoco
    if (actual_state.get('safety_valid')is not True or actual_state.get('native_abort')
            or actual_state.get('native_grasp')is not True):
        raise PermissionError('SAFE_ACTUAL_HELD_SOURCE_STATE_BEFORE_HANDOVER')
    np.testing.assert_array_equal(engine.data.qpos,actual_state['qpos'])
    np.testing.assert_array_equal(engine.data.qvel,actual_state['qvel'])
    specification=mujoco.mjtState.mjSTATE_INTEGRATION
    def state(data):
        value=np.empty(mujoco.mj_stateSize(engine.model,specification))
        mujoco.mj_getState(engine.model,data,value,specification)
        return value.tolist()
    def value(name):
        item=getattr(engine,name,None)
        return item.tolist()if isinstance(item,np.ndarray)else deepcopy(item)
    snapshot=dict(schema='SOURCE_HELD_HANDOVER_FULL_INTEGRATION_AND_CONTROLLER_V1',
        native_state_specification=int(specification),physics_version=mujoco.__version__,
        data_integration=state(engine.data),IK_data_integration=state(engine.ik_data),
        controller={n:value(n)for n in('arm_targets','finger_targets','_ik_target','_cached_ik_solution',
                    '_cached_ik_error','last_ik_error','ik_solve_calls','ik_cache_hits','steps','safety','mode','finger_driver_version')},
        actual_state={k:(v.tolist()if isinstance(v,np.ndarray)else deepcopy(v))for k,v in actual_state.items()},
        no_reset_no_integration_no_controller_change=True)
    body=json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(snapshot=snapshot,sha256=hashlib.sha256(body).hexdigest())
