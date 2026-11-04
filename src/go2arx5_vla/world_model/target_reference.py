"""Target reference clock contract. No controller, ROS or physics construction."""
import hashlib
import json
import math
from scripts.pi0294.task_clock import actual_source_age

SCHEMA = 'WM02_TARGET_REFERENCE_CONSUMPTION_V1'
TASK_START = 2503
ATOM_TICKS = 25
DT = .002


def deadline(source):
    if isinstance(source, bool) or not math.isfinite(source):
        raise ValueError('FINITE_ACTUAL_SOURCE_REQUIRED')
    return source + .4


def require_live_window(source, native_time, ticks):
    if (type(ticks) is not int or ticks <= 0 or ticks % ATOM_TICKS
            or not math.isfinite(native_time) or actual_source_age(native_time,source)<0):
        raise ValueError('ACTUAL_SOURCE_AND_COMPLETE_RELATIVE_ATOMS_REQUIRED')
    end = native_time + ticks*DT
    # The receiver expires at >=deadline-1e-10, not 400ms after install.
    if end >= deadline(source)-1e-10:
        raise ValueError('WINDOW_REACHES_REFERENCE_DEADLINE_RENEW_FIRST')
    return dict(source_mpc_s=source, native_start_s=native_time,
                native_end_s=end, deadline_s=deadline(source),
                remaining_at_install_s=deadline(source)-native_time,
                expiry_margin_s=deadline(source)-end)


def require_install_clock(source, native_time):
    deadline(source)
    if not math.isfinite(native_time) or not 0 <= actual_source_age(native_time,source) <= .025:
        raise ValueError('INSTALL_REQUIRES_FRESH_MEASURED_SOURCE')


def trajectory_identity(times, states, inputs):
    value=dict(times=list(times), poses=[list(x) for x in states],
               inputs=[list(x) for x in inputs])
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def installed_target_identities(events, ref):
    # Original receiver republishes its rolling trajectory under the same
    # accepted reference. Match real sent content, never just the first grid.
    return {trajectory_identity(e['times'],e['poses'],e['inputs']):e
            for e in events if e.get('kind')=='installed'and e.get('ref_id')==ref}


def measured_base_hold(document, pose):
    """Existing native 14D pose reference, never fabricated BASE velocities."""
    import copy
    pose=list(pose)
    if len(pose)!=7 or not all(math.isfinite(x)for x in pose)or abs(sum(x*x for x in pose[3:])-1)>1e-5:
        raise ValueError('MEASURED_NATIVE_BASE_POSE7_REQUIRED')
    result=copy.deepcopy(document)
    result['base']=dict(frame='world',start_delay=0.,lead_in=0.,waypoints=[dict(time=k*.05,position=pose[:3],quaternion_xyzw=pose[3:])for k in range(41)])
    return result


def relevant_loaded_path(path):
    # Graphics/DDS shared-memory files commonly appear as '/dev/zero
    # (deleted)'. They are not executable/codegen identity artifacts.
    relevant=any(x in path for x in('/ocs2-','/candidate-','/clock-v1/','/install/','libmujoco'))
    if relevant and '(deleted)'in path:
        raise RuntimeError('LOADED_EXECUTOR_ARTIFACT_DELETED:'+path)
    return relevant


def fixed_goal_poses(initial, goal, *, renewed):
    """Predetermined 1mm ramp then hold; renew the planned suffix, no reanchor."""
    import numpy as np
    initial=np.asarray(initial,dtype=float);goal=np.asarray(goal,dtype=float)
    if initial.shape!=(4,4)or goal.shape!=(4,4)or not np.isfinite([initial,goal]).all():
        raise ValueError('FINITE_ACTUAL_START_AND_PREDECLARED_GOAL_REQUIRED')
    result=np.repeat(goal[None],40,axis=0)
    if not renewed:
        weight=np.minimum((np.arange(40)+1)/8,1.)[:,None]
        result[:,:3,3]=initial[:3,3]+weight*(goal[:3,3]-initial[:3,3])
    return result


def ttl_pause_points(source):
    at=round(deadline(source)/DT)
    if abs(at*DT-deadline(source))>1e-8 or not TASK_START<at-1<at+1<=2706:
        raise ValueError('ACTUAL_TTL_BOUNDARIES_OUTSIDE_FINITE_DIAGNOSTIC')
    return [at-1,at,at+1,2706]


def dynamic_limit(phase, previous):
    """Versioned finite diagnostic, never accept arbitrary metadata limits."""
    table = {
        'g2-hold-short': ('g2-reference-qualification', TASK_START+25),
        'g2-hold-renew': ('g2-hold-short', TASK_START+400),
        'g2-eef-x': ('g2-hold-renew', TASK_START+200),
        'g2-eef-y': ('g2-eef-x', TASK_START+200),
        'g2-standing-hold-short-v2': ('g2-hold-short', TASK_START+25),
        'g2-standing-hold-renew-v2': ('g2-standing-hold-short-v2', TASK_START+400),
        'g2-standing-eef-x-v2': ('g2-standing-hold-renew-v2', TASK_START+200),
        'g2-standing-eef-y-v2': ('g2-standing-eef-x-v2', TASK_START+200),
        'g2-standing-ttl-v2': ('g2-standing-eef-y-v2',2706),
    }
    for batch,count,first in(('a',8,'g2-standing-eef-y-v2'),('b',4,'collect-target-a-08')):
        for index in range(1,count+1):
            name=f'collect-target-{batch}-{index:02d}'
            table[name]=(first if index==1 else f'collect-target-{batch}-{index-1:02d}',TASK_START+600)
    for index in(1,2):
        table[f'collect-target-parallel-{index:02d}']=('g2-standing-eef-y-v2',TASK_START+600)
    if phase not in table:
        raise PermissionError('UNKNOWN_FINITE_DYNAMIC_PHASE')
    parent, limit = table[phase]
    if (previous is None or previous.get('phase') != parent
            or previous.get('status') != 'NATIVE_STAGE_PASSED'
            or not previous.get('shutdown_clean')
            or not previous.get('owned_process_exits')
            or any(p['exit_code'] != 0 for p in previous['owned_process_exits'])):
        raise PermissionError('QUALIFIED_FINITE_DYNAMIC_PREDECESSOR_REQUIRED')
    return limit
