"""Static-only cube-layout candidates with an unchanged robot and home posture.

The saved robot sweep is a hypothetical geometric check, never new transitions,
controller restoration, TRAIN data, or permission to execute a candidate.
"""
from pathlib import Path
import struct
import time

import numpy as np

BASE_SCENE = '40408bbcbef2469396187c741750905a6ba9ed7161dc031d14bfab5fed62ddd2'
LAYOUTS = {
    'TRAIN01': (-.03,-.02), 'TRAIN02': (-.03,.02),
    'TRAIN03': (0.,-.04), 'TRAIN04': (0.,.04),
    'TRAIN05': (.03,-.02), 'TRAIN06': (.03,.02),
    'DEV01': (.015,.03), 'DEV02': (.015,-.03),
    'DEV03': (-.015,.03), 'DEV04': (-.015,-.03),
}


def need(ok, reason):
    if not ok:
        raise ValueError('STABLE_LAYOUT_STATIC_' + reason)


def arrays(model):
    return {k: v.copy() for k in dir(model)
            if isinstance((v := getattr(model,k)),np.ndarray)}


def move_cube_home_only(model, offset_xy):
    """Before any simulation: modify only the free cube's home XY values."""
    delta = np.asarray(offset_xy,float)
    need(delta.shape == (2,) and np.isfinite(delta).all()
         and 0 < np.linalg.norm(delta) <= .04 + 1e-12, 'FINITE_PREREGISTERED_SMALL_XY')
    need((model.nq,model.nv,model.nu,model.nkey) == (34,32,20,1)
         and model.key(0).name == 'home' and model.opt.timestep == .002,
         'ORIGINAL_ABI_HOME_AND_DT')
    cube = model.body('task_cube').id
    joint = int(model.body_jntadr[cube]); qa = int(model.jnt_qposadr[joint])
    need(model.joint(joint).name == 'task_cube_free' and qa == 27,
         'ONLY_ORIGINAL_CUBE_FREE_JOINT')
    before = arrays(model)
    model.key_qpos[0,qa:qa+2] += delta
    verify_only_cube_home_change(before,model,delta)
    return dict(cube_qpos_address=qa,offset_xy_m=delta.tolist(),
        original_home_cube_pose=before['key_qpos'][0,qa:qa+7].tolist(),
        candidate_home_cube_pose=model.key_qpos[0,qa:qa+7].tolist(),
        changed_model_arrays=['key_qpos'],robot_startup_joints_changed=False,
        robot_dynamics_changed=False,physics_dt_s=.002)


def verify_only_cube_home_change(before, candidate, offset_xy):
    after = arrays(candidate)
    need(set(before) == set(after), 'EXACT_MODEL_ARRAY_MEMBERS')
    expected = before['key_qpos'].copy()
    expected[0,27:29] += np.asarray(offset_xy,float)
    for name, value in before.items():
        wanted = expected if name == 'key_qpos' else value
        need(wanted.shape == after[name].shape and wanted.dtype == after[name].dtype
             and np.array_equal(wanted,after[name]), 'ONLY_HOME_CUBE_XY_CHANGED_' + name)
    need(candidate.opt.timestep == .002, 'ORIGINAL_DT')


def sweep(model, journal, offset_xy, *, expected_ticks, finite_check):
    """FK/collision on EVERY saved pose, with only hypothetical cube XY shifted.

    Caller must pin the original CLOSED0 journal before this function. No
    hypothetical packet may be exported as a policy input or observed transition.
    """
    import mujoco
    from scripts.pi0294.task_state import ActualState, parse_state
    from .native_replay import forbid_integration
    need(mujoco.__version__ == '3.2.7' and mujoco.get_mjcb_control() is None,
         'ORIGINAL_FORWARD_ONLY_ENGINE')
    state = ActualState(model); robot = set(range(1,model.nbody)) - {model.body('task_cube').id}
    cube = model.body('task_cube').id; count = 0; minimum_TCP = float('inf')
    start,cpu = time.monotonic(),time.process_time()
    with forbid_integration(mujoco) as integrations, Path(journal).open('rb') as stream:
        while header := stream.read(4):
            if count % 128 == 0:
                finite_check()
            need(len(header) == 4, 'COMPLETE_JOURNAL_HEADER')
            length, = struct.unpack('<I',header)
            need(40 <= length <= 2*1024**2, 'BOUNDED_ORIGINAL_PACKET')
            raw = stream.read(length); need(len(raw) == length, 'COMPLETE_PACKET')
            packet = parse_state(raw); count += 1
            need(packet['step'] == count and abs(packet['time_s']-count*.002) <= 1e-8,
                 'COMPLETE_CONTIGUOUS_ORIGINAL_CLOCK')
            # This is an isolated FK copy. The original journal is never changed.
            hypothetical = dict(packet,qpos=packet['qpos'].copy())
            hypothetical['qpos'][27:29] += np.asarray(offset_xy,float)
            state.update(hypothetical)
            for contact in state.data.contact:
                a,b = (int(model.geom_bodyid[int(i)]) for i in (contact.geom1,contact.geom2))
                need(not ((a == cube and b in robot) or (b == cube and a in robot)),
                     'HYPOTHETICAL_ROBOT_CUBE_CONTACT_AT_' + str(count))
            distance = np.linalg.norm(state.pose('tool')[:3,3]-state.pose('cube')[:3,3])
            minimum_TCP = min(minimum_TCP,float(distance))
        need(count == expected_ticks and not integrations, 'FULL_SAVED_SWEEP_ZERO_INTEGRATIONS')
    return dict(saved_pose_FK_calls=count,contact_free_at_saved_poses=True,
        minimum_hypothetical_TCP_cube_distance_m=minimum_TCP,
        static_hypothetical_geometry_not_new_data=True,new_transitions=0,
        integrations=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)


def compile_candidate(base_path, output, layout):
    """Create a fresh candidate; sweep and new native admission remain required."""
    import mujoco
    from .storage import digest
    base_path,output = Path(base_path),Path(output)
    need(layout in LAYOUTS and digest(base_path) == BASE_SCENE
         and not base_path.is_symlink(), 'EXACT_FROZEN_T06_AND_LAYOUT')
    need(output.is_absolute() and output.parent.resolve() == output.parent
         and output.name == 'SCENE.private.mjb' and not output.exists(), 'FRESH_SCENE_FILE')
    need(mujoco.__version__ == '3.2.7', 'EXACT_NATIVE_MUJOCO_VERSION')
    model = mujoco.MjModel.from_binary_path(str(base_path)); original = arrays(model)
    changes = move_cube_home_only(model,LAYOUTS[layout])
    mujoco.mj_saveModel(model,str(output),None)
    reloaded = mujoco.MjModel.from_binary_path(str(output))
    verify_only_cube_home_change(original,reloaded,LAYOUTS[layout])
    need(digest(base_path) == BASE_SCENE, 'FROZEN_INPUT_UNCHANGED')
    return dict(schema='WM02_STABLE_LAYOUT_COMPILED_NOT_QUALIFIED_V1',
        layout=layout,split='TRAIN' if layout.startswith('TRAIN') else 'DEV',
        parent='STABLE_TASK_LAYOUT_' + layout,base_scene_sha256=BASE_SCENE,
        scene_sha256=digest(output),changes=changes,static_sweep_completed=False,
        native_execution_authorized=False,native_integrations=0,model_updates=0)
