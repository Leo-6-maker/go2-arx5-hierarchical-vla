"""Recover an episode-fixed mount frame from already saved native evidence.

Only robot qpos and its calibrated MJB establish the mount. mj_forward is used
on an independent read-only model; no integration or image rendering occurs.
"""
import json
from pathlib import Path
import numpy as np

from .codec_v1 import rigid
from ..world_model.storage import digest
from scripts.pi0291_arx5_data.native_dataset import matrix_from10
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.task_state import read_state


def pinned(pin):
    p = Path(pin['path'])
    if digest(p) != pin['sha256']:
        raise ValueError('ORIGINAL_CALIBRATION_EVIDENCE_CHANGED')
    return json.loads(p.read_bytes())


def episode_frame(dataset_root, episode, T_policy_from_world):
    import mujoco
    if mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None:
        raise ValueError('UNCHANGED_CALLBACK_FREE_NATIVE_MODEL_REQUIRED')
    spec = pinned(episode['proof']['spec'])
    capture_index = pinned(episode['proof']['capture_index'])
    ready = pinned(episode['proof']['start_readiness'])
    first = capture_index['records'][0]
    body = unseal(pinned(first))
    packet_path = Path(first['path']).parent/'STATE.private.bin'
    packet_sha = digest(packet_path)
    if packet_sha != body['files']['STATE.private.bin']:
        raise ValueError('ORIGINAL_POST_HOLD_PACKET_HASH')
    packet = read_state(packet_path)
    scene_sha = digest(spec['model_path'])
    if (packet['step'] != ready['native_step'] or not ready['passes']
            or packet['step'] != first['native_step'] or packet['time_s'] != body['native_time_s']
            or body['world']['scene_sha256'] != scene_sha
            or body['world']['execution_spec_sha256'] != episode['proof']['spec']['sha256']):
        raise ValueError('ACTUAL_EPISODE_START_AND_ORIGINAL_SCENE_REQUIRED')
    policy_from_world = rigid(T_policy_from_world)
    if not np.array_equal(policy_from_world, spec['task_phase_source_from_target']):
        raise ValueError('ORIGINAL_POLICY_TARGET_TRANSFORM_CHANGED')
    model = mujoco.MjModel.from_binary_path(spec['model_path'])
    data = mujoco.MjData(model)
    mount = model.body('x5_base_link').id
    if model.body_parentid[mount] != model.body('base_link').id or model.body_jntnum[mount] != 0:
        raise ValueError('CALIBRATED_FIXED_ARX_MOUNT_TO_BASE_REQUIRED')
    for key in ('qpos', 'qvel', 'ctrl'):
        getattr(data, key)[:] = packet[key]
    data.time = packet['time_s']
    mujoco.mj_forward(model, data)
    world_from_mount = np.eye(4)
    world_from_mount[:3, :3] = data.xmat[mount].reshape(3, 3)
    world_from_mount[:3, 3] = data.xpos[mount]
    world_from_tcp = np.eye(4)
    tcp = model.site('x5_ee').id
    world_from_tcp[:3, :3] = data.site_xmat[tcp].reshape(3, 3)
    world_from_tcp[:3, 3] = data.site_xpos[tcp]
    with np.load(Path(dataset_root)/episode['file'], allow_pickle=False) as z:
        first_state = z['actual_state10'][0]
    if not np.allclose(matrix_from10(first_state), policy_from_world @ world_from_tcp, rtol=0, atol=1e-10):
        raise ValueError('CANONICAL_TCP_AND_POLICY_FRAME_MUST_MATCH_SAVED_NATIVE_FK')
    return dict(T_policy_from_mount=rigid(policy_from_world @ world_from_mount).tolist(),
                mount_body='x5_base_link', native_step=packet['step'], native_time_s=packet['time_s'],
                source_spec=episode['proof']['spec'], first_capture=first, scene_sha256=scene_sha,
                packet_sha256=packet_sha, fixed_for_episode=True, object_truth_used=False,
                additional_TCP_offset_m=0., native_integrations=0)
