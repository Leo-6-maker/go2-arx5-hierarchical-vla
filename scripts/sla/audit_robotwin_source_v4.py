"""Inspect a bounded legacy RoboTwin sample without admitting action training.

Endpose is explicitly treated as recorded achieved EEF. Algebraic codec
probes at row indices do not certify physical time or desired-command FK.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from go2arx5_vla.sla_adapter.codec_v1 import pad32, roundtrip_error
from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON, GripLogitCodec
from scripts.pi0291_arx5_data.native_dataset import pose10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_probe(pose7, gripper):
    if (pose7.shape != (40, 7) or gripper.shape != (40,)
            or not np.isfinite(pose7).all() or not np.isfinite(gripper).all()
            or not np.allclose(np.linalg.norm(pose7[:, 3:], axis=1), 1, rtol=0, atol=1e-6)
            or ((gripper < 0) | (gripper > 1)).any()):
        raise ValueError('FINITE_SOURCE_POSE_WXYZ_AND_ORIGINAL_GRIP_RANGE_REQUIRED')
    transforms = np.broadcast_to(np.eye(4), (40, 4, 4)).copy()
    transforms[:, :3, 3] = pose7[:, :3]
    transforms[:, :3, :3] = Rotation.from_quat(pose7[:, [4, 5, 6, 3]]).as_matrix()
    return pose10(transforms, gripper)


def audit(directory, files, *, source_binding):
    began = time.time()
    directory = Path(directory)
    descriptors = json.loads(Path(files).read_bytes())
    if len(descriptors) != 20:
        raise ValueError('THIS_BOUNDARY_AUDIT_REQUIRES_EXACT_FIRST_TWENTY_EPISODES')
    info = json.loads((directory / 'scene_info.json').read_bytes())
    episodes, candidates = [], []
    arrays = {}
    for number, row in enumerate(descriptors):
        path = directory / row['file']
        if path.resolve().parent != directory.resolve() or sha(path) != row['sha256']:
            raise ValueError('FIXED_SAFE_EPISODE_BYTES_REQUIRED')
        arm = info[f'episode_{number}']['info']['{a}']
        if arm not in ('left', 'right'):
            raise ValueError('EXPLICIT_SINGLE_ACTIVE_ARM_REQUIRED')
        other = 'right' if arm == 'left' else 'left'
        with h5py.File(path, 'r') as f:
            fields = []
            f.visititems(lambda name, value: fields.append(name) if isinstance(value, h5py.Dataset) else None)
            pose = np.asarray(f[f'endpose/{arm}_endpose'])
            grip = np.asarray(f[f'endpose/{arm}_gripper'])
            joints = np.asarray(f[f'joint_action/{arm}_arm'])
            inactive = np.asarray(f[f'joint_action/{other}_arm'])
            count = len(pose)
            if (pose.shape != (count, 7) or grip.shape != (count,) or count < 40
                    or joints.shape != (count, 6) or not np.isfinite(joints).all()
                    or not np.isfinite(inactive).all()):
                raise ValueError('ALIGNED_COMPLETE_LEGACY_SOURCE_ARRAYS_REQUIRED')
            cameras = [camera for camera in ('head_camera', 'left_camera', 'right_camera')
                       if f'observation/{camera}/rgb' in f]
            if any(len(f[f'observation/{camera}/rgb']) != count for camera in cameras):
                raise ValueError('SOURCE_IMAGE_ARRAY_LENGTH_MISMATCH')
            time_fields = [name for name in fields if any(word in name.lower()
                           for word in ('time', 'stamp', 'tick', 'frequency'))]
            episodes.append(dict(episode=number, file_sha256=row['sha256'], frames=count,
                active_arm=arm, cameras=cameras, time_fields=time_fields,
                inactive_joint_target_max_span_rad=float(np.ptp(inactive, axis=0).max()),
                source_gripper_min=float(grip.min()), source_gripper_max=float(grip.max()),
                close_transitions=int((np.diff(grip) < -1e-8).sum()),
                open_transitions=int((np.diff(grip) > 1e-8).sum())))
            # Fixed index-space probes only. No 20Hz clock is invented.
            anchors = np.unique(np.linspace(0, count - 40, 4, dtype=int)).tolist()
            if len(anchors) != 4:
                raise ValueError('FOUR_UNIQUE_INDEX_PROBES_PER_EPISODE_REQUIRED')
            candidates.extend((number, anchor) for anchor in anchors[:3])
            arrays[number] = (path, arm, pose, grip, anchors[-1])
    candidates.extend((number, arrays[number][-1]) for number in range(4))
    if len(set(candidates)) != len(candidates) or len(candidates) != 64:
        raise ValueError('EXACT_SIXTY_FOUR_UNIQUE_FIXED_PROBES_REQUIRED')
    codec = GripLogitCodec(np.eye(4))
    records, image_shapes = [], set()
    max_error = dict(xyz_m=0., so3_rad=0., gripper=0.)
    for number, anchor in candidates:
        path, arm, pose, grip, _ = arrays[number]
        physical = canonical_probe(pose[anchor:anchor+40], grip[anchor:anchor+40])
        _, latent = codec.encode(physical[0], physical)
        error = roundtrip_error(physical, codec.decode(physical[0], pad32(latent)))
        if error['xyz_m'] > 1e-5 or error['so3_rad'] > 1e-5 or error['gripper'] > EPSILON + 1e-12:
            raise ValueError('AUXILIARY_POSE_CODEC_NUMERIC_FAILURE')
        for key, value in error.items():
            max_error[key] = max(max_error[key], value)
        hashes = []
        with h5py.File(path, 'r') as f:
            for camera in ('head_camera', arm + '_camera'):
                raw = bytes(f[f'observation/{camera}/rgb'][anchor])
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.mode != 'RGB':
                        raise ValueError('ACTUAL_SOURCE_RGB_REQUIRED')
                    image_shapes.add(image.size)
                hashes.append(hashlib.sha256(raw).hexdigest())
        records.append(dict(episode=number, row_anchor=anchor, source_frame_indices=[anchor, anchor+39],
                            cameras=['head_camera', arm+'_camera'], RGB_sha256=hashes, error=error))
    return dict(schema='SLA_V4_LEGACY_ROBOTWIN_SOURCE_AUDIT_ACTUAL',
        source_binding=source_binding, episodes_checked=20, original_frame_rows=sum(x['frames'] for x in episodes),
        episodes=episodes, auxiliary_index_codec_probes=64, actual_RGB_decodes=128,
        original_RGB_sizes=sorted(image_shapes), max_auxiliary_roundtrip_error=max_error,
        numerical_probe_frame='SOURCE_LOCAL_IDENTITY_FOR_ALGEBRA_ONLY_NOT_TCP_CALIBRATION',
        numerical_probe_semantics='RECORDED_ACHIEVED_EEF_PLUS_RECORDED_GRIPPER_FIELD_NOT_DESIRED_EEF',
        physical_timestamps_present=any(x['time_fields'] for x in episodes),
        classification='S_aux', S_cmd_episodes_admitted=0, desired_command_windows_converted=0,
        training_windows_admitted=0, source_physical_replay_episodes=0,
        isolation_reasons=['ARCHIVE_COLLECTION_CODE_REVISION_UNBOUND',
            'NO_PHYSICAL_TIMESTAMP_OR_TICK_RECORDS', 'COMMAND_JOINT_FK_AND_TOOL_FRAME_NOT_VERIFIED',
            'RECORDED_ENDPOSE_IS_ACHIEVED_EEF_NOT_DESIRED_COMMAND',
            'MEASURED_CURRENT_GRIPPER_STATE_NOT_ESTABLISHED'],
        records=records, optimizer_updates=0, model_calls=0, physical_requests=0,
        wall_s=time.time()-began)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--files', type=Path, required=True)
    p.add_argument('--source-binding', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    binding = json.loads(a.source_binding.read_bytes())
    result = audit(a.directory, a.files, source_binding=binding)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('episodes', 'records', 'source_binding')}))
