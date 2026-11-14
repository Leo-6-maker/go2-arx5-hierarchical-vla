"""Audit four fixed episodes from each V4.1 source task; no action admission."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image

from scripts.sla.audit_robotwin_source_v4 import canonical_probe
from go2arx5_vla.sla_adapter.codec_v1 import pad32, roundtrip_error
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec, EPSILON

TASKS = ('move_pillbottle_pad', 'move_stapler_pad', 'place_mouse_pad',
         'place_empty_cup', 'place_object_stand')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(descriptor_path, binding):
    began = time.perf_counter()
    rows = json.loads(Path(descriptor_path).read_bytes())
    if [(r['task'], r['episode']) for r in rows] != [(t, i) for t in TASKS for i in range(4)]:
        raise ValueError('FIXED_FIVE_TASKS_FOUR_EPISODES_EACH_NOT_TWENTY_PER_TASK')
    episodes, arrays, probes = [], [], []
    for number, row in enumerate(rows):
        path = Path(row['file'])
        if sha(path) != row['sha256']:
            raise ValueError('PINNED_SOURCE_EPISODE_BYTES_CHANGED')
        metadata = json.loads(Path(row['scene_info']).read_bytes())[f"episode_{row['episode']}"]['info']
        with h5py.File(path, 'r') as f:
            fields, attrs = [], []
            def inspect(name, value):
                if isinstance(value, h5py.Dataset):
                    fields.append(name)
                attrs.extend(name+'/'+key for key in value.attrs)
            f.visititems(inspect)
            attrs.extend('/'+key for key in f.attrs)
            version = f['data_format_version'][()].decode()
            if version != 'v1.0' or f.attrs.get('source_format') != 'RoboTwin':
                raise ValueError('EXPLICIT_DEFAULT_ARCHIVE_XPOLICYLAB_V1_SCHEMA_REQUIRED')
            frequency = int(f['additional_info/frequency'][()])
            if frequency <= 0:
                raise ValueError('POSITIVE_ORIGINAL_SERIALIZED_FREQUENCY_REQUIRED')
            joints = {arm: np.asarray(f[f'action/{arm}_arm_joint_states']) for arm in ('left', 'right')}
            spans = {arm: float(np.ptp(a, axis=0).max()) for arm, a in joints.items()}
            arm = metadata.get('{a}')
            if arm is None:
                moving = [a for a, span in spans.items() if span > 1e-8]
                if len(moving) != 1:
                    raise ValueError('EXACTLY_ONE_ACTIVE_ARM_REQUIRED_FOR_METADATA_FREE_EPISODE')
                arm = moving[0]
            if arm not in ('left', 'right'):
                raise ValueError('EXPLICIT_SINGLE_ACTIVE_ARM_REQUIRED')
            inactive = 'right' if arm == 'left' else 'left'
            if spans[arm] <= 1e-8 or spans[inactive] > 1e-8:
                raise ValueError('DUAL_ARM_DEPENDENCY_EXCLUDED_FROM_THIS_FIXED_PILOT')
            pose = np.asarray(f[f'action/{arm}_ee_poses'])
            grip = np.asarray(f[f'action/{arm}_ee_joint_states'])[:, 0]
            current_pose = np.asarray(f[f'state/{arm}_ee_poses'])
            current_grip = np.asarray(f[f'state/{arm}_ee_joint_states'])[:, 0]
            count = len(pose)
            cameras = ['cam_head', 'cam_'+arm+'_wrist']
            if (pose.shape != (count, 7) or grip.shape != (count,) or count < 43
                    or current_pose.shape != (count, 7) or current_grip.shape != (count,)
                    or any(not np.isfinite(a).all() for a in (pose, grip, current_pose, current_grip))
                    or any(((a < 0) | (a > 1)).any() for a in (grip, current_grip))
                    or any(not np.allclose(np.linalg.norm(a[:, 3:], axis=1), 1, rtol=0, atol=1e-6)
                           for a in (pose, current_pose))
                    or any(not np.isfinite(a).all() or len(a) != count for a in joints.values())
                    or any(len(f[f'vision/{c}/colors']) != count for c in cameras)):
                raise ValueError('ALIGNED_FINITE_SOURCE_EEF_JOINT_AND_RGB_ARRAYS_REQUIRED')
            for side in ('left', 'right'):
                for suffix in ('arm_joint_states', 'ee_joint_states', 'ee_poses'):
                    before = np.asarray(f[f'state/{side}_{suffix}'])
                    after = np.asarray(f[f'action/{side}_{suffix}'])
                    if not np.array_equal(after[:-1], before[1:]):
                        raise ValueError('RECORDED_STATE_ACTION_ONE_ROW_SHIFT_MUST_BE_EXPLICIT')
            clocks = [s for s in fields+attrs if any(k in s.lower() for k in ('time', 'stamp', 'tick', 'frequency'))]
            anchors = np.unique(np.linspace(0, count-40, 4, dtype=int)).tolist()
            if len(anchors) != 4:
                raise ValueError('FOUR_DISTINCT_FIXED_INDEX_PROBES_REQUIRED')
            arrays.append((pose, grip, anchors, current_pose, current_grip))
            probes.extend((number, a) for a in anchors[:3])
            episodes.append(dict(task=row['task'], episode=row['episode'], file_sha256=row['sha256'],
                frames=count, active_arm=arm, arm_from_scene_metadata='{a}' in metadata,
                inactive_joint_target_max_span_rad=spans[inactive],
                gripper_min=float(grip.min()), gripper_max=float(grip.max()),
                close_transitions=int((np.diff(grip)<-1e-8).sum()),
                open_transitions=int((np.diff(grip)>1e-8).sum()), physical_clock_fields=clocks,
                data_format_version=version, declared_frequency_hz=frequency,
                action_state_adjacent_row_shift_exact=True,
                hdf5_attribute_names=attrs, camera_order=cameras))
    probes.extend((number, arrays[number][2][-1]) for number in range(4))
    if len(probes) != len(set(probes)) or len(probes) != 64:
        raise ValueError('EXACT_64_FIXED_INDEX_PROBES_OVER_TWENTY_EPISODES')
    codec = GripLogitCodec(np.eye(4))
    maximum = dict(xyz_m=0., so3_rad=0., gripper=0.)
    records, sizes = [], set()
    for number, anchor in probes:
        pose, grip, _, current_pose, current_grip = arrays[number]
        value = canonical_probe(pose[anchor:anchor+40], grip[anchor:anchor+40])
        state = canonical_probe(np.repeat(current_pose[anchor:anchor+1], 40, axis=0),
                                np.repeat(current_grip[anchor], 40))[0]
        _, latent = codec.encode(state, value)
        error = roundtrip_error(value, codec.decode(state, pad32(latent)))
        if error['xyz_m'] > 1e-5 or error['so3_rad'] > 1e-5 or error['gripper'] > EPSILON+1e-12:
            raise ValueError('AUXILIARY_INDEX_CODEC_NUMERIC_FAILURE')
        for k, v in error.items():
            maximum[k] = max(maximum[k], v)
        hashes = []
        with h5py.File(rows[number]['file'], 'r') as f:
            for camera in episodes[number]['camera_order']:
                raw = bytes(f[f'vision/{camera}/colors'][anchor])
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.mode != 'RGB':
                        raise ValueError('ACTUAL_ORIGINAL_SOURCE_RGB_REQUIRED')
                    sizes.add(image.size)
                hashes.append(hashlib.sha256(raw).hexdigest())
        records.append(dict(task=rows[number]['task'], episode=rows[number]['episode'],
                            row_anchor=anchor, RGB_sha256=hashes, error=error))
    return dict(schema='SLA_V4_1_FIXED_FIVE_TASK_SOURCE20_AUDIT_ACTUAL', source_binding=binding,
        episodes_checked=20, episodes_per_task=dict(Counter(e['task'] for e in episodes)),
        default_embodiment='aloha_agilex', episodes=episodes,
        original_frame_rows=sum(e['frames'] for e in episodes), auxiliary_index_codec_probes=64,
        actual_RGB_decodes=128, original_RGB_sizes=sorted(sizes), max_auxiliary_roundtrip_error=maximum,
        numerical_probe_semantics='ACHIEVED_EEF_INDEX_ALGEBRA_ONLY_NOT_PHYSICAL_COMMAND_H40',
        classification='S_aux', S_cmd_episodes_admitted=0, desired_command_windows_converted=0,
        source_physical_replay_episodes=0, training_windows_admitted=0,
        declared_frequency_hz=sorted({e['declared_frequency_hz'] for e in episodes}),
        physical_timestamps_present=any(any('frequency' not in k.lower()
            for k in e['physical_clock_fields']) for e in episodes),
        frequency_bound_to_native_capture_clock=False,
        isolation_reasons=['ARCHIVE_COLLECTION_REVISION_UNBOUND', 'PHYSICAL_CLOCK_NOT_ESTABLISHED',
            'DESIRED_JOINT_COMMAND_FK_AND_TOOL_FRAME_UNVERIFIED',
            'RECORDED_ENDPOSE_IS_ACHIEVED_EEF', 'MEASURED_CURRENT_GRIPPER_NOT_ESTABLISHED'],
        records=records, optimizer_updates=0, optimization_visits=0, model_calls=0,
        physical_requests=0, wall_s=time.perf_counter()-began)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.episodes, json.loads(args.binding.read_bytes()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('episodes', 'records', 'source_binding')}))
