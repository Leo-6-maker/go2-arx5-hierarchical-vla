"""CPU-only draft object-task boundaries; never a physical-task PASS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
from go2arx5_vla.adapters.groot_to_wbc import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
from go2arx5_vla.data.readiness import StateV1
from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix, rotation_matrix_to_rot6d_rows


def width_to_unit(width_m):
    value = np.asarray(width_m, dtype=float)
    if not np.isfinite(value).all() or np.any((value < 0) | (value > .08)):
        raise ValueError('draft tool width outside 0..80mm; no clipping')
    return value / .08


def check_window(packet):
    inputs = packet['inputs']
    if set(inputs) != {'body_rgb', 'wrist_rgb', 'state_v1', 'language'}:
        raise ValueError('policy input allowlist: no privileged fields')
    for key in ('body_rgb', 'wrist_rgb'):
        image = np.asarray(inputs[key])
        if image.shape != (96, 96, 3) or image.dtype != np.uint8:
            raise ValueError('RGB must be uint8 96x96x3')
    if not isinstance(inputs['language'], str) or not inputs['language'].strip():
        raise ValueError('missing task instruction')
    state = StateV1.from_array(inputs['state_v1']).as_array()
    labels = np.asarray(packet['commanded_labels'], dtype=float)
    times = np.asarray(packet['command_timestamps'], dtype=float)
    tick = float(packet['observation_timestamp'])
    actual = np.asarray(packet['actual_tcp_world'], dtype=float)
    anchor = np.asarray(packet['anchor_world_m'], dtype=float)
    if labels.shape != (40, 10) or not np.isfinite(labels).all():
        raise ValueError('finite commanded H40x10 required')
    if actual.shape != (9,) or anchor.shape != (3,) or not np.isfinite(actual).all() or not np.isfinite(anchor).all():
        raise ValueError('measured TCP and explicit anchor required')
    if not np.isfinite(tick) or times.shape != (40,) or not np.allclose(times, tick + .05 * np.arange(40), atol=1e-9, rtol=0):
        raise ValueError('H40 must use actual observation clock')
    stamps = np.asarray(packet['sensor_timestamps'], dtype=float)
    if stamps.shape != (3,) or not np.isfinite(stamps).all() or not np.allclose(stamps, tick, atol=1e-9, rtol=0):
        raise ValueError('draft requires same-tick RGB/RGB/state')
    if not packet['clock_id'] or not packet['source_frame'] or not packet['tcp_semantic_id']:
        raise ValueError('clock/frame/TCP identity missing')
    if not np.allclose(state[:3] + anchor, actual[:3], atol=1e-6, rtol=0) or not np.allclose(state[3:9], actual[3:9], atol=1e-6, rtol=0):
        raise ValueError('RGB/state measured TCP pairing inconsistent')
    if abs(state[9] - float(width_to_unit(packet['measured_gripper_width_m']))) > 1e-6:
        raise ValueError('state must contain measured opening, not command')
    if np.any((labels[:, 9] < 0) | (labels[:, 9] > 1)):
        raise ValueError('gripper commands outside declared new-task range')
    # ponytail: reuse canonical quaternion boundary; adapter convenience input is column-packed.
    chunk = TrajectoryChunkV1(
        chunk_id=packet['episode_id'], source='SYNTHETIC_CPU_FIXTURE',
        observation_timestamp=tick, point_timestamps=times,
        eef_position=labels[:, :3] + anchor,
        eef_orientation=matrix_to_quat(rot6d_rows_to_matrix(labels[:, 3:9])),
        source_frame=packet['source_frame'], tcp_semantic_id=packet['tcp_semantic_id'],
        validity_start=times[0], validity_end=times[-1], gripper=labels[:, 9])
    return chunk


def demo():
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    row = rotation_matrix_to_rot6d_rows(rotation)
    labels = np.tile(np.r_[.1, .2, .3, row, 1.], (40, 1))
    labels[8:24, 9] = .2
    anchor = np.array([.4, -.1, .2])
    packet = dict(
        inputs=dict(body_rgb=np.zeros((96, 96, 3), np.uint8), wrist_rgb=np.zeros((96, 96, 3), np.uint8),
                    state_v1=np.r_[.1, .2, .3, row, .75], language='Move the red object into the right bin, release it, and withdraw.'),
        commanded_labels=labels, command_timestamps=2 + .05 * np.arange(40), observation_timestamp=2.,
        actual_tcp_world=np.r_[labels[0, :3] + anchor, row], anchor_world_m=anchor,
        sensor_timestamps=[2., 2., 2.], measured_gripper_width_m=.06,
        clock_id='fixture_only', source_frame='OBJECT_TASK_WORLD', tcp_semantic_id='draft_tool_palm_face', episode_id='fixture_only')
    chunk = check_window(packet)
    adapter = GR00TToWBCAdapter(ReferenceAdapterConfig(chunk.source_frame, chunk.source_frame, chunk.tcp_semantic_id, stale_timeout_s=.5))
    for offset, expected in ((0., 1.), (.4, .2), (1.2, 1.)):
        preview, grip = adapter.adapt(chunk, controller_timestamp=2 + offset)
        assert np.allclose(rot6d_rows_to_matrix(preview.eef_rot6d), rotation, atol=1e-12)
        assert np.allclose(preview.eef_position, labels[0, :3] + anchor)
        assert grip.normalized_command == expected
    assert adapter.adapt(chunk, controller_timestamp=3.2)[1].stale
    assert np.allclose(width_to_unit([0, .04, .08]) * .08, [0, .04, .08])
    failures = 0
    for change in (
        {'inputs': dict(packet['inputs'], teacher_phase='CLOSE')},
        {'sensor_timestamps': [2, 2.02, 2]},
        {'command_timestamps': packet['command_timestamps'] + .01},
        {'measured_gripper_width_m': .08},
        {'actual_tcp_world': packet['actual_tcp_world'] + .01},
        {'commanded_labels': np.full((40, 10), np.nan)},
    ):
        try:
            check_window(dict(packet, **change))
        except ValueError:
            failures += 1
        else:
            raise AssertionError('negative fixture accepted')
    try:
        width_to_unit(.081)
    except ValueError:
        failures += 1
    else:
        raise AssertionError('width silently clipped')
    return dict(status='PASS_CPU_DRAFT_FORMAT_ROTATION_CLOCK_GRIPPER_LEAKAGE', negative_cases=failures,
                evidence_domain='SYNTHETIC_CPU_FIXTURE_NOT_PHYSICS', launch_allowed=False,
                official_processor_roundtrip='NOT_RUN', teacher_physics_success='NOT_RUN',
                model_queries=0, optimizer_updates=0, physics_steps=0, renders=0, teacher_calls=0,
                selected_checkpoint='NONE', deployment_selected_checkpoint='NONE',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))
