"""Actual parser/PNG validation with synthetic FK; no renderer/model/native run."""
import copy
import hashlib
import io
import json
import struct
import subprocess
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.pi0292.relay_protocol import sealed, unseal
from scripts.pi0294.task_state import ActualState
from go2arx5_vla.world_model.fusion_observation import (
    CAMERAS, TargetShadowBinding, shadow_inference_payload, transform_hash)
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows


def png(color):
    out = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), color, dtype=np.uint8)).save(out, format='PNG')
    return out.getvalue()


def fixture(step=3703, mapped=True):
    yaw, angle = .4, -.3
    transform = np.eye(4)
    transform[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    transform[:3, 3] = [.2, -.1, .04]
    tcp = np.eye(4)
    tcp[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    tcp[:3, 3] = [.26, -.02, .71]
    names = dict(body_rgb=0, wrist_rgb=1)
    model = SimpleNamespace(camera=lambda name: SimpleNamespace(id=names[name]),
        body=lambda index: SimpleNamespace(name=('base_link', 'x5_link6')[index]),
        cam_bodyid=np.array([0, 1]), cam_pos=np.array([[.25, -.24, .28], [.1124, .1, .055]]),
        cam_quat=np.array([[1., 0, 0, 0], [1., 0, 0, 0]]), cam_fovy=np.array([75., 75.]))
    data = SimpleNamespace(cam_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
                           cam_xpos=np.array([[.25, -.24, .68], [.3, .1, .72]]))
    reader = object.__new__(ActualState)
    reader.model, reader.data = model, data
    measured = dict(native_step=step, time_s=step * .002, tcp=tcp, gripper_fraction=.6,
                    safety_valid=True, gripper_state_valid=True, native_abort=False,
                    object_pose='PRIVILEGED_MUST_NOT_ENTER_INPUT', teacher_stage='FORBIDDEN')
    values = np.zeros(86, dtype='<f8'); values[3] = 1.
    raw = struct.pack('<8sQdIII', b'GRTSTATE', step, step * .002, 34, 32, 20) + values.tobytes() + struct.pack('<I', 0)
    calibration = {}
    for name, cid in names.items():
        optical = np.diag([1., -1., -1., 1.]); optical[:3, 3] = data.cam_xpos[cid]
        focal = 128 / np.tan(np.deg2rad(75.) / 2)
        calibration[name] = dict(timestamp_s=step * .002, T_world_optical=optical.tolist(),
                                K=[[focal, 0, 128], [0, focal, 128], [0, 0, 1]])
    cal = json.dumps(calibration).encode()
    images = dict(body_rgb=png(20), wrist_rgb=png(80))
    binding = TargetShadowBinding('ACTUAL_TARGET_WORLD', '1' * 64, '2' * 64,
        transform if mapped else None, transform_hash(transform) if mapped else None)
    args = dict(actual_state=reader, native_state_bytes=raw, expected_state_sha256=hashlib.sha256(raw).hexdigest(),
        capture=dict(index=48, native_step=step, time_s=step * .002),
        calibration_bytes=cal, expected_calibration_sha256=hashlib.sha256(cal).hexdigest(),
        image_bytes=images, expected_image_sha256={k: hashlib.sha256(v).hexdigest() for k, v in images.items()},
        language='Original observed instruction', snapshot_uuid=str(uuid.uuid4()))
    return binding, args, measured, transform


class FusionTargetObservation(unittest.TestCase):
    def test_nonlegacy_native_steps_and_absolute_clock_with_explicit_nonunit_transform(self):
        for step in (3503, 3703, 4103):
            with self.subTest(step=step):
                binding, args, measured, transform = fixture(step)
                original = args['native_state_bytes']
                with patch.object(ActualState, 'update', return_value=measured) as update:
                    record = binding.bind(**args)
                body = unseal(record); payload = shadow_inference_payload(record)
                expected = transform @ measured['tcp']
                np.testing.assert_allclose(payload['state_eef10'][:3], expected[:3, 3], atol=1e-14)
                np.testing.assert_allclose(payload['state_eef10'][3:9], rotation_matrix_to_rot6d_rows(expected[:3, :3]), atol=1e-14)
                self.assertEqual(payload['state_eef10'][9], .6)
                self.assertEqual(body['native_time_s'], step * .002)
                self.assertEqual(body['native_step'], step)
                self.assertEqual(args['native_state_bytes'], original)
                self.assertEqual(update.call_count, 1)
                self.assertEqual(update.call_args.args[0]['step'], step)
                self.assertNotIn('PRIVILEGED', json.dumps(record))
                self.assertNotIn('teacher_stage', payload)
                self.assertEqual(set(payload), {'snapshot_uuid', 'language', 'state_eef10', 'images', 'frame', 'tcp'})

    def test_actual_target_cameras_and_role_mapping_do_not_claim_equivalence_or_rewrite_bytes(self):
        binding, args, measured, _ = fixture()
        with patch.object(ActualState, 'update', return_value=measured):
            record = binding.bind(**args)
        body = unseal(record); target = body['target_observation']; p = body['provenance']
        self.assertEqual(set(target['cameras']), set(CAMERAS))
        self.assertEqual(target['cameras']['body_rgb']['parent_body'], 'base_link')
        self.assertEqual(target['cameras']['wrist_rgb']['parent_body'], 'x5_link6')
        self.assertEqual(target['cameras']['body_rgb']['fovy_deg'], 75.)
        self.assertEqual(p['camera_role_mapping'], {'external_rgb': 'body_rgb', 'wrist_rgb': 'wrist_rgb'})
        self.assertFalse(p['camera_equivalence'])
        self.assertFalse(p['target_embodiment_qualified'])
        self.assertFalse(p['native_execution_authorized'])
        self.assertTrue(p['shadow_only'])
        self.assertEqual(p['embodiment_id'], 10)
        self.assertEqual(p['desired_outcome_provenance'], 'SEMANTICS_UNRESOLVED')
        self.assertFalse(p['source_outcomes_supplied'])
        payload = shadow_inference_payload(record)
        self.assertEqual(payload['images'][0]['sha256'], args['expected_image_sha256']['body_rgb'])
        self.assertEqual(payload['images'][1]['sha256'], args['expected_image_sha256']['wrist_rgb'])
        self.assertEqual(p['source_H40_command_offsets_s'][0], 0.)
        self.assertAlmostEqual(p['source_H40_command_offsets_s'][-1], 1.95)

    def test_missing_frame_relation_retains_target_input_without_faking_source_payload(self):
        binding, args, measured, _ = fixture(mapped=False)
        with patch.object(ActualState, 'update', return_value=measured):
            record = binding.bind(**args)
        body = unseal(record)
        self.assertIsNone(body['source_policy_view'])
        self.assertEqual(body['target_observation']['frame'], 'ACTUAL_TARGET_WORLD')
        self.assertEqual(body['provenance']['frame_relation'], 'SEMANTICS_UNRESOLVED')
        with self.assertRaisesRegex(PermissionError, 'UNRESOLVED_FRAME'):
            shadow_inference_payload(record)

    def test_tampered_original_bytes_fail_before_fk(self):
        for field in ('native_state_bytes', 'calibration_bytes'):
            binding, args, measured, _ = fixture(); args[field] += b'changed'
            with self.subTest(field=field), patch.object(ActualState, 'update') as update:
                with self.assertRaises(ValueError): binding.bind(**args)
                update.assert_not_called()
        for camera in CAMERAS:
            binding, args, measured, _ = fixture(); args['image_bytes'][camera] += b'changed'
            with self.subTest(camera=camera), patch.object(ActualState, 'update') as update:
                with self.assertRaises(ValueError): binding.bind(**args)
                update.assert_not_called()

    def test_changed_snapshot_clocks_or_original_png_shape_rejected(self):
        for fault in ('capture_step', 'capture_time', 'capture_bool', 'retimed_native', 'PNG_shape', 'PNG_camera'):
            binding, args, measured, _ = fixture()
            if fault == 'capture_step': args['capture']['native_step'] = 3700
            elif fault == 'capture_time': args['capture']['time_s'] = 0.
            elif fault == 'capture_bool': args['capture']['index'] = True
            elif fault == 'retimed_native':
                raw = bytearray(args['native_state_bytes']); struct.pack_into('<d', raw, 16, .0)
                args['native_state_bytes'] = bytes(raw); args['expected_state_sha256'] = hashlib.sha256(raw).hexdigest()
            elif fault == 'PNG_shape':
                out = io.BytesIO(); Image.new('RGB', (128, 256)).save(out, format='PNG')
                args['image_bytes']['body_rgb'] = out.getvalue()
                args['expected_image_sha256']['body_rgb'] = hashlib.sha256(out.getvalue()).hexdigest()
            else: args['image_bytes']['external_rgb'] = args['image_bytes'].pop('body_rgb')
            with self.subTest(fault=fault), patch.object(ActualState, 'update') as update:
                with self.assertRaises(ValueError): binding.bind(**args)
                update.assert_not_called()

    def test_self_consistent_calibration_cannot_override_actual_camera_fk_or_intrinsics(self):
        for fault in ('time', 'extrinsic', 'K', 'bool_K', 'nan', 'extra_camera'):
            binding, args, measured, _ = fixture(); cal = json.loads(args['calibration_bytes'])
            if fault == 'time': cal['body_rgb']['timestamp_s'] = 0.
            elif fault == 'extrinsic': cal['body_rgb']['T_world_optical'][0][3] += .001
            elif fault == 'K': cal['body_rgb']['K'][0][0] += 1.
            elif fault == 'bool_K': cal['body_rgb']['K'][2][2] = True
            elif fault == 'nan': cal['body_rgb']['K'][0][0] = float('nan')
            else: cal['external_rgb'] = cal.pop('body_rgb')
            args['calibration_bytes'] = json.dumps(cal).encode()
            args['expected_calibration_sha256'] = hashlib.sha256(args['calibration_bytes']).hexdigest()
            with self.subTest(fault=fault), patch.object(ActualState, 'update', return_value=measured):
                with self.assertRaises(ValueError): binding.bind(**args)

    def test_bad_measured_feedback_not_replaced_with_desired_or_clipped_state(self):
        for change in (dict(time_s=0.), dict(native_step=2500), dict(safety_valid=False),
                       dict(gripper_state_valid=False), dict(native_abort=True), dict(gripper_fraction=1.001)):
            binding, args, measured, _ = fixture(); measured.update(change)
            with self.subTest(change=change), patch.object(ActualState, 'update', return_value=measured):
                with self.assertRaises(ValueError): binding.bind(**args)

    def test_transform_bytes_and_full_sidecar_are_required_for_inference(self):
        binding, args, measured, transform = fixture()
        transform[0, 3] += .001
        with self.assertRaises(ValueError):
            TargetShadowBinding('ACTUAL_TARGET_WORLD', '1' * 64, '2' * 64, transform, binding.expected_transform_sha256)
        with patch.object(ActualState, 'update', return_value=measured):
            record = binding.bind(**args)
        for fault in ('unsealed', 'missing_sidecar', 'claim_qualified', 'transform', 'state', 'image_binding'):
            changed = copy.deepcopy(record)
            if fault == 'unsealed': changed['body']['native_step'] += 1
            else:
                body = changed['body']
                if fault == 'missing_sidecar': body['provenance'] = {}
                elif fault == 'claim_qualified': body['provenance']['target_embodiment_qualified'] = True
                elif fault == 'transform': body['source_policy_view']['T_source_from_target'][0][3] += .001
                elif fault == 'state': body['source_policy_view']['payload']['state_eef10'][0] += .001
                else: body['target_observation']['image_sha256']['body_rgb'] = '0' * 64
                changed = sealed(body)
            with self.subTest(fault=fault), self.assertRaises((ValueError, KeyError)):
                shadow_inference_payload(changed)

    def test_native_binding_import_stays_free_of_torch_ros_and_simulator(self):
        code = ('import sys; import go2arx5_vla.world_model.fusion_observation; '
                'assert not any(k in sys.modules for k in ("torch", "mujoco", "rclpy"))')
        completed = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
