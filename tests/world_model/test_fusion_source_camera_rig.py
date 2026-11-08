"""Pure geometry/schema fixtures; no renderer or native qualification evidence."""
import copy
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from go2arx5_vla.world_model import fusion_source_camera_rig as rig


FRAME = np.array([
    [0.9994979245410603, -0.004770103779283492, -0.031323233358717985, 0.1184559088991858],
    [0.006100452971270548, 0.9990772300043939, 0.04251438533354902, -0.015560812187311514],
    [0.03109153118865576, -0.042684125816034195, 0.9986047176393984, 0.06448618628506314],
    [0., 0., 0., 1.]])
WRIST_Q = [0.6804555622516456, -0.19230243836414793, -0.5039547070762974, -0.4960137631312696]
BODY_AXES = '0.8448190755542285 0.5350520811843448 0.0 -0.25584293802661273 0.40396253372623064 0.8782702672909946'
WRIST_AXES = '0.0 -0.48120721016036144 0.8766068793305708 0.8688540861438091 0.4339802379671197 0.23823041377038456'


def K(fov):
    focal = 128 / math.tan(math.radians(fov)/2)
    return [[focal, 0., 128.], [0., focal, 128.], [0., 0., 1.]]


def fixture():
    body = dict(attributes=dict(name='body_rgb', pos='1.0 -0.9 1.1', xyaxes=BODY_AXES, fovy='55'),
                parent_chain=[dict(tag='worldbody', attributes={})])
    wrist = dict(attributes=dict(name='wrist_rgb', pos='0.1124 0.1 0.055', xyaxes=WRIST_AXES, fovy='75'),
                 parent_chain=[dict(tag='body', attributes=dict(name='x5_link6'))])
    external = np.eye(4); external[:3, :3] = rig._xyaxes(BODY_AXES) @ rig.OPTICAL_FROM_MUJOCO
    external[:3, 3] = [1., -.9, 1.1]
    source_cal = {
        'body_rgb': dict(K=K(55), T_world_optical=external.tolist(), parent='world_fixed_rig',
                         snapshot_id=rig.SOURCE_CASE+':0', timestamp_s=0.),
        'wrist_rgb': dict(K=K(75), T_world_optical=np.eye(4).tolist(), parent='x5_link6',
                          snapshot_id=rig.SOURCE_CASE+':0', timestamp_s=0.)}
    audit = dict(schema='WM02_SOURCE71558_CAMERA_IDENTITY_CHAIN_RO_V1',
        status='PASS_EXACT_71558_EPISODE_SCENE_CAMERA_CHAIN', errors=[], read_only=True, effects=0,
        checks={str(i): True for i in range(32)}, source_observation_sha256=rig.SOURCE_OBSERVATION_SHA256,
        source_XML_sha256=rig.SOURCE_XML_SHA256, source_code_sha=rig.SOURCE_CODE_SHA,
        source_episode_id=rig.SOURCE_CASE,
        calibration_provenance='PINNED_STATIC_QUALIFICATION_REUSED_SAME_INITIAL_RENDER_PAIR',
        files={'/fixture/episodes/'+rig.SOURCE_CASE+'/STATIC_QUALIFICATION.json':
               dict(sha256=rig.SOURCE_STATIC_SHA256)},
        camera_elements=dict(body_rgb=body, wrist_rgb=wrist), source_calibration=source_cal)
    native = {
        'body_rgb': dict(name='body_rgb', parent_body='base_link', position_local_m=[.25, -.24, .28],
            quaternion_local_wxyz=[1., 0., 0., 0.], fovy_deg=75., width=256, height=256,
            T_world_optical=np.eye(4).tolist(), K=K(75)),
        'wrist_rgb': dict(name='wrist_rgb', parent_body='x5_link6', position_local_m=[.1124, .1, .055],
            quaternion_local_wxyz=WRIST_Q, fovy_deg=75., width=256, height=256,
            T_world_optical=np.eye(4).tolist(), K=K(75))}
    native['wrist_rgb']['T_world_optical'][0][3] = .5
    return audit, native


def build(audit, native, frame=FRAME, pin=None):
    raw = json.dumps(audit, allow_nan=False).encode()
    with patch.object(rig, 'SOURCE_IDENTITY_SHA256', hashlib.sha256(raw).hexdigest()):
        return rig.SourceCameraRig(raw, frame, native,
            expected_native_camera_records_sha256=rig.record_hash(native) if pin is None else pin)


class SourceCameraRigTests(unittest.TestCase):
    def setUp(self):
        self.audit, self.native = fixture()

    def test_inverse_frame_transport_and_optical_axes(self):
        value = build(self.audit, self.native)
        source = self.audit['source_calibration']['body_rgb']['T_world_optical']
        np.testing.assert_allclose(FRAME @ value.external_T_target_world_optical, source, rtol=0, atol=1e-12)
        np.testing.assert_allclose(value.external_T_target_world_mujoco[:3, :3] @ rig.OPTICAL_FROM_MUJOCO,
                                   value.external_T_target_world_optical[:3, :3], rtol=0, atol=1e-12)
        self.assertEqual(value.external_fovy_deg, 55.)
        np.testing.assert_allclose(value.external_K, K(55), rtol=0, atol=1e-12)

    def test_native_body_and_wrist_records_are_preserved_and_distinct(self):
        value = build(self.audit, self.native); declaration = value.declaration()
        self.assertEqual(declaration['native_target_camera_records'], self.native)
        self.assertEqual(declaration['native_target_camera_records']['body_rgb']['fovy_deg'], 75.)
        overlay = declaration['declared_rendering_sensors'][rig.OVERLAY_CAMERA]
        self.assertEqual(overlay['fovy_deg'], 55.)
        self.assertEqual(declaration['declared_rendering_sensors']['wrist_rgb']['T_world_optical'],
                         self.native['wrist_rgb']['T_world_optical'])
        self.assertNotEqual(self.native['wrist_rgb']['T_world_optical'],
                            self.audit['source_calibration']['wrist_rgb']['T_world_optical'])
        self.assertEqual(declaration['policy_camera_role_mapping']['external_rgb'], rig.OVERLAY_CAMERA)

    def test_no_render_or_capture_authority_from_declaration(self):
        declaration = build(self.audit, self.native).declaration()
        json.dumps(declaration, allow_nan=False)
        self.assertEqual(declaration['status'], 'DECLARATION_ONLY_NOT_RENDERED')
        self.assertTrue(all(x == 0 for x in declaration['effects'].values()))
        for key in ('native_camera_equivalence','image_domain_equivalence','occlusion_equivalence',
                    'target_embodiment_qualified','target_task_qualified','native_execution_authorized',
                    'source_images_supplied','observation_payload_created'):
            self.assertIs(declaration[key], False)
        self.assertTrue(declaration['requirements']['new_versioned_overlay_binding'])
        self.assertTrue(declaration['requirements']['source_PNG_replacement_forbidden'])

    def test_declaration_keeps_camera_geometry_without_source_object_diagnostics(self):
        for camera in self.audit['source_calibration'].values():
            camera.update(object_point_ids=list(range(8)), object_point_visible=[True]*8,
                          object_points_uv=[[20., 30.]]*8,
                          optical_axis_convention='OpenCV +X right +Y down +Z forward')
        original = copy.deepcopy(self.audit)
        value = build(self.audit, self.native).declaration()['source_saved_camera_provenance']
        self.assertEqual(value['cameras'], self.audit['camera_elements'])
        for name, camera in value['calibration'].items():
            self.assertEqual(set(camera), {'K', 'T_world_optical', 'parent', 'snapshot_id',
                                           'timestamp_s', 'optical_axis_convention'})
            self.assertEqual(camera['K'], self.audit['source_calibration'][name]['K'])
        self.assertNotIn('object_point', json.dumps(value))
        self.assertEqual(self.audit, original)

    def test_copy_patch_separates_slot_from_sensor_and_is_pure(self):
        value = build(self.audit, self.native); original = copy.deepcopy(self.native)
        descriptor = value.render_copy_patch()
        self.assertEqual(descriptor['existing_render_copy_slot'], 'body_rgb')
        self.assertEqual(descriptor['declared_sensor_name'], rig.OVERLAY_CAMERA)
        self.assertEqual(descriptor['parent_body_id'], 0)
        self.assertIs(descriptor['original_native_model_write_authorized'], False)
        self.assertEqual(descriptor['status'], 'PATCH_DESCRIPTOR_ONLY_NOT_APPLIED')
        self.assertEqual(self.native, original)
        with self.assertRaisesRegex(ValueError, 'EXTERNAL_RENDER_COPY_SLOT'):
            value.render_copy_patch('wrist_rgb')

    def test_arrays_and_returned_metadata_cannot_mutate_rig(self):
        value = build(self.audit, self.native)
        for array in (value.external_T_target_world_optical, value.external_T_target_world_mujoco, value.external_K):
            with self.assertRaises(ValueError): array.setflags(write=True)
            with self.assertRaises(ValueError): array.flat[0] = 10
        with self.assertRaises(dataclasses.FrozenInstanceError): value.external_fovy_deg = 75
        self.native['body_rgb']['fovy_deg'] = 90
        changed = value.declaration(); changed['native_target_camera_records']['wrist_rgb']['fovy_deg'] = 90
        self.assertEqual(value.declaration()['native_target_camera_records']['wrist_rgb']['fovy_deg'], 75)
        self.assertEqual(value.declaration()['native_target_camera_records']['body_rgb']['fovy_deg'], 75)

    def test_exact_audit_bytes_and_native_record_hash(self):
        raw = json.dumps(self.audit).encode()
        with self.assertRaisesRegex(ValueError, 'AUDIT_BYTES'):
            rig.SourceCameraRig(raw, FRAME, self.native, expected_native_camera_records_sha256=rig.record_hash(self.native))
        with self.assertRaisesRegex(ValueError, 'NATIVE_CAMERA_RECORDS'):
            build(self.audit, self.native, pin='0'*64)

    def test_source_identity_and_same_pair_negatives(self):
        cases = [
            ('status', 'FAILED'), ('errors', ['missing']), ('read_only', False), ('effects', 1),
            ('source_observation_sha256', '0'*64), ('source_XML_sha256', '0'*64),
            ('source_code_sha', '0'*40), ('source_episode_id', 'other'), ('calibration_provenance', 'ASSUMED')]
        for key, replacement in cases:
            with self.subTest(key=key):
                audit = copy.deepcopy(self.audit); audit[key] = replacement
                with self.assertRaises(ValueError): build(audit, self.native)
        for field, replacement in (('timestamp_s', .002), ('snapshot_id', 'later'), ('parent', 'base_link')):
            with self.subTest(field=field):
                audit = copy.deepcopy(self.audit); audit['source_calibration']['body_rgb'][field] = replacement
                with self.assertRaisesRegex(ValueError, 'SAME_INITIAL_SOURCE_RENDER_PAIR'): build(audit, self.native)

    def test_source_fov_pose_and_static_proof_negatives(self):
        for mode in ('static', 'parent', 'fov', 'world_pose', 'K', 'nonorthogonal'):
            with self.subTest(mode=mode):
                audit = copy.deepcopy(self.audit)
                if mode == 'static': audit['files'] = {}
                if mode == 'parent': audit['camera_elements']['body_rgb']['parent_chain'][0]['tag'] = 'body'
                if mode == 'fov': audit['camera_elements']['body_rgb']['attributes']['fovy'] = '75'
                if mode == 'world_pose': audit['source_calibration']['body_rgb']['T_world_optical'][0][3] += .001
                if mode == 'K': audit['source_calibration']['body_rgb']['K'][0][0] += .1
                if mode == 'nonorthogonal': audit['camera_elements']['body_rgb']['attributes']['xyaxes'] = '1 0 0 1 0 0'
                with self.assertRaises(ValueError): build(audit, self.native)

    def test_native_wrist_must_have_same_local_geometry(self):
        for mode in ('parent', 'position', 'rotation', 'fov'):
            with self.subTest(mode=mode):
                native = copy.deepcopy(self.native); wrist = native['wrist_rgb']
                if mode == 'parent': wrist['parent_body'] = 'base_link'
                if mode == 'position': wrist['position_local_m'][0] += .001
                if mode == 'rotation': wrist['quaternion_local_wxyz'] = [1., 0., 0., 0.]
                if mode == 'fov': wrist['fovy_deg'] = 55.; wrist['K'] = K(55)
                with self.assertRaisesRegex(ValueError, 'WRIST_LOCAL_GEOMETRY'): build(self.audit, native)

    def test_frame_map_is_fixed_rigid_numeric_and_exactly_pinned(self):
        variants = [np.eye(4), FRAME[:3], FRAME.copy(), FRAME.copy(), FRAME.astype(object)]
        variants[2][0, 0] += .01; variants[3][0, 3] = np.nan; variants[4][3, 3] = True
        for frame in variants:
            with self.subTest(frame=str(frame)):
                with self.assertRaises(ValueError): build(self.audit, self.native, frame=frame)

    def test_original_native_record_validation_negatives(self):
        for mode in ('resolution', 'K', 'quaternion', 'reflection', 'missing'):
            with self.subTest(mode=mode):
                native = copy.deepcopy(self.native)
                if mode == 'resolution': native['body_rgb']['width'] = 320
                if mode == 'K': native['body_rgb']['K'][0][0] += 1
                if mode == 'quaternion': native['body_rgb']['quaternion_local_wxyz'] = [2, 0, 0, 0]
                if mode == 'reflection': native['body_rgb']['T_world_optical'][0][0] = -1
                if mode == 'missing': native.pop('body_rgb')
                with self.assertRaises(ValueError): build(self.audit, native)

    def test_import_has_no_runtime_side_effect_dependencies(self):
        code = "import sys;import go2arx5_vla.world_model.fusion_source_camera_rig;assert not any(x in sys.modules for x in ('torch','mujoco','rclpy','OpenGL','PIL'))"
        reply = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(reply.returncode, 0, reply.stderr)


if __name__ == '__main__':
    unittest.main()
