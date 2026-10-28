"""CPU fixtures only: never import a physics engine or integrate MuJoCo."""
from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from .scene_cards import build_scene_plan, scene_group, task_card, validate_groups, digest, near_duplicate_report, legacy_layouts
from .scene import rendered_field_of_view
from .score import BOOL_FIELDS, score, maximum_run_duration
from .episode import validate_reservation, run_episode


def push_fixture():
    card = task_card(scene_group('COMMISSIONING', 'F0', 0), 'T2')
    times = np.arange(641)/20
    obj = np.repeat(np.eye(4)[None], 641, axis=0)
    obj[:, :2, 3] = np.asarray(card['source_xy'])+np.minimum(1., times/10)[:, None]*(np.asarray(card['goal_xy'])-card['source_xy'])
    obj[:, 2, 3] = P['table_top']+.02
    tcp = obj.copy()
    tcp[:, 2, 3] += .13
    data = dict(time_s=times, object_pose=obj, tcp_world=tcp,
                object_linear_velocity=np.zeros((641, 3)), object_angular_velocity=np.zeros((641, 3)))
    for name in BOOL_FIELDS:
        data[name] = np.full(641, name in ('object_present', 'contact_valid', 'gripper_state_valid', 'safety_valid'), dtype=bool)
    data['robot_object_contact'] = (times > 1) & (times < 10)
    data['contact_time_s'] = np.arange(1, 16001)*.002
    micro = np.zeros((16000, 5))
    micro[:, 0] = (data['contact_time_s'] > 1) & (data['contact_time_s'] < 10)
    micro[:, 3] = P['table_top']+.02
    data['contact_audit'] = micro
    return card, data


class PhysicsContracts(unittest.TestCase):
    def test_counts_pairing_extension(self):
        small, large = build_scene_plan(), build_scene_plan('v3.1')
        self.assertEqual((len(small['groups']), len(small['cards'])), (40, 80))
        self.assertEqual((len(large['groups']), len(large['cards'])), (112, 224))
        self.assertTrue({digest(c) for c in small['cards']} <= {digest(c) for c in large['cards']})
        for group in large['groups']:
            a, b = task_card(group, 'T1'), task_card(group, 'T2')
            for key in ('id', 'task', 'task_id', 'instruction'):
                a.pop(key)
                b.pop(key)
            self.assertEqual(a, b)
        report = near_duplicate_report(large['groups'])
        self.assertFalse(report['conflicts'])
        self.assertEqual(validate_groups(large['groups'])['groups'], 112)
        legacy, _ = legacy_layouts()
        self.assertFalse(near_duplicate_report([*large['groups'], *large['confirmation_groups']], legacy)['conflicts'])
        self.assertEqual(large['geometric_uniqueness']['legacy_layouts_checked'], 96)
        self.assertTrue(large['prefreeze_rejected_candidates'])

    def test_original_cross_split_conflict_rejected(self):
        a, b = scene_group('TRAIN', 'F2', 9), scene_group('DEV', 'F2', 0)
        with self.assertRaises(ValueError):
            validate_groups([a, b])

    def test_development_cards_preserved_as_actual_exclusions(self):
        attempted = [task_card(scene_group('COMMISSIONING', 'F0', i), 'T2') for i in (0, 1)]
        hashes = [digest(c) for c in attempted]
        plan = build_scene_plan(preserved_groups=[{'card': c, 'fixture': False} for c in attempted])
        self.assertEqual(hashes, [digest(c) for c in attempted])
        self.assertFalse(near_duplicate_report([*plan['groups'], *plan['confirmation_groups']], attempted)['conflicts'])
        self.assertTrue(plan['preserved_groups_sha256'])

    def test_legacy_conflict(self):
        group = scene_group('TRAIN', 'F0', 0)
        with self.assertRaises(ValueError):
            validate_groups([group], [group])

    def test_visual_near_duplicates_are_grouped(self):
        group = scene_group('TRAIN', 'F0', 0)
        variant = copy.deepcopy(group)
        variant['group_id'], variant['split'], variant['family'] = 'PI0294/DEV/F2/fake', 'DEV', 'F2'
        variant['source_xy'][0] += .001
        variant['goal_xy'][0] += .001
        variant['cube_yaw'] += .01
        variant['visual']['light_scale'] = 1.2
        report = near_duplicate_report([group, variant])
        self.assertEqual(report['conflicts'][0]['disposition'], 'CROSS_SPLIT_LEAKAGE')
        variant['group_id'], variant['split'] = group['group_id'], group['split']
        self.assertFalse(near_duplicate_report([group, variant])['conflicts'])

    def test_rendered_visibility_has_real_target_contract(self):
        card = task_card(scene_group('COMMISSIONING', 'F0', 0), 'T2')
        image = np.zeros((256, 256, 3), np.uint8)+100
        image[10:30, 10:30] = [240, 10, 10]
        image[40:60, 40:60] = [10, 30, 240]
        transform = np.eye(4)
        transform[:3, :3] = np.diag([1., -1., -1.])
        transform[:3, 3] = [*card['goal_xy'], P['table_top']+.5]
        calibration = dict(K=[[100., 0., 128.], [0., 100., 128.], [0., 0., 1.]],
                           T_world_optical=transform.tolist(), object_point_visible=[True]*8, timestamp_s=0.)
        images = dict(body_rgb=image.copy(), wrist_rgb=image.copy())
        geometry = dict(body_rgb=calibration, wrist_rgb=calibration)
        self.assertTrue(rendered_field_of_view(images, geometry, card)['passed'])
        images['body_rgb'][40:60, 40:60] = 100
        self.assertFalse(rendered_field_of_view(images, geometry, card)['passed'])

    def test_push_strict_gates(self):
        card, data = push_fixture()
        self.assertTrue(score(data, card)['persistent_success'])
        for failure in ('lift', 'grasp', 'no_contact', 'not_separated', 'partial', 'no_highrate', 'unsafe'):
            bad = copy.deepcopy(data)
            if failure == 'lift':
                bad['contact_audit'][2000, 3] += .0051
            elif failure == 'grasp':
                bad['contact_audit'][2000:2150, 1] = 1
            elif failure == 'no_contact':
                bad['contact_audit'][:, 0] = 0
                bad['robot_object_contact'][:] = False
            elif failure == 'not_separated':
                bad['robot_object_contact'][580] = True
            elif failure == 'partial':
                bad['time_s'] = bad['time_s'][:-1]
                for k in ('object_pose', 'tcp_world', 'object_linear_velocity', 'object_angular_velocity', 'robot_object_contact', *BOOL_FIELDS):
                    bad[k] = bad[k][:-1]
            elif failure == 'no_highrate':
                bad.pop('contact_audit')
            elif failure == 'unsafe':
                bad['contact_audit'][100, 4] = 1
            self.assertFalse(score(bad, card)['persistent_success'], failure)

    def test_run_duration_exact(self):
        self.assertAlmostEqual(maximum_run_duration(np.arange(150)*.002, np.ones(150, bool), .002), .3)

    def test_static_rejection_preserves_one_frame_without_stepping(self):
        card, physics = push_fixture()
        tcp, obj = physics['tcp_world'][0], physics['object_pose'][0]
        state = dict(time_s=0., tcp=tcp, object_pose=obj, contacts=[], qpos=np.zeros(15), qvel=np.zeros(14),
                     finger_q=np.full(2, .044), finger_qd=np.zeros(2), gripper_fraction=1., gripper_speed=0.,
                     gripper_gap_m=.089048, actuator_force=np.zeros(8), arm_q=np.zeros(6), arm_qd=np.zeros(6),
                     object_linear_velocity=np.zeros(3), object_angular_velocity=np.zeros(3),
                     opposing_contact=False, relative_motion_valid=True, object_points=np.zeros((8, 3)))
        state.update({k: bool(v[0]) for k, v in physics.items() if k in BOOL_FIELDS})
        engine = SimpleNamespace(xml='<mujoco/>', snapshot=lambda: state.copy(), steps=0, safety=[],
            geoms={'cube_geom': 0}, finger_targets=np.full(2, .044), model=SimpleNamespace(nq=15, nv=14, nu=8),
            render=lambda s: ({c: np.zeros((256, 256, 3), np.uint8) for c in ('body_rgb', 'wrist_rgb')}, {}),
            close=Mock(), step=Mock(side_effect=AssertionError('Static rejection must never integrate')))
        teacher = SimpleNamespace(observe=lambda s: None, stage='approach', events=[], lifted=False, carried=False)
        gl = SimpleNamespace(GL_VENDOR=1, GL_RENDERER=2, glGetString=lambda token: b'Mesa' if token == 1 else b'llvmpipe')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'episodes').mkdir()
            output = root/'episodes'/card['id']
            reservation = dict(schema='PI0294_LIVE_RESERVATION_V1', fixture=False, id=card['id'], card=card,
                card_sha256=digest(card), source_sha='a'*40, root=str(root), output=str(output),
                deadline_epoch=time.time()+60, cpus=[min(os.sched_getaffinity(0))],
                reservation=dict(physics_starts=1, physics_steps=16000, rgb_pairs=641, cpu_s=180.,
                                 output_bytes=512*1024**2, memory_bytes=4*1024**3))
            with patch.dict(os.environ, {'LIBGL_ALWAYS_SOFTWARE': '1', 'CUDA_VISIBLE_DEVICES': '',
                    '__EGL_VENDOR_LIBRARY_FILENAMES': '/usr/share/glvnd/egl_vendor.d/50_mesa.json'}), \
                 patch.dict(sys.modules, {'OpenGL': SimpleNamespace(GL=gl), 'mujoco': SimpleNamespace()}), \
                 patch.object(sys.modules[run_episode.__module__].os, 'sched_setaffinity'), \
                 patch.object(sys.modules[run_episode.__module__], 'make_engine', return_value=engine), \
                 patch.object(sys.modules[run_episode.__module__], 'make_teacher', return_value=teacher), \
                 patch.object(sys.modules[run_episode.__module__], 'qualify_engine', return_value={'status': 'STATIC_FAIL'}) as qualify:
                result = run_episode(card, output, '/fixture/not-an-asset', 'a'*40, reservation)
            self.assertEqual((result['physics_starts'], result['physics_steps'], result['rgb_pairs'], result['paired_samples']), (1, 0, 1, 1))
            self.assertEqual(result['status'], 'FAILED_PRESERVED')
            self.assertFalse(result['score']['persistent_success'])
            self.assertEqual(result['output_bytes'], sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
            self.assertTrue((output/'STATIC_QUALIFICATION.json').is_file())
            engine.step.assert_not_called()
            self.assertIs(qualify.call_args.args[0], engine)
            self.assertIn('rendered', qualify.call_args.kwargs)

    def test_reservation_rejection_before_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            card = task_card(scene_group('COMMISSIONING', 'F0', 0), 'T2')
            output = root/'episodes'/card['id']
            r = dict(schema='PI0294_LIVE_RESERVATION_V1', fixture=False, id=card['id'], card=card,
                     card_sha256=digest(card), source_sha='a'*40, root=str(root), output=str(output),
                     deadline_epoch=time.time()+60, cpus=[0],
                     reservation=dict(physics_starts=1, physics_steps=16000, rgb_pairs=641, output_bytes=512*1024**2))
            validate_reservation(r, card, output, 'a'*40)
            r['fixture'] = True
            with self.assertRaises(ValueError):
                validate_reservation(r, card, output, 'a'*40)


if __name__ == '__main__':
    unittest.main()
