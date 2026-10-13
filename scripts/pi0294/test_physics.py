"""CPU fixtures only: never import a physics engine or integrate MuJoCo."""
from __future__ import annotations

import copy
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from .scene_cards import build_scene_plan, scene_group, task_card, validate_groups, digest
from .score import BOOL_FIELDS, score, maximum_run_duration
from .episode import validate_reservation


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
        self.assertEqual(validate_groups(large['groups'])['groups'], 112)

    def test_legacy_conflict(self):
        group = scene_group('TRAIN', 'F0', 0)
        with self.assertRaises(ValueError):
            validate_groups([group], [group])

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
