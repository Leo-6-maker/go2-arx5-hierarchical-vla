"""Geometry/recovery/sampling regressions; no model or physics execution."""
from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.pi0294.multitask.object_relations import (
    axis_rotation, book_insertion, door_turn, inverse, pose)
from scripts.pi0294.multitask.reader import bind_source_reader
from scripts.pi0294.multitask.recipe import proposed_recipe, trainer_arguments
from scripts.pi0294.multitask.sampling import EVENTS, TASKS, SharedTaskSampler


def translated(x, y, z):
    p = np.eye(4)
    p[:3, 3] = [x, y, z]
    return p


class GeometryTests(unittest.TestCase):
    def door(self, **changes):
        h = translated(.4, -.2, .6) @ axis_rotation([1, 0, 0], .2)
        angle = .35
        handle = h @ axis_rotation([0, 0, 1], angle) @ translated(.4, .02, .1)
        tool = handle @ translated(-.03, .01, .02) @ axis_rotation([0, 1, 0], .3)
        kw = dict(hinge_world=h, axis_hinge=[0, 0, 1], actual_tcp_world=tool,
                  actual_handle_world=handle, actual_angle_rad=angle,
                  target_angle_rad=1.1, actual_grasp_valid=True, angle_limits_rad=[-.2, 1.4])
        kw.update(changes)
        return kw, door_turn(**kw)

    def test_door_keeps_actual_grasp_and_exact_hinge_arc(self):
        k, r = self.door()
        grasp = inverse(k['actual_tcp_world']) @ k['actual_handle_world']
        h = k['hinge_world']
        hz = inverse(axis_rotation(k['axis_hinge'], k['actual_angle_rad'])) @ inverse(h) @ k['actual_handle_world']
        for tool, theta in zip(r.tcp_world, r.hinge_angles_rad):
            np.testing.assert_allclose(tool @ grasp, h @ axis_rotation(k['axis_hinge'], theta) @ hz, atol=1e-9)
        np.testing.assert_allclose(r.tcp_world[0], k['actual_tcp_world'], atol=1e-9)
        self.assertAlmostEqual(r.hinge_angles_rad[-1], 1.1)
        self.assertLessEqual(np.linalg.norm(np.diff(r.tcp_world[:, :3, 3], axis=0), axis=1).max()/.05, .06+1e-9)
        self.assertLessEqual(np.abs(np.diff(r.hinge_angles_rad)).max()/.05, .3+1e-9)

    def test_door_world_frame_equivariance(self):
        k, original = self.door()
        world = translated(-2, .8, 1.2) @ axis_rotation([0, 1, 0], -.7)
        for field in ('hinge_world', 'actual_tcp_world', 'actual_handle_world'):
            k[field] = world @ k[field]
        r = door_turn(**k)
        np.testing.assert_allclose(r.tcp_world, world @ original.tcp_world, atol=1e-9)

    def test_actual_regrasp_changes_tool_path(self):
        k, old = self.door()
        k['actual_tcp_world'] = k['actual_tcp_world'] @ translated(.02, -.01, .01)
        recovered = door_turn(**k)
        np.testing.assert_allclose(recovered.tcp_world[0], k['actual_tcp_world'], atol=1e-9)
        self.assertGreater(np.linalg.norm(recovered.tcp_world[-1, :3, 3]-old.tcp_world[-1, :3, 3]), .01)

    def test_open_direction_is_not_forced_positive(self):
        _, r = self.door(target_angle_rad=-.1)
        self.assertTrue(np.all(np.diff(r.hinge_angles_rad) < 0))

    def test_door_refuses_lost_grasp_and_out_of_range(self):
        for update in (dict(actual_grasp_valid=False), dict(target_angle_rad=1.5),
                       dict(axis_hinge=[0, 0, 2]), dict(max_linear_m_s=float('nan'))):
            with self.assertRaises(ValueError):
                self.door(**update)

    def book(self, **changes):
        shelf = translated(.6, .3, .4) @ axis_rotation([0, 0, 1], .6)
        goal = shelf @ translated(.14, 0, .16)
        actual_book = translated(.3, -.2, .6) @ axis_rotation([0, 0, 1], -.2)
        actual_tool = actual_book @ translated(-.03, .01, .08) @ axis_rotation([0, 1, 0], .4)
        k = dict(actual_tcp_world=actual_tool, actual_book_world=actual_book,
                 target_book_world=goal, shelf_world=shelf, book_size_m=[.2, .04, .3],
                 cavity_bounds_shelf_m=[[0, .35], [-.04, .04], [0, .4]], actual_grasp_valid=True)
        k.update(changes)
        return k, book_insertion(**k)

    def test_book_targets_correct_object_pose_under_offset_grasp(self):
        k, r = self.book()
        np.testing.assert_allclose(r.insert_tcp_world @ r.actual_tool_to_book, k['target_book_world'], atol=1e-9)
        pre_book = inverse(k['shelf_world']) @ r.preinsert_tcp_world @ r.actual_tool_to_book
        self.assertAlmostEqual(pre_book[0, 3]+.1, -.08)
        self.assertGreaterEqual(r.final_corner_clearance_m.min(), 0)

    def test_book_rotated_slot_and_grasp_equivariance(self):
        k, r = self.book()
        world = translated(2, 1, -.3) @ axis_rotation([1, 0, 0], -.2)
        for key in ('actual_tcp_world', 'actual_book_world', 'target_book_world', 'shelf_world'):
            k[key] = world @ k[key]
        moved = book_insertion(**k)
        np.testing.assert_allclose(moved.insert_tcp_world, world @ r.insert_tcp_world, atol=1e-9)
        np.testing.assert_allclose(moved.preinsert_tcp_world, world @ r.preinsert_tcp_world, atol=1e-9)

    def test_book_rejects_center_only_fit_that_collides_at_corners(self):
        k, _ = self.book()
        k['target_book_world'] = k['shelf_world'] @ translated(.14, 0, .16) @ axis_rotation([0, 0, 1], .4)
        with self.assertRaisesRegex(ValueError, 'DOES_NOT_FIT'):
            book_insertion(**k)

    def test_invalid_pose_and_book_grasp(self):
        invalid = np.eye(4); invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            pose(invalid)
        with self.assertRaises(ValueError):
            self.book(actual_grasp_valid=False)


def dataset(recovery=False):
    rows, episodes = [], []
    for task in TASKS:
        for lane in ('normal', 'recovery') if recovery else ('normal',):
            for i, length in enumerate((5, 40)):
                row = dict(id=f'{task}-{lane}-{i}', task=task, learning_lane=lane,
                           split='TRAIN', domain='SOURCE_CONTACT_PHYSICS', admitted_success=True,
                           sha256=str(i+1)*64, event_anchors={sorted(EVENTS[task])[0]: [1, 3]})
                if lane == 'recovery':
                    row['recovery'] = dict(origin='SOURCE_STUDENT_ACTUAL_VISIT',
                        failure_state_sha256='a'*64, restored_state_sha256='a'*64,
                        student_policy_sha256='b'*64, expert_execution_sha256='c'*64,
                        expert_replanned_from_actual_state=True, recovery_physically_verified=True)
                episodes.append(row)
                rows.extend((row, anchor) for anchor in range(length))
    return SimpleNamespace(rows=rows, manifest_sha='d'*64, statistics_sha='e'*64,
        split='TRAIN', for_optimization=True, manifest=dict(episodes=episodes,
        learning_scope='SOURCE_SHARED_FOUR_TASK_EEF10_V1', shared_task_statistics_fit=dict(
        split='TRAIN', episode_sha256={r['id']: r['sha256'] for r in episodes if r['learning_lane'] == 'normal'},
        tasks=list(TASKS), statistics_sha256='e'*64)))


class SamplerTests(unittest.TestCase):
    def test_exact_per_update_task_and_event_counts(self):
        for recovery in (False, True):
            sampler = SharedTaskSampler(dataset(recovery), seed=29, stop=320, recovery_round=recovery)
            for start in range(0, 320, 32):
                rows = [sampler.selection_at(c) for c in range(start, start+32)]
                self.assertEqual(Counter(r['task'] for r in rows), {t: 8 for t in TASKS})
                for task in TASKS:
                    selected = [r for r in rows if r['task'] == task]
                    self.assertEqual(sum(r['mode'] == 'event' for r in selected), 4)
                    self.assertEqual(sum(r['lane'] == 'recovery' for r in selected), 2 if recovery else 0)
                    self.assertTrue(all(r['anchor'] in (1, 3) for r in selected if r['mode'] == 'event'))

    def test_resume_is_absolute_cursor_and_same_identity(self):
        data = dataset(True)
        full = SharedTaskSampler(data, seed=29, stop=320, recovery_round=True)
        resumed = SharedTaskSampler(data, seed=29, start=160, stop=320, recovery_round=True)
        self.assertEqual(list(full)[160:], list(resumed))
        self.assertEqual(full.state_dict(160), resumed.state_dict(160))
        with self.assertRaises(ValueError):
            full.state_dict(159)

    def test_long_episode_does_not_gain_episode_weight(self):
        sampler = SharedTaskSampler(dataset(), seed=29, stop=32000)
        counts = Counter(sampler.selection_at(c)['episode_id'] for c in range(32000))
        # One parent has 8x the windows; episode-first selection remains ~1:1.
        for task in TASKS:
            ratio = counts[task+'-normal-0']/counts[task+'-normal-1']
            self.assertTrue(.85 < ratio < 1.15)

    def test_recovery_rejects_stale_or_unverified_correction(self):
        changes = [dict(restored_state_sha256='f'*64), dict(expert_replanned_from_actual_state=False),
                   dict(recovery_physically_verified=False), dict(origin='OLD_TEACHER_TIMESTAMP')]
        for change in changes:
            data = dataset(True)
            row = next(r for r in data.manifest['episodes'] if r['learning_lane'] == 'recovery')
            row['recovery'].update(change)
            with self.assertRaisesRegex(ValueError, 'RECOVERY_REQUIRES'):
                SharedTaskSampler(data, seed=29, recovery_round=True)

    def test_no_silent_missing_task_or_event_fallback(self):
        for change in ('task', 'event', 'foreign_event', 'test', 'target', 'not_admitted'):
            data = dataset()
            if change == 'task':
                data.rows = [(r, a) for r, a in data.rows if r['task'] != 'DOOR']
            else:
                row = data.rows[0][0]
                if change == 'event': row['event_anchors'] = {}
                elif change == 'foreign_event': row['event_anchors'] = {'handle_grasp': [1]}
                elif change == 'test': row['split'] = 'TEST'
                elif change == 'target': row['domain'] = 'TARGET_WBC'
                else: row['admitted_success'] = False
            with self.assertRaises(ValueError):
                SharedTaskSampler(data, seed=29)

    def test_reader_view_uses_original_processed_input_and_new_sampler(self):
        class Reader:
            def __init__(self): self.__dict__.update(dataset().__dict__)
            def __len__(self): return len(self.rows)
            def __getitem__(self, index): return {'processed': ('ORIGINAL_READER', index)}
        original = Reader()
        view = bind_source_reader(original, seed=29)
        self.assertEqual(view[(0, 2)], {'processed': ('ORIGINAL_READER', (0, 2))})
        self.assertEqual(len(list(view.make_sampler(view, stop=32))), 32)
        self.assertFalse(hasattr(original, 'make_sampler'))
        original.manifest['shared_task_statistics_fit']['episode_sha256'].pop('BOOK-normal-0')
        with self.assertRaises(ValueError):
            bind_source_reader(original, seed=29)

    def test_metadata_changes_resume_identity_and_no_sample_contains_geometry(self):
        data = dataset(True)
        first = SharedTaskSampler(data, seed=29, recovery_round=True)
        data.manifest['episodes'][0]['sha256'] = 'f'*64
        second = SharedTaskSampler(data, seed=29, recovery_round=True)
        self.assertNotEqual(first.state_dict(0), second.state_dict(0))
        self.assertEqual(set(second.selection_at(0)), {'task', 'lane', 'episode_id', 'mode', 'anchor', 'index'})

    def test_recovery_reuses_common_four_task_statistics_without_refit(self):
        data = dataset(True)
        view = bind_source_reader(data, seed=29, recovery_round=True)
        self.assertEqual(view.statistics_sha, 'e'*64)
        self.assertEqual(view.make_sampler(view, stop=32).state_dict(0)['statistics_sha256'], 'e'*64)


class RecipeTests(unittest.TestCase):
    def test_microbatch_and_accumulation_consume_actual_32(self):
        for micro, accum in ((1, 32), (2, 16), (4, 8)):
            r = proposed_recipe(manifest_sha256='a'*64, statistics_sha256='b'*64,
                                seed=29, microbatch=micro, accumulation=accum)
            args = trainer_arguments(r)
            self.assertEqual(args['per_device_train_batch_size']*args['gradient_accumulation_steps'], 32)
            self.assertEqual(args['warmup_steps']/args['max_steps'], .05)
            self.assertFalse(r['new_training_budget_adopted'])
        with self.assertRaises(ValueError):
            proposed_recipe(manifest_sha256='a'*64, statistics_sha256='b'*64,
                            seed=29, microbatch=2, accumulation=32)

    def test_recovery_sampling_does_not_authorize_a_second_10k_raw_restart(self):
        r = proposed_recipe(manifest_sha256='a'*64, statistics_sha256='b'*64,
                            seed=29, recovery_round=True)
        with self.assertRaises(ValueError):
            trainer_arguments(r)


if __name__ == '__main__':
    unittest.main()
