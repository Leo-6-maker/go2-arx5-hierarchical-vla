import copy
import unittest

import numpy as np

from .rules import Envelope, candidate_statistics, split_groups


class RulesTests(unittest.TestCase):
    def row(self, index, task='T1', **kwargs):
        return dict(id=str(index)+task, scene_group=str(index), geometry_sha256=str(index),
                    generator_identity=str(index), task=task, direction='+Y', family='F0',
                    split='TRAIN', nominal_clearance_m=.004, **kwargs)

    def test_split_stable_paired_and_order_independent(self):
        rows = [self.row(i, t) for i in range(8) for t in ('T1', 'T2')]
        actual, strata = split_groups(rows)
        self.assertEqual(actual, split_groups(list(reversed(rows)))[0])
        self.assertEqual(strata[0]['diagnostic_groups'], 2)
        for i in range(8):
            self.assertEqual(len({r['role'] for r in actual if r['scene_group'] == str(i)}), 1)

    def test_transitive_visual_and_reference_siblings(self):
        rows = [self.row(i) for i in range(4)]
        rows[1]['geometry_sha256'] = '0'
        rows[2]['generator_identity'] = '1'
        result, _ = split_groups(rows)
        self.assertEqual(len({r['lineage'] for r in result[:3]}), 1)

    def test_small_strata_diagnostic_and_no_window_split(self):
        result, _ = split_groups([self.row(0), self.row(1)])
        self.assertEqual({r['role'] for r in result}, {'FIT', 'DIAGNOSTIC'})
        self.assertEqual(split_groups([self.row(0)])[0][0]['role'], 'FIT')

    def test_one_formula_only_effective_minmax_and_preserve_inputs(self):
        stats = {'new_embodiment': {'relative_action': {'eef_9d': dict(min=[0.]*9, max=[1.]*9,
                                                                   mean=[.5]*9, q01=[.1]*9)},
                                    'state': {'untouched': [5]}}}
        before = copy.deepcopy(stats)
        env = Envelope((9,)); env.add(np.full((40, 9), 2.))
        result, report = candidate_statistics(stats, dict(min=np.zeros(9), max=np.ones(9)), env)
        self.assertEqual(stats, before)
        self.assertTrue(np.allclose(result['new_embodiment']['relative_action']['eef_9d']['min'], -.1))
        self.assertTrue(np.allclose(result['new_embodiment']['relative_action']['eef_9d']['max'], 2.1))
        self.assertEqual(report['padding_searches'], 0)
        self.assertEqual(result['new_embodiment']['state'], stats['new_embodiment']['state'])

    def test_per_offset_shape_not_flattened(self):
        env = Envelope((40, 9)); values = np.arange(360).reshape(40, 9)
        env.add(values)
        np.testing.assert_array_equal(env.minimum, values)

    def test_empty_and_nonfinite_and_bad_shape_rejected(self):
        env = Envelope((9,))
        with self.assertRaises(ValueError): env.add(np.full((40, 9), np.nan))
        with self.assertRaises(ValueError): Envelope((10,))
        with self.assertRaises(ValueError): candidate_statistics(
            {'new_embodiment': {'relative_action': {'eef_9d': {'min':[0.]*9, 'max':[1.]*9}}}},
            dict(min=np.zeros(9), max=np.ones(9)), env)


if __name__ == '__main__': unittest.main()
