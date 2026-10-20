import copy
import json
from pathlib import Path
import unittest

from scripts.pi0294.data_cov1.plan import preflight_score_geometry
from scripts.pi0294.e1.scope import STATS
from .plan import AUTH, GO, R1, CARD, bank_gate, candidate, cards, pool, slots, validate_group
from .scope import authority


class PlanTests(unittest.TestCase):
    def groups(self):
        return [candidate(*slot, 0) for slot in slots()]

    def test_finite_deterministic_new_namespace_and_scorer_compatible(self):
        groups = self.groups()
        self.assertEqual(len(groups), 12)
        for slot in slots():
            for index in range(32):
                group = candidate(*slot, index)
                self.assertEqual(group, candidate(*slot, index))
                self.assertGreaterEqual(preflight_score_geometry(group), .15)
                self.assertTrue(group['group_id'].startswith('PI0294/CR2_FIXED_R1/TRAIN/'))
        self.assertEqual(pool([]), pool([]))
        self.assertFalse(pool([])['task_results_seen'])

    def test_first_four_and_pairs(self):
        result = cards(self.groups())
        self.assertEqual(len({c['id'] for c in result}), 24)
        self.assertEqual([(c['appearance_profile'], c['direction'], c['distance_bin'], c['task']) for c in result[:4]],
            [('AP0', '-Y', 'NEAR', 'T1'), ('AP0', '-Y', 'NEAR', 'T2'),
             ('AP2', '-Y', 'FAR', 'T1'), ('AP2', '-Y', 'FAR', 'T2')])
        self.assertTrue(all(c['nominal_clearance_m'] == .004 for c in result))

    def test_wrong_namespace_geometry_and_candidates_rejected(self):
        for index in (-1, 32, True):
            with self.assertRaises(PermissionError): candidate('AP0', '+Y', 'NEAR', index)
        for mutation in (dict(split='DEV'), dict(family='DATA_COV1'), dict(group_id='PI0294/PV1/TRAIN/F0/000'),
                         dict(goal_xy=[.45, -.08])):
            group = self.groups()[0]; group.update(mutation)
            with self.assertRaises((PermissionError, ValueError)): validate_group(group)
        with self.assertRaises(PermissionError): cards(self.groups()[:-1])

    def test_unchanged_score_boundary(self):
        for value, accepted in ((.149999, False), (.15, True), (.150001, True)):
            card = dict(source_xy=[0., 0.], goal_xy=[value, 0.])
            if accepted: self.assertGreaterEqual(preflight_score_geometry(card), .15)
            else:
                with self.assertRaises(PermissionError): preflight_score_geometry(card)

    def rows(self):
        return [dict(id=c['id'], task_success=True, admitted=True, evidence_qualified=True,
                     native_faults=0, unknown_effects=[], R0_PASS_to_R1_FAIL=False) for c in cards(self.groups())]

    def test_bank_requires_complete_original_denominator(self):
        rows = self.rows()
        self.assertEqual(bank_gate(rows, self.groups())['status'], 'PASS_CR2_DATA_BANK_GATE')
        self.assertEqual(bank_gate(rows[:4], self.groups())['status'], 'NOT_READY')
        with self.assertRaises(PermissionError): bank_gate(rows + rows[:1], self.groups())

    def test_valid_failures_do_not_become_admission_or_safety(self):
        for field, value in (('native_faults', 1), ('unknown_effects', ['UNKNOWN']), ('R0_PASS_to_R1_FAIL', True),
                             ('evidence_qualified', False)):
            rows = self.rows(); rows[0][field] = value
            self.assertEqual(bank_gate(rows, self.groups())['status'], 'NOT_READY')
        rows = self.rows()
        for i in (0, 2, 4): rows[i].update(task_success=False, admitted=False)
        self.assertEqual(bank_gate(rows, self.groups())['status'], 'NOT_READY')


class ScopeTests(unittest.TestCase):
    def adoption(self):
        limits = json.loads((Path(__file__).resolve().parents[3] / CARD).read_text())['limits']
        return dict(schema='PI0294_CR2_FIXED_R1_ADOPTION_V1', owner='EXISTING_LOCAL_PC',
            protocol_release=AUTH, approval_ref=GO, limits=limits, R0_statistics_sha256=STATS, R1_statistics_sha256=R1,
            prior_closed_scopes_reused=False, PV1_budget_or_numerics_changed=False, worker_cpu_pool=[18, 19],
            PV1_reserved_CPUs=[5], T0_epoch=100, stop_dispatch_epoch=100+40*3600,
            stop_compute_epoch=100+44*3600, hard_close_epoch=100+48*3600)

    def test_exact_new_authority_and_old_go_rejected(self):
        authority(self.adoption(), 101)
        for key, value in (('approval_ref', 'OLD_COV1_GO'), ('R1_statistics_sha256', STATS),
                           ('worker_cpu_pool', [5, 19]), ('PV1_budget_or_numerics_changed', True)):
            a = self.adoption(); a[key] = value
            with self.assertRaises(PermissionError): authority(a, 101)

    def test_budget_inflation_and_clock_reset_not_allowed(self):
        for key in ('episodes', 'CoreLab_CPU_s', 'GPU', 'physical_replacements'):
            a = self.adoption(); a['limits'][key] += 1
            with self.assertRaises(PermissionError): authority(a, 101)
        a = self.adoption(); a['stop_compute_epoch'] += 1
        with self.assertRaises(PermissionError): authority(a, 101)
        with self.assertRaises(PermissionError): authority(self.adoption(), 100+44*3600)
        authority(self.adoption(), 100+44*3600, closeout=True)


if __name__ == '__main__': unittest.main()
