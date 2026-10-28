"""CPU-only analysis fixtures; no model, physics, remote job or native claim."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pi0294.pv1 import paired_development as paired
from scripts.pi0294.pv1.evaluation_platform import ENDPOINTS
from scripts.pi0294.test_pv1_baseline import fixtures, raw
from scripts.pi0294.test_pv1_evaluation_plan import groups


def candidate_rows(arm, successes=(), bank=294101):
    rows = fixtures()[:24] if bank == 294101 else fixtures()[24:]
    for index, item in enumerate(rows):
        row = json.loads(item['adjudication'])
        row.update(model=arm, attempt_id=row['attempt_id'].replace('B0_CORR-', arm+'-'))
        success = index in successes
        row.update(status='SUCCESS' if success else 'VALID_TASK_FAIL',
                   failure_stage='COMPLETE_SUCCESS' if success else 'OBJECT_TASK_RESULT')
        result = json.loads(item['result']);result['id'] = row['attempt_id']
        item['result'] = raw(result)
        row['result_sha256'] = hashlib.sha256(item['result']).hexdigest()
        item['adjudication'] = raw(row)
        audit = json.loads(item['audit'])
        audit.update(attempt_id=row['attempt_id'], adjudication=row,
                     score=dict(complete_32s=True, clock_valid=True, safety_valid=True,
                                persistent_success=success))
        item['audit'] = raw(audit)
        ready = json.loads(item['model_ready'])
        ready['label'] = arm+'-DEV-N'+str(bank)
        ready['identity'].update(label=arm, global_step=4096,
                                 checkpoint_manifest_sha256=ENDPOINTS[arm])
        item['model_ready'] = raw(ready)
        chain = json.loads(item['causal_chain'])
        chain.update(attempt_id=row['attempt_id'], identity=ready['identity'],
                     adjudication_sha256=hashlib.sha256(item['adjudication']).hexdigest())
        item['causal_chain'] = raw(chain)
    return rows


def analyze(a, b, **kwargs):
    population = raw(dict(groups=groups()))
    with patch('scripts.pi0294.pv1.baseline.POPULATION',
               hashlib.sha256(population).hexdigest()):
        return paired.compare(population,
            evidence_by_candidate=dict(A_CONTROL=a, B_COVERAGE=b), **kwargs)


class PairedDevelopment(unittest.TestCase):
    def test_first_one_arm_negative_is_not_an_effect_or_48_failures(self):
        a = candidate_rows('A_CONTROL')[:1]
        result = analyze(a, [])
        self.assertEqual(result['status'], 'EXPLORATORY_PARTIAL')
        self.assertEqual(result['paired_conditions'], 0)
        self.assertIsNone(result['effect_estimates'])
        self.assertIsNone(result['intervals'])
        self.assertEqual(result['by_candidate_task']['A_CONTROL']['T1']['valid_task_failures'], 1)
        self.assertIsNone(result['by_candidate_task']['B_COVERAGE']['T1']['observed_success_rate'])
        missing = [r for r in result['cells'] if r['task_success'] is None]
        self.assertEqual(len(missing), 47)
        self.assertEqual({r['status'] for r in missing}, {'NO_SUPPLIED_TERMINAL_EVIDENCE'})

    def test_partial_pair_uses_matched_denominator_without_cluster_interval(self):
        result = analyze(candidate_rows('A_CONTROL')[:4], candidate_rows('B_COVERAGE', [0])[:1])
        self.assertEqual(result['paired_by_task']['T1']['pairs'], 1)
        self.assertEqual(result['paired_by_task']['T1']['descriptive_B_minus_A'], 1.)
        self.assertEqual(result['paired_by_task']['T2']['pairs'], 0)
        self.assertIsNone(result['paired_by_task']['T2']['descriptive_B_minus_A'])
        self.assertIsNone(result['intervals'])

    def test_complete_all_negative_is_complete_evidence_not_success_or_equivalence(self):
        result = analyze(candidate_rows('A_CONTROL'), candidate_rows('B_COVERAGE'))
        self.assertTrue(result['complete_supplied_pair'])
        self.assertEqual(result['paired_conditions'], 24)
        self.assertEqual(set(result['effect_estimates'].values()), {0.})
        self.assertTrue(all(v == [0., 0.] for v in result['intervals'].values()))
        self.assertFalse(result['training_or_selection_admission'])
        self.assertFalse(result['complete_DEV48_per_candidate'])
        self.assertTrue(result['remote_attempt_inventory_complete_not_proven_here'])
        self.assertTrue(result['resampling']['small_sample_or_degenerate_interval_is_not_equivalence'])

    def test_opposite_task_changes_cancel_macro_and_keep_discordant_pairs(self):
        result = analyze(candidate_rows('A_CONTROL', range(1,24,2)),
                         candidate_rows('B_COVERAGE', range(0,24,2)))
        self.assertEqual(result['effect_estimates'],
            dict(T1_B_minus_A=1., T2_B_minus_A=-1., macro_B_minus_A=0.))
        self.assertEqual(result['paired_by_task']['T1']['B_only_success'], 12)
        self.assertEqual(result['paired_by_task']['T2']['A_only_success'], 12)
        self.assertEqual(result['intervals']['macro_B_minus_A'], [0.,0.])

    def test_cluster_interval_resamples_geometry_with_both_tasks_together(self):
        result = analyze(candidate_rows('A_CONTROL'), candidate_rows('B_COVERAGE', [0,1]),
                         bootstrap_draws=2000, seed=17)
        self.assertEqual(result['resampling']['blocks'], 12)
        # A whole-geometry paired change makes all three bootstrap columns
        # identical. Independent task/cell resampling would break this.
        self.assertEqual(result['intervals']['T1_B_minus_A'], result['intervals']['macro_B_minus_A'])
        self.assertEqual(result['intervals']['T2_B_minus_A'], result['intervals']['macro_B_minus_A'])
        self.assertAlmostEqual(result['effect_estimates']['macro_B_minus_A'], 1/12)
        draws = np.random.default_rng(17).binomial(12, 1/12, 200000)/12
        expected = np.quantile(draws, [.025,.975])
        np.testing.assert_allclose(result['intervals']['macro_B_minus_A'], expected, atol=1/12)

    def test_duplicate_or_foreign_model_and_missing_raw_causal_evidence_fail_closed(self):
        a = candidate_rows('A_CONTROL');b = candidate_rows('B_COVERAGE')
        for bad in (a+[a[0]], candidate_rows('B_COVERAGE')):
            with self.subTest(kind='duplicate_or_foreign'), self.assertRaises(PermissionError):
                analyze(bad, b)
        bad = deepcopy(a);bad[0].pop('causal_chain')
        with self.assertRaises(PermissionError):analyze(bad,b)
        bad = deepcopy(a);bad[0]['audit'] = {}
        with self.assertRaises(PermissionError):analyze(bad,b)

    def test_original_population_cannot_be_replaced_by_summary(self):
        with self.assertRaisesRegex(PermissionError, 'FROZEN_POPULATION'):
            paired.compare(raw(dict(groups=groups())),
                evidence_by_candidate=dict(A_CONTROL=[], B_COVERAGE=[]))

    def test_second_noise_bank_is_rejected_instead_of_silently_filtered(self):
        with self.assertRaisesRegex(PermissionError, 'FIRST_ORIGINAL_NOISE_BANK'):
            analyze(candidate_rows('A_CONTROL', bank=294102), [])

    def test_diagnosed_engineering_is_missing_without_fabricated_task_outcome(self):
        item = candidate_rows('A_CONTROL')[0]
        row = json.loads(item['adjudication'])
        row.update(status='ENGINEERING_INCOMPLETE', pipeline_trustworthy=False,
                   failure_stage='TRANSPORT')
        item['adjudication'] = raw(row)
        audit = json.loads(item['audit']);audit['adjudication'] = row
        item['audit'] = raw(audit);item.pop('causal_chain')
        item['engineering_diagnosis'] = raw(dict(
            status='CONFIRMED_ENGINEERING_FAILURE_NO_POLICY_REPLAY',
            original_adjudication_sha256=hashlib.sha256(item['adjudication']).hexdigest(),
            original_result_sha256=row['result_sha256'], native_process_terminal_verified=True,
            unknown_effects=[], policy_outcome_valid=False,
            evidence_sha256=dict(native_terminal='a'*64)))
        result = analyze([item], candidate_rows('B_COVERAGE', [0])[:1])
        self.assertEqual(result['by_candidate_task']['A_CONTROL']['T1']['engineering_missing'], 1)
        self.assertEqual(result['by_candidate_task']['A_CONTROL']['T1']['valid_task_failures'], 0)
        self.assertEqual(result['paired_conditions'], 0)
        self.assertIsNone(result['effect_estimates'])
        self.assertIsNone(result['cells'][0]['task_success'])

    def test_native_safety_error_cannot_be_omitted_to_produce_a_favorable_effect(self):
        a = candidate_rows('A_CONTROL')
        row = json.loads(a[0]['adjudication'])
        row.update(status='ACTUAL_SAFETY_FAIL', live_native_failures=1)
        a[0]['adjudication'] = raw(row)
        with self.assertRaises(PermissionError):
            analyze(a, candidate_rows('B_COVERAGE', range(24)))

    def test_bounds_reject_bool_and_implicit_seed_or_arm_substitution(self):
        for kwargs in (dict(bootstrap_draws=True), dict(bootstrap_draws=99),
                       dict(seed=True), dict(seed=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(PermissionError):
                analyze([], [], **kwargs)
        with self.assertRaises(PermissionError):
            paired.compare(b'{}', evidence_by_candidate=dict(A_CONTROL=[], B0_CORR=[]))


if __name__ == '__main__':unittest.main()
