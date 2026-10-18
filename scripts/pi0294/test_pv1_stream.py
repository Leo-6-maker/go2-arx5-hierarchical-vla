"""Predeclared prefix and bank admission; synthetic rows are not real data."""
import unittest

from scripts.pi0294.test_pv1_dataset import manifest
from scripts.pi0294.pv1.stream import bank_manifest,common_support_upper_bound,ready_case,MASTER


def rows():
    result=manifest()['episodes']
    for row in result:
        row.update(status='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN',admitted_for_training=True,
            codec=dict(status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'))
    return result


class StreamContracts(unittest.TestCase):
    def test_future_source_cannot_escape_the_registered_sibling_children(self):
        with self.assertRaises(PermissionError):
            ready_case(dict(source_root='/outside/unregistered'),'/tmp/PV1-fixture/children/stream')

    def test_one_qualified_ordered_prefix_freezes_both_arms_bank(self):
        r=rows();m=bank_manifest(r,'CLEARANCE_PAIR',len(r))
        self.assertEqual(m['master_capture_plan_sha256'],MASTER)
        self.assertEqual(len(m['independent_support']['common_T2_groups']),64)
        self.assertFalse(m['later_captures_hot_injected'])
        self.assertFalse(m['evaluation_rows_used_for_training'])
        with self.assertRaises(PermissionError):bank_manifest(r,'CLEARANCE_PAIR',1)

    def test_correct_actions_that_fail_frozen_codec_cannot_be_silently_admitted(self):
        r=rows();r[10]['codec']['status']='BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR'
        with self.assertRaises(PermissionError):bank_manifest(r,'CLEARANCE_PAIR',192)
        r=rows();r[0]['admitted_for_training']=False
        with self.assertRaises(PermissionError):bank_manifest(r,'CLEARANCE_PAIR',192)

    def test_windows_cannot_replace_independent_direction_or_common_support(self):
        r=rows();r.pop()
        with self.assertRaises(PermissionError):bank_manifest(r,'CLEARANCE_PAIR',192)
        self.assertEqual(len(bank_manifest(r,'COVERAGE_FIRST',288)['independent_support']['original_T2_groups']),64)

    def test_fallback_uses_finite_population_upper_bound_not_guessing_all_later_fail(self):
        from scripts.pi0294.pv1.dataset import TRAIN_GROUPS
        failed=[dict(scene_group=g,task='T2',admitted_for_training=False) for g in list(TRAIN_GROUPS)[:2]]
        self.assertTrue(common_support_upper_bound(failed)['can_still_meet_margin_gate'])
        failed=[dict(scene_group=g,task='T2',admitted_for_training=False) for g in list(TRAIN_GROUPS)[:33]]
        self.assertFalse(common_support_upper_bound(failed)['can_still_meet_margin_gate'])
        negative=[g for g,d in TRAIN_GROUPS.items() if d=='-Y'][:3]
        bound=common_support_upper_bound([dict(scene_group=g,task='T2',admitted_for_training=False) for g in negative])
        self.assertEqual(bound['total'],93)
        self.assertEqual(bound['directions']['-Y'],15)
        self.assertFalse(bound['can_still_meet_margin_gate'])
