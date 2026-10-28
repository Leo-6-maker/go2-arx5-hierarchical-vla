import unittest

from scripts.pi0294.scene_cards import digest, scene_group, task_card
from scripts.pi0294.pv1.plan import BANKS, COUNTS, candidate_pool, evaluation_cards, original_group, slots


class PV1PlanTests(unittest.TestCase):
    def test_counts_and_train_direction(self):
        self.assertEqual(len(slots()),128)
        groups=[original_group(**slot,candidate=0) for slot in slots()]
        self.assertEqual(sum(g['split']=='TRAIN' for g in groups),96)
        f1=[g for g in groups if g['split']=='TRAIN' and g['family']=='F1']
        self.assertEqual(sum(g['direction']=='-Y' for g in f1),18)
        self.assertEqual(sum(g['direction']=='+Y' for g in f1),18)
        self.assertEqual(len(evaluation_cards(groups,'DEV')),48)
        self.assertEqual(len(evaluation_cards(groups,'LOCKBOX')),80)

    def test_original_generator_not_new_distribution(self):
        for family in ('F0','F1','F2'):
            got=original_group('TRAIN',family,0,2)
            old=scene_group('CONFIRM_CORE',family,2000,candidate_index=2)
            for key in ['source_xy','goal_xy','cube_yaw','visual','seed']:
                self.assertEqual(got[key],old[key])
            a,b=task_card(got,'T1'),task_card(got,'T2')
            for key in ['source_xy','goal_xy','cube_yaw','visual','initial_state_sha256']:
                self.assertEqual(a[key],b[key])

    def test_finite_pool_no_result_input_or_seed_redraw(self):
        p=candidate_pool([])
        self.assertEqual(len(p['slots']),128)
        self.assertTrue(all(len(s['candidates'])==32 for s in p['slots']))
        self.assertFalse(p['policy_or_Teacher_success_screening'])
        self.assertEqual(digest(p),digest(candidate_pool([])))
        with self.assertRaises(PermissionError): original_group('TRAIN','F3',0,0)
        with self.assertRaises(PermissionError): original_group('TRAIN','F0',0,32)

    def test_partial_evaluation_population_rejected(self):
        with self.assertRaises(PermissionError): evaluation_cards([],'LOCKBOX')
        with self.assertRaises(PermissionError): evaluation_cards([],'TRAIN')
