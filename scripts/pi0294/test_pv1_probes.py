from copy import deepcopy
import unittest

from scripts.pi0294.pv1.probes import plan,LANES,ANCHORS


def fixture(margin):
    rows=[]
    for lane in LANES:
        for i in range(2):
            task='T1' if lane.endswith('T1') else 'T2'
            row=dict(id=lane+'-'+str(i)+('-'+margin if lane=='new_T2' else ''),
                scene_group='PV1/TRAIN/group-'+str(i),PV1_lane=lane,task=task,split='TRAIN')
            rows.extend((row,anchor) for anchor in ANCHORS)
    return rows


class FixedProbeContracts(unittest.TestCase):
    def test_twelve_fixed_train_only_draws_not_task_success_or_extra_physics(self):
        v=plan(fixture('C04'))
        self.assertEqual(v['count'],12);self.assertEqual(sum(r['task']=='T1' for r in v['windows']),6)
        self.assertEqual({r['split'] for r in v['windows']},{'TRAIN'})
        self.assertEqual(v['physical_rollouts'],0);self.assertFalse(v['DEV_or_LOCKBOX_used'])
        self.assertTrue(v['query_plan_is_not_a_task_success_or_retraining_gate'])

    def test_teacher_variant_ids_cannot_change_paired_parent_anchor_or_noise(self):
        self.assertEqual(plan(fixture('C04')),plan(fixture('C12')))

    def test_missing_lane_anchor_or_wrong_task_rejected_before_any_draw(self):
        for change in ('lane','anchor','task'):
            rows=deepcopy(fixture('C04'))
            if change=='lane':rows=[(r,a) for r,a in rows if r['PV1_lane']!='new_T2']
            if change=='anchor':rows=rows[1:]
            if change=='task':rows[0][0]['task']='T2'
            with self.subTest(change=change),self.assertRaises(PermissionError):plan(rows)
