from collections import Counter
from types import SimpleNamespace
import unittest

from scripts.pi0294.pv1.sampler import MatchedSampler,lane_at


def dataset(route,arm):
    rows=[]
    for lane in ['original_T1','original_T2','C1_T1','new_T1','new_T2']:
        task='T1' if lane.endswith('T1') else 'T2'
        for parent in range(4):
            identity=lane+'-'+str(parent)
            if lane=='new_T2':identity+='-C12' if arm=='B_12MM' else '-C04'
            row=dict(PV1_lane=lane,task=task,split='TRAIN',id=identity,scene_group='NEW-'+str(parent))
            rows.extend((row,anchor) for anchor in [0,5,600])
    return SimpleNamespace(PV1_route=route,PV1_arm=arm,rows=rows,manifest_sha='FIXTURE_FROZEN_MANIFEST')


class SamplerTests(unittest.TestCase):
    def test_exact_batch32_task_and_bank_exposure(self):
        expected={('COVERAGE_FIRST','A_CONTROL'):dict(original_T1=8,C1_T1=8,original_T2=16),
            ('COVERAGE_FIRST','B_COVERAGE'):dict(original_T1=4,new_T1=4,C1_T1=8,original_T2=8,new_T2=8),
            ('CLEARANCE_PAIR','A_4MM'):dict(original_T1=4,new_T1=4,C1_T1=8,original_T2=8,new_T2=8),
            ('CLEARANCE_PAIR','B_12MM'):dict(original_T1=4,new_T1=4,C1_T1=8,original_T2=8,new_T2=8)}
        for (route,arm),wanted in expected.items():
            for block in [0,1,1023,4095]:
                rows=[lane_at(block*32+n,route,arm) for n in range(32)]
                self.assertEqual(Counter(t for t,_ in rows),dict(T1=16,T2=16))
                self.assertEqual(Counter(l for _,l in rows),wanted)

    def test_common_C1_and_retained_original_visits_match_exactly(self):
        a=MatchedSampler(dataset('COVERAGE_FIRST','A_CONTROL'),stop=512)
        b=MatchedSampler(dataset('COVERAGE_FIRST','B_COVERAGE'),stop=512)
        for cursor in range(512):
            left,right=a.selection_at(cursor),b.selection_at(cursor)
            if not right['lane'].startswith('new_'):
                self.assertEqual(left,right)

    def test_margin_only_changes_T2_variant_not_common_parent_or_window(self):
        a=MatchedSampler(dataset('CLEARANCE_PAIR','A_4MM'),stop=512)
        b=MatchedSampler(dataset('CLEARANCE_PAIR','B_12MM'),stop=512)
        for cursor in range(512):
            self.assertEqual(a.selection_at(cursor),b.selection_at(cursor))
        self.assertEqual(a.support_sha256,b.support_sha256)

    def test_exact_resume_is_suffix_not_new_random_sequence(self):
        d=dataset('CLEARANCE_PAIR','A_4MM')
        full=list(MatchedSampler(d,stop=512))
        self.assertEqual(full[128:],list(MatchedSampler(d,start=128,stop=512)))
        self.assertNotEqual(full,list(MatchedSampler(d,stop=512,seed=20261020)))
        with self.assertRaises(PermissionError):MatchedSampler(d,stop=2048*64+1)
        with self.assertRaises(PermissionError):MatchedSampler(d,seed=20261017)

    def test_eval_rows_and_missing_bank_are_rejected(self):
        d=dataset('CLEARANCE_PAIR','A_4MM')
        d.rows[0][0]['split']='DEV'
        with self.assertRaises(PermissionError):MatchedSampler(d)
        d=dataset('CLEARANCE_PAIR','A_4MM')
        d.rows=[(r,a) for r,a in d.rows if r['PV1_lane']!='C1_T1']
        with self.assertRaises(PermissionError):MatchedSampler(d)
