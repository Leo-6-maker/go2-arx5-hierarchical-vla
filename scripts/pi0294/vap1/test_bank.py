from copy import deepcopy
import hashlib
import unittest

from scripts.pi0294.cr2.bank import manifest,configuration
from scripts.pi0294.vap1 import plan


def fixture():
    groups=[plan.candidate(*slot,0) for slot in plan.slots()];rows=[]
    for i,c in enumerate(plan.cards(groups)):
        rows.append(dict(id=c['id'],task=c['task'],scene_group=c['group_id'],task_success=True,
            admitted=True,evidence_qualified=True,native_faults=0,unknown_effects=[],R0_PASS_to_R1_FAIL=False,
            status='PAIRED_FIXED_R0_R1_SAVED_DATA_COMPLETE',transition='FAIL->PASS',
            source_raw_manifest_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
            R1=dict(status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP',actual_statistics_sha256=plan.R1),
            official_reader=dict(R1=dict(statistics_sha256=plan.R1,actual_anchors=[0,80,160,320,480,600],
                actual_dual_camera_frames=True,mask_H40_EEF10_zero_padding=True,deterministic_repeat=True))))
    return groups,rows


class BankTests(unittest.TestCase):
    def make(self,groups,rows,profile):
        return manifest(groups,rows,population_sha256='a'*64,adoption_sha256='b'*64,profile=profile)

    def test_exact_batch_denominators_never_promote_partial(self):
        groups,rows=fixture()
        for profile,count in (('VAP1_D24',24),('VAP1_D48',48)):
            value=self.make(groups,rows[:count],profile)
            self.assertEqual(len(value['episodes']),count)
            self.assertEqual(value['bank'],'VAP1_R1_D'+str(count))
            self.assertEqual(value['gate']['planned'],count)
            self.assertEqual(value['statistics_sha256'],plan.R1)
            self.assertIn('all_paired_attempts_identity',value)
            self.assertNotIn('all_24_paired_attempts_identity',value)
            self.assertFalse(value['policy_success_claim'])
            with self.assertRaises(PermissionError):self.make(groups,rows[:count-1],profile)
        with self.assertRaises(PermissionError):self.make(groups,rows,'VAP1_D24')
        with self.assertRaises(PermissionError):self.make(groups,rows[:24],'VAP1_D48')

    def test_original_guard_and_unknown_profiles_preserved(self):
        groups,rows=fixture()
        with self.assertRaises(PermissionError):self.make(groups,rows[:24],None)
        for name in ('VAP1_D72','VAP1',False,{}):
            with self.assertRaises(PermissionError):configuration(name)
        self.assertEqual(configuration()['bank'],'CR2_FIXED_R1_P12')

    def test_valid_failure_remains_denominator_not_positive_data(self):
        groups,rows=fixture();rows[0].update(task_success=False,admitted=False)
        value=self.make(groups,rows[:24],'VAP1_D24')
        self.assertEqual(value['gate']['accounted'],24)
        self.assertEqual(len(value['episodes']),23)
        self.assertNotIn(rows[0]['id'],{r['id'] for r in value['episodes']})

    def test_fake_reader_duplicate_raw_or_wrong_R1_rejected(self):
        for mutation in (lambda r:r[0]['R1'].update(actual_statistics_sha256='0'*64),
            lambda r:r[0]['official_reader']['R1'].update(actual_dual_camera_frames=False),
            lambda r:r[0].update(source_raw_manifest_sha256=r[1]['source_raw_manifest_sha256']),
            lambda r:r[0].update(unknown_effects=['UNKNOWN'])):
            groups,rows=fixture();mutation(rows)
            with self.assertRaises(PermissionError):self.make(groups,rows[:24],'VAP1_D24')

    def test_same_total_D_T_caps_and_own_frozen_names(self):
        for p in ('VAP1_D24','VAP1_D48'):
            c=configuration(p)
            self.assertEqual(c['auth'],plan.AUTH)
            self.assertEqual(c['output_cap'],32*1024**3)
            self.assertNotEqual(c['index_schema'],configuration()['index_schema'])
            self.assertEqual(c['attempts'],'ALL_PAIRED_ATTEMPTS.json')


if __name__=='__main__':unittest.main()
