import unittest

from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.pv1.dataset import admit_new_support,TRAIN_GROUPS,POPULATION


def manifest(count=64):
    rows=[]
    for i,(group,direction) in enumerate(list(TRAIN_GROUPS.items())[:count]):
        for task,margin in [('T1',.004),('T2',.004),('T2',.012)]:
            rows.append(dict(id=f'PV1-{i}-{task}-{margin}',scene_group=group,
                task=task,nominal_clearance_m=margin,split='TRAIN',full_task_success=True,
                independent_raw_admitted=True,windows=601,direction=direction))
    return dict(schema='PI0294_PV1_ADMITTED_TRAIN_BANK_V1',domain='FIXED_BASE_ARX5_TASK_SIM',
        statistics_sha256=STATS,statistics_refit=False,population_plan_sha256=POPULATION,
        elementwise_codec_admission='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP',
        evaluation_rows_used_for_training=False,episodes=rows)


class DatasetAdmissionTests(unittest.TestCase):
    def test_common_success_parents_not_duplicated_windows(self):
        v=admit_new_support(manifest(),'CLEARANCE_PAIR')
        self.assertEqual(len(v['common_T2_groups']),64)
        self.assertEqual(len(v['T1_groups']),64)

    def test_insufficient_pair_support_allows_only_qualified_coverage(self):
        m=manifest()
        m['episodes']=[r for r in m['episodes'] if r['id']!='PV1-63-T2-0.012']
        with self.assertRaises(PermissionError):admit_new_support(m,'CLEARANCE_PAIR')
        self.assertEqual(len(admit_new_support(m,'COVERAGE_FIRST')['original_T2_groups']),64)

    def test_failures_eval_labels_and_refit_never_admitted(self):
        for key,value in [('split','DEV'),('full_task_success',False),('independent_raw_admitted',False),('windows',600)]:
            m=manifest();m['episodes'][0][key]=value
            with self.assertRaises(PermissionError):admit_new_support(m,'CLEARANCE_PAIR')
        m=manifest();m['statistics_refit']=True
        with self.assertRaises(PermissionError):admit_new_support(m,'CLEARANCE_PAIR')

    def test_independent_negative_direction_support_cannot_be_replaced_by_more_positive(self):
        m=manifest()
        for r in m['episodes']:r['direction']='+Y'
        with self.assertRaises(PermissionError):admit_new_support(m,'CLEARANCE_PAIR')

    def test_same_prefix_is_not_membership_in_the_sealed_train_population(self):
        m=manifest();m['episodes'][0]['scene_group']='PI0294/PV1/TRAIN/F1/063'
        with self.assertRaises(PermissionError):admit_new_support(m,'CLEARANCE_PAIR')

    def test_common_pair_not_only_original_bank_needs_negative_direction_coverage(self):
        m=manifest(96)
        remove={r['id'] for r in m['episodes'] if r['task']=='T2' and r['nominal_clearance_m']==.012 and r['direction']=='-Y'}
        remove=set(sorted(remove)[:3])
        m['episodes']=[r for r in m['episodes'] if r['id'] not in remove]
        self.assertGreater(len(set(r['scene_group'] for r in m['episodes'] if r['nominal_clearance_m']==.012)),64)
        with self.assertRaises(PermissionError):admit_new_support(m,'CLEARANCE_PAIR')
        admit_new_support(m,'COVERAGE_FIRST')
