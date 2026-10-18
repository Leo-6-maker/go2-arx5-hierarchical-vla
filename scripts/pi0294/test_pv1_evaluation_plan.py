from copy import deepcopy
import unittest
import uuid

from scripts.pi0294.pv1.evaluation_plan import development,remaining_panel,validate_registry
from scripts.pi0294.pv1.plan import COUNTS,original_group


def groups():
    return [original_group(split,family,i,0) for split,counts in COUNTS.items()
        for family,count in counts.items() for i in range(count)]


class DevelopmentContracts(unittest.TestCase):
    def setUp(self):
        self.plan=development(groups(),route='COVERAGE_FIRST',candidate='B0_CORR',new_bank_manifest_sha256='a'*64)
        self.spec=next(iter(self.plan['panels'].values()));self.card=self.spec['cards'][0]

    def row(self,**extra):
        return dict(condition_id=self.card['condition_id'],model='B0_CORR',attempt_id=self.card['id'],
            status='VALID_TASK_FAIL',failure_stage='S1_PRE_EXECUTION_REJECT',pipeline_trustworthy=True,
            live_native_failures=0,**extra)

    def test_exact_balanced_48_new_dev_conditions_no_lockbox_or_success_screen(self):
        cards=[c for p in self.plan['panels'].values() for c in p['cards']]
        self.assertEqual(len(cards),48);self.assertEqual(len({c['condition_id'] for c in cards}),48)
        self.assertEqual(len({c['scene_group'] for c in cards}),12)
        self.assertEqual({c['split'] for c in cards},{'DEV'})
        self.assertEqual(sum(c['task']=='T1' for c in cards),24)
        self.assertFalse(self.plan['evaluation_conditions_teacher_success_screened'])
        self.assertFalse(self.plan['final_LOCKBOX_opened'])
        self.assertTrue(all(not c['teacher_takeover'] for c in cards))
        self.assertEqual(len(remaining_panel([],self.spec)),24)

    def test_arm_matching_has_identical_geometry_noise_and_language_not_separate_task_weights(self):
        a=development(groups(),route='CLEARANCE_PAIR',candidate='A_4MM',new_bank_manifest_sha256='a'*64)
        b=development(groups(),route='CLEARANCE_PAIR',candidate='B_12MM',new_bank_manifest_sha256='a'*64)
        normalize=lambda p:[{k:v for k,v in c.items() if k not in ('id','policy_candidate','pv1_model_label','pv1_session_label')}
            for s in p['panels'].values() for c in s['cards']]
        self.assertEqual(normalize(a),normalize(b))
        with self.assertRaises(PermissionError):development(groups(),route='CLEARANCE_PAIR',candidate='B_COVERAGE',new_bank_manifest_sha256='a'*64)

    def test_valid_policy_negatives_and_successes_are_not_repeated(self):
        for status in ('SUCCESS','VALID_TASK_FAIL'):
            row=self.row();row['status']=status
            self.assertEqual(len(remaining_panel([row],self.spec)),23)
        row=self.row();bad=deepcopy(row);bad['attempt_id']+='-REPLACEMENT-01'
        with self.assertRaises(PermissionError):remaining_panel([row,bad],self.spec)

    def test_only_one_evidence_confirmed_engineering_replacement_is_possible(self):
        row=self.row();row.update(status='ENGINEERING_INCOMPLETE',failure_stage='ENGINEERING_OR_UNKNOWN',pipeline_trustworthy=False)
        with self.assertRaisesRegex(PermissionError,'SETTLED_EVIDENCE'):remaining_panel([row],self.spec)
        row.update(confirmed_engineering_failure=True,engineering_diagnosis_sha256='c'*64)
        cards=remaining_panel([row],self.spec);replacement=next(c for c in cards if c['condition_id']==self.card['condition_id'])
        self.assertEqual(replacement['pv1_replacement_of'],self.card['id'])
        second=deepcopy(row);second['attempt_id']=replacement['id']
        self.assertEqual(len(remaining_panel([row,second],self.spec)),23)
        missing=deepcopy(row);missing.pop('confirmed_engineering_failure')
        with self.assertRaisesRegex(PermissionError,'ORIGINAL_SETTLED_DIAGNOSIS'):
            remaining_panel([missing,second],self.spec)
        third=deepcopy(second);third['attempt_id']+='-AGAIN'
        with self.assertRaises(PermissionError):remaining_panel([row,second,third],self.spec)

    def test_native_safety_or_s1_integrity_is_not_an_engineering_replacement(self):
        for change in ('ACTUAL_SAFETY_FAIL','S1_INTEGRITY'):
            row=self.row()
            if change=='ACTUAL_SAFETY_FAIL':row['status']=change
            else:row['failure_stage']=change
            with self.assertRaises(PermissionError):remaining_panel([row],self.spec)

    def test_live_registry_cannot_change_task_noise_order_or_checkpoint_between_candidates(self):
        identity=dict(global_step=2048,relay_identity=dict(profile='B1'))
        peer={k:'a'*40 if k.endswith('source_sha') else 'b'*64
            for k in ('core_source_sha','hopper_source_sha','core_manifest_sha256','hopper_manifest_sha256')}
        registry=dict(run_uuid='fixture',job_id='123',checkpoint_identity=identity,deadline_epoch=10000,cases={},**peer)
        for i,card in enumerate(self.spec['cards']):
            episode=str(uuid.uuid4());registry['cases'][episode]=dict(episode_uuid=episode,attempt_id=card['id'],run_uuid='fixture',
                identity=identity['relay_identity'],card=card,layout_id=card['scene_group']+'/'+card['task'],panel_index=i,
                deadline_epoch=10000,hopper_session_relative='evidence/'+card['pv1_session_label']+'/resident')
        def check(r):return validate_registry(r,self.spec,identity,run_uuid='fixture',job_id='123',peer=peer,prior_rows=[])
        self.assertTrue(check(registry))
        for change in ('candidate_noise','wrong_task','extra_condition','wrong_weight','order','peer'):
            r=deepcopy(registry);case=next(iter(r['cases'].values()))
            if change=='candidate_noise':case['layout_id']='OTHER_CANDIDATE/'+case['layout_id']
            if change=='wrong_task':case['card']['task']='T2' if case['card']['task']=='T1' else 'T1'
            if change=='extra_condition':r['cases'].pop(next(iter(r['cases'])))
            if change=='wrong_weight':r['checkpoint_identity']['global_step']=1024
            if change=='order':case['panel_index']=23
            if change=='peer':r['core_source_sha']='c'*40
            with self.subTest(change=change),self.assertRaises(PermissionError):check(r)
