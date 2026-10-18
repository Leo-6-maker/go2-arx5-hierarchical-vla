from copy import deepcopy
import hashlib
import unittest

from scripts.pi0294 import test_confirmation_plan as fixture
from scripts.pi0294.mp1 import confirmation_evidence as e
from scripts.pi0294.mp1.confirmation_plan import evaluation_plan


class ConfirmationEvidenceTests(unittest.TestCase):
    def setup_plan(self):
        candidate,groups,static=fixture.ConfirmationPlanTests().fixture()
        return evaluation_plan(groups,candidate,static,[])

    def row(self,plan,*,status='SUCCESS',replacement=False):
        label,spec=next(iter(plan['panels'].items()));card=spec['cards'][0]
        return dict(attempt_id=card['id']+('-REPLACEMENT-01' if replacement else ''),scene_group=card['scene_group'],
            task=card['task'],model=label,status=status,failure_stage='NONE' if status=='SUCCESS' else 'S1_PRE_EXECUTION_REJECT',
            pipeline_trustworthy=status in ['SUCCESS','VALID_TASK_FAIL'],main_integrations=16000 if status=='SUCCESS' else 200,
            duration_s=32 if status=='SUCCESS' else .4,RGB_pairs=641 if status=='SUCCESS' else 9,
            S1_rejections=1 if status=='VALID_TASK_FAIL' else 0,live_native_failures=0,teacher_takeover=False,split=e.config.SPLIT)

    def audit(self,row):return dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',adjudication=deepcopy(row),video=dict(sha256='a'*64))

    def test_missing_and_unaudited_successes_are_not_fabricated_success(self):
        plan=self.setup_plan();empty=e.coverage(plan,[],{})
        self.assertEqual(len(empty['cells']),40);self.assertEqual(empty['tasks']['T1']['planned_conditions'],20)
        self.assertFalse(empty['paired_noise_trials_are_independent_scenes']);self.assertEqual(empty['independent_scene_groups'],10)
        row=self.row(plan);value=e.coverage(plan,[row],{})
        self.assertEqual(value['cells'][0]['status'],'PENDING_INDEPENDENT_AUDIT');self.assertEqual(value['tasks']['T1']['successes'],0)
        good=e.coverage(plan,[row],{row['attempt_id']:self.audit(row)})
        self.assertEqual(good['tasks']['T1']['successes'],1);self.assertEqual(good['tasks']['T1']['qualified_policy_outcomes'],1)

    def test_original_early_S1_negative_is_valid_and_must_not_repeat(self):
        plan=self.setup_plan();row=self.row(plan,status='VALID_TASK_FAIL');audit=self.audit(row)
        value=e.coverage(plan,[dict(row,route_failure=dict(message='KNOWN_CORE_TERMINAL'))],{row['attempt_id']:audit})
        self.assertTrue(value['cells'][0]['valid_policy_outcome']);self.assertEqual(value['tasks']['T1']['successes'],0)
        self.assertEqual(value['tasks']['T1']['S1_rejections'],1)
        repeat=self.row(plan,status='SUCCESS',replacement=True)
        with self.assertRaises(PermissionError):e.coverage(plan,[row,repeat],{})

    def test_single_engineering_replacement_and_audit_join(self):
        plan=self.setup_plan();old=self.row(plan,status='ENGINEERING_INCOMPLETE');new=self.row(plan,replacement=True)
        value=e.coverage(plan,[old,new],{new['attempt_id']:self.audit(new)})
        self.assertEqual(value['actual_attempts'],2);self.assertTrue(value['cells'][0]['engineering_replacement_used'])
        self.assertEqual(value,e.coverage(plan,[new,old],{new['attempt_id']:self.audit(new)}))
        changed=self.audit(new);changed['adjudication']['main_integrations']=1
        self.assertFalse(e.coverage(plan,[old,new],{new['attempt_id']:changed})['cells'][0]['valid_policy_outcome'])

    def test_native_safety_is_preserved_not_filtered_as_missing_data(self):
        plan=self.setup_plan();row=self.row(plan,status='ACTUAL_SAFETY_FAIL');row['live_native_failures']=1
        value=e.coverage(plan,[row],{row['attempt_id']:self.audit(row)})
        self.assertTrue(value['actual_safety_stop']);self.assertEqual(value['tasks']['T1']['live_native_failures'],1)
        self.assertEqual(value['cells'][0]['status'],'ACTUAL_SAFETY_FAIL');self.assertFalse(value['cells'][0]['success'])

    def test_short_success_foreign_rows_and_unknown_effects_rejected_or_reported(self):
        plan=self.setup_plan();row=self.row(plan);row['duration_s']=1
        with self.assertRaises(ValueError):e.coverage(plan,[row],{})
        row=self.row(plan);row['model']='DEV8'
        with self.assertRaises(ValueError):e.coverage(plan,[row],{})
        row=self.row(plan);value=e.coverage(plan,[],{},unresolved=[row['attempt_id']])
        self.assertEqual(value['cells'][0]['status'],'UNRESOLVED_PHYSICAL_OUTCOME')
        with self.assertRaises(ValueError):e.coverage(plan,[],{},unresolved=['foreign'])

    def test_complete_means_all40_audited_cells_not_an_admission_rate_gate(self):
        plan=self.setup_plan();rows=[];audits={}
        for label,spec in plan['panels'].items():
            for card in spec['cards']:
                row=self.row(plan,status='VALID_TASK_FAIL');row.update(attempt_id=card['id'],scene_group=card['scene_group'],task=card['task'],model=label)
                rows.append(row);audits[row['attempt_id']]=self.audit(row)
        value=e.coverage(plan,rows,audits)
        self.assertEqual(value['status'],'COMPLETE_ACTUAL_AUDITED_CONFIRM40')
        self.assertEqual(value['actual_attempts'],40)
        for task in ['T1','T2']:
            self.assertEqual(value['tasks'][task]['qualified_policy_outcomes'],20)
            self.assertEqual(value['tasks'][task]['successes'],0)
        self.assertEqual(value['deployment'],'NONE')

    def test_actual_receipts_fixed_noise_seed_precision_and_format(self):
        plan=self.setup_plan();spec=next(iter(plan['panels'].values()));card=spec['cards'][0];sequence=3
        text=spec['noise_namespace']+'/'+card['scene_group']+'/'+str(sequence)
        seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')%(2**63)
        shape=dict(shape=[1,40,132],dtype='torch.float32')
        receipt=dict(seed=seed,inference_precision=dict(mode='ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST',cuda_matmul_allow_tf32=False,bf16_autocast=True),
            noise=dict(shape,sha256='b'*64),action_sha256='c'*64,arrays_sha256='d'*64)
        value=e.draw_receipt(spec,card,sequence,receipt,expected_noise_format=shape)
        self.assertFalse(value['GPU_RNG_reexecuted']);self.assertEqual(e.paired_noise([value,value])['unique_seeds'],1)
        for change in ['seed','precision','format','fingerprint']:
            bad=deepcopy(receipt)
            if change=='seed':bad['seed']+=1
            elif change=='precision':bad['inference_precision']['cuda_matmul_allow_tf32']=True
            elif change=='format':bad['noise']['dtype']='different'
            else:bad['noise']['sha256']='not-a-sha'
            with self.subTest(change=change),self.assertRaises(ValueError):e.draw_receipt(spec,card,sequence,bad,expected_noise_format=shape)
        with self.assertRaises(ValueError):e.paired_noise([value,dict(value,noise_sha256='f'*64)])


if __name__=='__main__':unittest.main()
