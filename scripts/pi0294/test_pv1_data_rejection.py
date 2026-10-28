from copy import deepcopy
import unittest

from scripts.pi0294.pv1.data_rejection import contact_force_only,retired_alternate


class DataRejectionContracts(unittest.TestCase):
    def setUp(self):
        self.result=dict(status='VALID_COMPLETE',physics_steps=16000,rgb_pairs=641,model_calls=0,
            score=dict(safety_valid=True,complete_32s=True,persistent_success=True))
        self.audit=dict(status='AUDIT_FAIL_PRESERVED',error_type='ValueError',integrations=0,render_calls=0,model_loads=0,
            error='Independent reconstruction error 5.884880998208164e-05 exceeds 1e-05',
            traceback='max_force = max(max_force, near(recorded["normal_force_N"], force[0], 1e-5))')
        self.s1=dict(live_native_failures=0,pre_execution_rejections=0,actions_modified=False,maximum_live_shadow_difference=1e-13,
            shadow_invocations=80,shadow_integrations=16000)

    def test_known_contact_force_readside_failure_is_exclusion_not_success_admission_or_replay(self):
        self.assertTrue(contact_force_only(self.result,self.audit,self.s1))
        self.assertFalse(self.audit.get('admitted',False))
        for change in ('other_field','safety','changed_action','unexplained','partial','wrong_gate'):
            r,a,s=deepcopy(self.result),deepcopy(self.audit),deepcopy(self.s1)
            if change=='other_field':a['traceback']='near(recorded["qpos"], qpos, 1e-5)'
            if change=='safety':s['live_native_failures']=1
            if change=='changed_action':s['actions_modified']=True
            if change=='unexplained':a['error']='different error'
            if change=='partial':r['physics_steps']=15999
            if change=='wrong_gate':a['error']=a['error'].replace('exceeds 1e-05','exceeds 0.0001')
            with self.subTest(change=change):self.assertFalse(contact_force_only(r,a,s))

    def test_retirement_cannot_remove_original_teacher_or_preempt_route_decision(self):
        binding=dict(retired_unattempted_alternate_source='/not/a/sibling',
            card=dict(task='T2',nominal_clearance_m=.004),card_sha256='a'*64)
        with self.assertRaises(PermissionError):retired_alternate(binding,'/tmp/campaign/children/reader','COVERAGE_FIRST')
        binding['card']['nominal_clearance_m']=.012
        with self.assertRaises(PermissionError):retired_alternate(binding,'/tmp/campaign/children/reader','CLEARANCE_PAIR')
