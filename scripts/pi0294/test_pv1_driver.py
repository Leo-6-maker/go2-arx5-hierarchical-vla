from copy import deepcopy
import unittest

from scripts.pi0294.pv1.driver import registry,LIVE_READINESS
from scripts.pi0294.pv1.evaluation_plan import development
from scripts.pi0294.test_pv1_evaluation_plan import groups


class DriverContracts(unittest.TestCase):
    def setUp(self):
        plan=development(groups(),route='COVERAGE_FIRST',candidate='B0_CORR',new_bank_manifest_sha256='b'*64)
        self.spec=next(iter(plan['panels'].values()))
        self.ready=dict(status='PV1_NATIVE_MODEL_READY',label=self.spec['cards'][0]['pv1_session_label'],run_uuid='fixture',
            job_id='123.hopper-m-02',allocation_end_epoch=10000,identity=dict(label='B0_CORR',global_step=2048,relay_identity=dict(profile='B1')),
            native_pre_model_qualification=dict(status='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK',receipt_sha256='c'*64,
                heartbeat_elapsed_s=600.1,transport_cycles=80,new_model_queries=0,new_physics=0))
        self.peers={k:'a'*40 if k.endswith('source_sha') else 'b'*64
            for k in ('core_source_sha','hopper_source_sha','core_manifest_sha256','hopper_manifest_sha256')}

    def test_actual_24_case_panel_has_original_requests_exact_sources_and_candidate_independent_noise(self):
        value=registry(self.spec,self.ready,self.peers,[],deadline_epoch=7300,now=1000)
        self.assertEqual(len(value['cases']),24);self.assertEqual(value['job_id'],self.ready['job_id'])
        for case in value['cases'].values():
            self.assertEqual(case['layout_id'],case['card']['scene_group']+'/'+case['card']['task'])
            self.assertEqual(case['core_request']['live_readiness_sha256'],LIVE_READINESS)
            self.assertFalse(case['card']['teacher_takeover'])
        self.assertEqual(value['native_pre_model_receipt_sha256'],'c'*64)

    def test_no_cpu_fake_qualification_other_checkpoint_or_extra_walltime(self):
        for change in ('no_proof','wrong_model','step','short','deadline'):
            r=deepcopy(self.ready);end=7300
            if change=='no_proof':r.pop('native_pre_model_qualification')
            if change=='wrong_model':r['identity']['label']='OTHER'
            if change=='step':r['identity']['global_step']=1024
            if change=='short':r['native_pre_model_qualification']['heartbeat_elapsed_s']=599.99
            if change=='deadline':end=7300.01
            with self.subTest(change=change),self.assertRaises(PermissionError):registry(self.spec,r,self.peers,[],deadline_epoch=end,now=1000)
