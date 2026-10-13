import copy
import unittest

from .allocation_admission import PARALLEL_AUTHORITY
from .pbs_main_eval import SOURCE, admit, capacity_ledger_view


class TestLastAllocationAdmission(unittest.TestCase):
    def setUp(self):
        self.now=1791886000
        self.ledger=dict(PBS_submissions=[dict(request_id='S3-smoke-01',status='SUBMIT_FAILED_PRESERVED',job_id=None),
            dict(request_id='S3-smoke-02',status='RECONCILED_TERMINAL'),
            dict(request_id='S4-S6-main-eval-01',stage='main-eval',status='SUBMITTED',job_id='124.server',GPU_reserved_s=18000,CPU_reserved_s=72000)],
            GPU_allocated_s=600,CPU_s=1000,CPU_upper_s=660,stages=[],main_updates=0)
        self.request=dict(request_id='S4-S6-main-eval-01',stage='main-eval',source_sha=SOURCE,
            parallel_authority=PARALLEL_AUTHORITY,parallel_peer_ids=['123.server'],walltime_s=18000,predicted_remaining_peak_bytes=40*1024**3)
        self.job=dict(Job_Owner='owner@host',job_state='R',project='CFP04-CF-007',
            Resource_List=dict(ngpus=1,walltime='05:00:00'),exec_host='hopper-42/1*4',stime=self.now-10)
        self.plan=dict(status='CAPACITY_ESTIMATE_PASSES_REQUIRES_BOUND_LIVE_ADMISSION',blockers=[],
            main_and_eval_share_next_allocation=True,main_optimizer_updates=4096,effective_batch=32,
            combined_walltime_seconds=18000,remaining_peak_bytes=40*1024**3,required_free_headroom_bytes=60*1024**3)
        self.args=dict(request=self.request,ledger=self.ledger,adoption=dict(release='261bc5b37f20aaff1ad7d5742252d3dc1a91fa65',
            T0_epoch=1791860304,new_work_cutoff_epoch=1791943104,deadline_epoch=1791946704,reset_on_resume=False),
            job_id='124.server',jobs={'124.server':self.job},user='owner',host='hopper-42',
            project=dict(project=dict(name='CFP04-CF-007',active=True,users=['owner'],cluster=['hopper-m-02'],
                start_date='2026-03-13',end_date='2027-03-12'),observed_epoch=self.now-10,balance_su=1),
            quota=dict(user='owner',used_bytes=10*1024**3,hard_bytes=100*1024**3,report_epoch=self.now-10),plan=self.plan,now=self.now)

    def test_exact_last_allocation_with_or_without_fixed_peer(self):
        self.assertEqual(admit(**self.args)['status'],'LAST_ALLOCATION_LIVE_MAIN_ADMISSION_PASS')
        self.args['jobs']['123.server']=dict(Job_Owner='owner@host',job_state='R',project='CFP04-CF-009',Job_Name='pi05_relative',Resource_List=dict(ngpus=1))
        self.assertEqual(admit(**self.args)['allocation_end_epoch'],self.now-10+18000)

    def test_wrong_resource_project_clock_and_capacity_refused(self):
        mutations=[('host',None,'login'),('request','walltime_s',18001),('request','source_sha','other'),
            ('ledger','main_updates',1),('ledger','GPU_allocated_s',18000),('ledger','pending_updates',1),
            ('project','balance_su',float('nan')),('quota','report_epoch',self.now-901),
            ('plan','blockers',['CAPACITY_HOLD']),('plan','main_optimizer_updates',2048),
            ('adoption','reset_on_resume',True)]
        for section,key,value in mutations:
            with self.subTest(section=section,key=key):
                args=copy.deepcopy(self.args)
                if key is None:args[section]=value
                else:args[section][key]=value
                with self.assertRaises(PermissionError):admit(**args)

    def test_preceding_smoke_must_be_reconciled_and_first_failure_retained(self):
        for key,value in [('status','RUNNING'),('CPU_reserved_s',1),('GPU_reserved_s',1)]:
            with self.subTest(key=key):
                args=copy.deepcopy(self.args);args['ledger']['PBS_submissions'][1][key]=value
                with self.assertRaises(PermissionError):admit(**args)
        self.ledger['PBS_submissions'][0]['job_id']='not-a-refused-job'
        with self.assertRaises(PermissionError):admit(**self.args)

    def test_capacity_view_keeps_all_other_ledger_state_and_does_not_mutate(self):
        before=copy.deepcopy(self.ledger);view=capacity_ledger_view(self.ledger)
        self.assertEqual(self.ledger,before)
        self.assertEqual(len(view['PBS_submissions']),2)
        self.assertEqual(view['GPU_allocated_s'],self.ledger['GPU_allocated_s'])
        self.assertEqual(view['CPU_upper_s'],self.ledger['CPU_upper_s'])
