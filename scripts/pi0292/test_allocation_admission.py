import copy
import json
from types import SimpleNamespace
import unittest
from scripts.pi0292.allocation_admission import GIB, PROJECT, RELEASE, SOURCE, current_project, owned_jobs, validate


class TestAllocationAdmission(unittest.TestCase):
    def setUp(self):
        self.now=1791883000
        self.job=dict(Job_Owner='owner@host',job_state='R',project=PROJECT,
            Resource_List=dict(ngpus=1,walltime='00:30:00'))
        self.arguments=dict(now=self.now,user='owner',job_id='123.server',jobs={'123.server':self.job},
            quota=dict(user='owner',used_bytes=10*GIB,hard_bytes=200*GIB,report_epoch=self.now-10),
            adoption=dict(release=RELEASE,T0_epoch=1791860304,new_work_cutoff_epoch=1791943104,
                deadline_epoch=1791946704,reset_on_resume=False),
            project_receipt=dict(project=dict(name=PROJECT,active=True,users=['owner'],cluster=['hopper-m-02'],
                start_date='2026-03-13',end_date='2027-03-12'),balance_su=1,observed_epoch=self.now-10),
            dependency=dict(status='PASS_MIGRATED_DEPENDENCY_FILES_AND_CPU_PROCESSOR',model_qualification=False,
                bindings_sha256='fixed',original_T0=1791860304,original_deadline=1791946704),
            request=dict(request_id='S3-smoke-02',source_sha=SOURCE,stage='smoke',walltime_s=1800,
                predicted_remaining_peak_bytes=80*GIB),binding_sha256='fixed')

    def test_pass_is_not_model_qualification(self):
        result=validate(**self.arguments)
        self.assertIn('NOT_MODEL_QUALIFICATION',result['status'])

    def test_reject_other_project_job_without_cancelling(self):
        self.arguments['jobs']['999.server']=dict(self.job,project='CFP04-CF-009')
        with self.assertRaisesRegex(PermissionError,'NO_OVERLAPPING'):validate(**self.arguments)

    def test_reject_each_changed_bound_fact(self):
        mutations=[('quota','report_epoch',self.now-901),('quota','report_epoch',self.now+1),
            ('quota','hard_bytes',90*GIB),('quota','used_bytes',True),('quota','user','other'),
            ('adoption','reset_on_resume',True),('adoption','T0_epoch',self.now),
            ('dependency','bindings_sha256','changed'),('dependency','model_qualification',True),
            ('request','source_sha','changed'),('request','walltime_s',3600),
            ('project_receipt','balance_su',float('nan')),('project_receipt','observed_epoch',self.now-1801)]
        for section,key,value in mutations:
            with self.subTest(section=section,key=key):
                arguments=copy.deepcopy(self.arguments);arguments[section][key]=value
                with self.assertRaises(PermissionError):validate(**arguments)

    def test_reject_wrong_running_allocation_and_project(self):
        for key,value in [('project','CFP04-CF-009'),('job_state','Q'),('Job_Owner','other@host')]:
            with self.subTest(key=key):
                self.job[key]=value
                with self.assertRaises(PermissionError):validate(**self.arguments)
                self.setUp()

    def test_correct_user_inventory_command_and_exact_job_read(self):
        calls=[]
        def query(command,**kwargs):
            calls.append(command)
            output='123.server\n' if command[0].endswith('qselect') else json.dumps({'Jobs':{'123.server':self.job}})
            return SimpleNamespace(returncode=0,stdout=output,stderr='')
        self.assertEqual(owned_jobs('owner',query),{'123.server':self.job})
        self.assertEqual(calls[0][1:],['-u','owner'])
        self.assertEqual(calls[1][1:],['-f','-F','json','123.server'])

    def test_unknown_scheduler_does_not_mean_empty(self):
        def query(*args,**kwargs):return SimpleNamespace(returncode=1,stdout='',stderr='offline')
        with self.assertRaises(PermissionError):owned_jobs('owner',query)

    def test_project_is_requeried_not_refreshed_from_stale_receipt(self):
        calls=[]
        def query(command,**kwargs):
            calls.append(command)
            return json.dumps([self.arguments['project_receipt']['project']]) if 'ls project' in command[-1] else "{'su': 42}"
        receipt=current_project(query)
        self.assertEqual(receipt['balance_su'],42)
        self.assertGreater(receipt['observed_epoch'],self.now)
        self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
