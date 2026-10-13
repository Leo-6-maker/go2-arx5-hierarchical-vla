import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from scripts.pi0292.allocation_admission import ENGINEERING_AUTHORITY, GIB, PARALLEL_AUTHORITY, PROJECT, RELEASE, SOURCE, allocation_project_receipt, current_project, owned_jobs, sha, validate, validate_engineering_revision


def engineering_revision():
    return dict(schema='PI0292_SINGLE_ENGINEERING_PBS_REVISION_V1',
        authority=ENGINEERING_AUTHORITY, run_uuid='20261013T025824Z-localpc-pi0292',
        science_release=RELEASE, original_submission_cap=3, revised_submission_cap=4,
        added_engineering_smoke_submissions=1, GPU_cap_s=28800,
        max_PI0292_concurrent_GPU=1, original_T0_epoch=1791860304,
        new_work_cutoff_epoch=1791943104, deadline_epoch=1791946704,
        reset_clock_or_budget=False, recipe_changed=False, science_gates_changed=False)


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
            return json.dumps([self.arguments['project_receipt']['project']]) if command[1]=='ls' else "{'su': 42}"
        receipt=current_project(query,executable='/qualified/native/bin/amgr')
        self.assertEqual(receipt['balance_su'],42)
        self.assertGreater(receipt['observed_epoch'],self.now)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0],['/qualified/native/bin/amgr','ls','project','--name',PROJECT,'--json-info'])
        self.assertEqual(calls[1],['/qualified/native/bin/amgr','checkbalance','project','--name',PROJECT])

    def test_third_smoke_requires_exact_human_revision_without_other_expansion(self):
        self.arguments['request'].update(request_id='S3-smoke-03',engineering_revision=engineering_revision())
        self.assertIn('NOT_MODEL_QUALIFICATION',validate(**self.arguments)['status'])
        for key,value in [('revised_submission_cap',5),('GPU_cap_s',28801),
                ('reset_clock_or_budget',True),('deadline_epoch',1791946705),('recipe_changed',True)]:
            with self.subTest(key=key):
                bad=engineering_revision();bad[key]=value
                with self.assertRaises(PermissionError):validate_engineering_revision(bad)
        self.arguments['request'].pop('engineering_revision')
        with self.assertRaises(PermissionError):validate(**self.arguments)

    def test_bound_login_receipt_needs_no_compute_amgr_but_does_not_admit_a_job(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'PROJECT_ADMISSION.private.json'
            nonce='c9e9f761-f1e8-4cf4-9851-68adb68d1f21'
            receipt=dict(self.arguments['project_receipt'],
                schema='PI0292_LOGIN_NATIVE_PROJECT_ATTESTATION_V1',
                source='HOPPER_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE',
                run_uuid='20261013T025824Z-localpc-pi0292',request_id='S3-smoke-02',submission_nonce=nonce)
            path.write_text(json.dumps(receipt))
            request=dict(self.arguments['request'],submission_nonce=nonce,
                project_admission_mode='FRESH_LOGIN_RECEIPT_NATIVE_PBS_V1',project_admission_sha256=sha(path))
            self.assertEqual(allocation_project_receipt(request,path),receipt)
            arguments=copy.deepcopy(self.arguments);arguments['project_receipt']=receipt
            for state in ('Q','F','X'):
                with self.subTest(state=state):
                    arguments['jobs']['123.server']['job_state']=state
                    with self.assertRaises(PermissionError):validate(**arguments)
            for key,value in [('project_admission_sha256','changed'),('submission_nonce',None),
                    ('submission_nonce','not-a-uuid'),('project_admission_mode','implicit')]:
                with self.subTest(key=key):
                    wrong=dict(request);wrong[key]=value
                    with self.assertRaises(PermissionError):allocation_project_receipt(wrong,path)
            for key,value in [('request_id','another-request'),('run_uuid','another-run'),
                    ('source','TIMESTAMP_ONLY'),('submission_nonce','98943063-6064-4354-b0a7-c3b0c524e6db')]:
                with self.subTest(key=key):
                    bad=dict(receipt);bad[key]=value;path.write_text(json.dumps(bad))
                    wrong=dict(request,project_admission_sha256=sha(path))
                    with self.assertRaises(PermissionError):allocation_project_receipt(wrong,path)

    def test_login_attestation_never_refreshes_old_or_future_native_observation(self):
        for observed in (self.now-1801,self.now+1):
            with self.subTest(observed=observed):
                arguments=copy.deepcopy(self.arguments)
                arguments['project_receipt']['observed_epoch']=observed
                with self.assertRaises(PermissionError):validate(**arguments)

    def test_explicit_two_job_revision_keeps_exact_single_gpu_peers(self):
        self.arguments['request'].update(parallel_authority=PARALLEL_AUTHORITY,parallel_peer_ids=['999.server'])
        peer=dict(self.job,Job_Name='pi05_relative',project='CFP04-CF-009')
        self.arguments['jobs']['999.server']=peer
        self.assertEqual(validate(**self.arguments)['own_current_jobs'],2)
        for mutation in ('third-job','wrong-project','two-gpu','foreign-owner','same-project','wrong-authority'):
            with self.subTest(mutation=mutation):
                altered=copy.deepcopy(self.arguments)
                if mutation=='third-job':altered['jobs']['888.server']=peer
                if mutation=='wrong-project':altered['jobs']['999.server']['project']='unknown'
                if mutation=='same-project':altered['jobs']['999.server']['project']=PROJECT
                if mutation=='two-gpu':altered['jobs']['999.server']['Resource_List']['ngpus']=2
                if mutation=='foreign-owner':altered['jobs']['999.server']['Job_Owner']='other@host'
                if mutation=='wrong-authority':altered['request']['parallel_authority']='inferred'
                with self.assertRaises(PermissionError):validate(**altered)
        del self.arguments['jobs']['999.server']
        self.assertEqual(validate(**self.arguments)['own_current_jobs'],1)


if __name__=='__main__':unittest.main()
