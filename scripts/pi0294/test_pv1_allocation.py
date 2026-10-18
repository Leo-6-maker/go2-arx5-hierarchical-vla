"""Native metadata fixtures do not prove any real allocation or permission."""
import unittest

from scripts.pi0294.pv1.allocation import validate_native,PROJECT_SOURCE
from scripts.pi0294.pv1.decision import AUTHORITY


def fixture():
    t0=1792348196;now=t0+1000;jobid='1.hopper-m-02';user='OWN_TEST_ACCOUNT'
    a=dict(authority_release=AUTHORITY,run_uuid='PV1-FIXTURE',T0_epoch=t0,
        stop_new_jobs_epoch=t0+144*3600,stop_new_training_epoch=t0+156*3600,
        stop_new_physics_epoch=t0+160*3600,hard_closeout_epoch=t0+168*3600)
    r=dict(schema='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1',run_uuid=a['run_uuid'],authority_release=AUTHORITY,
        request_id='alloc-01',created_epoch=t0+800,stage='P2_PRIMARY',data_bank_status='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED',
        new_bank_manifest_sha256='a'*64,numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',walltime_s=7200,
        project='PROJECT-B',job_name='pi0294_pv1_01',submission_nonce='fixture',ncpus=12,additional_output_upper_bytes=64*1024**3)
    job=dict(job_state='R',Job_Owner=user+'@fixture',project=r['project'],Job_Name=r['job_name'],
        exec_host='hopper-c-01/0',stime=now-60,Resource_List=dict(ngpus=1,ncpus=12,
            select='1:ngpus=1:gpu_model=H200:ncpus=12:mem=120GB',walltime='02:00:00'))
    ledger=dict(used=dict(H200_s=0,PBS_submissions=1),reserved=dict(H200_s=7200),
        PBS_submissions=[dict(request_id=r['request_id'],job_id=jobid,project=r['project'],reserved_H200_s=7200)],
        new_storage_peak_bytes=0)
    proof=dict(schema='PI0294_PV1_RUNNING_JOB_PROJECT_ATTESTATION_V1',source=PROJECT_SOURCE,
        run_uuid=a['run_uuid'],request_id=r['request_id'],job_id=jobid,submission_nonce=r['submission_nonce'],
        observed_epoch=now-30,balance_su=1.,project=dict(name=r['project'],active=True,users=[user],
            cluster=['hopper-m-02'],start_date='2026-01-01',end_date='2026-12-31'))
    return dict(now=now,user=user,hostname='hopper-c-01',job_id=jobid,jobs={jobid:job},
        quota=dict(user=user,used_bytes=0,hard_bytes=250*1024**3,report_epoch=now-30),adoption=a,request=r,
        ledger=ledger,project_proof=proof,queue_proof=dict(source='NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS',
            queue='auto',destination='small',maximum_wall_s=7200,observed_epoch=t0+790))


class AllocationTests(unittest.TestCase):
    def test_eligible_alternate_project_not_more_budget_or_unrelated_job_cancellation(self):
        f=fixture();f['jobs']['other']=dict(job_state='R',Job_Name='unrelated_healthy')
        p=validate_native(**f)
        self.assertEqual(p['project'],'PROJECT-B')
        self.assertFalse(p['unrelated_owned_jobs_cancelled'])

    def test_before_data_queue_stale_project_login_and_foreign_node_fail_closed(self):
        for change in ('data','proof','project','node','wall','clock','quota','overlap','GPUmodel','CPU'):
            f=fixture()
            if change=='data':f['request']['data_bank_status']='PENDING'
            if change=='proof':f['project_proof']['observed_epoch']=f['now']-1801
            if change=='project':f['project_proof']['project']['users']=[]
            if change=='node':f['hostname']='hopper-l-01'
            if change=='wall':f['request']['walltime_s']=28801
            if change=='clock':f['adoption']['T0_epoch']+=1
            if change=='quota':f['quota']['hard_bytes']=64*1024**3
            if change=='overlap':f['jobs']['2']=dict(job_state='R',Job_Name='pi0294_pv1_other_project')
            if change=='GPUmodel':f['jobs'][f['job_id']]['Resource_List']['select']='1:ngpus=1:gpu_model=A100:ncpus=12'
            if change=='CPU':f['jobs'][f['job_id']]['Resource_List']['ncpus']=4
            with self.subTest(change=change):
                with self.assertRaises(PermissionError):validate_native(**f)

    def test_other_project_does_not_unlock_final_reserve(self):
        f=fixture();f['ledger']['used']['H200_s']=76*3600
        with self.assertRaises(PermissionError):validate_native(**f)
        f=fixture();f['queue_proof']['maximum_wall_s']=3600
        with self.assertRaises(PermissionError):validate_native(**f)

    def test_registered_child_keeps_original_campaign_identity(self):
        f=fixture();f['request']['campaign_run_uuid']=f['adoption']['run_uuid']
        f['request']['run_uuid']='policy-01';f['project_proof']['run_uuid']='policy-01'
        f['root_run_uuid']='policy-01'
        validate_native(**f)
        f['request']['campaign_run_uuid']='A_NEW_CLOCK'
        with self.assertRaises(PermissionError):validate_native(**f)
