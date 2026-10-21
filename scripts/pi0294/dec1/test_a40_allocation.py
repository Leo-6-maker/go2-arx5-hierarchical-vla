"""Synthetic native metadata only; no allocation, model, training or auth calls."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from . import a40_allocation as a
from . import a40_future as future
from . import dual_platform as dual
from . import future_budget as budget
from .alternate_accounting import digest
from .test_dual_platform import fixture as dual_fixture
from .test_future_budget import fixtures as forecast_fixture, amendment, NOW
from .long_training import POLICY_SHA


def raw(value):return (json.dumps(value,sort_keys=True)+'\n').encode()


def fixture():
    grant,_,_=dual_fixture();job_id='17.stdct-mgmt-02';user='fixture';now=NOW
    r=dict(schema='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1',platform_training_version=a.VERSION,
        native_platform=a.PLATFORM,authority_release=a.PARENT_AUTHORITY,authority_amendment=a.AUTHORITY,
        scientific_admission_version=a.SCIENCE_VERSION,campaign_run_uuid='20261018T182956Z-localpc-pv1',
        original_T0_epoch=a.T0,hard_close_epoch=a.HARD_CLOSE,stage='P2_PRIMARY',project='CFP04-CF-007',
        run_uuid='source-A-fixture',submit_count=13,job_name='pi0294_dec1_a40_train',
        data_bank_status='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED',
        parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.R0,new_bank_manifest_sha256=a.BANK,
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',active_cpu_limit=4,requested_ncpus=4,
        requested_memory_bytes=32*a.GIB,compatibility_queries=17,TRAIN_probe_queries=12,new_robot_physics=0,
        LOCKBOX_opened=False,training_block=dict(arm='A_CONTROL',replica=False,
            same_allocation_1024_probes=True,stop_after_step=4096,resume_checkpoint_relative=None),
        created_epoch=now-120,source_sha='a'*40,run_manifest_sha256='b'*64,expected_native_start_step=0,
        request_id='40d177f7-f689-4f4e-99e4-5c0d64b0e5a1',submission_nonce='c589d094-56b9-4b39-827e-e650986e839f',
        walltime_s=28800,CPU_upper_s=115560,additional_output_upper_bytes=52*10**9,
        allowed_native_capacity_profiles=[dict(ncpus=c,memory_bytes=m) for c,m in a.CAPACITY_PROFILES])
    req=raw(r);request_sha=hashlib.sha256(req).hexdigest()
    row=dict(request_id=r['request_id'],job_id=job_id,native_platform=a.PLATFORM,global_submit_count=13,
        platform_training_version=a.VERSION,scientific_admission_version=a.SCIENCE_VERSION,
        project=r['project'],job_name=r['job_name'],source_sha=r['source_sha'],request_sha256=request_sha,
        run_manifest_sha256=r['run_manifest_sha256'],submission_nonce=r['submission_nonce'],arm='A_CONTROL',
        reserved_H200_s=28800,accounting_pad_s=60,reserved_optimizer_updates=4096,reserved_neural_queries=29,
        CPU_upper_s=115560,physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,
        legacy_GPU_seconds_field_is_projection=True,parent_checkpoint_manifest_sha256=a.PARENT,
        statistics_sha256=a.R0,new_bank_manifest_sha256=a.BANK,
        additional_output_upper_bytes=r['additional_output_upper_bytes'],status='SUBMITTED_PENDING_NATIVE',
        dec1_work_id='TRAIN:A_CONTROL:LONG')
    old=[dict(request_id='old-'+str(n)) for n in range(12)]
    h=dict(schema='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1',revision=1,
        authority_release=a.PARENT_AUTHORITY,campaign_run_uuid=r['campaign_run_uuid'],unknown_effects=[],pending_updates=[],
        optimizer_updates=0,generation_queries=1054,
        used=dict(H200_s=20745,PBS_submissions=13,optimizer_updates=0,neural_queries=1054),
        reserved=dict(H200_s=28860,optimizer_updates=4096,neural_queries=29),PBS_submissions=old+[row],
        CPU_s=32590.,CPU_reserved_s=115560,new_storage_current_bytes=62*a.GIB,
        new_storage_peak_bytes=62*a.GIB,
        storage_reserved_bytes=r['additional_output_upper_bytes'])
    job=dict(job_state='R',Job_Owner=user+'@host',project=r['project'],Job_Name=r['job_name'],
        queue='batch_gpu',Rerunable=False,run_count=1,
        Variable_List=dict(PV1_REQUEST_ID=r['request_id'],PV1_SUBMISSION_NONCE=r['submission_nonce']),
        Resource_List=dict(select='1:ngpus=1:ncpus=4:mem=32gb',ngpus=1,ncpus=4,mem='33554432kb',walltime='08:00:00'),
        stime=now-60,exec_host='gn-a40-01/0')
    view=dict(status='PASS_TWO_PLATFORM_CAMPAIGN_VIEW',grant_sha256=hashlib.sha256(grant).hexdigest(),
        campaign_running_by_platform=dict(VANDA_A40=1),
        observed_epoch=now,campaign_jobs=[dict(platform=a.PLATFORM,job_id=job_id,state='R',project=r['project'],name=r['job_name'])])
    project=dict(schema='VAP1_A40_NATIVE_RUNNING_PROJECT_ATTESTATION_V1',job_id=job_id,run_uuid=r['run_uuid'],
        request_id=r['request_id'],submission_nonce=r['submission_nonce'],source='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE',
        observed_epoch=now-10,balance_GPU_hours=100.,balance_CPU_hours=1000.,
        project=dict(name=r['project'],active=True,users=[user],cluster=['stdct-mgmt-02'],
            start_date='2026-01-01',end_date='2026-12-31'))
    args=dict(now=now,user=user,hostname='gn-a40-01',job_id=job_id,job=job,request_bytes=req,ledger=h,
        project_proof=project,quota=dict(user=user,scope='OWN_PROJECT_AND_FILESYSTEM_MINIMUM',
            observed_epoch=now,available_bytes=200*a.GIB),
        queue_proof=dict(source='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS',requested_queue='auto',
            destination='batch_gpu',observed_epoch=now-130,maximum_wall_s=28800),grant_bytes=grant,campaign_view=view)
    return args


def future_inputs(args):
    s,p,plan=forecast_fixture();r=json.loads(args['request_bytes']);h=args['ledger']
    for v in (s,p):v['campaign_run_uuid']=r['campaign_run_uuid']
    plan.update(native_platform=a.PLATFORM,performance_basis='UNMEASURED_A40_OWNER_CEILING_REVALIDATE_FIRST32',
        H200_timing_is_A40_measurement=False,first32_are_counted_A_CONTROL_updates=True,
        long_allocation_policy_sha256=POLICY_SHA)
    plan['training_jobs']=[]
    for arm in ('A_CONTROL','B_COVERAGE'):
        plan['training_jobs'].append(dict(work_id='TRAIN:'+arm+':LONG',arm=arm,start_step=0,stop_step=4096,
            same_allocation_1024_probes=True,walltime_upper_s=28800,
            resources=dict(H200_s=28860,PBS_submissions=1,optimizer_updates=4096,neural_queries=29,
                Hopper_CPU_s=115560,Core_CPU_s=180,local_CPU_s=120,engine_starts=3,
                transfer_bytes=10*a.GIB,Hopper_storage_bytes=r['additional_output_upper_bytes'])))
    plan['storage_lifecycle']['new_output_peak_upper_bytes']['Hopper_storage_bytes']=144*a.GIB
    d=dict(authority_release=a.AUTHORITY,campaign_run_uuid=r['campaign_run_uuid'],unknown_effects=[],
        GPU_allocated_s=251,new_submissions=[dict(request_id='old-compat'),h['PBS_submissions'][-1]],
        GPU_reserved_s=28860,optimizer_updates=0,optimizer_updates_reserved=4096,neural_queries=0,
        neural_queries_reserved=29,CPU_s=25920.,CPU_reserved_s=115560,episode_starts=0,physics_ticks=0,constructors=0,
        new_GPU_site_output_allocated_upper_bytes=34*a.GIB,GPU_site_output_reserved_bytes=r['additional_output_upper_bytes'])
    s['paired_source_training_platform']=a.PLATFORM
    s['used'].update(h['used'],Hopper_CPU_s=h['CPU_s'],Hopper_storage_bytes=h['new_storage_current_bytes'])
    s['reserved']=dict(h['reserved'],Hopper_CPU_s=h['CPU_reserved_s'],Hopper_storage_bytes=h['storage_reserved_bytes'])
    s['dec1_used'].update(H200_s=251,PBS_submissions=2,Hopper_CPU_s=d['CPU_s'],Hopper_storage_bytes=34*a.GIB)
    s['dec1_reserved']=dict(s['reserved'])
    s['storage_peaks']['Hopper_storage_bytes']=62*a.GIB;s['dec1_storage_peaks']['Hopper_storage_bytes']=34*a.GIB
    s['current_native_request_work_id']='TRAIN:A_CONTROL:LONG'
    s['work_reservations']=[dict(work_id=s['current_native_request_work_id'],job_id=args['job_id'],kind='TRAIN',
        submission_already_charged=True,resources=s['reserved'])]
    s['actual_Hopper_ledger_sha256']=hashlib.sha256(raw(h)).hexdigest()
    s['actual_DEC1_ledger_sha256']=hashlib.sha256(raw(d)).hexdigest()
    return dict(request_bytes=args['request_bytes'],ledger_bytes=raw(h),DEC1_ledger_bytes=raw(d),
        snapshot_bytes=raw(s),progress_bytes=raw(p),plan_bytes=raw(plan),amendment_bytes=raw(amendment()),
        now_epoch=args['now'],job_id=args['job_id'])


def prepared(args):
    inputs=future_inputs(args)
    with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(inputs['amendment_bytes']).hexdigest()):
        args['budget_proof']=future.recompute(**inputs)
    return args


class A40AllocationTests(unittest.TestCase):
    def validate(self,args):
        checksum=hashlib.sha256(args['grant_bytes']).hexdigest()
        with patch.object(dual,'GRANT_SHA',checksum),patch.object(a,'GRANT_SHA',checksum):
            return a.validate_native(**args)

    def test_real_type_distinguishes_a40_and_preserves_full_job_and_future_reserves(self):
        f=prepared(fixture());v=self.validate(f)
        self.assertEqual(v['status'],a.STATUS);self.assertEqual(v['physical_GPU'],'NVIDIA_A40')
        self.assertFalse(v['request_or_native_resources_modified']);self.assertEqual(v['actual_process_CPU_limit'],4)
        self.assertEqual(f['budget_proof']['forecast']['original_final_protection']['H200_s'],72000)
        self.assertEqual(f['budget_proof']['forecast']['future_required_additional']['PBS_submissions'],4)

    def test_capacity_rewrite_is_cold_declared_not_active_cpu_or_free_billing(self):
        f=prepared(fixture());limits=f['job']['Resource_List']
        limits.update(select='1:ngpus=1:ncpus=36:mem=240gb',ncpus=36,mem='240gb')
        v=self.validate(f);self.assertEqual(v['native_allocated_ncpus'],36)
        self.assertEqual(v['actual_process_CPU_limit'],4)
        f['project_proof']['balance_CPU_hours']=100
        with self.assertRaises(PermissionError):self.validate(f)

    def test_model_is_not_admitted_on_wrong_native_or_stale_auth_context(self):
        mutations=[lambda f:f.update(hostname='stdct-login-01'),lambda f:f['job'].update(job_state='Q'),
            lambda f:f['job'].update(project='OTHER'),lambda f:f['job'].update(run_count=2),
            lambda f:f['job']['Variable_List'].update(PV1_SUBMISSION_NONCE='other'),
            lambda f:f['project_proof'].update(observed_epoch=f['now']-61),
            lambda f:f['project_proof']['project'].update(users=[]),
            lambda f:f['queue_proof'].update(maximum_wall_s=1800),
            lambda f:f['job']['Resource_List'].update(select='1:ngpus=1:ncpus=12:mem=128gb',ncpus=12,mem='128gb'),
            lambda f:f['campaign_view'].update(observed_epoch=f['now']-121),
            lambda f:f['quota'].update(available_bytes=50*a.GIB)]
        for change in mutations:
            with self.subTest(change=change):
                f=prepared(fixture());change(f)
                with self.assertRaises(PermissionError):self.validate(f)

    def test_changed_request_missing_envelope_and_forged_budget_fail(self):
        changes=[lambda f:f['ledger']['reserved'].update(optimizer_updates=4095),
            lambda f:f['ledger'].update(CPU_reserved_s=0),
            lambda f:f['ledger'].update(unknown_effects=['unknown']),
            lambda f:f['budget_proof'].update(global_ledger_canonical_sha256='f'*64),
            lambda f:f['budget_proof']['forecast'].update(admissible_forecast=False),
            lambda f:f['budget_proof']['forecast']['mandatory_jobs'][0].update(stop_step=1024),
            lambda f:f.update(request_bytes=f['request_bytes']+b' ')]
        for change in changes:
            with self.subTest(change=change):
                f=prepared(fixture());change(f)
                with self.assertRaises(PermissionError):self.validate(f)

    def test_request_remains_source_only_and_no_new_clock_or_recipe(self):
        for key,value in [('platform_training_version','legacy'),('new_bank_manifest_sha256','f'*64),
            ('statistics_sha256','f'*64),('new_robot_physics',1),('LOCKBOX_opened',True),
            ('walltime_s',28801),('CPU_upper_s',0),('submit_count',0),('run_uuid','../escape'),
            ('hard_close_epoch',a.HARD_CLOSE+1)]:
            with self.subTest(key=key):
                f=prepared(fixture());r=json.loads(f['request_bytes']);r[key]=value;f['request_bytes']=raw(r)
                with self.assertRaises(PermissionError):self.validate(f)

    def test_resume_cannot_import_other_arm_or_duplicate_json_keys(self):
        f=fixture();r=json.loads(f['request_bytes']);r['expected_native_start_step']=32
        r['training_block']['resume_checkpoint_relative']='training/B_COVERAGE/checkpoint-32'
        with self.assertRaises(PermissionError):a.request_scope(r,now=f['now'])
        f=prepared(fixture());f['request_bytes']=f['request_bytes'].rstrip()[:-1]+b',"stage":"P2_PRIMARY"}'
        with self.assertRaises(PermissionError):self.validate(f)

    def test_future_recomputes_native_counters_and_whole_remaining_work(self):
        for change in ('global','DEC1','matrix','H200_as_A40','reserve','native_bytes'):
            with self.subTest(change=change):
                inputs=future_inputs(fixture())
                if change in ('global','DEC1','reserve','native_bytes'):
                    s=json.loads(inputs['snapshot_bytes'])
                    if change=='global':s['used']['H200_s']=0
                    if change=='DEC1':s['dec1_used']['PBS_submissions']=0
                    if change=='reserve':s['reserved']['optimizer_updates']=0
                    if change=='native_bytes':s['actual_Hopper_ledger_sha256']='f'*64
                    inputs['snapshot_bytes']=raw(s)
                else:
                    p=json.loads(inputs['plan_bytes'])
                    if change=='matrix':p['remaining_matrix_envelope']['core_cells']=32
                    if change=='H200_as_A40':p['H200_timing_is_A40_measurement']=True
                    inputs['plan_bytes']=raw(p)
                with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(inputs['amendment_bytes']).hexdigest()),self.assertRaises(PermissionError):
                    future.recompute(**inputs)


if __name__=='__main__':unittest.main()
