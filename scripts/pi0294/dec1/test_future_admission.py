"""CPU bridge fixtures; no actual quota, GPU, training or remote permissions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import future_admission as mature,neural
from scripts.pi0294.pv1.dataset import POPULATION
from scripts.pi0294.pv1.evaluation_plan import POPULATION_FILE
from scripts.pi0294.test_mp1_transport_fixture import put
from scripts.pi0294.test_pv1_future_budget import raw
from . import test_source_admission as source_fixture
from .test_future_budget import fixtures,NOW,T0
from .source_admission import COPY,VERSION
from .authority import AUTHORITY,PARENT_AUTHORITY,PARENT
from .future_admission import INDEX,bounded_process_cpu


class BridgeTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root,values,authority,request,_,self.path_sha=source_fixture.StageTests().fixture(tmp.name)
        self.values=values;self.authority=authority;self.request=request
        self.s,self.p,self.plan=fixtures()
        self.request.update(schema='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1',authority_release=PARENT_AUTHORITY,
            run_uuid=self.root.name,campaign_run_uuid='original',source_sha='b'*40,created_epoch=NOW,
            request_id='source-request',submission_nonce='source-nonce',expected_native_start_step=0,
            walltime_s=10800,additional_output_upper_bytes=26*10**9,job_name='pi0294_pv1_12',project='CFP04-CF-007',
            dec1_phase_Hopper_CPU_upper_s=40000)
        self.values[COPY]['accounting']=dict(original_remaining_after_spent_reservations_and_final=
            dict(GPU_submissions=13,neural_queries=37946,optimizer_updates=16384))
        put(self.root.parent.parent/'authority/DEC1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json',self.values[COPY],True)
        native_raw=(self.root.parent.parent/'authority/DEC1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json').read_bytes()
        self.values['plans/DEC1_SOURCE_TRAIN_ADMISSION.json']['adoption_sha256']=hashlib.sha256(native_raw).hexdigest()
        old_jobs=[dict(status='NATIVE_TERMINAL',job_id=str(i)+'.hopper-m-02',request_id=str(i),run_uuid='old-'+str(i),
                       request_sha256='c'*64,project='CFP04-CF-007') for i in range(11)]
        self.ledger=dict(schema='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1',authority_release=PARENT_AUTHORITY,
            campaign_run_uuid='original',unknown_effects=[],pending_updates=[],revision=1,CPU_s=7000,
            used=dict(H200_s=20494,PBS_submissions=11,optimizer_updates=0,neural_queries=1054),
            reserved={},generation_queries=1054,optimizer_updates=0,PBS_submissions=old_jobs,training_runs={})
        baseline=deepcopy(self.ledger);baseline['CPU_s']=6820
        self.s['used'].update(neural_queries=1054,optimizer_updates=0)
        self.work=dict(schema='DEC1_FUTURE_REQUEST_WORK_BINDING_V1',stage='P2_PRIMARY',arm='A_CONTROL',
            replica=False,start_step=0,stop_step=1024,probe_only=False,work_id='TRAIN:A_CONTROL:1')
        self.index=dict(schema='DEC1_SOURCE_FUTURE_BUDGET_INPUT_INDEX_V1',authority_release=AUTHORITY,
            campaign_run_uuid='original',T0_epoch=T0,runtime_refresh_policy='SAME_OWNER_SAME_JOB_FRESH_RECOMPUTATION_V1',
            plan='plans/dec1-plan.json',request_work_binding='plans/dec1-work.json',baseline_hopper_ledger='plans/baseline-hopper.json',
            performance_evidence='plans/performance.json',checkpoint_size_evidence='plans/checkpoint-sizes.json',
            storage_lifecycle_evidence='plans/storage-lifecycle.json')
        self.values.update({INDEX:self.index,self.index['request_work_binding']:self.work,
            self.index['baseline_hopper_ledger']:baseline,
            self.index['performance_evidence']:dict(schema='DEC1_PRIOR_PARENT_STEP_TIMING_BOUND_V1',
                parent_checkpoint_manifest_sha256=PARENT,not_current_PV1_measurement=True,
                new_model_loads=0,new_updates=0,new_physics=0,declared_upper_multiplier=1.2,
                step_logs=[dict(file_sha256='d'*64,first_step=1,last_step=2,count=2,maximum_s=3.)]),
            self.index['checkpoint_size_evidence']:dict(new_model_loads=0,checkpoint_members=[dict(name='model',bytes=1024)],total_payload_bytes=1024),
            self.index['storage_lifecycle_evidence']:dict(status='PREDECLARED_PLAN_NOT_RETIREMENT_OR_READBACK')})
        self.plan['performance_evidence_sha256']=hashlib.sha256(raw(self.values[self.index['performance_evidence']])).hexdigest()
        self.plan['checkpoint_size_evidence_sha256']=hashlib.sha256(raw(self.values[self.index['checkpoint_size_evidence']])).hexdigest()
        self.plan['storage_lifecycle']['implementation_or_execution_plan_sha256']=hashlib.sha256(raw(self.values[self.index['storage_lifecycle_evidence']])).hexdigest()
        self.values[self.index['plan']]=self.plan
        self.bind()

    def bind(self):
        for name,value in self.values.items():
            if name==COPY:
                put(self.root/name,value,True)
            else:
                path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():path.chmod(0o600)
                path.write_bytes(raw(value));path.chmod(0o400)
        self.population_raw=(self.root/POPULATION_FILE).read_bytes()
        self.actual_digest=hashlib.sha256
        digest=lambda b:POPULATION if b==self.population_raw else self.actual_digest(b).hexdigest()
        m=dict(schema='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1',owner='EXISTING_LOCAL_PC',run_uuid=self.root.name,
            authority=self.authority,code=dict(vla_commit='b'*40),limits=dict(PV1_T0_epoch=T0),
            execution=dict(sealed_inputs={n:digest((self.root/n).read_bytes()) for n in self.values}))
        put(self.root/'RUN_MANIFEST.json',m,True)
        self.request.update(run_manifest_sha256=digest((self.root/'RUN_MANIFEST.json').read_bytes()),sealed_inputs=m['execution']['sealed_inputs'])
        self.write_ledger()

    def write_ledger(self):
        path=self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';put(path,self.ledger)
        self.s.update(hopper_ledger_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hopper_ledger_revision=self.ledger['revision'])

    def recompute(self,purpose='PREPARE'):
        actual=hashlib.sha256
        digest=lambda b:POPULATION if b==self.population_raw else actual(b).hexdigest()
        with patch.object(mature,'_digest',side_effect=digest),patch.object(neural,'sha',side_effect=self.path_sha):
            return mature.recompute(self.root,request_bytes=raw(self.request),snapshot_bytes=raw(self.s),
                progress_bytes=raw(self.p),purpose=purpose,now_epoch=NOW)

    def test_new_bridge_accepts_partial_b0_only_with_full_native_amendment(self):
        result=self.recompute()
        self.assertEqual(result['authority_amendment'],AUTHORITY)
        self.assertTrue(result['full_B0_not_fabricated'])
        self.assertEqual(json.loads((self.root/'plans/PV1_B0_DEV_ADMISSION.json').read_text())['status'],'PARTIAL14')
        self.assertTrue(result['forecast']['admissible_forecast'])
        self.assertFalse(result['old_PV1_release_or_LOCKBOX_admitted'])

    def test_native_ledger_or_adoption_baseline_cannot_be_a_fake_zero(self):
        self.ledger['generation_queries']+=1
        put(self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json',self.ledger)
        with self.assertRaises(PermissionError):self.recompute()
        self.ledger['generation_queries']-=1
        self.write_ledger()
        self.values[self.index['baseline_hopper_ledger']]['generation_queries']=0;self.bind()
        with self.assertRaisesRegex(PermissionError,'BASELINE'):self.recompute()

    def test_request_phase_output_residency_and_all_remaining_arms_are_checked(self):
        for change in (
            lambda:self.request.update(dec1_phase_Hopper_CPU_upper_s=1),
            lambda:self.request.update(additional_output_upper_bytes=1),
            lambda:self.request.update(walltime_s=20000),
            lambda:self.s['dec1_used'].update(Hopper_storage_bytes=60*1024**3),
        ):
            oldr=deepcopy(self.request);olds=deepcopy(self.s);change()
            with self.assertRaises(PermissionError):self.recompute()
            self.request=oldr;self.s=olds
        self.plan['training_jobs'].pop();self.bind()
        with self.assertRaises(PermissionError):self.recompute()

    def test_forged_request_version_does_not_bypass_legacy_or_missing_authority(self):
        self.authority.clear();self.bind()
        with self.assertRaises(PermissionError):self.recompute()

    def test_process_cpu_limit_is_the_smaller_immutable_phase_or_global_bound(self):
        self.assertEqual(bounded_process_cpu(self.request,100000),40000)
        self.assertEqual(bounded_process_cpu(self.request,30000),30000)
        self.assertEqual(bounded_process_cpu({},100000),100000)
        with self.assertRaises(PermissionError):bounded_process_cpu(dict(self.request,dec1_phase_Hopper_CPU_upper_s=True),100000)

    def test_prior_timing_is_bound_and_cannot_claim_unmeasured_fast_updates(self):
        timing=self.values[self.index['performance_evidence']]
        for key,value in (('new_model_loads',1),('not_current_PV1_measurement',False),
                          ('parent_checkpoint_manifest_sha256','f'*64),('declared_upper_multiplier',1.0)):
            prior=timing[key];timing[key]=value
            self.plan['performance_evidence_sha256']=hashlib.sha256(raw(timing)).hexdigest();self.bind()
            with self.assertRaises(PermissionError):self.recompute()
            timing[key]=prior
        timing['step_logs'][0]['maximum_s']=5.7
        self.plan['performance_evidence_sha256']=hashlib.sha256(raw(timing)).hexdigest();self.bind()
        with self.assertRaisesRegex(PermissionError,'TIMING_UPPER'):self.recompute()

    def test_mature_same_job_native_attestation_recomputes_the_dec1_branch(self):
        job='new.hopper-m-02';reqsha=hashlib.sha256(raw(self.request)).hexdigest()
        self.ledger['PBS_submissions'].append(dict(job_id=job,request_id=self.request['request_id'],
            request_sha256=reqsha,status='SUBMITTED_PENDING_NATIVE',run_uuid=self.root.name,
            reserved_H200_s=10800,accounting_pad_s=60,additional_output_upper_bytes=26*10**9,
            scientific_admission_version=VERSION,dec1_work_id=self.work['work_id']))
        self.ledger['used']['PBS_submissions']+=1;self.ledger['reserved']=dict(H200_s=10860)
        self.s['used']['PBS_submissions']+=1;self.s['dec1_used']['PBS_submissions']=1
        self.s['reserved']=dict(H200_s=10860,Hopper_storage_bytes=26*10**9)
        self.s['dec1_reserved']=deepcopy(self.s['reserved']);self.s['current_native_request_work_id']=self.work['work_id']
        self.s['work_reservations']=[dict(work_id=self.work['work_id'],job_id=job,kind='TRAIN',
            submission_already_charged=True,resources=deepcopy(self.s['reserved']))];self.write_ledger()
        native=dict(job_state='R',Job_Name=self.request['job_name'],project=self.request['project'],
            Rerunable=False,run_count=1,Variable_List=dict(PV1_REQUEST_ID=self.request['request_id'],PV1_SUBMISSION_NONCE=self.request['submission_nonce']))
        actual=hashlib.sha256;digest=lambda b:POPULATION if b==self.population_raw else actual(b).hexdigest()
        with patch.object(mature,'_digest',side_effect=digest),patch.object(neural,'sha',side_effect=self.path_sha):
            value=mature.release_attestation(self.root,request_bytes=raw(self.request),snapshot_bytes=raw(self.s),
                progress_bytes=raw(self.p),native_job_bytes=raw(native),job_id=job,phase_started_epoch=NOW-1,now_epoch=NOW)
            value['recomputed_proof']={'status':'FORGED_SUMMARY_IGNORED'}
            verified=mature.verify_release(self.root,request_bytes=raw(self.request),attestation_bytes=raw(value),
                expected_job_id=job,phase_started_epoch=NOW-1,now_epoch=NOW)
            self.assertEqual(verified['authority_amendment'],AUTHORITY)


if __name__=='__main__':unittest.main()
