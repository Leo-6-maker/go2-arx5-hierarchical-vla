"""CPU-only fresh owner-budget fixtures, not native permissions or timing."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import future_admission as admission
from scripts.pi0294.test_pv1_future_budget import (
    NOW,T0,HASH,assumptions,baseline,case,observation,progress,raw,snapshot,training_jobs)
from scripts.pi0294.test_mp1_transport_fixture import put


class FutureAdmission(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)/'campaign/children/cold';self.root.mkdir(parents=True)
        self.inputs={};self.p=progress();self.s=snapshot();self.a=assumptions()
        self.a.update(case_floor=case()['declared_floor'],case_headroom_multiplier=1.25,
            unmeasured_audit_CPU_s_upper=120,unmeasured_audit_wall_s_upper=120)
        self.obs=observation();self.result=json.loads(self.obs['result'])
        manifest=raw(dict(files={'raw':'hash_fixture_only'}))
        self.cost=dict(schema='PI0294_PV1_TERMINAL_COST_READBACK_V1',
            status='PASS_READ_ONLY_TERMINAL_COST_READBACK',authority_release=admission.AUTHORITY,
            campaign_run_uuid='fixture',attempt_id=self.result['id'],observed_epoch=NOW-100,
            result_sha256=admission._digest(self.obs['result']),
            adjudication_sha256=admission._digest(self.obs['adjudication']),
            independent_audit_sha256=admission._digest(self.obs['audit']),
            causal_chain_sha256=admission._digest(self.obs['causal_chain']),
            raw_file_manifest_sha256=admission._digest(manifest),terminal_process_verified=True,
            unknown_effects=[],inventory_stable=True,regular_file_count=10,regular_file_bytes=10000000,
            allocated_file_bytes=12000000,inventory_metadata_sha256=HASH,observer_CPU_s=.1,
            observer_wall_s=.2,prior_audit_CPU_s=None,prior_audit_wall_s=None,
            new_integrations=0,new_model_queries=0,new_native_constructors=0)
        paths={k:'plans/actual/'+k+'.json' for k in admission.EVIDENCE_KEYS}
        for kind in ('result','adjudication','audit','causal_chain'):
            self.inputs[paths[kind]]=self.obs[kind]
        self.inputs[paths['cost_readback']]=raw(self.cost)
        self.inputs[paths['raw_file_manifest']]=manifest
        self.block=dict(arm='A_CONTROL',replica=False,stop_after_step=1024,resume_checkpoint_relative=None)
        self.work=dict(schema='PI0294_PV1_FUTURE_REQUEST_WORK_BINDING_V1',stage='P2_PRIMARY',
            arm='A_CONTROL',replica=False,start_step=0,stop_step=1024,
            forecast_candidate='A_CONTROL',work_id='TRAIN:A_CONTROL:1')
        jobs=training_jobs(self.p,case(),self.a)
        for entry in jobs:
            if entry['work_id']=='TRAIN:A_CONTROL':
                one=entry['blocks'][0];one['optimizer_updates']=1024;one['resources']['optimizer_updates']=1024
                two=deepcopy(one);two.update(optimizer_updates=3072,TRAIN_probes=0)
                two['resources']['optimizer_updates']=3072
                entry['blocks']=[one,two]
        self.index=dict(schema='PI0294_PV1_FUTURE_BUDGET_INPUT_INDEX_V1',authority_release=admission.AUTHORITY,
            runtime_refresh_policy='SAME_OWNER_SAME_JOB_FRESH_RECOMPUTATION_V1',
            campaign_run_uuid='fixture',T0_epoch=T0,replica_mode='REQUIRED',case_observations=[paths],
            baseline_admission='plans/baseline.json',assumptions='plans/assumptions.json',
            training_jobs='plans/training-jobs.json',request_work_binding='plans/work.json')
        self.inputs.update({self.index['baseline_admission']:baseline(),
            self.index['assumptions']:raw(self.a),self.index['training_jobs']:raw(dict(jobs=jobs)),
            self.index['request_work_binding']:raw(self.work),
            admission.INDEX:raw(self.index),'plans/PRIMARY_TRAIN_BLOCK.json':raw(self.block)})
        self.ledger=dict(schema='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1',authority_release=admission.AUTHORITY,
            campaign_run_uuid='fixture',unknown_effects=[],pending_updates=[],revision=1,CPU_s=0,
            used=dict(H200_s=0,PBS_submissions=0,optimizer_updates=0,neural_queries=0),reserved={},
            generation_queries=0,optimizer_updates=0,PBS_submissions=[],training_runs={})
        self.request=dict(schema='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1',authority_release=admission.AUTHORITY,
            run_uuid='cold',campaign_run_uuid='fixture',source_sha='b'*40,stage='P2_PRIMARY',created_epoch=NOW,
            request_id='fixture-request',submission_nonce='fixture-nonce',training_block=self.block,
            expected_native_start_step=0,walltime_s=18000,additional_output_upper_bytes=1,
            job_name='pi0294_pv1_01',project='CFP04-CF-007')
        self.bind()

    def bind(self):
        for name,value in self.inputs.items():
            path=self.root/name
            if path.exists():path.chmod(0o600)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value);path.chmod(0o400)
        m=dict(schema='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1',owner='EXISTING_LOCAL_PC',run_uuid='cold',
            authority=dict(protocol_ref=admission.AUTHORITY),code=dict(vla_commit='b'*40),
            limits=dict(PV1_T0_epoch=T0),execution=dict(sealed_inputs={n:admission._digest(v) for n,v in self.inputs.items()}))
        put(self.root/'RUN_MANIFEST.json',m,True)
        self.request['run_manifest_sha256']=admission._digest((self.root/'RUN_MANIFEST.json').read_bytes())
        self.request['sealed_inputs']=m['execution']['sealed_inputs']
        self.write_ledger()

    def write_ledger(self):
        put(self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json',self.ledger)
        self.s.update(hopper_ledger_sha256=admission._digest((self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes()),
            hopper_ledger_revision=self.ledger['revision'])

    def recompute(self,purpose='PREPARE',now=NOW):
        return admission.recompute(self.root,request_bytes=raw(self.request),snapshot_bytes=raw(self.s),
            progress_bytes=raw(self.p),purpose=purpose,now_epoch=now)

    def test_original_cost_readback_becomes_explicit_derived_view_not_original_worker(self):
        result=self.recompute();forecast=result['forecast']
        self.assertEqual(result['status'],'PASS_RECOMPUTED_FORECAST_NOT_NATIVE_ALLOCATION_OR_GO')
        evidence=forecast['case_cost_evidence']
        self.assertEqual(evidence[0]['result_sha256'],self.cost['result_sha256'])
        c=admission._case_cost(self.index,self.inputs,self.a,NOW)
        self.assertFalse(c['original_worker_receipt_claimed'])
        self.assertEqual(c['actual_read_side_cost_provenance'][0]['unmeasured_prior_audit_dimensions'],['CPU','wall'])
        self.assertGreaterEqual(c['resources']['Core_storage_bytes'],15000000)

    def test_original_cost_hashes_terminal_inventory_and_unknowns_cannot_be_faked_by_pass(self):
        original=deepcopy(self.cost);path=self.index['case_observations'][0]['cost_readback']
        for change in [dict(result_sha256='c'*64),dict(inventory_stable=False),
                dict(terminal_process_verified=False),dict(unknown_effects=['UNKNOWN']),
                dict(new_integrations=1)]:
            with self.subTest(change=change):
                self.inputs[path]=raw(dict(original,**change));self.bind()
                with self.assertRaisesRegex(PermissionError,'TERMINAL_COST'):self.recompute()

    def test_missing_unmeasured_audit_upper_is_not_fabricated_measurement(self):
        del self.a['unmeasured_audit_CPU_s_upper']
        self.inputs[self.index['assumptions']]=raw(self.a);self.bind()
        with self.assertRaises(KeyError):self.recompute()

    def test_full_baseline_and_fresh_same_campaign_native_projection_required(self):
        for change in [dict(unknown_effects=['UNKNOWN']),dict(pending_updates=['UPDATE']),
                dict(revision=2),dict(generation_queries=1)]:
            with self.subTest(change=change):
                prior=deepcopy(self.ledger);self.ledger.update(change)
                put(self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json',self.ledger)
                with self.assertRaises(PermissionError):self.recompute()
                self.ledger=prior;self.write_ledger()
        value=json.loads(baseline());value['condition_result_sha256'].pop('0')
        self.inputs[self.index['baseline_admission']]=raw(value);self.bind()
        with self.assertRaisesRegex(PermissionError,'COMPLETE_ACTUAL_B0'):self.recompute()

    def test_unsealed_symlink_or_changed_input_and_request_block_are_denied(self):
        path=self.root/'plans/work.json';original=path.read_bytes();path.chmod(0o600);path.write_bytes(b'{}');path.chmod(0o400)
        with self.assertRaisesRegex(PermissionError,'MANIFEST_SEALED'):self.recompute()
        path.unlink();target=self.root/'OTHER';target.write_bytes(original);path.symlink_to(target)
        with self.assertRaisesRegex(PermissionError,'SYMLINK'):self.recompute()
        path.unlink();path.write_bytes(original);path.chmod(0o400)
        self.request['training_block']=dict(self.block,stop_after_step=4096)
        with self.assertRaisesRegex(PermissionError,'EXACT_NATIVE_BLOCK'):self.recompute()

    def test_native_release_requires_fresh_job_nonce_phase_and_recomputes_ignoring_pass_summary(self):
        job='123.hopper-m-02';req_sha=admission._digest(raw(self.request))
        item=dict(request_id=self.request['request_id'],job_id=job,run_uuid='cold',
            request_sha256=req_sha,status='SUBMITTED_PENDING_NATIVE',reserved_H200_s=18000,accounting_pad_s=60)
        self.ledger['PBS_submissions']=[item];self.ledger['used']['PBS_submissions']=1
        self.ledger['reserved']=dict(H200_s=18060)
        self.s['used']['PBS_submissions']=1;self.s['reserved']=dict(H200_s=18060)
        self.s['work_reservations']=[dict(work_id='TRAIN:A_CONTROL:1',job_id=job,kind='TRAIN',
            submission_already_charged=True,resources=dict(H200_s=18060))]
        self.s['reserved_sequential_wall_upper_s']=18120;self.write_ledger()
        native=dict(job_state='R',Job_Name=self.request['job_name'],project=self.request['project'],
            Rerunable=False,run_count=1,Variable_List=dict(PV1_REQUEST_ID=self.request['request_id'],
                PV1_SUBMISSION_NONCE=self.request['submission_nonce']))
        value=admission.release_attestation(self.root,request_bytes=raw(self.request),snapshot_bytes=raw(self.s),
            progress_bytes=raw(self.p),native_job_bytes=raw(native),job_id=job,phase_started_epoch=NOW-1,now_epoch=NOW)
        value['recomputed_proof']={'status':'FORGED_PASS_SUMMARY_IGNORED'}
        verify=lambda v:admission.verify_release(self.root,request_bytes=raw(self.request),attestation_bytes=raw(v),
            expected_job_id=job,phase_started_epoch=NOW-1,now_epoch=NOW)
        self.assertEqual(verify(value)['purpose'],'NATIVE_PRETRAIN')
        for change in [dict(job_id='999.hopper-m-02'),dict(submission_nonce='wrong'),
                dict(phase_started_epoch=NOW-2),dict(observed_epoch=NOW-121)]:
            with self.subTest(change=change),self.assertRaises(PermissionError):verify(dict(value,**change))
        stale=deepcopy(value);s=deepcopy(self.s);s['observed_epoch']=NOW-2
        stale['raw_snapshot_base64']=base64.b64encode(raw(s)).decode()
        with self.assertRaisesRegex(PermissionError,'NOT_FRESH_NATIVE'):verify(stale)

    def test_queue_hook_preserves_original_request_bytes_and_does_not_admit_missing_budget(self):
        for name,data in ((admission.SNAPSHOT,raw(self.s)),(admission.PROGRESS,raw(self.p))):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);path.chmod(0o400)
        request_raw=(json.dumps(self.request,indent=2)+'\n').encode()
        with patch.object(admission.time,'time',return_value=NOW):
            proof=admission.prequeue(self.root,self.request,'SUBMIT',request_bytes=request_raw)
            self.assertEqual(proof['request_sha256'],hashlib.sha256(request_raw).hexdigest())
            with self.assertRaisesRegex(PermissionError,'EXACT_PREQUEUE_REQUEST'):
                admission.prequeue(self.root,self.request,'SUBMIT',request_bytes=b'{}')
        (self.root/admission.SNAPSHOT).unlink()
        with self.assertRaises(PermissionError):admission.prequeue(self.root,self.request,'PREPARE')
        self.assertIsNone(admission.prequeue(self.root,dict(stage='B0_DEV'),'PREPARE'))
        self.assertIsNone(admission.prequeue(self.root,dict(stage='P2_PRIMARY',
            training_block=dict(probe_only=True)),'PREPARE'))

    def test_native_wait_has_finite_no_effect_timeout_and_verifies_exact_actual_request(self):
        directory=self.root/'pbs/alloc-01';directory.mkdir(parents=True)
        request_raw=(json.dumps(self.request,indent=2)+'\n').encode()
        path=directory/'REQUEST.json';path.write_bytes(request_raw);path.chmod(0o400)
        epoch=[NOW];checks=[]
        allocation=dict(job_id='123.hopper-m-02',allocation_end_epoch=NOW+10000)
        clock=lambda:epoch[0]
        def sleep(seconds):epoch[0]+=seconds
        with self.assertRaisesRegex(TimeoutError,'NO_NEW_UPDATES_OR_QUERIES'):
            admission.wait_native_release(self.root,directory,allocation,lambda:checks.append(True),now=clock,sleep=sleep)
        ready=json.loads((directory/'PV1_FUTURE_PHASE_READY.json').read_bytes())
        self.assertEqual(ready['request_sha256'],hashlib.sha256(request_raw).hexdigest())
        self.assertEqual(ready['new_updates'],0);self.assertEqual(ready['new_queries'],0)
        self.assertEqual(len(checks),175)
        with self.assertRaises(FileExistsError):
            admission.wait_native_release(self.root,directory,allocation,lambda:None,now=clock,sleep=sleep)


if __name__=='__main__':unittest.main()
