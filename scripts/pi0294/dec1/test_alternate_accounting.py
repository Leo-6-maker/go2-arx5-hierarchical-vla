"""Synthetic ledger/native receipts, no host commands or model work."""
from copy import deepcopy
import unittest
import uuid
from unittest.mock import patch

from scripts.pi0294.pv1.neural_ledger import check
from scripts.pi0294.pv1.pbs_owner import dependency_for,native_chain
from . import alternate_accounting as accounting
from .authority import AUTHORITY,PARENT_AUTHORITY,HARD_CLOSE

NOW=1792530196


def fixture():
    value=dict(schema='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1',authority_release=PARENT_AUTHORITY,
        campaign_run_uuid='20261018T182956Z-localpc-pv1',unknown_effects=[],pending_updates=[],revision=100,
        CPU_s=16030.,generation_queries=1054,optimizer_updates=0,
        used=dict(H200_s=20494,PBS_submissions=11,neural_queries=1054,optimizer_updates=0),
        reserved=dict(H200_s=0,neural_queries=0),
        PBS_submissions=[dict(job_id=str(i)+'.hopper-m-02',status='NATIVE_TERMINAL',
            request_id='old-'+str(i)) for i in range(11)],new_storage_peak_bytes=9*1024**3,new_storage_current_bytes=9*1024**3)
    request=dict(schema='DEC1_ALTERNATE_QUALIFICATION_REQUEST_V1',scientific_admission_version=accounting.VERSION,
        authority_release=PARENT_AUTHORITY,authority_amendment=AUTHORITY,campaign_run_uuid=value['campaign_run_uuid'],
        original_T0_epoch=1792348196,hard_close_epoch=HARD_CLOSE,submit_count=12,walltime_s=1800,ncpus=4,memory_gb=32,
        created_epoch=NOW,model_service_or_training_or_physics=False,remaining_declared_source_and_matrix_GPU_s=155220,
        remaining_declared_source_and_matrix_PBS=7,DEC1_CPU_used_s=9540,source_sha='a'*40,
        run_uuid='dec1-vanda-compat-01',request_id=str(uuid.uuid4()),submission_nonce=str(uuid.uuid4()))
    native=dict(schema='DEC1_DUAL_SITE_EXCLUSIVE_PREQUEUE_PROOF_V1',observed_epoch=NOW,
        Hopper_owned_jobs=[],Vanda_owned_jobs=[],old_679029_terminal=True,original_parent_STOP=False,
        original_unknown_effects=[],Vanda_project_eligible_and_covers_request=True,single_existing_local_owner=True,
        ledger_sha256=accounting.digest(value),request_sha256=accounting.digest(request),
        run_manifest_sha256='d'*64,cold_input_audit_sha256='e'*64)
    return value,request,native


class AccountingTests(unittest.TestCase):
    def reserved(self):
        value,self.request,self.native=fixture()
        return accounting.reserve(value,self.request,now_epoch=NOW,fresh_native=self.native)

    def bound(self):
        return accounting.bind_job(self.reserved(),self.request['request_id'],'1469999.stdct-mgmt-02',now_epoch=NOW+1)

    def terminal(self,value):
        item=value['PBS_submissions'][-1]
        native=dict(schema='DEC1_NATIVE_VANDA_TERMINAL_V1',job_id=item['job_id'],job_state='F',project=item['project'],
            job_name=item['job_name'],owner='e1192548',ngpus=1,request_id=item['request_id'],
            submission_nonce=item['submission_nonce'],recorded_process_terminal=True,observed_epoch=NOW+200,
            conservative_GPU_s=120,conservative_CPU_s=87.,Exit_status=0)
        effects=dict(schema='DEC1_SAVED_TRAIN_EFFECT_SUMMARY_V1',request_id=item['request_id'],job_id=item['job_id'],
            request_sha256=item['request_sha256'],physical_or_optimizer_effects=0,query_attempts=17,
            model_load_attempts=1,unknown_query_intents=[])
        return native,effects

    def test_reservation_changes_same_global_totals_not_old_history(self):
        original,request,native=fixture();before=deepcopy(original)
        value=accounting.reserve(original,request,now_epoch=NOW,fresh_native=native)
        self.assertEqual(original,before);self.assertEqual(value['PBS_submissions'][:11],original['PBS_submissions'])
        self.assertEqual(value['used']['PBS_submissions'],12)
        self.assertEqual(value['reserved']['H200_s'],1860);self.assertEqual(value['reserved']['neural_queries'],17)
        self.assertEqual(value['generation_queries'],1054);self.assertEqual(value['optimizer_updates'],0)
        self.assertEqual(value['CPU_reserved_s'],7440)
        self.assertTrue(value['PBS_submissions'][-1]['legacy_GPU_seconds_field_is_projection'])
        check(value)

    def test_live_external_job_blocks_legacy_dependency_and_hopper_queue_reads(self):
        value=self.bound()
        with self.assertRaisesRegex(PermissionError,'FOREIGN_SERVER'):dependency_for(value)
        with patch('scripts.pi0294.pv1.pbs_owner.owned_jobs',side_effect=AssertionError('NO_HOPPER_QUERY')):
            with self.assertRaisesRegex(PermissionError,'FOREIGN_SERVER'):native_chain(value)

    def test_unknown_reply_keeps_all_reservations_and_cannot_resubmit(self):
        value=self.reserved()
        with self.assertRaises(PermissionError):accounting.bind_job(value,self.request['request_id'],'not-a-job',now_epoch=NOW+1)
        with self.assertRaises(PermissionError):accounting.reserve(value,self.request,now_epoch=NOW,fresh_native=self.native)
        self.assertIsNone(value['PBS_submissions'][-1]['job_id']);self.assertEqual(value['reserved']['H200_s'],1860)

    def test_fresh_site_permission_exclusion_and_historical_baseline_are_required(self):
        changes=[lambda v,r,n:n.update(Hopper_owned_jobs=['679029.hopper-m-02']),
            lambda v,r,n:n.update(Vanda_owned_jobs=['foreign.stdct-mgmt-02']),
            lambda v,r,n:n.update(old_679029_terminal=False),lambda v,r,n:n.update(Vanda_project_eligible_and_covers_request=False),
            lambda v,r,n:n.update(observed_epoch=NOW-121),lambda v,r,n:n.update(ledger_sha256='f'*64),
            lambda v,r,n:r.update(original_T0_epoch=NOW),lambda v,r,n:r.update(remaining_declared_source_and_matrix_PBS=1),
            lambda v,r,n:r.update(walltime_s=28800),lambda v,r,n:r.update(DEC1_CPU_used_s=96*3600),
            lambda v,r,n:v.update(unknown_effects=['unknown']),lambda v,r,n:v['used'].update(H200_s=0)]
        for change in changes:
            with self.subTest(change=change):
                v,r,n=fixture();change(v,r,n)
                with self.assertRaises(PermissionError):accounting.reserve(v,r,now_epoch=NOW,fresh_native=n)

    def test_terminal_native_and_actual_journal_settle_once_preserving_reserve(self):
        value=self.bound();native,effects=self.terminal(value)
        settled=accounting.settle(value,self.request['request_id'],native_proof=native,effects=effects,now_epoch=NOW+201)
        self.assertEqual(settled['used']['H200_s'],20494+120);self.assertEqual(settled['generation_queries'],1071)
        self.assertEqual(settled['used']['PBS_submissions'],12);self.assertEqual(settled['reserved']['H200_s'],0)
        self.assertEqual(settled['reserved']['neural_queries'],0);self.assertEqual(settled['CPU_reserved_s'],0)
        self.assertEqual(settled['CPU_s'],16030+87);self.assertEqual(settled['model_loads'],1)
        self.assertEqual(settled['PBS_submissions'][:11],value['PBS_submissions'][:11]);check(settled)
        with self.assertRaisesRegex(PermissionError,'ALREADY_SETTLED'):
            accounting.settle(settled,self.request['request_id'],native_proof=native,effects=effects,now_epoch=NOW+202)
        self.assertIsNone(dependency_for(settled))

    def test_wrong_job_nonce_or_possible_unsettled_query_never_looks_clean(self):
        value=self.bound();native,effects=self.terminal(value)
        for field,new in [('job_id','1469998.stdct-mgmt-02'),('submission_nonce',str(uuid.uuid4())),
                          ('recorded_process_terminal',False),('conservative_GPU_s',9999)]:
            with self.subTest(field=field):
                p=deepcopy(native);p[field]=new
                with self.assertRaises(PermissionError):
                    accounting.settle(value,self.request['request_id'],native_proof=p,effects=effects,now_epoch=NOW+201)
        effects.update(query_attempts=2,unknown_query_intents=[1])
        settled=accounting.settle(value,self.request['request_id'],native_proof=native,effects=effects,now_epoch=NOW+201)
        self.assertEqual(settled['generation_queries'],1056);self.assertTrue(settled['unknown_effects'])
        with self.assertRaises(PermissionError):check(settled)

    def test_dropped_query_reservation_or_untyped_foreign_entry_is_rejected(self):
        for change in (lambda v:v['reserved'].update(neural_queries=0),
                lambda v:v['PBS_submissions'][-1].update(native_platform='another-site'),
                lambda v:v['PBS_submissions'][-1].update(scientific_admission_version='legacy'),
                lambda v:v.update(CPU_s=160*3600)):
            with self.subTest(change=change):
                value=self.reserved();change(value)
                with self.assertRaises(PermissionError):check(value)


if __name__=='__main__':unittest.main()
