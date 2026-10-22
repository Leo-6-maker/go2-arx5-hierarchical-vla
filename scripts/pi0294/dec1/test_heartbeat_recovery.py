"""Small pure-contract fixtures. Written only; helper did not execute tests.

Fixed-receipt parsing is mocked explicitly in semantic-negative cases. The
separate actual grant test reads the published grant bytes after integration.
No fixture asserts a new native job or creates large payloads.
"""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch,Mock

from scripts.pi0294.dec1 import heartbeat_recovery as hr


def grant_value():
    return dict(schema='VAP1_A2_HEARTBEAT_RECOVERY_AUTHORIZATION_V1',
        status='HUMAN_AUTHORIZED_EXACT_PRESERVED_COST_RECOVERY',
        narrow_changes={'DEC1_cumulative_updates':[9216,9987],'DEC1_new_PBS':[11,12],
            'DEC1_GPU_site_output_GiB':[352,418],'global_GPU_site_output_GiB':[419,485]},
        original_T0_epoch=hr.T0,original_clock=dict(stop_new_jobs_epoch=1792866596,
            stop_training_epoch=1792909796,stop_physics_epoch=1792924196,hard_closeout_epoch=hr.HARD_CLOSE),
        failed_job=hr.FAILED_JOB,failed_request=hr.FAILED_REQUEST,
        closed_gpu_receipt_sha256=hr.CLOSURE_SHA,closed_control_receipt_sha256=hr.CONTROL_CLOSURE_SHA,
        seed_checkpoint2048_sha256=hr.SEED_CHECKPOINT_SHA,prior_storage_release=hr.storage.RELEASE,
        retirement_credit_bytes=0,all_checkpoints_preserved=True,
        recovery_semantics=dict(latest_complete_step=2048,saved_cursor=65536,failed_lineage_actual_step=2819,
            old_lost_updates=1024,new_lost_updates=771,all_charged_updates_before_recovery=3843,
            A_remaining_updates=2048,B_remaining_updates=4096,total_required_updates=9987,
            new_named_lineage_required=True,failed_lineage_or_cost_rollback=False,
            new_request_and_registration_required=True,B_independent_original_CORR=True,
            original12_A_probes_preserved_no_repeat=True,proposed_lineage_key=hr.LINEAGE,
            seed_full_state_step=2048,failed_lineage_preserved=hr.recovery.LINEAGE,
            seed_is_imported_state_not_new_charged_updates=True))


def request(arm='A_CONTROL'):
    return dict(heartbeat_recovery_release=hr.RELEASE,heartbeat_recovery_grant_sha256=hr.GRANT_SHA,
        recovery_authorization_release=hr.recovery.RELEASE,recovery_adoption_sha256=hr.recovery.GRANT_SHA,
        segmented_authorization_release=hr.seg.RELEASE,segmented_grant_sha256=hr.seg.GRANT_SHA,
        storage_authorization_release=hr.storage.RELEASE,storage_grant_sha256=hr.storage.GRANT_SHA,
        training_block=dict(arm=arm),training_lineage_key=hr.LINEAGE if arm=='A_CONTROL' else 'B_COVERAGE')


def fixtures():
    row=dict(request_id=hr.FAILED_REQUEST,job_id=hr.FAILED_JOB,status='NATIVE_TERMINAL',
        committed_optimizer_updates=771)
    h=dict(authority_release=hr.PARENT_AUTHORITY,campaign_run_uuid='20261018T182956Z-localpc-pv1',
        DEC1_source_pair_platform='VANDA_A40',optimizer_updates=3843,used=dict(optimizer_updates=3843),
        unknown_effects=[],pending_updates=[],reserved=dict(optimizer_updates=0,neural_queries=0),
        CPU_reserved_s=0,PBS_submissions=[deepcopy(row)],
        training_runs={'A_CONTROL':dict(last_native_step=1024,recipe_sha256=hr.recovery.RECIPE),
            hr.recovery.LINEAGE:dict(last_native_step=2819,recipe_sha256=hr.recovery.RECIPE)})
    d=dict(authority_release=hr.AUTHORITY,campaign_run_uuid=h['campaign_run_uuid'],optimizer_updates=3843,
        unknown_effects=[],pending_updates=[],GPU_reserved_s=0,CPU_reserved_s=0,new_submissions=[deepcopy(row)])
    gpu=dict(status='CLOSED_A2_HEARTBEAT_FAILURE_2819_PRESERVED',job_id=hr.FAILED_JOB,
        latest_complete_step=2048,checkpoints={'2048':dict(complete_sha256=hr.SEED_CHECKPOINT_SHA)},recoverable_full_state=False,
        prior_complete_state_preserved=True,retired_or_deleted_files=0,closed_epoch=1792680313.9795883,
        effects=dict(job_id=hr.FAILED_JOB,request_id=hr.FAILED_REQUEST,
            source_sha='3a1420e76c492196fa83905f4bde92924cc12c28',arm='A_CONTROL',start_native_step=2048,
            last_native_step=2819,optimizer_updates_committed=771,query_attempts=17,model_load_attempts=2,
            pending_updates=[],unknown_query_intents=[],recipe_sha256=hr.recovery.RECIPE,effect_journal_complete=True),
        settlement=dict(updates=3843,unknown_effects=[]))
    control=dict(status='CLOSED_CPU_A2_HEARTBEAT_FAILURE_CONTROL_AFTER_KNOWN_EFFECTS_SETTLEMENT',
        job_id=hr.FAILED_JOB,new_A_recovery_step=2819,old_A1024_costs_preserved=True,unknown_effects=[],
        new_GPU_PBS_models_queries_updates_physics=0,closed_epoch=1792680339.0070903,
        GPU_close=dict(lease_status='CLOSED',terminal_receipt_sha256=hr.CLOSURE_SHA,terminal_receipt=deepcopy(gpu)),
        settled_submission=deepcopy(row))
    return h,d,gpu,control


class HeartbeatRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.h,self.d,self.gpu,self.control=fixtures()
        self.grant=grant_value()
        self.old_validate=patch.object(hr.storage,'validate_adopted',return_value={}).start()
        self.old_marker=patch.object(hr.storage,'_marker',return_value={'adopted_epoch':hr.T0+1}).start()
        self.addCleanup(patch.stopall)
    def read(self,raw):
        return {b'grant':(self.grant,hr.GRANT_SHA),b'gpu':(self.gpu,hr.CLOSURE_SHA),
            b'control':(self.control,hr.CONTROL_CLOSURE_SHA)}[raw]
    def adopt(self):
        with patch.object(hr,'_read',side_effect=self.read):
            return hr.adopt(self.h,self.d,grant_bytes=b'grant',closure_bytes=b'gpu',
                control_closure_bytes=b'control',now_epoch=hr.APPROVED_EPOCH+1)

    def test_exact_published_grant_bytes(self):
        repo=Path(__file__).resolve().parents[3]
        path=repo/'reports/PI0294/design/parallel-science-20261022/A2_HEARTBEAT_RECOVERY_AUTHORIZATION_20261023.json'
        self.assertEqual(hr.grant(path.read_bytes())['failed_job'],hr.FAILED_JOB)
        with self.assertRaises(PermissionError):hr.grant(b'{}')

    def test_grant_four_adjustments_and_semantics_cannot_expand(self):
        for changes in [dict(self.grant['narrow_changes'],DEC1_new_PBS=[11,13]),
                dict(self.grant['narrow_changes'],global_GPU_site_output_GiB=[419,486])]:
            bad=deepcopy(self.grant);bad['narrow_changes']=changes
            with patch.object(hr,'_read',return_value=(bad,hr.GRANT_SHA)):
                with self.assertRaises(PermissionError):hr.grant(b'fixture')
        bad=deepcopy(self.grant);bad['recovery_semantics']['seed_is_imported_state_not_new_charged_updates']=False
        with patch.object(hr,'_read',return_value=(bad,hr.GRANT_SHA)):
            with self.assertRaises(PermissionError):hr.grant(b'fixture')

    def test_enabled_plan_or_snapshot_needs_only_exact_new_pair(self):
        self.assertTrue(hr.enabled(dict(heartbeat_recovery_release=hr.RELEASE,heartbeat_recovery_grant_sha256=hr.GRANT_SHA)))
        self.assertFalse(hr.enabled({}))
        for bad in [dict(heartbeat_recovery_release=None),dict(heartbeat_recovery_grant_sha256=hr.GRANT_SHA),
                dict(heartbeat_recovery_release=hr.RELEASE,heartbeat_recovery_grant_sha256='wrong')]:
            with self.assertRaises(PermissionError):hr.enabled(bad)

    def test_caps_use_only_new_exact_scope_else_old_storage_caps(self):
        self.assertEqual(hr.caps(request()),(485*hr.GIB,418*hr.GIB))
        with self.assertRaises(PermissionError):hr.caps(dict(heartbeat_recovery_release=hr.RELEASE,heartbeat_recovery_grant_sha256=hr.GRANT_SHA))
        with patch.object(hr.storage,'caps',return_value=(1,2)) as old:
            self.assertEqual(hr.caps({}),(1,2));old.assert_called_once_with({})

    def test_lineage_keeps_old_requests_and_independent_B(self):
        self.assertEqual(hr.lineage(request()),hr.LINEAGE)
        self.assertEqual(hr.lineage(request('B_COVERAGE')),'B_COVERAGE')
        with patch.object(hr.recovery,'lineage',return_value='OLD') as old:
            self.assertEqual(hr.lineage({'arm':'A_CONTROL'}),'OLD');old.assert_called_once()
        for key,value in [('training_lineage_key',hr.recovery.LINEAGE),('storage_grant_sha256','wrong'),
                ('segmented_grant_sha256','wrong'),('recovery_adoption_sha256','wrong')]:
            bad=request();bad[key]=value
            with self.assertRaises(PermissionError):hr.lineage(bad)

    def test_adopt_changes_only_two_markers_and_imported_A_seed(self):
        before_h,before_d=deepcopy(self.h),deepcopy(self.d)
        h,d=self.adopt()
        self.assertEqual(self.h,before_h);self.assertEqual(self.d,before_d)
        self.assertEqual(h[hr.FIELD],d[hr.FIELD]);self.assertEqual(h['optimizer_updates'],3843)
        self.assertEqual(h['training_runs'][hr.LINEAGE]['last_native_step'],2048)
        self.assertEqual(h['training_runs'][hr.recovery.LINEAGE]['last_native_step'],2819)
        stripped=deepcopy(h);stripped.pop(hr.FIELD);stripped['training_runs'].pop(hr.LINEAGE)
        self.assertEqual(stripped,before_h)
        stripped=deepcopy(d);stripped.pop(hr.FIELD);self.assertEqual(stripped,before_d)

    def test_duplicate_adopt_and_preexisting_new_lineage_rejected(self):
        self.h,self.d=self.adopt()
        with self.assertRaises(PermissionError):self.adopt()
        self.h,self.d,self.gpu,self.control=fixtures()
        self.h['training_runs'][hr.LINEAGE]=dict(last_native_step=2048,recipe_sha256=hr.recovery.RECIPE)
        with self.assertRaises(PermissionError):self.adopt()

    def test_bad_hash_active_or_unsettled_old_submission_rejected(self):
        with patch.object(hr,'_read',side_effect=lambda raw:(self.read(raw)[0],'wrong') if raw==b'gpu' else self.read(raw)):
            with self.assertRaises(PermissionError):hr.adopt(self.h,self.d,grant_bytes=b'grant',closure_bytes=b'gpu',control_closure_bytes=b'control',now_epoch=hr.APPROVED_EPOCH+1)
        for key,value in [('unknown_effects',['unknown']),('pending_updates',[{'pending':1}]),
                ('CPU_reserved_s',1),('reserved',dict(optimizer_updates=1))]:
            h=deepcopy(self.h);self.h[key]=value
            with self.assertRaises(PermissionError):self.adopt()
            self.h=h
        self.h['PBS_submissions'][0]['status']='R'
        with self.assertRaises(PermissionError):self.adopt()

    def test_wrong_GPU_progress_query_load_or_recipe_rejected(self):
        for key,value in [('last_native_step',2048),('optimizer_updates_committed',0),
                ('query_attempts',18),('model_load_attempts',3),('recipe_sha256','wrong')]:
            before=deepcopy(self.gpu);old_control=deepcopy(self.control)
            self.gpu['effects'][key]=value
            self.control['GPU_close']['terminal_receipt']=deepcopy(self.gpu)
            with self.assertRaises(PermissionError):self.adopt()
            self.gpu=before;self.control=old_control

    def test_control_lease_or_embedded_GPU_mismatch_rejected(self):
        for key,value in [('lease_status','ACTIVE'),('terminal_receipt_sha256','wrong')]:
            old=deepcopy(self.control);self.control['GPU_close'][key]=value
            with self.assertRaises(PermissionError):self.adopt()
            self.control=old
        self.control['new_A_recovery_step']=2048
        with self.assertRaises(PermissionError):self.adopt()

    def test_newjob_cutoff_blocks_adoption_and_checkpoint2048_hash_is_required(self):
        with patch.object(hr,'_read',side_effect=self.read):
            with self.assertRaises(PermissionError):hr.adopt(self.h,self.d,grant_bytes=b'grant',closure_bytes=b'gpu',control_closure_bytes=b'control',now_epoch=hr.NEW_JOBS_CUTOFF)
        self.gpu['checkpoints']['2048']['complete_sha256']='wrong'
        self.control['GPU_close']['terminal_receipt']=deepcopy(self.gpu)
        with self.assertRaises(PermissionError):self.adopt()

    def test_B_already_started_or_cost_rollback_rejected(self):
        self.h['training_runs']['B_COVERAGE']=dict(last_native_step=1,recipe_sha256=hr.recovery.MATCHED_RECIPES['B_COVERAGE'])
        with self.assertRaises(PermissionError):self.adopt()
        self.h['training_runs'].pop('B_COVERAGE');self.d['optimizer_updates']=3072
        with self.assertRaises(PermissionError):self.adopt()

    def test_marker_mismatch_or_extra_key_and_old_lineage_rewind_rejected(self):
        h,d=self.adopt();hr.validate_adopted(h,d)
        bad=deepcopy(h);bad[hr.FIELD]['extra']='not_authorized'
        with self.assertRaises(PermissionError):hr.validate_global(bad)
        bad=deepcopy(d);bad[hr.FIELD]['known_irrecoverable_optimizer_updates']=0
        with self.assertRaises(PermissionError):hr.validate_adopted(h,bad)
        bad=deepcopy(h);bad['training_runs'][hr.recovery.LINEAGE]['last_native_step']=2048
        with self.assertRaises(PermissionError):hr.validate_global(bad)

    def test_global_validation_does_not_call_storage_global_dispatcher(self):
        h,d=self.adopt()
        with patch.object(hr.storage,'global_cap_from_ledger',side_effect=AssertionError('RECURSION')):
            self.assertEqual(hr.validate_global(h),h[hr.FIELD])
        self.old_marker.assert_called()

    def test_admitted_requires_old_admission_manifest_seals_and_same_native_bytes(self):
        req=request();req['sealed_inputs']={hr.PLAN:hr.GRANT_SHA}
        manifest=dict(authority=dict(heartbeat_recovery_release=hr.RELEASE,
            heartbeat_recovery_grant_sha256=hr.GRANT_SHA),execution=dict(sealed_inputs={hr.PLAN:hr.GRANT_SHA}))
        leaf=Mock();leaf.read_bytes.return_value=b'grant'
        native=Mock();native.read_bytes.return_value=b'grant'
        with patch.object(hr.storage,'admitted',return_value=True) as old,patch.object(hr.storage,'_readonly',side_effect=[leaf,native]),patch.object(hr,'sha',return_value=hr.GRANT_SHA),patch.object(hr,'grant',return_value=self.grant):
            self.assertTrue(hr.admitted('/registered/children/new',manifest,req));old.assert_called_once()
        manifest['authority']['heartbeat_recovery_grant_sha256']='wrong'
        with patch.object(hr.storage,'admitted',return_value=True),patch.object(hr.storage,'_readonly',return_value=leaf),patch.object(hr,'sha',return_value=hr.GRANT_SHA):
            with self.assertRaises(PermissionError):hr.admitted('/registered/children/new',manifest,req)


if __name__=='__main__':unittest.main()
