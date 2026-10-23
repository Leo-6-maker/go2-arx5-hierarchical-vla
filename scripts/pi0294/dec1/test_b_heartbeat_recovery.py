"""Pure scope fixtures; no fixture is a native closure or admission."""
from contextlib import contextmanager,ExitStack
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from . import b_heartbeat_recovery as b

GRANT=Path(__file__).resolve().parents[3]/'reports/PI0294/design/parallel-science-20261022/B1_HEARTBEAT_RECOVERY_AUTHORIZATION_20261023.json'

def fixture():
    row=dict(status='NATIVE_TERMINAL',job_id=b.FAILED_JOB,request_id=b.FAILED_REQUEST,source_sha=b.FAILED_SOURCE,
        request_sha256=b.REQUEST_SHA,arm='B_COVERAGE',training_lineage_key='B_COVERAGE',start_native_step=0,
        committed_optimizer_updates=1829,query_attempts=29,conservative_H200_s=14196,conservative_CPU_s=15060,
        reserved_optimizer_updates=0,reserved_neural_queries=0,native_exit_status=1,
        terminal_native_sha256='c4ce568d3d5f56cb36b5a2c15d7d581f2dfaec7e88165ce9f1ace404deba9ffc',
        terminal_effect_summary_sha256='89515d0d921a84626843894a432ac26a97415a2c27668bcf2d98ee8fa580713f')
    h=dict(PBS_submissions=[row],optimizer_updates=7720,used=dict(optimizer_updates=7720,H200_s=75658),
        CPU_s=103291.82213385901,generation_queries=1180,unknown_effects=[],pending_updates=[],reserved={},
        CPU_reserved_s=0,storage_reserved_bytes=0,training_runs={
            'B_COVERAGE':dict(last_native_step=1829,recipe_sha256=b.RECIPE),
            b.zero.HR.LINEAGE:dict(last_native_step=4096,recipe_sha256=b.zero.recovery.RECIPE)},
        **{b.zero.FIELD:dict(adopted_epoch=b.APPROVED_EPOCH-100)})
    d=dict(new_submissions=[dict(status='NATIVE_TERMINAL',job_id=b.FAILED_JOB,request_id=b.FAILED_REQUEST)],
        optimizer_updates=7720,unknown_effects=[],GPU_reserved_s=0,CPU_reserved_s=0)
    e=dict(job_id=b.FAILED_JOB,request_id=b.FAILED_REQUEST,request_sha256=b.REQUEST_SHA,source_sha=b.FAILED_SOURCE,
        arm='B_COVERAGE',start_native_step=0,optimizer_updates_committed=1829,last_native_step=1829,
        query_attempts=29,model_load_attempts=4,pending_updates=[],unknown_query_intents=[],recipe_sha256=b.RECIPE,
        effect_journal_complete=True,new_physics=0,new_episodes=0)
    c=dict(status='CLOSED_B1_HEARTBEAT_FAILURE_1829_B1024_PRESERVED',job_id=b.FAILED_JOB,latest_complete_step=1024,
        checkpoints={'1024':dict(complete_sha256=b.SEED_SHA,members=15,actual_consumed_cursor=32768)},
        committed_updates_without_new_complete_state=805,recoverable_full_state=False,prior_complete_state_preserved=True,
        retired_or_deleted_files=0,effects=e,native_terminal=dict(job_state='F',Exit_status=1),
        settlement=dict(updates=7720,unknown_effects=[]),closed_epoch=b.APPROVED_EPOCH-10)
    cc=dict(status='CLOSED_CPU_B1_FAILED1829_CONTROL_AFTER_KNOWN_EFFECTS_SETTLEMENT',job_id=b.FAILED_JOB,
        new_B_native_step=1829,old_A1024_costs_preserved=True,unknown_effects=[],new_GPU_PBS_models_queries_updates_physics=0,
        settled_submission=deepcopy(row),GPU_close=dict(lease_status='CLOSED',terminal_receipt_sha256=b.CLOSURE_SHA,
            terminal_receipt=deepcopy(c)),closed_epoch=b.APPROVED_EPOCH-9)
    return h,d,c,cc

@contextmanager
def reads(c,cc):
    values={b'GPU':(c,b.CLOSURE_SHA),b'CPU':(cc,b.CONTROL_CLOSURE_SHA),b'BADGPU':(c,'0'*64),b'BADCPU':(cc,'0'*64)}
    with ExitStack() as s:
        s.enter_context(patch.object(b,'grant',return_value={}))
        s.enter_context(patch.object(b,'_read',side_effect=lambda x:deepcopy(values[x])))
        s.enter_context(patch.object(b.zero,'validate_adopted'))
        s.enter_context(patch.object(b.zero,'validate_global'))
        yield

def adopt(h,d,**kwargs):
    k=dict(grant_bytes=b'G',closure_bytes=b'GPU',control_closure_bytes=b'CPU',now_epoch=b.APPROVED_EPOCH+1)
    k.update(kwargs);return b.adopt(h,d,**k)

def scope():
    from .test_zero_update_recovery import scope as old
    r=old();r.update(arm='B_COVERAGE',training_lineage_key=b.LINEAGE,
        **dict(zip(b.SCOPE_KEYS,(b.RELEASE,b.GRANT_SHA))))
    return r

class BHeartbeatTests(unittest.TestCase):
    def test_exact_published_grant(self):
        raw=GRANT.read_bytes();v=b.grant(raw)
        self.assertEqual(v['recovery_semantics']['remaining_updates'],3072)
        with self.assertRaises(PermissionError):b.grant(raw+b' ')
        for key in ('narrow_changes','latest_full_state','old_B_lineage_preserved','recovery_semantics','original_final','original_clock'):
            wrong=deepcopy(v);wrong[key]={}
            with self.subTest(key=key),patch.object(b,'_read',return_value=(wrong,b.GRANT_SHA)),self.assertRaises(PermissionError):b.grant(b'X')

    def test_only_dual_marker_and_imported_seed_added_all_paid_costs_preserved(self):
        h,d,c,cc=fixture();before=deepcopy((h,d))
        with reads(c,cc):
            hh,dd=adopt(h,d);b.validate_adopted(hh,dd)
            self.assertEqual(hh[b.FIELD],dd[b.FIELD]);self.assertEqual(hh['optimizer_updates'],7720)
            seed=hh['training_runs'].pop(b.LINEAGE);self.assertEqual(seed,dict(last_native_step=1024,recipe_sha256=b.RECIPE))
            self.assertEqual({k:v for k,v in hh.items() if k!=b.FIELD},h)
            self.assertEqual({k:v for k,v in dd.items() if k!=b.FIELD},d)
            self.assertEqual((h,d),before)
            with self.assertRaises(PermissionError):adopt(hh,dd)

    def test_failures_never_rewind_failed_lineage_or_ignore_costs(self):
        for mode in ('old_step','old_recipe','A_not_closed','GPU_cost','CPU_cost','query_cost','updates','reserved',
            'unknown','active','GPU_hash','CPU_hash','wrong_checkpoint','lost805','wrong_source','pending','open_GPU','CPU_row','deadline'):
            h,d,c,cc=fixture();kw={}
            if mode=='old_step':h['training_runs']['B_COVERAGE']['last_native_step']=1024
            elif mode=='old_recipe':h['training_runs']['B_COVERAGE']['recipe_sha256']=b.zero.recovery.RECIPE
            elif mode=='A_not_closed':h['training_runs'][b.zero.HR.LINEAGE]['last_native_step']=2048
            elif mode=='GPU_cost':h['used']['H200_s']-=1
            elif mode=='CPU_cost':h['CPU_s']-=1
            elif mode=='query_cost':h['generation_queries']-=1
            elif mode=='updates':h['optimizer_updates']-=1
            elif mode=='reserved':d['CPU_reserved_s']=1
            elif mode=='unknown':h['unknown_effects']=[{}]
            elif mode=='active':d['new_submissions'][0]['status']='SUBMITTED_PENDING_NATIVE'
            elif mode=='GPU_hash':kw['closure_bytes']=b'BADGPU'
            elif mode=='CPU_hash':kw['control_closure_bytes']=b'BADCPU'
            elif mode=='wrong_checkpoint':c['checkpoints']['1024']['complete_sha256']='0'*64
            elif mode=='lost805':c['committed_updates_without_new_complete_state']=0
            elif mode=='wrong_source':c['effects']['source_sha']='0'*40
            elif mode=='pending':c['effects']['pending_updates']=[{}]
            elif mode=='open_GPU':cc['GPU_close']['lease_status']='ACTIVE'
            elif mode=='CPU_row':cc['settled_submission']['committed_optimizer_updates']=1024
            else:kw['now_epoch']=b.CUTOFF
            with self.subTest(mode=mode),reads(c,cc),self.assertRaises(PermissionError):adopt(h,d,**kw)

    def test_new_scope_lineage_is_only_B_and_requires_all_older_scopes(self):
        r=scope();self.assertEqual(b.lineage(r),b.LINEAGE);self.assertEqual(b.caps(r),(615*b.GIB,549*b.GIB))
        for key in (*b.SCOPE_KEYS,'zero_update_resume_grant_sha256','heartbeat_recovery_grant_sha256',
            'recovery_adoption_sha256','storage_grant_sha256','segmented_grant_sha256','training_lineage_key'):
            wrong=dict(r);wrong.pop(key)
            with self.subTest(key=key),self.assertRaises(PermissionError):b.lineage(wrong)
        wrong=dict(r,arm='A_CONTROL')
        with self.assertRaises(PermissionError):b.caps(wrong)

    def test_dual_markers_and_imported_progress_remain_bound_after_training(self):
        h,d,c,cc=fixture()
        with reads(c,cc):
            hh,dd=adopt(h,d)
            for step in (1024,2048,4096):
                hh['training_runs'][b.LINEAGE]['last_native_step']=step
                hh['optimizer_updates']=hh['used']['optimizer_updates']=dd['optimizer_updates']=7720+step-1024
                b.validate_adopted(hh,dd)
            for mode in ('missing_marker','mismatched','rollback_seed','wrong_seed_recipe'):
                ah,ad=deepcopy(hh),deepcopy(dd)
                if mode=='missing_marker':ad.pop(b.FIELD)
                elif mode=='mismatched':ad[b.FIELD]['known_irrecoverable_optimizer_updates']=0
                elif mode=='rollback_seed':ah['training_runs'][b.LINEAGE]['last_native_step']=0
                else:ah['training_runs'][b.LINEAGE]['recipe_sha256']=b.zero.recovery.RECIPE
                with self.subTest(mode=mode),self.assertRaises(PermissionError):b.validate_adopted(ah,ad)

if __name__=='__main__':unittest.main()

class BRecoveryResourceExtensionTests(unittest.TestCase):
    def test_campaign_submission_helpers_preserve_legacy24_and_fail_closed_on_partial_marker(self):
        from scripts.pi0294.pv1.resources import pbs_limit,recovery_resource_kwargs
        from scripts.pi0294.dec1 import b_heartbeat_recovery as runtime_b
        h,d,c,cc=fixture()
        self.assertEqual(pbs_limit(h),24)
        self.assertEqual(recovery_resource_kwargs(h),{})
        with reads(c,cc),patch.object(runtime_b.zero,'validate_global'):
            h,d=adopt(h,d)
            self.assertEqual(pbs_limit(h),25)
            self.assertIs(recovery_resource_kwargs(h)['b_recovery_global'],h)
            h[b.FIELD]['known_irrecoverable_optimizer_updates']=0
            for fn in (pbs_limit,recovery_resource_kwargs):
                with self.subTest(fn=fn.__name__),self.assertRaises(PermissionError):fn(h)

    def test_only_adopted_global_ledger_gets25_and_only_the_PBS_cap_changes(self):
        from scripts.pi0294.pv1.resources import admit_resources
        h,d,c,cc=fixture()
        from scripts.pi0294.dec1 import b_heartbeat_recovery as runtime_b
        with reads(c,cc),patch.object(runtime_b.zero,'validate_adopted'),patch.object(runtime_b.zero,'validate_global'):
            h,d=adopt(h,d)
            h['used']['PBS_submissions']=24
            self.assertEqual(admit_resources(h['used'],h['reserved'],dict(PBS_submissions=1),
                b_recovery_ledgers=(h,d))['status'],'PASS_PV1_BUDGET_ADMISSION_NOT_NATIVE_EXECUTION')
            with self.assertRaises(PermissionError):admit_resources(h['used'],h['reserved'],dict(PBS_submissions=1))
            with self.assertRaises(PermissionError):admit_resources(h['used'],h['reserved'],dict(PBS_submissions=2),b_recovery_ledgers=(h,d))
            with self.assertRaises(PermissionError):admit_resources(h['used'],h['reserved'],dict(H200_s=1000000),b_recovery_ledgers=(h,d))
            fake=dict(h['used'],PBS_submissions=0)
            with self.assertRaises(PermissionError):admit_resources(fake,h['reserved'],{},b_recovery_ledgers=(h,d))
            with self.assertRaises(PermissionError):admit_resources(fake,h['reserved'],{},b_recovery_global=h)
            h.pop(b.FIELD)
            with self.assertRaises(PermissionError):admit_resources(h['used'],h['reserved'],{},b_recovery_global=h)
