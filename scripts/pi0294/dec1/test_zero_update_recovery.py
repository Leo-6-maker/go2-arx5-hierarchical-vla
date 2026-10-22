"""CPU fixture text only; not executed by its author.

Root copies module/test into scripts/pi0294/dec1 before running in a guarded
CPU process. Real grant bytes come from the fixed published local card.
Baseline HR validation is mocked only in focused synthetic adoption fixtures;
existing HR integration tests must remain in the owner's integration suite.
"""
from contextlib import ExitStack,contextmanager
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.pi0294.dec1 import zero_update_recovery as z

GRANT_PATH=Path(__file__).resolve().parents[3]/'reports/PI0294/design/parallel-science-20261022/ZERO_UPDATE_RESUME_AUTHORIZATION_20261023.json'

def fixture():
    row=dict(status='NATIVE_TERMINAL',job_id=z.FAILED_JOB,request_id=z.FAILED_REQUEST,source_sha=z.FAILED_SOURCE,
        request_sha256=z.REQUEST_SHA,training_lineage_key=z.HR.LINEAGE,arm='A_CONTROL',start_native_step=2048,
        committed_optimizer_updates=0,query_attempts=0,reserved_optimizer_updates=0,reserved_neural_queries=0,
        terminal_native_sha256=z.NATIVE_SHA,terminal_effect_summary_sha256=z.EFFECT_SHA,
        conservative_H200_s=754,conservative_CPU_s=259)
    h=dict(PBS_submissions=[row],optimizer_updates=3843,used=dict(optimizer_updates=3843,H200_s=49117),
        CPU_s=72668.82213385901,generation_queries=1134,unknown_effects=[],pending_updates=[],
        reserved=dict(optimizer_updates=0,H200_s=0,neural_queries=0),CPU_reserved_s=0,storage_reserved_bytes=0,
        training_runs={z.HR.LINEAGE:dict(last_native_step=2048,recipe_sha256=z.recovery.RECIPE),
            z.recovery.LINEAGE:dict(last_native_step=2819,recipe_sha256=z.recovery.RECIPE),
            'A_CONTROL':dict(last_native_step=1024,recipe_sha256=z.recovery.RECIPE)},
        **{z.HR.FIELD:dict(adopted_epoch=z.APPROVED_EPOCH-10)})
    d=dict(new_submissions=[dict(status='NATIVE_TERMINAL',job_id=z.FAILED_JOB,request_id=z.FAILED_REQUEST)],
        optimizer_updates=3843,unknown_effects=[],GPU_reserved_s=0,CPU_reserved_s=0,
        **{z.HR.FIELD:deepcopy(h[z.HR.FIELD])})
    effects=dict(schema='VAP1_A40_NATIVE_SOURCE_EFFECT_SUMMARY_V1',job_id=z.FAILED_JOB,request_id=z.FAILED_REQUEST,
        request_sha256=z.REQUEST_SHA,source_sha=z.FAILED_SOURCE,arm='A_CONTROL',start_native_step=2048,
        last_native_step=2048,optimizer_updates_committed=0,query_attempts=0,model_load_attempts=0,
        pending_updates=[],unknown_query_intents=[],effect_journal_complete=True,recipe_sha256=z.recovery.RECIPE,
        new_physics=0,new_episodes=0)
    closure=dict(status='CLOSED_RECOVERY2_PREMODEL_HEARTBEAT_FAILURE_PRESERVED',job_id=z.FAILED_JOB,
        latest_complete_step=2048,checkpoints={},recoverable_full_state=False,prior_complete_state_preserved=True,
        committed_updates_without_new_complete_state=0,retired_or_deleted_files=0,effects=effects,
        settlement=dict(status='EXACT_RECOVERY2_PREMODEL_HEARTBEAT_FAILURE_SETTLED',job_id=z.FAILED_JOB,
            request_id=z.FAILED_REQUEST,updates=3843,unknown_effects=[]),closed_epoch=z.APPROVED_EPOCH-5)
    control=dict(status='CLOSED_CPU_RECOVERY2_PREMODEL_FAILURE_CONTROL_AFTER_KNOWN_EFFECTS_SETTLEMENT',
        job_id=z.FAILED_JOB,old_A1024_costs_preserved=True,new_A_recovery_step=2048,unknown_effects=[],
        new_GPU_PBS_models_queries_updates_physics=0,settled_submission=deepcopy(row),closed_epoch=z.APPROVED_EPOCH-4,
        GPU_close=dict(lease_status='CLOSED',terminal_receipt_sha256=z.CLOSURE_SHA,terminal_receipt=deepcopy(closure)))
    return h,d,closure,control

@contextmanager
def metadata_reads(closure,control):
    # Isolate semantics from pinned digest verification, then test bad hashes
    # separately. No real closure receipt is manufactured or written.
    values={b'GPU':(closure,z.CLOSURE_SHA),b'CPU':(control,z.CONTROL_CLOSURE_SHA),
        b'BADGPU':(closure,'0'*64),b'BADCPU':(control,'0'*64)}
    with ExitStack() as stack:
        stack.enter_context(patch.object(z,'grant',return_value={}))
        stack.enter_context(patch.object(z,'_read',side_effect=lambda raw:deepcopy(values[raw])))
        stack.enter_context(patch.object(z.HR,'validate_global',return_value={}))
        stack.enter_context(patch.object(z.HR,'validate_adopted',return_value={}))
        yield

def adopt(h,d,**changes):
    args=dict(grant_bytes=b'GRANT',closure_bytes=b'GPU',control_closure_bytes=b'CPU',now_epoch=z.APPROVED_EPOCH+1)
    args.update(changes);return z.adopt(h,d,**args)

def scope():
    return dict(zero_update_resume_release=z.RELEASE,zero_update_resume_grant_sha256=z.GRANT_SHA,
        heartbeat_recovery_release=z.HR.RELEASE,heartbeat_recovery_grant_sha256=z.HR.GRANT_SHA,
        recovery_authorization_release=z.recovery.RELEASE,recovery_adoption_sha256=z.recovery.GRANT_SHA,
        storage_authorization_release=z.HR.storage.RELEASE,storage_grant_sha256=z.HR.storage.GRANT_SHA,
        segmented_authorization_release=z.HR.seg.RELEASE,segmented_grant_sha256=z.HR.seg.GRANT_SHA,
        training_lineage_key=z.HR.LINEAGE,arm='A_CONTROL')

class ZeroUpdateRecoveryTests(unittest.TestCase):
    def test_real_fixed_grant_and_bad_bytes_or_narrow_change_rejected(self):
        raw=GRANT_PATH.read_bytes();value=z.grant(raw)
        self.assertEqual(value['updates_cap_unchanged'],9987)
        with self.assertRaises((PermissionError,ValueError)):z.grant(raw+b' ')
        for key in ('status','narrow_changes','prior_heartbeat_release','recovery_semantics'):
            bad=deepcopy(value);bad[key]={} if isinstance(value[key],dict) else 'WRONG'
            with self.subTest(key=key),patch.object(z,'_read',return_value=(bad,z.GRANT_SHA)),self.assertRaises(PermissionError):z.grant(b'BAD')
        with patch.object(z,'_read',return_value=(value,'0'*64)),self.assertRaises(PermissionError):z.grant(b'BADHASH')

    def test_adopt_only_adds_identical_marker_and_preserves_all_state(self):
        h,d,c,cc=fixture();before=deepcopy((h,d))
        with metadata_reads(c,cc):
            hh,dd=adopt(h,d);self.assertEqual(hh[z.FIELD],dd[z.FIELD]);z.validate_adopted(hh,dd)
            self.assertEqual({k:v for k,v in hh.items() if k!=z.FIELD},h)
            self.assertEqual({k:v for k,v in dd.items() if k!=z.FIELD},d)
            self.assertEqual((h,d),before)
            with self.assertRaises(PermissionError):adopt(hh,dd)
            for target in ('global','DEC1'):
                ah,ad=deepcopy(hh),deepcopy(dd);(ah if target=='global' else ad).pop(z.FIELD)
                with self.subTest(target=target),self.assertRaises(PermissionError):z.validate_adopted(ah,ad)

    def test_exact_scope_caps_and_missing_original_scope_refused_without_recursion(self):
        r=scope()
        with patch.object(z.HR,'caps',side_effect=AssertionError('NO_RECURSION')):
            self.assertEqual(z.caps(r),(550*z.GIB,483*z.GIB));self.assertEqual(z.pbs_limit(r),13)
            old={k:v for k,v in r.items() if not k.startswith('zero_update_resume_')}
            self.assertEqual(z.caps(old),(485*z.GIB,418*z.GIB))
        for key in ('zero_update_resume_release','zero_update_resume_grant_sha256','heartbeat_recovery_grant_sha256',
            'segmented_grant_sha256','storage_grant_sha256','recovery_adoption_sha256'):
            bad=dict(r);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(PermissionError):z.caps(bad)

    def test_refuses_changed_closure_or_bad_hash_and_no_claim_of_new_seed(self):
        for kind in ('gpu_hash','cpu_hash','checkpoint','query','load','update','source','unsettled','open_gpu',
            'wrong_cpu_row','wrong_cpu_job','cpu_unknown','deleted','changed_seed','deadline'):
            h,d,c,cc=fixture();kwargs={}
            if kind=='gpu_hash':kwargs['closure_bytes']=b'BADGPU'
            elif kind=='cpu_hash':kwargs['control_closure_bytes']=b'BADCPU'
            elif kind=='checkpoint':c['checkpoints']={'2048':{'new':'NOT_A_SEED'}}
            elif kind=='query':c['effects']['query_attempts']=1
            elif kind=='load':c['effects']['model_load_attempts']=1
            elif kind=='update':c['effects']['optimizer_updates_committed']=1
            elif kind=='source':c['effects']['source_sha']='0'*40
            elif kind=='unsettled':c['settlement']['status']='UNKNOWN'
            elif kind=='open_gpu':cc['GPU_close']['lease_status']='ACTIVE'
            elif kind=='wrong_cpu_row':cc['settled_submission']['committed_optimizer_updates']=1
            elif kind=='wrong_cpu_job':cc['job_id']='1.stdct-mgmt-02'
            elif kind=='cpu_unknown':cc['unknown_effects']=[{}]
            elif kind=='deleted':c['retired_or_deleted_files']=1
            elif kind=='changed_seed':h['training_runs'][z.HR.LINEAGE]['last_native_step']=2819
            else:kwargs['now_epoch']=z.CUTOFF
            with self.subTest(kind=kind),metadata_reads(c,cc),self.assertRaises(PermissionError):adopt(h,d,**kwargs)

    def test_no_unknown_unclosed_unreserved_B_started_or_cost_loss(self):
        for kind in ('unknown','reserved','active','unsettled','B_started','lost_updates','lost_GPU','lost_CPU','lost_queries'):
            h,d,c,cc=fixture()
            if kind=='unknown':h['unknown_effects']=[{}]
            elif kind=='reserved':d['CPU_reserved_s']=1
            elif kind=='active':d['new_submissions'][0]['status']='SUBMITTED_PENDING_NATIVE'
            elif kind=='unsettled':h['PBS_submissions'][0]['status']='NATIVE_RUNNING_QUALIFICATION'
            elif kind=='B_started':h['training_runs']['B_COVERAGE']=dict(last_native_step=1)
            elif kind=='lost_updates':h['optimizer_updates']=3842
            elif kind=='lost_GPU':h['used']['H200_s']=0
            elif kind=='lost_CPU':h['CPU_s']=0
            else:h['generation_queries']=0
            with self.subTest(kind=kind),metadata_reads(c,cc),self.assertRaises(PermissionError):adopt(h,d)

if __name__=='__main__':unittest.main()
