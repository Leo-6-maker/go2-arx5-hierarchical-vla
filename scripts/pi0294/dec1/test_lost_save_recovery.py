"""Pure synthetic native metadata; never a scheduler, model or real recovery."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from . import lost_save_recovery as rec
from . import a40_accounting as accounting
from . import a40_allocation as a
from . import a40_future as future
from . import future_budget as budget
from . import dual_platform as dual
from .test_a40_accounting import prequeue,terminal
from .test_a40_allocation import fixture,future_inputs,raw

CARD=Path(__file__).resolve().parents[3]/'reports/PI0294/design/VAP1_A40_LOST_SAVE_RECOVERY_AUTHORIZATION_20261022.json'


@contextmanager
def recovered():
    f=prequeue();h,d=deepcopy(f['ledger']),deepcopy(f['DEC1'])
    h['optimizer_updates']=h['used']['optimizer_updates']=1024;d['optimizer_updates']=1024
    h['DEC1_source_pair_platform']='VANDA_A40'
    for row in h['PBS_submissions']:row['status']='NATIVE_TERMINAL'
    h['training_runs']={'A_CONTROL':dict(last_native_step=1024,recipe_sha256=rec.RECIPE)}
    h['PBS_submissions'][-1].update(request_id=rec.FAILED_REQUEST,job_id='1474116.stdct-mgmt-02',
        status='NATIVE_TERMINAL',committed_optimizer_updates=1024,
        terminal_native_sha256='43c0ffec6b3d3adff703d561b7489580d0b9a723f464bace2da23ee1d6955d0b',
        terminal_effect_summary_sha256='e5b707657d517773c9f2521dd1f57d0d252935f1ad2a04dc95b04952678644c9')
    d['new_submissions'].append(dict(request_id=rec.FAILED_REQUEST,job_id='1474116.stdct-mgmt-02',status='NATIVE_TERMINAL'))
    closure=raw(dict(job_id='1474116.stdct-mgmt-02',recoverable_full_state=False,checkpoint_complete_files=[],
        effects=dict(request_id=rec.FAILED_REQUEST,optimizer_updates_committed=1024),settlement=dict(unknown_effects=[])))
    with patch.object(rec,'CLOSURE_SHA',hashlib.sha256(closure).hexdigest()):
        yield f,h,d,closure


def inputs_for(f,h,d):
    args=fixture();values=future_inputs(args)
    r=json.loads(f['request_bytes']);r.update(recovery_authorization_release=rec.RELEASE,
        recovery_adoption_sha256=rec.GRANT_SHA,training_lineage_key=rec.LINEAGE,
        CPU_envelope_profile=a.BOUNDED_CPU_PROFILE,CPU_upper_s=80000)
    s=json.loads(values['snapshot_bytes']);p=json.loads(values['progress_bytes']);plan=json.loads(values['plan_bytes'])
    s['used'].update(h['used'],Hopper_CPU_s=h['CPU_s'],Hopper_storage_bytes=h['new_storage_current_bytes'])
    s['reserved']={};s['dec1_reserved']={};s['work_reservations']=[];s.pop('current_native_request_work_id')
    s['dec1_used'].update(H200_s=d['GPU_allocated_s'],PBS_submissions=len(d['new_submissions']),
        optimizer_updates=d['optimizer_updates'],neural_queries=d['neural_queries'],Hopper_CPU_s=d['CPU_s'],
        Hopper_storage_bytes=d['new_GPU_site_output_allocated_upper_bytes'],episode_starts=d['episode_starts'],
        physics_ticks=d['physics_ticks'],engine_starts=d['constructors'],transfer_bytes=0)
    s['source_recovery_grant_sha256']=rec.GRANT_SHA
    p.update(source_recovery_authorization_release=rec.RELEASE,known_irrecoverable_optimizer_updates=1024)
    for row in plan['training_jobs']:
        row['CPU_envelope_profile']=a.BOUNDED_CPU_PROFILE;row['resources']['Hopper_CPU_s']=80000
    values.update(request_bytes=raw(r),ledger_bytes=raw(h),DEC1_ledger_bytes=raw(d),snapshot_bytes=raw(s),
        progress_bytes=raw(p),plan_bytes=raw(plan),purpose='PREQUEUE',job_id=None,recovery_bytes=CARD.read_bytes())
    s['actual_Hopper_ledger_sha256']=hashlib.sha256(values['ledger_bytes']).hexdigest()
    s['actual_DEC1_ledger_sha256']=hashlib.sha256(values['DEC1_ledger_bytes']).hexdigest();values['snapshot_bytes']=raw(s)
    return values


class RecoveryTests(unittest.TestCase):
    def test_original_B_recipe_settles_to_B_without_accepting_A_or_changing_old_loss(self):
        with recovered() as (f,h,d,c):
            h,d=rec.adopt(h,d,grant_bytes=CARD.read_bytes(),closure_bytes=c,now_epoch=f['now'])
            values=inputs_for(f,h,d);r=json.loads(values['request_bytes'])
            r['training_block']['arm']='B_COVERAGE';r['training_lineage_key']='B_COVERAGE'
            values['request_bytes']=raw(r);f['request_bytes']=values['request_bytes']
            with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(values['amendment_bytes']).hexdigest()):
                proof=future.recompute(**values)
            with patch.object(dual,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()):
                v,dd=accounting.reserve(h,d,request_bytes=f['request_bytes'],budget_proof=proof,
                    grant_bytes=f['grant_bytes'],campaign_view=f['campaign_view'],now_epoch=f['now'])
            v,dd=accounting.bind_job(v,dd,request_id=r['request_id'],job_id=f['job_id'],now_epoch=f['now'])
            native,effects=terminal(f);effects.update(arm='B_COVERAGE',recipe_sha256=rec.MATCHED_RECIPES['B_COVERAGE'])
            before,dbefore=deepcopy(v),deepcopy(dd)
            for wrong in (rec.RECIPE,'f'*64):
                bad=deepcopy(effects);bad['recipe_sha256']=wrong
                with self.subTest(wrong_recipe=wrong),self.assertRaisesRegex(PermissionError,'MATCHED_RECIPE'):
                    accounting.settle(v,dd,request_id=r['request_id'],native_proof=native,effects=bad,now_epoch=f['now'])
                self.assertEqual(v,before);self.assertEqual(dd,dbefore)
            settled,ds=accounting.settle(v,dd,request_id=r['request_id'],native_proof=native,effects=effects,now_epoch=f['now'])
            self.assertEqual(settled['training_runs']['A_CONTROL'],h['training_runs']['A_CONTROL'])
            self.assertNotIn(rec.LINEAGE,settled['training_runs'])
            self.assertEqual(settled['training_runs']['B_COVERAGE'],dict(last_native_step=32,
                recipe_sha256=rec.MATCHED_RECIPES['B_COVERAGE']))
            self.assertEqual(settled['optimizer_updates'],1056);self.assertEqual(ds['optimizer_updates'],1056)
            self.assertEqual(v,before);self.assertEqual(dd,dbefore)
            rec.validate_adopted(settled,ds)

    def test_exact_grant_and_old_cap_unchanged_without_opt_in(self):
        self.assertEqual(rec.grant(CARD.read_bytes())['cumulative_DEC1_optimizer_update_cap']['new'],9216)
        self.assertEqual(budget.effective_caps()['optimizer_updates'],8192)
        with self.assertRaises(PermissionError):rec.grant(CARD.read_bytes()+b' ')
        with self.assertRaises(PermissionError):budget.effective_caps(recovery_bytes=CARD.read_bytes())

    def test_adoption_is_once_and_preserves_every_old_spent_and_progress_field(self):
        with recovered() as (f,h,d,c):
            before=deepcopy(h);dh=deepcopy(d)
            a1,d1=rec.adopt(h,d,grant_bytes=CARD.read_bytes(),closure_bytes=c,now_epoch=f['now'])
            self.assertEqual(h,before);self.assertEqual(d,dh)
            self.assertEqual({k:v for k,v in a1.items() if k!=rec.FIELD},h)
            self.assertEqual({k:v for k,v in d1.items() if k!=rec.FIELD},d)
            rec.validate_adopted(a1,d1)
            with self.assertRaises(PermissionError):rec.adopt(a1,d1,grant_bytes=CARD.read_bytes(),closure_bytes=c,now_epoch=f['now'])
            for bad in ('unknown','reserved','counter','closure','old-progress'):
                v=deepcopy(h);dd=deepcopy(d);cc=c
                if bad=='unknown':v['unknown_effects']=['uncertain']
                if bad=='reserved':v['reserved']['H200_s']=1
                if bad=='counter':dd['optimizer_updates']=0
                if bad=='closure':cc+=b' '
                if bad=='old-progress':v['training_runs']['A_CONTROL']['last_native_step']=0
                with self.subTest(bad=bad),self.assertRaises(PermissionError):
                    rec.adopt(v,dd,grant_bytes=CARD.read_bytes(),closure_bytes=cc,now_epoch=f['now'])

    def test_cold_whole_pair_forecast_accounts_for_known_loss_not_fake_checkpoint(self):
        with recovered() as (f,h,d,c):
            h,d=rec.adopt(h,d,grant_bytes=CARD.read_bytes(),closure_bytes=c,now_epoch=f['now'])
            values=inputs_for(f,h,d)
            with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(values['amendment_bytes']).hexdigest()):
                result=future.recompute(**values)
                self.assertTrue(result['forecast']['admissible_forecast'])
                self.assertEqual(result['forecast']['effective_DEC1_VAP1_caps']['optimizer_updates'],9216)
                self.assertEqual(result['forecast']['future_required_additional']['optimizer_updates'],8192)
                self.assertEqual(result['forecast']['original_final_protection']['H200_s'],72000)
                for bad in ('grant','lost-count','lineage','marker'):
                    v=deepcopy(values)
                    if bad=='grant':v['recovery_bytes']=None
                    elif bad=='lost-count':
                        p=json.loads(v['progress_bytes']);p['known_irrecoverable_optimizer_updates']=0;v['progress_bytes']=raw(p)
                    elif bad=='lineage':
                        r=json.loads(v['request_bytes']);r['training_lineage_key']='A_CONTROL';v['request_bytes']=raw(r)
                    else:
                        hh=json.loads(v['ledger_bytes']);hh[rec.FIELD]['known_irrecoverable_optimizer_updates']=0;v['ledger_bytes']=raw(hh)
                    with self.subTest(bad=bad),self.assertRaises(PermissionError):future.recompute(**v)

    def test_replacement_commits_to_new_key_without_rollback_or_double_loss_credit(self):
        with recovered() as (f,h,d,c):
            h,d=rec.adopt(h,d,grant_bytes=CARD.read_bytes(),closure_bytes=c,now_epoch=f['now'])
            values=inputs_for(f,h,d);f.update(ledger=h,DEC1=d,request_bytes=values['request_bytes'])
            with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(values['amendment_bytes']).hexdigest()):proof=future.recompute(**values)
            with patch.object(dual,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()):
                v,dd=accounting.reserve(h,d,request_bytes=f['request_bytes'],budget_proof=proof,
                    grant_bytes=f['grant_bytes'],campaign_view=f['campaign_view'],now_epoch=f['now'])
            r=json.loads(f['request_bytes']);v,dd=accounting.bind_job(v,dd,request_id=r['request_id'],job_id=f['job_id'],now_epoch=f['now'])
            native,effects=terminal(f);effects['recipe_sha256']=rec.RECIPE
            before,dbefore=deepcopy(v),deepcopy(dd)
            bad=deepcopy(effects);bad['recipe_sha256']=rec.MATCHED_RECIPES['B_COVERAGE']
            with self.assertRaisesRegex(PermissionError,'MATCHED_RECIPE'):
                accounting.settle(v,dd,request_id=r['request_id'],native_proof=native,effects=bad,now_epoch=f['now'])
            self.assertEqual(v,before);self.assertEqual(dd,dbefore)
            v,dd=accounting.settle(v,dd,request_id=r['request_id'],native_proof=native,effects=effects,now_epoch=f['now'])
            self.assertEqual(v['optimizer_updates'],1056);self.assertEqual(dd['optimizer_updates'],1056)
            self.assertEqual(v['training_runs']['A_CONTROL']['last_native_step'],1024)
            self.assertEqual(v['training_runs'][rec.LINEAGE]['last_native_step'],32)
            rec.validate_adopted(v,dd)


if __name__=='__main__':unittest.main()
