"""Synthetic source escrow lifecycle, never a scheduler or model invocation."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from . import a40_accounting as accounting
from . import a40_allocation as allocation
from . import a40_future as future
from . import future_budget as budget
from . import dual_platform as dual
from .alternate_accounting import validate_items, require_hopper_active_chain
from scripts.pi0294.pv1.neural_ledger import check as check_global_ledger
from .test_a40_allocation import fixture, future_inputs, raw


def prequeue():
    f=fixture();inputs=future_inputs(f)
    h=json.loads(inputs['ledger_bytes']);d=json.loads(inputs['DEC1_ledger_bytes']);s=json.loads(inputs['snapshot_bytes'])
    h['PBS_submissions'].pop();h['used']['PBS_submissions']-=1
    h.update(reserved={},CPU_reserved_s=0,storage_reserved_bytes=0)
    d['new_submissions'].pop()
    for key in ('GPU_reserved_s','CPU_reserved_s','optimizer_updates_reserved','neural_queries_reserved','GPU_site_output_reserved_bytes'):d[key]=0
    s['used']['PBS_submissions']-=1;s['reserved']={};s['dec1_used']['PBS_submissions']-=1;s['dec1_reserved']={}
    s['work_reservations']=[];s.pop('current_native_request_work_id')
    inputs.update(ledger_bytes=raw(h),DEC1_ledger_bytes=raw(d),purpose='PREQUEUE',job_id=None)
    s['actual_Hopper_ledger_sha256']=hashlib.sha256(inputs['ledger_bytes']).hexdigest()
    s['actual_DEC1_ledger_sha256']=hashlib.sha256(inputs['DEC1_ledger_bytes']).hexdigest();inputs['snapshot_bytes']=raw(s)
    with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(inputs['amendment_bytes']).hexdigest()):
        proof=future.recompute(**inputs)
    f.update(ledger=h,DEC1=d,budget_proof=proof);f['campaign_view']['campaign_jobs']=[]
    f['campaign_view']['campaign_running_by_platform']={}
    return f


def terminal(f):
    r=json.loads(f['request_bytes']);job=f['job_id'];now=f['now']
    native=dict(schema='VAP1_NATIVE_A40_SOURCE_TERMINAL_V1',job_id=job,job_state='F',project=r['project'],
        job_name=r['job_name'],owner='e1192548',ngpus=1,request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        recorded_process_terminal=True,observed_epoch=now,conservative_GPU_s=1000,conservative_CPU_s=2000,
        new_resident_allocated_upper_bytes=1000000,Exit_status=0)
    effects=dict(schema='VAP1_A40_NATIVE_SOURCE_EFFECT_SUMMARY_V1',job_id=job,request_id=r['request_id'],
        request_sha256=hashlib.sha256(f['request_bytes']).hexdigest(),source_sha=r['source_sha'],arm='A_CONTROL',
        start_native_step=0,new_physics=0,new_episodes=0,effect_journal_complete=True,
        optimizer_updates_committed=32,last_native_step=32,query_attempts=17,model_load_attempts=2,
        pending_updates=[],unknown_query_intents=[],recipe_sha256='d'*64)
    return native,effects


class SourceAccountingTests(unittest.TestCase):
    def test_stricter_profile_is_reserved_and_settled_without_reducing_other_budgets(self):
        f=prequeue();r=json.loads(f['request_bytes'])
        r.update(CPU_envelope_profile=allocation.BOUNDED_CPU_PROFILE,CPU_upper_s=80000)
        f['request_bytes']=raw(r)
        f['budget_proof']['request_sha256']=hashlib.sha256(f['request_bytes']).hexdigest()
        # Reserve consumes the already-verified pure proof; native recomputation
        # separately checks the matching sealed plan and all future job limits.
        v,d=self.reserve(f)
        self.assertEqual(v['CPU_reserved_s'],80000);self.assertEqual(d['CPU_reserved_s'],80000)
        self.assertEqual(v['reserved']['H200_s'],28860);self.assertEqual(v['reserved']['optimizer_updates'],4096)
        row=v['PBS_submissions'][-1]
        self.assertEqual(row['CPU_envelope_profile'],allocation.BOUNDED_CPU_PROFILE)
        v,d=accounting.bind_job(v,d,request_id=r['request_id'],job_id=f['job_id'],now_epoch=f['now'])
        n,e=terminal(f)
        closed,dc=accounting.settle(v,d,request_id=r['request_id'],native_proof=n,effects=e,now_epoch=f['now'])
        self.assertEqual(closed['CPU_reserved_s'],0);self.assertEqual(dc['CPU_reserved_s'],0)
        row=deepcopy(row);row['CPU_upper_s']=115560
        with self.assertRaises(PermissionError):accounting.validate_item(row)

    def reserve(self,f):
        with patch.object(dual,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()):
            return accounting.reserve(f['ledger'],f['DEC1'],request_bytes=f['request_bytes'],
                budget_proof=f['budget_proof'],grant_bytes=f['grant_bytes'],campaign_view=f['campaign_view'],now_epoch=f['now'])

    def bound(self):
        f=prequeue();v,d=self.reserve(f);r=json.loads(f['request_bytes'])
        v,d=accounting.bind_job(v,d,request_id=r['request_id'],job_id=f['job_id'],now_epoch=f['now'])
        return f,v,d

    def test_whole_source_block_reserved_before_qsub_and_bound_once(self):
        f=prequeue();before=deepcopy(f['ledger']);v,d=self.reserve(f)
        self.assertEqual(f['ledger'],before)
        self.assertEqual(v['used']['PBS_submissions'],13);self.assertEqual(v['reserved']['H200_s'],28860)
        self.assertEqual(v['reserved']['optimizer_updates'],4096);self.assertEqual(d['optimizer_updates_reserved'],4096)
        self.assertEqual(v['DEC1_source_pair_platform'],'VANDA_A40');validate_items(v);check_global_ledger(v)
        with self.assertRaises(PermissionError):require_hopper_active_chain(v)
        r=json.loads(f['request_bytes']);v,d=accounting.bind_job(v,d,request_id=r['request_id'],job_id=f['job_id'],now_epoch=f['now'])
        with self.assertRaises(PermissionError):accounting.bind_job(v,d,request_id=r['request_id'],job_id=f['job_id'],now_epoch=f['now'])

    def test_native_terminal_charges_only_real_32_updates_keeps_all_other_costs(self):
        f,v,d=self.bound();r=json.loads(f['request_bytes']);n,e=terminal(f)
        closed,dc=accounting.settle(v,d,request_id=r['request_id'],native_proof=n,effects=e,now_epoch=f['now'])
        validate_items(closed);check_global_ledger(closed)
        self.assertEqual(closed['used']['H200_s'],21745);self.assertEqual(closed['optimizer_updates'],32)
        self.assertEqual(closed['generation_queries'],1071);self.assertEqual(closed['reserved']['H200_s'],0)
        self.assertEqual(closed['reserved']['optimizer_updates'],0);self.assertEqual(dc['optimizer_updates'],32)
        self.assertEqual(closed['training_runs']['A_CONTROL']['last_native_step'],32)
        with self.assertRaises(PermissionError):
            accounting.settle(closed,dc,request_id=r['request_id'],native_proof=n,effects=e,now_epoch=f['now'])

    def test_unsettled_query_and_optimizer_intents_remain_reserved_and_block_future_work(self):
        f,v,d=self.bound();r=json.loads(f['request_bytes']);n,e=terminal(f)
        e['pending_updates']=[dict(arm='A_CONTROL',request_id=r['request_id'],local_step_before=32)]
        e['unknown_query_intents']=[16]
        closed,dc=accounting.settle(v,d,request_id=r['request_id'],native_proof=n,effects=e,now_epoch=f['now'])
        validate_items(closed)
        check_global_ledger(closed,allow_unknown=True)
        self.assertEqual(closed['reserved']['optimizer_updates'],1);self.assertEqual(dc['neural_queries_reserved'],1)
        self.assertTrue(closed['unknown_effects']);self.assertTrue(closed['pending_updates'])

    def test_missing_envelope_wrong_nonce_live_job_or_overrun_preserve_original_reservation(self):
        for field in ('running','nonce','cpu','GPU','source','step','query','output'):
            f,v,d=self.bound();r=json.loads(f['request_bytes']);n,e=terminal(f);before=deepcopy(v)
            if field=='running':n['job_state']='R'
            if field=='nonce':n['submission_nonce']='other'
            if field=='cpu':n['conservative_CPU_s']=200000
            if field=='GPU':n['conservative_GPU_s']=30000
            if field=='source':e['source_sha']='f'*40
            if field=='step':e['last_native_step']=33
            if field=='query':e['query_attempts']=30
            if field=='output':n['new_resident_allocated_upper_bytes']=100*1024**3
            with self.subTest(field=field),self.assertRaises(PermissionError):
                accounting.settle(v,d,request_id=r['request_id'],native_proof=n,effects=e,now_epoch=f['now'])
            self.assertEqual(v,before)

    def test_changed_global_state_or_foreign_pair_platform_cannot_reuse_forecast(self):
        for field in ('PBS','CPU','platform','DEC1','unknown'):
            f=prequeue()
            if field=='PBS':f['ledger']['used']['PBS_submissions']+=1
            if field=='CPU':f['ledger']['CPU_s']+=1
            if field=='platform':f['ledger']['DEC1_source_pair_platform']='HOPPER_H200'
            if field=='DEC1':f['DEC1']['CPU_s']+=1
            if field=='unknown':f['ledger']['unknown_effects']=['unknown']
            with self.subTest(field=field),self.assertRaises(PermissionError):self.reserve(f)


if __name__=='__main__':unittest.main()
