"""CPU-only segmented lifecycle and full-work arithmetic; no real dispatch."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from . import segmented_training as seg
from . import a40_allocation as a
from . import a40_accounting as accounting
from . import a40_future as future
from . import future_budget as budget
from . import dual_platform as dual
from . import lost_save_recovery as rec
from .test_lost_save_recovery import recovered,inputs_for
from .test_a40_accounting import terminal
from .test_a40_allocation import raw

CARD=Path(__file__).resolve().parents[3]/'reports/PI0294/design/VAP1_A40_SEGMENTED_RECOVERY_AUTHORIZATION_20261022.json'


def setup(f,h,d,c):
    h,d=rec.adopt(h,d,grant_bytes=(CARD.parent/'VAP1_A40_LOST_SAVE_RECOVERY_AUTHORIZATION_20261022.json').read_bytes(),
        closure_bytes=c,now_epoch=f['now'])
    h['optimizer_updates']=h['used']['optimizer_updates']=d['optimizer_updates']=1056
    h['training_runs'][rec.LINEAGE]=dict(last_native_step=32,recipe_sha256=rec.RECIPE)
    while len(h['PBS_submissions'])<15:h['PBS_submissions'].append(dict(request_id='closed-'+str(len(h['PBS_submissions'])),status='NATIVE_TERMINAL'))
    while len(d['new_submissions'])<4:d['new_submissions'].append(dict(request_id='dec-closed-'+str(len(d['new_submissions'])),status='NATIVE_TERMINAL'))
    h['used']['PBS_submissions']=15
    closure=raw(dict(fixture='CLOSED_NATIVE32'))
    with patch.object(seg,'CLOSURE_SHA',hashlib.sha256(closure).hexdigest()):
        h,d=seg.adopt(h,d,grant_bytes=CARD.read_bytes(),closure_bytes=closure,now_epoch=f['now'])
    values=inputs_for(f,h,d);r=json.loads(values['request_bytes']);s=json.loads(values['snapshot_bytes'])
    p=json.loads(values['progress_bytes']);plan=json.loads(values['plan_bytes'])
    r.update(segmented_authorization_release=seg.RELEASE,segmented_grant_sha256=seg.GRANT_SHA,
        CPU_envelope_profile=seg.CPU_PROFILE,CPU_upper_s=54000,walltime_s=22500,submit_count=16,
        expected_native_start_step=32,additional_output_upper_bytes=70*10**9)
    r['training_block'].update(stop_after_step=2048,resume_checkpoint_relative='training/A_CONTROL/checkpoint-32')
    p['candidates']['A_CONTROL'].update(optimizer_updates_committed=32,checkpoint_complete_sha256='c'*64)
    s['segmented_grant_sha256']=seg.GRANT_SHA
    s['dec1_storage_peaks']['Hopper_storage_bytes']=d['new_GPU_site_output_allocated_upper_bytes']
    plan.update(segmented_authorization_release=seg.RELEASE,performance_basis='MEASURED_A40_FIRST32_MAX_WITH_HEADROOM_V1',
        seconds_per_update_upper=seg.RATE,qualification_load_save_close_upper_s=3600,training_jobs=[])
    for arm,start,stop in (('A_CONTROL',32,2048),('A_CONTROL',2048,4096),('B_COVERAGE',0,2048),('B_COVERAGE',2048,4096)):
        plan['training_jobs'].append(dict(work_id=f'TRAIN:{arm}:SEGMENT:{stop}',arm=arm,start_step=start,stop_step=stop,
            same_allocation_1024_probes=True,segmented_authorization_release=seg.RELEASE,segmented_grant_sha256=seg.GRANT_SHA,
            CPU_envelope_profile=seg.CPU_PROFILE,walltime_upper_s=22500,
            resources=dict(H200_s=22560,PBS_submissions=1,optimizer_updates=stop-start,neural_queries=29,
                Hopper_CPU_s=54000,Core_CPU_s=180,local_CPU_s=120,engine_starts=3,
                transfer_bytes=2*a.GIB,Hopper_storage_bytes=70*10**9)))
    plan['storage_lifecycle']['new_output_peak_upper_bytes']['Hopper_storage_bytes']=176*a.GIB
    values.update(segment_bytes=CARD.read_bytes(),request_bytes=raw(r),snapshot_bytes=raw(s),progress_bytes=raw(p),plan_bytes=raw(plan))
    return h,d,values


class SegmentedTrainingTests(unittest.TestCase):
    def test_exact_grant_no_implicit_cap_change_or_minted_permission(self):
        self.assertEqual(seg.grant(CARD.read_bytes())['training_wall_s_each'],22500)
        with self.assertRaises(PermissionError):seg.grant(CARD.read_bytes()+b' ')
        with self.assertRaises(PermissionError):budget.effective_caps(segment_bytes=CARD.read_bytes())
        self.assertEqual(budget.effective_caps()['PBS_submissions'],8)
        self.assertEqual(budget.effective_caps()['H200_s'],172800)
        with recovered() as (f,h,d,c):
            h,d,values=setup(f,h,d,c)
            with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(values['amendment_bytes']).hexdigest()):
                proof=future.recompute(**values)
                self.assertTrue(proof['forecast']['admissible_forecast'])
                self.assertEqual(proof['forecast']['effective_DEC1_VAP1_caps']['PBS_submissions'],11)
                self.assertEqual(proof['forecast']['effective_DEC1_VAP1_caps']['H200_s'],187200)
                self.assertEqual(proof['forecast']['future_required_additional']['optimizer_updates'],8160)
                self.assertEqual(proof['forecast']['original_final_protection']['H200_s'],72000)

    def test_four_precise_segments_and_real_CPU_hardlimit_not_arbitrary_partial_training(self):
        with recovered() as (f,h,d,c):
            h,d,values=setup(f,h,d,c);r=json.loads(values['request_bytes'])
            a.request_scope(r,now=f['now']);self.assertEqual(a.process_cpu_limit(r),53760)
            for key,value in (('expected_native_start_step',0),('CPU_upper_s',80000),('walltime_s',28800),('segmented_grant_sha256','x')):
                changed=deepcopy(r);changed[key]=value
                with self.subTest(key=key),self.assertRaises(PermissionError):a.request_scope(changed,now=f['now'])
            self.assertEqual(h['training_runs']['A_CONTROL']['last_native_step'],1024)
            self.assertEqual(h['training_runs'][rec.LINEAGE]['last_native_step'],32)

    def test_reservation_and_terminal_only_charge_32_to2048_not_restart4096(self):
        with recovered() as (f,h,d,c):
            h,d,values=setup(f,h,d,c);r=json.loads(values['request_bytes'])
            with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(values['amendment_bytes']).hexdigest()):proof=future.recompute(**values)
            with patch.object(dual,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()):
                v,dd=accounting.reserve(h,d,request_bytes=values['request_bytes'],budget_proof=proof,
                    grant_bytes=f['grant_bytes'],campaign_view=f['campaign_view'],now_epoch=f['now'])
            self.assertEqual(v['reserved']['optimizer_updates'],2016);self.assertEqual(v['CPU_reserved_s'],54000)
            self.assertEqual(v['PBS_submissions'][-1]['dec1_work_id'],'TRAIN:A_CONTROL:SEGMENT:2048')
            v,dd=accounting.bind_job(v,dd,request_id=r['request_id'],job_id=f['job_id'],now_epoch=f['now'])
            ff=deepcopy(f);ff['request_bytes']=values['request_bytes'];n,e=terminal(ff)
            e.update(start_native_step=32,last_native_step=2048,optimizer_updates_committed=2016,query_attempts=29,recipe_sha256=rec.RECIPE)
            v,dd=accounting.settle(v,dd,request_id=r['request_id'],native_proof=n,effects=e,now_epoch=f['now'])
            self.assertEqual(v['optimizer_updates'],3072)
            self.assertEqual(v['training_runs']['A_CONTROL']['last_native_step'],1024)
            self.assertEqual(v['training_runs'][rec.LINEAGE]['last_native_step'],2048)
            self.assertEqual(v['reserved']['optimizer_updates'],0)

    def test_failed_gate_old_rate_or_missing_adoption_cannot_fund_continuation(self):
        with recovered() as (f,h,d,c):
            h,d,values=setup(f,h,d,c)
            for kind in ('grant','marker','rate','clock'):
                v=deepcopy(values)
                if kind=='grant':v['segment_bytes']=None
                elif kind=='marker':
                    hh=json.loads(v['ledger_bytes']);hh.pop(seg.FIELD);v['ledger_bytes']=raw(hh)
                elif kind=='rate':
                    plan=json.loads(v['plan_bytes']);plan['seconds_per_update_upper']=6.1;v['plan_bytes']=raw(plan)
                else:v['now_epoch']=a.T0+144*3600
                with self.subTest(kind=kind),patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(values['amendment_bytes']).hexdigest()):
                    with self.assertRaises(PermissionError):future.recompute(**v)

    def test_same_job1024_probes_then2048_and_separate_full_state_reentry4096(self):
        events=[]
        def train(**kw):events.append(('train',kw));return dict(global_step=kw['stop'])
        kw=dict(train=train,probe=lambda:events.append('12-original-probes'),unload=lambda:events.append('unload'),
            guard=lambda:events.append('guard'),load=lambda path,trainable:events.append(('load',str(path),trainable)),checkpoint='checkpoint-1024')
        result,_=seg.execute(start_step=32,stop_step=2048,resume_checkpoint='checkpoint-32',**kw)
        self.assertEqual(result['global_step'],2048)
        self.assertEqual([x[1]['stop'] for x in events if isinstance(x,tuple) and x[0]=='train'],[1024,2048])
        self.assertEqual(events.count('12-original-probes'),1)
        events.clear();result,_=seg.execute(start_step=2048,stop_step=4096,resume_checkpoint='checkpoint-2048',**kw)
        self.assertEqual(events,[('train',dict(stop=4096,phase='SEGMENT_REENTRY',checkpoint='checkpoint-2048'))])

    def test_partial_first_phase_does_not_probe_or_continue(self):
        never=lambda *a,**k:self.fail('must stop at known partial')
        result,records=seg.execute(start_step=32,stop_step=2048,train=lambda **kw:dict(global_step=90),
            probe=never,unload=never,load=never,guard=never,checkpoint='checkpoint-1024',resume_checkpoint='checkpoint-32')
        self.assertEqual(result['global_step'],90);self.assertEqual(len(records),1)
