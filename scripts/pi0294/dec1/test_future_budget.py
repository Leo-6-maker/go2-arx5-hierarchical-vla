"""Synthetic arithmetic only: no claim of native timing, GPU or target readiness."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import future_budget as old
from scripts.pi0294.test_pv1_future_budget import raw,T0,HASH
from .authority import AUTHORITY,BANK,PARENT_AUTHORITY
from .future_budget import forecast
from . import future_budget as budget

NOW=T0+48*3600
GIB=1024**3


def amendment():
    return dict(schema='VAP1_ADOPTION_AND_DELTA_BUDGET_V1',authority_release=budget.VAP1,
        original_T0_epoch=1792348196,original_hard_close_epoch=budget.HARD_CLOSE,
        cumulative_subcap_delta=dict(local_CPU_s=[28800,57600],CoreLab_CPU_s=[345600,576000],
            GPU_site_output_bytes=[64*GIB,160*GIB],CoreLab_output_bytes=[64*GIB,128*GIB],transfer_bytes=[64*GIB,128*GIB]),
        old_clocks_and_counters_reset=False,GPU_global_hours=96,GPU_DEC1_hours=48,
        global_PBS_limit=24,DEC1_new_PBS_limit=8,concurrent_GPUs=1,final_GPU_protected_s=72000)


def fixtures():
    clock=dict(stop_new_jobs_epoch=T0+144*3600,stop_training_epoch=T0+156*3600,
               stop_physics_epoch=T0+160*3600,hard_closeout_epoch=T0+168*3600)
    s=dict(schema='DEC1_SINGLE_OWNER_FRESH_RESOURCE_SNAPSHOT_V1',authority_release=AUTHORITY,
        original_authority_release=PARENT_AUTHORITY,campaign_run_uuid='original',T0_epoch=T0,
        observed_epoch=NOW,clock=clock,unknown_effects=[],native_safety_faults=0,new_bank_manifest_sha256=BANK,
        schedule=dict(target_ready=False,interfaces_ready=False,core_matrix_complete=False,evidence_sha256=HASH),
        used=dict(H200_s=20494,PBS_submissions=11,Core_CPU_s=140000,Hopper_CPU_s=7000,local_CPU_s=26500,
                  Hopper_storage_bytes=40*GIB,Core_storage_bytes=20*GIB,local_storage_bytes=2*GIB),
        reserved={},dec1_used=dict(Core_CPU_s=360,Hopper_CPU_s=180,local_CPU_s=1200),dec1_reserved={},
        work_reservations=[],storage_peaks=dict(Hopper_storage_bytes=40*GIB,Core_storage_bytes=20*GIB,local_storage_bytes=2*GIB),
        dec1_storage_peaks=dict(Hopper_storage_bytes=0,Core_storage_bytes=0,local_storage_bytes=0))
    p=dict(schema='DEC1_SOURCE_TRAIN_OWNER_PROGRESS_V1',authority_release=AUTHORITY,
        campaign_run_uuid='original',T0_epoch=T0,observed_epoch=NOW,new_bank_manifest_sha256=BANK,
        unknown_effects=[],native_safety_faults=0,
        candidates={a:dict(optimizer_updates_committed=0,probe_1024_complete=False) for a in ('A_CONTROL','B_COVERAGE')})
    plan=dict(schema='DEC1_PRESEALED_SOURCE_AND_MATRIX_RESOURCE_PLAN_V1',authority_release=AUTHORITY,
        old_PV1_release_or_LOCKBOX_admitted=False,target_cost_is_measured=False,
        matrix_is_budget_ceiling_not_feasibility_proof=True,accounting_pad_s=60,site_wall_limit_s=28800,
        qualification_load_save_close_upper_s=3300,seconds_per_update_upper=4.,seconds_per_archived_probe_upper=10.,
        performance_evidence_sha256=HASH,full_checkpoint_payload_upper_bytes=26*10**9,
        checkpoint_size_evidence_sha256=HASH,queue_delay_upper_s_per_job=3600,package_closeout_wall_s=3600,
        training_jobs=[],remaining_matrix_envelope=dict(core_cells=64,commissioning_upper=8,
            engineering_replacements_upper=16,valid_failure_replays=0,
            resources=dict(H200_s=24*3600+180,PBS_submissions=3,episode_starts=88,neural_queries=7040,
                physics_ticks=2816000,engine_starts=7128,Core_CPU_s=200000,Hopper_CPU_s=80000,local_CPU_s=3600,
                transfer_bytes=5*GIB,Hopper_storage_bytes=2*GIB,Core_storage_bytes=20*GIB,local_storage_bytes=GIB)),
        storage_lifecycle=dict(schema='DEC1_PRESEALED_CHECKPOINT_STORAGE_LIFECYCLE_V1',
            full_resume_members_preserved=True,retire_only_after_actual_independent_readback=True,
            actual_transition_guard_before_later_checkpoint=True,implementation_or_execution_plan_sha256=HASH,
            new_output_peak_upper_bytes=dict(Hopper_storage_bytes=64*GIB,Core_storage_bytes=48*GIB,local_storage_bytes=2*GIB)))
    for arm in p['candidates']:
        for n,(start,stop,wall,cpu) in enumerate(((0,1024,10800,40000),(1024,4096,18000,60000)),1):
            plan['training_jobs'].append(dict(work_id=f'TRAIN:{arm}:{n}',arm=arm,start_step=start,stop_step=stop,
                walltime_upper_s=wall,resources=dict(H200_s=wall+60,PBS_submissions=1,optimizer_updates=stop-start,
                    neural_queries=12 if stop==1024 else 0,Hopper_CPU_s=cpu,Core_CPU_s=180,local_CPU_s=120,
                    engine_starts=2,transfer_bytes=10*GIB,Hopper_storage_bytes=26*10**9)))
    return s,p,plan


class ForecastTests(unittest.TestCase):
    def test_explicit_exact_amendment_changes_only_existing_non_GPU_subcaps(self):
        value=amendment();data=raw(value)
        with patch.object(budget,'VAP1_ADOPTION',hashlib.sha256(data).hexdigest()):
            caps=budget.effective_caps(data)
            self.assertEqual(caps['Hopper_storage_bytes'],160*GIB)
            self.assertEqual(caps['Core_storage_bytes'],128*GIB)
            self.assertEqual(caps['local_CPU_s'],16*3600)
            for key in ('H200_s','PBS_submissions','optimizer_updates','neural_queries','Hopper_CPU_s','episode_starts','physics_ticks'):
                self.assertEqual(caps[key],budget.CAPS[key])
            s,p,q=fixtures();q['storage_lifecycle']['new_output_peak_upper_bytes']['Hopper_storage_bytes']=128*GIB
            with self.assertRaises(PermissionError):forecast(snapshot=s,progress_bytes=raw(p),plan=q,now_epoch=NOW)
            v=forecast(snapshot=s,progress_bytes=raw(p),plan=q,now_epoch=NOW,amendment_bytes=data)
            self.assertTrue(v['admissible_forecast']);self.assertEqual(v['original_final_protection']['H200_s'],72000)
            self.assertEqual(v['budget_amendment_sha256'],hashlib.sha256(data).hexdigest())
            for key,wrong in (('concurrent_GPUs',2),('GPU_DEC1_hours',96),('old_clocks_and_counters_reset',True)):
                bad=deepcopy(value);bad[key]=wrong
                with self.subTest(key=key),self.assertRaises(PermissionError):budget.effective_caps(raw(bad))
        with self.assertRaises(PermissionError):budget.effective_caps(data)
        self.assertEqual(budget.effective_caps(),budget.CAPS)

    def test_pre_b0_source_is_funded_but_target_feasibility_not_fabricated(self):
        s,p,plan=fixtures();v=forecast(snapshot=s,progress_bytes=raw(p),plan=plan,now_epoch=NOW)
        self.assertTrue(v['admissible_forecast'])
        self.assertEqual(v['future_required_additional']['optimizer_updates'],8192)
        self.assertEqual(v['future_required_additional']['PBS_submissions'],7)
        self.assertEqual(v['future_required_additional']['Hopper_storage_bytes'],64*GIB)
        self.assertFalse(v['old_PV1_release_and_LOCKBOX_admitted'])
        self.assertEqual(v['target_cost_feasibility'],'UNKNOWN_UNTIL_NATIVE_COMMISSIONING')
        self.assertEqual(v['original_final_protection']['H200_s'],72000)

    def test_same_native_allocation_is_credited_once_not_subtracted_twice(self):
        s,p,plan=fixtures();job=plan['training_jobs'][0]
        s['used']['PBS_submissions']+=1;s['dec1_used']['PBS_submissions']=1
        s['reserved']=dict(H200_s=10860);s['dec1_reserved']=dict(H200_s=10860)
        s['current_native_request_work_id']=job['work_id']
        s['work_reservations']=[dict(work_id=job['work_id'],job_id='new.hopper-m-02',kind='TRAIN',
            submission_already_charged=True,resources=dict(H200_s=10860))]
        v=forecast(snapshot=s,progress_bytes=raw(p),plan=plan,now_epoch=NOW)
        self.assertTrue(v['admissible_forecast']);self.assertEqual(v['matching_reserved_credits']['PBS_submissions'],1)
        self.assertEqual(v['future_required_additional']['PBS_submissions'],6)
        s['work_reservations'].append(deepcopy(s['work_reservations'][0]))
        with self.assertRaises(PermissionError):forecast(snapshot=s,progress_bytes=raw(p),plan=plan,now_epoch=NOW)

    def test_existing_cold_core_reservation_not_a_free_gpu_or_submission(self):
        s,p,plan=fixtures();s['reserved']=dict(Core_CPU_s=180)
        s['dec1_reserved']=dict(Core_CPU_s=180);s['current_native_request_work_id']=plan['training_jobs'][0]['work_id']
        s['work_reservations']=[dict(work_id=s['current_native_request_work_id'],job_id=None,
            kind='TRAIN',submission_already_charged=False,reservation_scope='COLD_NON_GPU_ONLY',resources=dict(Core_CPU_s=180))]
        v=forecast(snapshot=s,progress_bytes=raw(p),plan=plan,now_epoch=NOW)
        self.assertTrue(v['admissible_forecast']);self.assertEqual(v['matching_reserved_credits']['PBS_submissions'],0)

    def test_missing_arm_probe_skipped_phase_clock_and_unfunded_matrix_rejected(self):
        for change in (
            lambda s,p,q:q['training_jobs'].pop(),
            lambda s,p,q:q['training_jobs'][0].update(stop_step=4096),
            lambda s,p,q:q['remaining_matrix_envelope'].update(core_cells=32),
            lambda s,p,q:s['clock'].update(hard_closeout_epoch=1),
            lambda s,p,q:s.update(unknown_effects=['UNKNOWN']),
            lambda s,p,q:s['schedule'].update(target_ready=True,interfaces_ready=True),
            lambda s,p,q:q['storage_lifecycle'].update(retire_only_after_actual_independent_readback=False),
        ):
            with self.subTest(change=change):
                s,p,q=fixtures();change(s,p,q)
                with self.assertRaises(PermissionError):forecast(snapshot=s,progress_bytes=raw(p),plan=q,now_epoch=NOW)

    def test_original_and_dec1_bounds_are_both_enforced(self):
        s,p,q=fixtures();s['used']['H200_s']=200000
        v=forecast(snapshot=s,progress_bytes=raw(p),plan=q,now_epoch=NOW)
        self.assertFalse(v['admissible_forecast']);self.assertIn('H200_s',v['global_shortfalls'])
        s,p,q=fixtures();s['used']['PBS_submissions']=19;s['dec1_used']['PBS_submissions']=3
        v=forecast(snapshot=s,progress_bytes=raw(p),plan=q,now_epoch=NOW)
        self.assertFalse(v['admissible_forecast']);self.assertIn('PBS_submissions',v['DEC1_shortfalls'])


if __name__=='__main__':
    unittest.main()
