"""Synthetic CPU-only forecast arithmetic; not native timing or permission proof."""
from copy import deepcopy
import hashlib
import json
import math
import unittest

from scripts.pi0294.pv1 import future_budget as budget
from scripts.pi0294.pv1.decision import AUTHORITY


T0 = 1792348196
NOW = T0+10000
HASH = 'a'*64


def raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def observation(identifier='fixture-T2'):
    result = dict(id=identifier, status='VALID_COMPLETE', source_sha='b'*40,
        fixture=False, physics_steps=16000, rgb_pairs=641,
        error=None, wall_s=720.25, cpu_s=60.25)
    result_bytes = raw(result)
    row = dict(attempt_id=identifier, result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        pipeline_trustworthy=True, status='VALID_TASK_FAIL', main_integrations=16000,
        duration_s=32, live_native_failures=0, teacher_takeover=False)
    row_bytes = raw(row)
    audit = dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT', attempt_id=identifier,
        errors=[], actual_native_safety_flag_rows=0, model_autonomous_evaluation=True,
        execution_source_sha='b'*40,
        checks={key:True for key in budget.AUDIT_CHECKS}, adjudication=row)
    chain = dict(status='PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN', attempt_id=identifier,
        adjudication_sha256=hashlib.sha256(row_bytes).hexdigest(), draws=80, executed_ACKs=80,
        new_integrations=0, new_model_queries=0, core_proof_sha256=HASH, hopper_proof_sha256=HASH)
    worker=dict(result,wall_s=720.5,cpu_s=60.5,owned_bytes=10000000,output_bytes=10000000)
    return dict(result=result_bytes, audit=raw(audit), adjudication=row_bytes,
        causal_chain=raw(chain), worker_receipt=raw(worker), observed_epoch=NOW)


def case():
    return budget.case_upper([observation()], now_epoch=NOW, multiplier=1.25,
        floor=dict(H200_s=900, Core_CPU_s=120, Hopper_CPU_s=30, local_CPU_s=5,
            transfer_bytes=3000000, Hopper_storage_bytes=1000000,
            Core_storage_bytes=10000000, local_storage_bytes=100000, engine_starts=81))


def baseline():
    return raw(dict(schema='PI0294_PV1_B0_ACTUAL_DEV_ADMISSION_V1',
        status='PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE', authority_release=AUTHORITY,
        checkpoint_manifest_sha256=budget.PARENT, population_plan_sha256=budget.POPULATION,
        new_bank_manifest_sha256=HASH,
        condition_result_sha256={str(i):HASH for i in range(48)},
        independent_audit_sha256={str(i):HASH for i in range(48)}, real_native_faults=0,
        Teacher_takeover=False, all_condition_evidence_audits_PASS=True,
        task_success_rate_is_not_training_admission_gate=True, successes=dict(T1=0, T2=0)))


def progress(replica='REQUIRED', optional='NOT_SELECTED'):
    arms = ['A_CONTROL', 'B_COVERAGE']
    if optional == 'SELECTED':
        arms += ['C_CONTROL', 'D_INTERVENTION']
    if replica == 'REQUIRED':
        arms.append('REPLICA')
    return dict(schema='PI0294_PV1_OWNER_VALIDATED_PROGRESS_V1', authority_release=AUTHORITY,
        campaign_run_uuid='fixture', T0_epoch=T0, observed_epoch=NOW,
        new_bank_manifest_sha256=HASH,
        unknown_effects=[], native_safety_faults=0, route='COVERAGE_FIRST',
        optional_pair=optional, replica_recipe_updates=4096,
        resource_critical_replica_omission_evidence_sha256=HASH,
        candidates={arm:dict(optimizer_updates_committed=0,
            valid_DEV_condition_result_sha256={}) for arm in arms})


def snapshot():
    return dict(authority_release=AUTHORITY, campaign_run_uuid='fixture', T0_epoch=T0,
        new_bank_manifest_sha256=HASH,
        observed_epoch=NOW, unknown_effects=[], used={}, reserved={}, work_reservations=[],
        storage_peaks={key:0 for key in ('Hopper_storage_bytes', 'Core_storage_bytes', 'local_storage_bytes')},
        reserved_sequential_wall_upper_s=0, clock=dict(stop_new_jobs_epoch=T0+144*3600,
            stop_training_epoch=T0+156*3600, stop_physics_epoch=T0+160*3600,
            hard_closeout_epoch=T0+168*3600))


def assumptions():
    return dict(site_wall_limit_s=8*3600, accounting_pad_s=60,
        per_job_overhead=dict(H200_s=900, Core_CPU_s=30, Hopper_CPU_s=30, local_CPU_s=30,
            engine_starts=2), per_probe_upper=dict(H200_s=10, Hopper_CPU_s=5,
                local_CPU_s=1, neural_queries=1, Hopper_storage_bytes=100000),
        queue_delay_upper_s_per_job=60,
        package_closeout_wall_s=3600, package_closeout_resources=dict(local_CPU_s=60),
        DEV_engineering_cases_per_candidate_upper=0)


def training_jobs(p, c, a):
    jobs = []
    for need in budget.requirements(raw(p), replica_mode='REQUIRED' if 'REPLICA' in p['candidates'] else 'OMITTED_RESOURCE_CRITICAL')['work']:
        if need['kind'] != 'TRAIN':
            continue
        resources = {key:a['per_job_overhead'].get(key, 0)+need['TRAIN_probes']*a['per_probe_upper'].get(key,0)
            for key in budget.CAPS}
        resources.update(H200_s=18060, PBS_submissions=1,
            optimizer_updates=need['optimizer_updates'])
        jobs.append(dict(work_id=need['work_id'], blocks=[dict(walltime_upper_s=18000,
            optimizer_updates=need['optimizer_updates'], TRAIN_probes=need['TRAIN_probes'],
            resources=resources)]))
    return jobs


def run(**changes):
    p=progress();c=case();a=assumptions()
    arguments=dict(snapshot=snapshot(), progress_bytes=raw(p), baseline_admission_bytes=baseline(),
        case_cost=c, assumptions=a, training_jobs=training_jobs(p,c,a), replica_mode='REQUIRED', now_epoch=NOW)
    arguments.update(changes)
    return budget.forecast(**arguments)


class FutureBudgetContracts(unittest.TestCase):
    def test_original_bytes_full_cases_and_explicit_headroom_not_success_rate(self):
        c=case()
        self.assertEqual(c['resources']['H200_s'],901)
        self.assertEqual(c['resources']['Core_CPU_s'],120)
        self.assertEqual(c['resources']['Core_storage_bytes'],12500000)
        self.assertEqual(c['resources']['physics_ticks'],32000)
        self.assertTrue(c['not_statistical_or_native_upper_bound'])
        self.assertTrue(run()['admissible_forecast'])
        self.assertEqual(json.loads(baseline())['successes'],dict(T1=0,T2=0))

    def test_partial_case_cannot_underestimate_full_remaining_conditions(self):
        o=observation();r=json.loads(o['result']);r['physics_steps']=14400;o['result']=raw(r)
        with self.assertRaisesRegex(PermissionError,'FULL_32S'):
            budget.case_upper([o], floor=case()['declared_floor'], multiplier=1.25, now_epoch=NOW)
        with self.assertRaisesRegex(PermissionError,'UNIQUE_TERMINAL'):
            budget.case_upper([observation(),observation()],floor=case()['declared_floor'],multiplier=1.25,now_epoch=NOW)

    def test_all_dimensions_and_final_protection_are_inside_the_total(self):
        out=run();future=out['future_required_additional']
        final=out['mandatory_jobs'][-1]
        self.assertEqual(set(future),set(budget.CAPS))
        for key,minimum in budget.FINAL.items():
            self.assertGreaterEqual(final['resources'][key],minimum)
        self.assertGreaterEqual(final['resources']['PBS_submissions'],3)
        self.assertEqual(future['optimizer_updates'],3*4096)
        self.assertEqual(len([j for j in out['mandatory_jobs'] if j['kind']=='DEV']),6)
        self.assertEqual(out['maximum_possible_grade'],'PV1_VALIDATED')
        self.assertTrue(out['estimates_are_not_native_allocation_or_timing_guarantees'])

    def test_mandatory_arms_and_full_baseline_cannot_be_omitted(self):
        p=progress();p['candidates'].pop('B_COVERAGE')
        with self.assertRaisesRegex(PermissionError,'MISSING_MANDATORY'):
            run(progress_bytes=raw(p))
        b=json.loads(baseline());b['condition_result_sha256'].pop('0')
        with self.assertRaisesRegex(PermissionError,'COMPLETE_ACTUAL_B0'):
            run(baseline_admission_bytes=raw(b))
        with self.assertRaisesRegex(PermissionError,'UPPER_MISSING'):
            run(training_jobs=[])

    def test_explicit_resource_critical_replica_omission_caps_grade(self):
        p=progress(replica='OMITTED_RESOURCE_CRITICAL');c=case();a=assumptions()
        out=run(progress_bytes=raw(p),replica_mode='OMITTED_RESOURCE_CRITICAL',training_jobs=training_jobs(p,c,a))
        self.assertEqual(out['maximum_possible_grade'],'PV1_BETA')
        self.assertEqual(out['future_required_additional']['optimizer_updates'],8192)
        p.pop('resource_critical_replica_omission_evidence_sha256')
        with self.assertRaisesRegex(PermissionError,'SHA256'):
            run(progress_bytes=raw(p),replica_mode='OMITTED_RESOURCE_CRITICAL')

    def test_selected_optional_pair_is_charged_not_silently_dropped(self):
        p=progress(optional='SELECTED');c=case();a=assumptions()
        out=run(progress_bytes=raw(p),training_jobs=training_jobs(p,c,a))
        self.assertEqual(out['future_required_additional']['optimizer_updates'],16384)
        self.assertEqual(len([j for j in out['mandatory_jobs'] if j['kind']=='TRAIN']),5)
        self.assertFalse(out['admissible_forecast'])
        self.assertIn('H200_s',out['resource_shortfalls'])

    def test_progress_uses_actual_committed_terminal_steps_not_planned_steps(self):
        p=progress();p['candidates']['A_CONTROL'].update(optimizer_updates_committed=1024,terminal_evidence_sha256=HASH)
        s=snapshot();s['used']['optimizer_updates']=1024
        c=case();a=assumptions();out=run(snapshot=s,progress_bytes=raw(p),training_jobs=training_jobs(p,c,a))
        self.assertEqual(out['future_required_additional']['optimizer_updates'],11264)
        p['candidates']['A_CONTROL']['valid_DEV_condition_result_sha256']={'x':HASH}
        with self.assertRaisesRegex(PermissionError,'INTERMEDIATE_CHECKPOINT'):
            run(progress_bytes=raw(p))

    def test_fixed_train_probes_are_archived_single_queries_not_live_episodes(self):
        out=run()
        jobs=[j for j in out['mandatory_jobs'] if j['kind']=='TRAIN']
        self.assertEqual(len(jobs),3)
        for job in jobs:
            resources=job['resources']
            self.assertEqual(resources['neural_queries'],12)
            self.assertEqual(resources['episode_starts'],0)
            self.assertEqual(resources['physics_ticks'],0)
            self.assertEqual(resources['engine_starts'],2)
        a=assumptions();a['per_probe_upper']['physics_ticks']=1
        with self.assertRaisesRegex(PermissionError,'ARCHIVED_WINDOW_DRAW_NO_PHYSICS'):
            run(assumptions=a)

    def test_resource_shortfalls_report_each_dimension_without_relabeling_as_allocation(self):
        s=snapshot();s['used'].update(H200_s=75*3600,Core_CPU_s=319*3600,transfer_bytes=299*budget.GIB)
        out=run(snapshot=s)
        self.assertFalse(out['admissible_forecast'])
        self.assertEqual(out['status'],'STOP_NEW_CANDIDATE_INSUFFICIENT_FORECAST_RUNWAY')
        self.assertTrue({'H200_s','Core_CPU_s'}<=set(out['resource_shortfalls']))

    def test_clock_has_separate_submission_training_physics_closeout_deadlines(self):
        late=T0+143*3600
        p=progress();p['observed_epoch']=late;s=snapshot();s['observed_epoch']=late
        c=case();c['observed_epoch']=late
        out=run(snapshot=s,progress_bytes=raw(p),case_cost=c,now_epoch=late)
        self.assertFalse(out['admissible_forecast'])
        self.assertIn('LAST_REQUIRED_SUBMISSION_AFTER_144H',out['clock_failures'])
        self.assertIn('REQUIRED_EVALUATION_AFTER_160H',out['clock_failures'])
        self.assertIn('PACKAGE_CLOSEOUT_AFTER_168H',out['clock_failures'])
        s['clock']['hard_closeout_epoch']+=1
        with self.assertRaisesRegex(PermissionError,'IMMUTABLE_ORIGINAL_CLOCK'):
            run(snapshot=s,progress_bytes=raw(p),case_cost=c,now_epoch=late)

    def test_reserved_work_is_credited_once_not_paid_twice(self):
        original=run();job=next(j for j in original['mandatory_jobs'] if j['kind']=='TRAIN')
        s=snapshot();resources=deepcopy(job['resources']);resources['PBS_submissions']=0
        s.update(reserved=resources,reserved_sequential_wall_upper_s=resources['H200_s']+60,
            work_reservations=[dict(work_id=job['work_id'],job_id='fixture.pbs',kind='TRAIN',
                submission_already_charged=True,resources=resources)])
        s['used']['PBS_submissions']=1
        out=run(snapshot=s)
        self.assertEqual(out['future_required_additional']['H200_s'],
            original['future_required_additional']['H200_s']-job['resources']['H200_s'])
        self.assertEqual(out['future_required_additional']['PBS_submissions'],
            original['future_required_additional']['PBS_submissions']-1)
        s['work_reservations'].append(deepcopy(s['work_reservations'][0]))
        with self.assertRaisesRegex(PermissionError,'UNIQUE_WORK'):
            run(snapshot=s)

    def test_partial_or_unknown_reservation_cannot_hide_timing_or_cost(self):
        job=next(j for j in run()['mandatory_jobs'] if j['kind']=='TRAIN')
        s=snapshot();resources=deepcopy(job['resources']);resources['H200_s']-=1;resources['PBS_submissions']=0
        s.update(reserved=resources,reserved_sequential_wall_upper_s=resources['H200_s']+60,
            work_reservations=[dict(work_id=job['work_id'],job_id='fixture.pbs',kind='TRAIN',
                submission_already_charged=True,resources=resources)])
        s['used']['PBS_submissions']=1
        with self.assertRaisesRegex(PermissionError,'FULLY_FUND'):
            run(snapshot=s)
        s['reserved']=deepcopy(s['reserved']);s['reserved']['H200_s']+=1
        with self.assertRaisesRegex(PermissionError,'RECONCILED'):
            run(snapshot=s)

    def test_native_gpu_only_reserve_does_not_credit_unreserved_cpu_updates_or_queries(self):
        original=run();job=next(j for j in original['mandatory_jobs'] if j['kind']=='TRAIN')
        s=snapshot();resources=dict(H200_s=job['resources']['H200_s'])
        s.update(reserved=resources,reserved_sequential_wall_upper_s=resources['H200_s']+60,
            work_reservations=[dict(work_id=job['work_id'],job_id='fixture.pbs',kind='TRAIN',
                submission_already_charged=True,resources=resources)])
        s['used']['PBS_submissions']=1;out=run(snapshot=s)
        for key in ('optimizer_updates','neural_queries','Core_CPU_s','Hopper_CPU_s','local_CPU_s'):
            self.assertEqual(out['future_required_additional'][key],original['future_required_additional'][key])
        self.assertEqual(out['future_required_additional']['H200_s'],
            original['future_required_additional']['H200_s']-resources['H200_s'])

    def test_stale_unknown_nonfinite_boolean_and_unknown_dimensions_fail_closed(self):
        for mutate in (lambda s:s.update(observed_epoch=NOW-901),
                       lambda s:s.update(unknown_effects=['unsettled_query']),
                       lambda s:s['used'].update(H200_s=math.inf),
                       lambda s:s['used'].update(PBS_submissions=True),
                       lambda s:s['used'].update(A40_s=0)):
            with self.subTest(mutate=mutate):
                s=snapshot();mutate(s)
                with self.assertRaises(PermissionError):run(snapshot=s)
        with self.assertRaisesRegex(PermissionError,'DUPLICATE_JSON_KEY'):
            budget.requirements(b'{"schema":1,"schema":2}',replica_mode='REQUIRED')

    def test_qualification_overhead_and_site_limit_count_per_integer_job(self):
        a=assumptions();a['per_job_overhead']['H200_s']=599
        with self.assertRaisesRegex(PermissionError,'600S_QUALIFICATION'):
            run(assumptions=a)
        a=assumptions();a['site_wall_limit_s']=1000
        with self.assertRaisesRegex(PermissionError,'NO_FULL_CASE'):
            run(assumptions=a)
        a=assumptions();a['accounting_pad_s']=0
        with self.assertRaisesRegex(PermissionError,'PAD_AT_LEAST_60S'):
            run(assumptions=a)

    def test_future_storage_uses_current_residency_and_checks_historical_peaks_separately(self):
        s=snapshot();s['storage_peaks']['Hopper_storage_bytes']=240*budget.GIB
        self.assertTrue(run(snapshot=s)['admissible_forecast'])
        s['used']['Hopper_storage_bytes']=230*budget.GIB
        self.assertIn('Hopper_storage_bytes',run(snapshot=s)['resource_shortfalls'])
        s['storage_peaks']['Hopper_storage_bytes']=1
        with self.assertRaisesRegex(PermissionError,'HISTORICAL_PEAK'):
            run(snapshot=s)
