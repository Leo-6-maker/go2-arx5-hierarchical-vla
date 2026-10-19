"""Pure, conservative PV1 runway forecasts; never an allocation or scientific GO.

The owner must first validate the original panel/checkpoint evidence. This helper
binds that owner's projection to bytes, derives mandatory remaining work, and
does not acquire locks, write a ledger, contact a site, or authorize a rollout.
"""
import hashlib
import json
import math
import re
from copy import deepcopy

from .decision import AUTHORITY
from .baseline import AUDIT_CHECKS
from .dataset import POPULATION
from .recipe import PARENT
from .resources import TOTAL
from .sampler import ARMS


GIB = 1024**3
CAPS = dict(TOTAL, Core_CPU_s=320*3600, Hopper_CPU_s=160*3600,
    local_CPU_s=48*3600, transfer_bytes=300*GIB, Hopper_storage_bytes=250*GIB,
    Core_storage_bytes=300*GIB, local_storage_bytes=20*GIB)
FINAL = dict(H200_s=20*3600, episode_starts=100, neural_queries=9000,
    physics_ticks=3200000, Hopper_storage_bytes=40*GIB)
COUNTERS = set(CAPS)-{'H200_s', 'Core_CPU_s', 'Hopper_CPU_s', 'local_CPU_s'}
SHA = re.compile('[0-9a-f]{64}')


def _fail(message):
    raise PermissionError('PV1_FUTURE_BUDGET_'+message)


def _number(value, *, integer=False, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or positive and value == 0 or integer and type(value) is not int):
        _fail('FINITE_NONNEGATIVE_EXACT_COUNTERS')
    return value


def _vector(value):
    if type(value) is not dict or set(value)-set(CAPS):
        _fail('KNOWN_RESOURCE_DIMENSIONS_ONLY')
    return {key: _number(value.get(key, 0), integer=key in COUNTERS) for key in CAPS}


def _read(raw):
    if type(raw) is not bytes:
        _fail('ACTUAL_EVIDENCE_BYTES_REQUIRED')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail('DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _: _fail('NONFINITE_JSON'))
    except (ValueError, UnicodeError) as exc:
        raise PermissionError('PV1_FUTURE_BUDGET_VALID_JSON_OBJECT') from exc
    if type(value) is not dict:
        _fail('EVIDENCE_JSON_OBJECT')
    return value, hashlib.sha256(raw).hexdigest()


def _sha(value):
    if type(value) is not str or SHA.fullmatch(value) is None:
        _fail('EXACT_EVIDENCE_SHA256')
    return value


def _fresh(observed, now, max_age):
    _number(observed);_number(now);_number(max_age, positive=True)
    if not 0 <= now-observed <= max_age:
        _fail('FRESH_SNAPSHOT_NOT_FUTURE_OR_STALE')


def case_upper(observations, *, floor, multiplier, now_epoch, max_age_s=900):
    """Use original full-duration result/audit/chain bytes, plus explicit margins.

    ``floor`` supplies unmeasured CPU, storage, transfer and constructor uppers.
    A measured maximum is not a statistical bound. Floors/multiplier remain
    owner-declared assumptions and must be revised from later actual evidence.
    """
    upper = _vector(floor)
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or multiplier < 1:
        _fail('EXPLICIT_HEADROOM_MULTIPLIER_AT_LEAST_ONE')
    if (not observations or upper['H200_s'] <= 0 or upper['Core_CPU_s'] <= 0
            or upper['Hopper_CPU_s'] <= 0 or upper['local_CPU_s'] <= 0
            or upper['Hopper_storage_bytes'] <= 0 or upper['Core_storage_bytes'] <= 0
            or upper['local_storage_bytes'] <= 0 or upper['transfer_bytes'] <= 0
            or upper['engine_starts'] < 81):
        _fail('NO_ZERO_OR_UNKNOWN_UNMEASURED_CASE_UPPERS')
    if any(upper[key] for key in ('PBS_submissions', 'optimizer_updates')):
        _fail('CASE_NOT_A_SUBMISSION_OR_TRAINING_UPDATE')
    evidence = [];seen = set();maximum_wall = maximum_cpu = maximum_bytes = 0
    for item in observations:
        _fresh(item['observed_epoch'], now_epoch, max_age_s)
        result, result_sha = _read(item['result'])
        worker, worker_sha = _read(item['worker_receipt'])
        audit, audit_sha = _read(item['audit'])
        chain, chain_sha = _read(item['causal_chain'])
        row = audit.get('adjudication', {})
        identifier = result.get('id')
        if not identifier or identifier in seen:
            _fail('UNIQUE_TERMINAL_CASE_COST_OBSERVATIONS')
        seen.add(identifier)
        if (result.get('fixture') is not False or result.get('status') != 'VALID_COMPLETE'
                or result.get('physics_steps') != 16000
                or result.get('rgb_pairs') != 641 or result.get('error')
                or audit.get('status') != 'PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT'
                or audit.get('attempt_id') != identifier or audit.get('errors') or audit.get('error')
                or audit.get('actual_native_safety_flag_rows') != 0
                or audit.get('model_autonomous_evaluation') is not True
                or not audit.get('checks') or not set(AUDIT_CHECKS) <= set(audit['checks'])
                or any(v is not True for v in audit['checks'].values())
                or audit.get('execution_source_sha') != result.get('source_sha')
                or re.fullmatch('[0-9a-f]{40}', result.get('source_sha', '')) is None
                or row.get('attempt_id') != identifier or row.get('result_sha256') != result_sha
                or row.get('pipeline_trustworthy') is not True
                or row.get('status') not in ('SUCCESS', 'VALID_TASK_FAIL')
                or row.get('main_integrations') != 16000 or row.get('duration_s') != 32
                or row.get('live_native_failures') != 0 or row.get('teacher_takeover') is not False
                or chain.get('status') != 'PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN'
                or chain.get('attempt_id') != identifier or chain.get('draws') != 80
                or chain.get('executed_ACKs') != 80 or chain.get('new_integrations') != 0
                or chain.get('new_model_queries') != 0):
            _fail('FULL_32S_80_ACK_TERMINAL_AUDITED_CASE_ONLY')
        # The audit embeds the adjudication; the chain binds its original bytes.
        adjudication, adjudication_sha = _read(item['adjudication'])
        if adjudication != row or chain.get('adjudication_sha256') != adjudication_sha:
            _fail('ORIGINAL_ADJUDICATION_BYTES_AND_CAUSAL_BINDING')
        for key in ('core_proof_sha256', 'hopper_proof_sha256'):
            _sha(chain.get(key))
        wall = _number(result.get('wall_s'), positive=True)
        cpu = _number(result.get('cpu_s'), positive=True)
        if any(worker.get(key) != result.get(key) for key in
                ('id', 'status', 'source_sha', 'fixture', 'physics_steps', 'rgb_pairs', 'error')):
            _fail('ACTUAL_WORKER_RECEIPT_SAME_ORIGINAL_TERMINAL_CASE')
        worker_wall = _number(worker.get('wall_s'), positive=True)
        worker_cpu = _number(worker.get('cpu_s'), positive=True)
        if worker_wall < wall or worker_cpu < cpu:
            _fail('WORKER_CLOSEOUT_COST_CANNOT_PRECEDE_RESULT')
        wall = worker_wall;cpu = worker_cpu
        size = _number(worker.get('owned_bytes'), integer=True, positive=True)
        if worker.get('output_bytes') != size:
            _fail('ACTUAL_WORKER_OUTPUT_BYTES_AGREE')
        maximum_wall = max(maximum_wall, wall)
        maximum_cpu = max(maximum_cpu, cpu)
        maximum_bytes = max(maximum_bytes, size)
        evidence.append(dict(attempt_id=identifier, result_sha256=result_sha,
            worker_receipt_sha256=worker_sha,
            audit_sha256=audit_sha, causal_chain_sha256=chain_sha,
            adjudication_sha256=adjudication_sha))
    upper['H200_s'] = max(upper['H200_s'], math.ceil(maximum_wall*multiplier))
    upper['Core_CPU_s'] = max(upper['Core_CPU_s'], math.ceil(maximum_cpu*multiplier))
    upper['Core_storage_bytes'] = max(upper['Core_storage_bytes'], math.ceil(maximum_bytes*multiplier))
    upper['episode_starts'] = max(upper['episode_starts'], 1)
    upper['neural_queries'] = max(upper['neural_queries'], 80)
    upper['physics_ticks'] = max(upper['physics_ticks'], 32000)
    return dict(schema='PI0294_PV1_CASE_COST_FORECAST_V1', resources=upper,
        evidence=evidence, measured_maximum_wall_s=maximum_wall,
        measured_maximum_Core_CPU_s=maximum_cpu, multiplier=multiplier,
        declared_floor=_vector(floor), observed_epoch=now_epoch,
        not_statistical_or_native_upper_bound=True)


def _baseline(raw):
    value, digest = _read(raw)
    results = value.get('condition_result_sha256', {})
    audits = value.get('independent_audit_sha256', {})
    if (value.get('schema') != 'PI0294_PV1_B0_ACTUAL_DEV_ADMISSION_V1'
            or value.get('status') != 'PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE'
            or value.get('authority_release') != AUTHORITY
            or value.get('checkpoint_manifest_sha256') != PARENT
            or value.get('population_plan_sha256') != POPULATION or len(results) != 48
            or set(results) != set(audits) or value.get('real_native_faults') != 0
            or value.get('Teacher_takeover') is not False
            or value.get('all_condition_evidence_audits_PASS') is not True
            or value.get('task_success_rate_is_not_training_admission_gate') is not True):
        _fail('COMPLETE_ACTUAL_B0_DEV48_BEFORE_TRAINING_NO_SUCCESS_GATE')
    for digest_value in list(results.values())+list(audits.values()):
        _sha(digest_value)
    _sha(value.get('new_bank_manifest_sha256'))
    return value, digest


def requirements(progress_bytes, *, replica_mode):
    """Derive remaining A/B, optional-pair and conditional replica obligations.

    The sole owner builds this projection ONLY from validated terminal raw
    evidence/checkpoints. Its hash is recorded, not promoted to an independent
    scientific audit. An omitted replica needs a recorded resource-critical
    decision; it caps the possible grade at BETA, never VALIDATED.
    """
    progress, digest = _read(progress_bytes)
    if (progress.get('schema') != 'PI0294_PV1_OWNER_VALIDATED_PROGRESS_V1'
            or progress.get('authority_release') != AUTHORITY
            or progress.get('unknown_effects') != [] or progress.get('native_safety_faults') != 0
            or progress.get('route') not in ARMS):
        _fail('OWNER_VALIDATED_PROGRESS_IDENTITY_AND_EFFECTS')
    if replica_mode not in ('REQUIRED', 'OMITTED_RESOURCE_CRITICAL'):
        _fail('EXPLICIT_REPLICA_SCOPE')
    arms = {arm: 4096 for arm in ARMS[progress['route']]}
    optional = progress['optional_pair']
    if optional not in ('NOT_SELECTED', 'SKIPPED', 'SELECTED'):
        _fail('EXPLICIT_OPTIONAL_PAIR_STATE')
    if optional == 'SELECTED':
        arms.update(C_CONTROL=2048, D_INTERVENTION=2048)
    replica_steps = progress['replica_recipe_updates']
    if replica_steps not in (2048, 4096) or optional != 'SELECTED' and replica_steps != 4096:
        _fail('REPLICA_REPEATS_P2_OR_SELECTED_P3_LOCAL_RECIPE')
    if replica_mode == 'REQUIRED':
        arms['REPLICA'] = replica_steps
    else:
        _sha(progress.get('resource_critical_replica_omission_evidence_sha256'))
    if set(progress['candidates']) != set(arms):
        _fail('NO_MISSING_MANDATORY_OR_EXTRA_CANDIDATE')
    work = []
    for candidate, target in arms.items():
        row = progress['candidates'][candidate]
        step = _number(row['optimizer_updates_committed'], integer=True)
        results = row['valid_DEV_condition_result_sha256']
        if step > target or type(results) is not dict or len(results) > 48:
            _fail('COMMITTED_PROGRESS_WITHIN_FIXED_RECIPE_AND_DEV48')
        if step or results:
            _sha(row['terminal_evidence_sha256'])
        if results and step != target:
            _fail('NO_INTERMEDIATE_CHECKPOINT_DEV_SELECTION')
        for value in results.values():
            _sha(value)
        if step < target:
            work.append(dict(work_id='TRAIN:'+candidate, kind='TRAIN', candidate=candidate,
                optimizer_updates=target-step, TRAIN_probes=12 if target == 4096 and step < 1024 else 0))
        if len(results) < 48:
            work.append(dict(work_id='DEV:'+candidate, kind='DEV', candidate=candidate,
                conditions=48-len(results)))
    return dict(work=work, progress_sha256=digest, campaign_run_uuid=progress['campaign_run_uuid'],
        new_bank_manifest_sha256=_sha(progress.get('new_bank_manifest_sha256')),
        committed_updates=sum(row['optimizer_updates_committed'] for row in progress['candidates'].values()),
        T0_epoch=progress['T0_epoch'], observed_epoch=progress['observed_epoch'],
        replica_mode=replica_mode, maximum_grade='PV1_VALIDATED' if replica_mode == 'REQUIRED' else 'PV1_BETA')


def forecast(*, snapshot, progress_bytes, baseline_admission_bytes, case_cost,
             assumptions, training_jobs, replica_mode, now_epoch, max_age_s=900):
    """Return a deterministic forecast or reject; caller performs actual admission.

    This is a pre-final candidate/training gate, not a replacement for P5's
    remaining-condition accounting. ``training_jobs`` are explicit cold-block
    upper envelopes, including fixed archived single-query TRAIN probes,
    qualification/load/checkpoint/close costs and retained outputs.
    Storage is conservatively added to CURRENT residency (not historical peak).
    A matching work reservation offsets only its reserved resources; an already
    charged submission is credited separately, once, by its unique job ID.
    """
    baseline, baseline_sha = _baseline(baseline_admission_bytes)
    required = requirements(progress_bytes, replica_mode=replica_mode)
    _fresh(snapshot['observed_epoch'], now_epoch, max_age_s)
    _fresh(required['observed_epoch'], now_epoch, max_age_s)
    _fresh(case_cost['observed_epoch'], now_epoch, max_age_s)
    t0 = _number(snapshot['T0_epoch'], integer=True)
    if (snapshot['authority_release'] != AUTHORITY or snapshot['unknown_effects'] != []
            or required['campaign_run_uuid'] != snapshot['campaign_run_uuid']
            or required['new_bank_manifest_sha256'] != baseline['new_bank_manifest_sha256']
            or snapshot['new_bank_manifest_sha256'] != baseline['new_bank_manifest_sha256']
            or required['T0_epoch'] != t0 or not t0 <= now_epoch < t0+144*3600):
        _fail('ORIGINAL_SINGLE_CAMPAIGN_AND_NEW_JOB_CUTOFF')
    clock = dict(stop_new_jobs_epoch=t0+144*3600, stop_training_epoch=t0+156*3600,
        stop_physics_epoch=t0+160*3600, hard_closeout_epoch=t0+168*3600)
    if snapshot['clock'] != clock:
        _fail('IMMUTABLE_ORIGINAL_CLOCK')
    used = _vector(snapshot['used']);reserved = _vector(snapshot['reserved'])
    if used['optimizer_updates'] < required['committed_updates']:
        _fail('COMMITTED_PROGRESS_ALREADY_CHARGED_IN_GLOBAL_USAGE')
    if set(snapshot['storage_peaks']) != {'Hopper_storage_bytes', 'Core_storage_bytes', 'local_storage_bytes'}:
        _fail('ALL_SITE_PEAKS_REQUIRED')
    for key, peak in snapshot['storage_peaks'].items():
        _number(peak, integer=True)
        if not used[key] <= peak <= CAPS[key]:
            _fail('CURRENT_RESIDENCY_AND_HISTORICAL_PEAK')
    case = _vector(case_cost['resources'])
    for key, minimum in dict(episode_starts=1, neural_queries=80, physics_ticks=32000, engine_starts=81).items():
        if case[key] < minimum:
            _fail('FULL_CASE_PHYSICAL_AND_QUERY_UPPER_FLOORS')
    if (case_cost.get('schema') != 'PI0294_PV1_CASE_COST_FORECAST_V1' or not case_cost.get('evidence')
            or case['H200_s'] <= 0 or case_cost.get('not_statistical_or_native_upper_bound') is not True):
        _fail('MEASURED_CASE_FORECAST_NOT_NATIVE_GUARANTEE')
    site_wall = _number(assumptions['site_wall_limit_s'], integer=True, positive=True)
    wall_limit = min(8*3600, site_wall)
    job_overhead = _vector(assumptions['per_job_overhead'])
    if (job_overhead['H200_s'] < 600 or job_overhead['engine_starts'] < 2
            or any(job_overhead[k] for k in ('PBS_submissions', 'optimizer_updates',
                'episode_starts', 'neural_queries', 'physics_ticks'))):
        _fail('EACH_COLD_JOB_600S_QUALIFICATION_PLUS_EXPLICIT_OVERHEAD')
    probe = _vector(assumptions['per_probe_upper'])
    if (probe['H200_s'] <= 0 or probe['Hopper_CPU_s'] <= 0 or probe['neural_queries'] != 1
            or any(probe[k] for k in ('PBS_submissions', 'optimizer_updates', 'episode_starts',
                'physics_ticks', 'engine_starts', 'Core_CPU_s', 'Core_storage_bytes'))):
        _fail('TRAIN_PROBE_ONE_ARCHIVED_WINDOW_DRAW_NO_PHYSICS_OR_EPISODE')
    pad = _number(assumptions['accounting_pad_s'], integer=True)
    if pad < 60:
        _fail('NATIVE_ACCOUNTING_PAD_AT_LEAST_60S')
    queue = _number(assumptions['queue_delay_upper_s_per_job'])
    close_wall = _number(assumptions['package_closeout_wall_s'], positive=True)
    eng = _number(assumptions['DEV_engineering_cases_per_candidate_upper'], integer=True)
    if eng > 48:
        _fail('ONE_ENGINEERING_REPLACEMENT_PER_DEV_CONDITION')
    slots = math.floor((wall_limit-job_overhead['H200_s'])/case['H200_s'])
    if slots < 1:
        _fail('NO_FULL_CASE_FITS_ONE_SITE_LIMITED_ALLOCATION')
    jobs = [];train_by_id = {}
    for job in training_jobs:
        if job['work_id'] in train_by_id:
            _fail('DUPLICATE_TRAINING_WORK_ID')
        train_by_id[job['work_id']] = job
    for item in required['work']:
        if item['kind'] == 'TRAIN':
            supplied = train_by_id.pop(item['work_id'], None)
            if supplied is None:
                _fail('FUTURE_REQUIRED_TRAINING_UPPER_MISSING')
            blocks = supplied['blocks']
            if not blocks:
                _fail('WHOLE_INTEGER_TRAINING_JOBS_REQUIRED')
            updates = probes = 0
            for index, block in enumerate(blocks):
                resources = _vector(block['resources'])
                wall = _number(block['walltime_upper_s'], integer=True, positive=True)
                count = _number(block['optimizer_updates'], integer=True, positive=True)
                probe_count = _number(block['TRAIN_probes'], integer=True)
                if (wall > wall_limit or wall < job_overhead['H200_s']+probe_count*probe['H200_s']
                        or resources['H200_s'] != wall+pad or resources['PBS_submissions'] != 1
                        or resources['optimizer_updates'] != count):
                    _fail('EXACT_TRAINING_BLOCK_UPPER_INCLUDES_QUALIFICATION_PROBES_AND_PAD')
                for key in CAPS:
                    minimum = job_overhead[key]+probe_count*probe[key]
                    if key not in ('H200_s', 'PBS_submissions', 'optimizer_updates') and resources[key] < minimum:
                        _fail('ALL_DIMENSION_TRAINING_PROBE_UPPERS')
                jobs.append(dict(work_id=item['work_id']+':'+str(index+1), kind='TRAIN',
                    walltime_upper_s=wall, resources=resources))
                updates += count;probes += probe_count
            if updates != item['optimizer_updates'] or probes != item['TRAIN_probes']:
                _fail('NO_DROPPED_UPDATES_OR_REQUIRED_FIXED_TRAIN_PROBES')
        else:
            remaining = item['conditions']+eng
            for index in range(math.ceil(remaining/slots)):
                count = min(slots, remaining-index*slots)
                resources = {key:job_overhead[key]+count*case[key] for key in CAPS}
                wall = math.ceil(resources['H200_s'])
                if wall > wall_limit:
                    _fail('ROUNDED_FULL_CASE_JOB_EXCEEDS_SITE_WALL_LIMIT')
                resources['H200_s'] = wall+pad;resources['PBS_submissions'] = 1
                jobs.append(dict(work_id=item['work_id']+':'+str(index+1), kind='DEV',
                    walltime_upper_s=wall, conditions=count, resources=resources))
    if train_by_id:
        _fail('NO_EXTRA_OR_UNSELECTED_TRAINING_WORK')
    # Final confirmation allows 80 regular plus at most 20 diagnosed replacements.
    final_jobs = max(math.ceil(100/slots), math.ceil(FINAL['H200_s']/wall_limit))
    final = {key:100*case[key]+final_jobs*job_overhead[key] for key in CAPS}
    for key, minimum in FINAL.items():
        final[key] = max(final[key], minimum)
    final['PBS_submissions'] = final_jobs
    final['H200_s'] += final_jobs*pad
    jobs.append(dict(work_id='P5_FINAL_AND_CLOSEOUT', kind='FINAL',
        walltime_upper_s=final['H200_s']-final_jobs*pad, resources=final))
    final_extra = _vector(assumptions['package_closeout_resources'])
    if any(final_extra[k] for k in ('H200_s', 'PBS_submissions', 'optimizer_updates', 'neural_queries', 'episode_starts', 'physics_ticks', 'engine_starts')):
        _fail('PACKAGE_CLOSEOUT_NOT_UNDECLARED_SCIENCE')
    for key in CAPS:
        jobs[-1]['resources'][key] += final_extra[key]
    funded = {};reserved_sum = {key:0 for key in CAPS};seen_jobs = set()
    for row in snapshot['work_reservations']:
        key = row['work_id']
        if key in funded or row['job_id'] in seen_jobs:
            _fail('UNIQUE_WORK_AND_NATIVE_JOB_RESERVATION_IDS')
        seen_jobs.add(row['job_id']);resources = _vector(row['resources'])
        if resources['PBS_submissions'] or type(row['submission_already_charged']) is not bool:
            _fail('SUBMISSION_CHARGED_ONCE_SEPARATELY_FROM_FUTURE_RESERVE')
        funded[key] = row
        for dimension in CAPS:
            reserved_sum[dimension] += resources[dimension]
    if reserved_sum != reserved or sum(r['submission_already_charged'] for r in funded.values()) > used['PBS_submissions']:
        _fail('ALL_RESERVATIONS_RECONCILED_NO_HIDDEN_OR_DOUBLE_CREDIT')
    additions = {key:0 for key in CAPS};credited = {key:0 for key in CAPS}
    submission_times = [];training_finish = physics_finish = now_epoch
    reserved_wall = _number(snapshot['reserved_sequential_wall_upper_s'])
    if reserved_wall < reserved['H200_s']+queue*sum(r['resources'].get('H200_s', 0)>0 for r in funded.values()):
        _fail('FULL_RESERVED_CHAIN_WALL_AND_QUEUE_FORECAST')
    cursor = now_epoch+reserved_wall
    for job in jobs:
        row = funded.get(job['work_id']);resources = job['resources']
        if row:
            if row['kind'] != job['kind']:
                _fail('RESERVED_WORK_KIND_CANNOT_BE_REASSIGNED')
            credit = _vector(row['resources'])
            credit['PBS_submissions'] = int(row['submission_already_charged'])
            if any(credit[key] < resources[key] for key in CAPS):
                _fail('MATCHING_RESERVATION_MUST_FULLY_FUND_SAME_REMAINING_WORK')
        else:
            credit = {key:0 for key in CAPS}
            count = resources['PBS_submissions']
            for _ in range(count):
                submission_times.append(cursor)
                cursor += queue+job['walltime_upper_s']/count
        for key in CAPS:
            claimed = min(resources[key], credit[key])
            credited[key] += claimed;additions[key] += resources[key]-claimed
        if job['kind'] == 'TRAIN':
            training_finish = max(training_finish, cursor)
        elif job['kind'] in ('DEV', 'FINAL'):
            physics_finish = max(physics_finish, cursor)
    finish = cursor+close_wall
    over = {key:used[key]+reserved[key]+additions[key]-CAPS[key] for key in CAPS
        if used[key]+reserved[key]+additions[key] > CAPS[key]}
    time_errors = []
    if submission_times and max(submission_times) >= clock['stop_new_jobs_epoch']:
        time_errors.append('LAST_REQUIRED_SUBMISSION_AFTER_144H')
    if training_finish >= clock['stop_training_epoch']:
        time_errors.append('REQUIRED_TRAINING_AFTER_156H')
    if physics_finish >= clock['stop_physics_epoch']:
        time_errors.append('REQUIRED_EVALUATION_AFTER_160H')
    if finish >= clock['hard_closeout_epoch']:
        time_errors.append('PACKAGE_CLOSEOUT_AFTER_168H')
    return dict(schema='PI0294_PV1_FUTURE_REQUIRED_WORK_FORECAST_V1',
        status='PASS_FORECAST_ONLY_NOT_ALLOCATION_OR_GO' if not over and not time_errors else 'STOP_NEW_CANDIDATE_INSUFFICIENT_FORECAST_RUNWAY',
        admissible_forecast=not over and not time_errors, authority_release=AUTHORITY,
        campaign_run_uuid=snapshot['campaign_run_uuid'], observed_epoch=now_epoch,
        baseline_admission_sha256=baseline_sha, progress_sha256=required['progress_sha256'],
        case_cost_evidence=deepcopy(case_cost['evidence']), case_cost_forecast=deepcopy(case_cost),
        replica_mode=replica_mode,
        maximum_possible_grade=required['maximum_grade'], clock=clock,
        used=used, reserved=reserved, future_required_additional=additions,
        matching_reserved_credits=credited, caps=dict(CAPS), final_protection_minimum=dict(FINAL),
        mandatory_jobs=jobs, projected_finish_epoch=finish,
        resource_shortfalls=over, clock_failures=time_errors,
        owner_declared_assumptions=deepcopy(assumptions),
        estimates_are_not_native_allocation_or_timing_guarantees=True,
        site_permissions_quota_peer_lease_and_scientific_guards_still_required=True,
        optional_pair_not_automatically_authorized=True,
        snapshot_cumulative_seconds_storage_current_residency=True)
