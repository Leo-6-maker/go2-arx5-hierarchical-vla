"""Evidence-based endpoint decisions for the fixed continuous-task generation.

This module does not train, select a subset of DEV, submit jobs, or grant a
physical scope. The existing owner supplies audited terminal evidence. The
decision is recomputed from the evidence when a later endpoint is admitted.
"""
import json
import math
from pathlib import Path

import numpy as np

from .storage import digest
from .task_learning_grasp_v1 import need
from .task_learning_probes_v1 import PHASES, contract as probe_contract, artifact, matched_count

PLAN = '7dbf07edb8104696a6c9974f9c18f4313cab0fcc'
SCHEMA = 'TASK_LEARNING_ENDPOINT_SCIENTIFIC_CONTRACT_V2'
CHANNELS = ('XYZ_m', 'SO3_rad', 'gripper')
KEY_PHASES = ('CLOSE', 'LIFT', 'HOLD')


def contract():
    """Freeze before any formal endpoint result is read; no fitted thresholds."""
    return dict(schema=SCHEMA, plan_sha=PLAN, endpoints=[2000, 4000, 8000],
        probe_selection=probe_contract(), key_phases=list(KEY_PHASES),
        aggregation='EQUAL_WEIGHT_REAL_PROBE_ROWS_WITHIN_EACH_PHASE_THEN_EQUAL_WEIGHT_KEY_PHASES',
        channels=list(CHANNELS), horizons=['H40', 'first'],
        first2000_open_loop_baseline='SAME_GENERATION_STEP0_TARGET_PROCESSOR_BEFORE_AND_BEFORE_DEV',
        first2000_native_baseline='COMMON_FULL_PANEL_ORIGINAL_B0_ALL40',
        later_baseline='PREVIOUS_FULLY_EVALUATED_SAME_GENERATION_B1',
        current_complete_common_DEV_required=80, per_method_requested=40,
        missing_or_nonfinite_metric='TECHNICAL_DIAGNOSIS_NO_SCIENTIFIC_PROMOTION',
        maximum_absolute_channel_degradation=dict(XYZ_m=.0001, SO3_rad=.001, gripper=.001),
        XYZ_relative_improvement=.10, XYZ_absolute_improvement_floor_m=.0001,
        TRAIN_XYZ_fit_ceiling_m=.001,
        allowed_progress_OR=[
            'COMPLETE_NATIVE_SUCCESS_COUNT_GAIN_AT_LEAST4_OF40',
            'KEY_DEV_H40_XYZ_IMPROVES10PERCENT_WITH_0P1MM_FLOOR_AND_LEGAL_HOLD_COUNT_GAIN_AT_LEAST4_OF40'],
        legal_hold_progress=dict(source='ORIGINAL_TASK_SCORE',minimum_contiguous_grasp_lift_hold_s=.1,
            minimum_lift_m=.020,standing_passes=True,unsafe_samples=0),
        shared_guards=['COMPLETE_COMMON_DEV', 'HARD_SAFETY0', 'TECHNICAL_FAILURE0',
            'TRAIN_KEY_XYZ_FITS_OR_IMPROVES', 'NO_KEY_PHASE_CHANNEL_DEGRADATION_H40_OR_FIRST',
            'NATIVE_SUCCESS_COUNT_NOT_LOWER'],
        later_B0_success_count_absolute_drift_allowance=4,
        local_confirmation_threshold_successes=32,
        endpoints_after8000=[], partial_action='CONTINUE_UNEXECUTED_OR_REPAIR_ONLY',
        overlapping_future_preview_consistency='NOT_APPLICABLE_FULL_H40_EXECUTED_WITHOUT_OVERLAP',
        statistical_significance_claim=False)


def descriptor(path):
    path = Path(path)
    return dict(path=str(path), sha256=digest(path))


def load(pin):
    path = Path(pin['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and digest(path) == pin['sha256'], 'PINNED_ENDPOINT_EVIDENCE')
    return json.loads(path.read_bytes())


def probe_summary(folder, *, split, global_step, generation_id, dataset_sha256, statistics_sha256):
    """Recompute errors from actual retained H40 arrays, never a fitted metric."""
    from scripts.pi0292.eval_contract import error_metrics
    from scripts.pi0292.relay_protocol import digest as object_hash
    folder = Path(folder)
    result = json.loads((folder/'RESULT.json').read_bytes())
    need(result.get('selection_contract') == probe_contract(), 'DECLARED_REAL_PHASE_PROBES')
    matched_count(result, result)
    rows, values = result['records'], []
    for index,row in enumerate(rows):
        identity = row['model_identity']
        need(row['split'] == split and row['model_global_step'] == global_step
             and identity.get('global_step') == global_step
             and identity.get('generation_id') == generation_id
             and identity.get('dataset_sha256') == dataset_sha256
             and identity.get('statistics_sha256') == statistics_sha256
             and row['Processor_decode_calls'] == 1 and row['seed']==2026110900+index,
             'SAME_GENERATION_PHASE_PROBE_IDENTITIES')
        path = artifact(folder, row)
        need(digest(path) == row['sha256'], 'ACTUAL_RETAINED_PROBE_ARRAY_HASH')
        with np.load(path, allow_pickle=False) as data:
            prediction, expert, state = (data[k] for k in ('physical_H40', 'expert_policy_H40', 'actual_state10'))
            need(prediction.shape == expert.shape == (40, 10) and state.shape == (10,)
                 and np.isfinite(prediction).all() and np.isfinite(expert).all()
                 and np.isfinite(state).all() and int(data['noise_seed']) == row['seed'],
                 'FINITE_REAL_PROBE_ARRAYS_AND_NOISE_SEED')
            error = error_metrics(prediction, expert, state)
        pointwise = error['pointwise']
        entry = dict(phase=row['phase_audit_only'])
        for horizon, indices in (('H40', slice(None)), ('first', slice(0, 1))):
            entry[horizon] = {}
            for channel, field in zip(CHANNELS, ('XYZ_euclidean_m', 'SO3_rad', 'gripper_absolute')):
                vector = pointwise[field]
                need(len(vector) == 40 and all(type(v) in (float, int) and math.isfinite(v)
                    and v >= 0 for v in vector), 'FINITE_PHASE_ERROR_' + channel)
                entry[horizon][channel] = float(np.mean(vector[indices]))
        values.append(entry)
    phases = {phase: {h: {ch: float(np.mean([r[h][ch] for r in values if r['phase'] == phase]))
                for ch in CHANNELS} for h in ('H40', 'first')} for phase in PHASES}
    key = {h: {ch: float(np.mean([phases[p][h][ch] for p in KEY_PHASES]))
                  for ch in CHANNELS} for h in ('H40', 'first')}
    matched = [{k: r[k] for k in ('episode', 'anchor', 'seed', 'noise', 'phase_audit_only',
                                 'genuine_recovery_audit_only')} for r in rows]
    return dict(split=split, global_step=global_step, generation_id=generation_id,
        dataset_sha256=dataset_sha256, statistics_sha256=statistics_sha256,
        actual_unique_probes=len(rows), matched_observation_noise_sha256=object_hash(matched),
        result=descriptor(folder/'RESULT.json'), artifacts=[dict(name=artifact(folder,r).name,sha256=r['sha256'])for r in rows],
        phases=phases, key=key)


def native_counts(panel, outcomes):
    from .task_learning_student_v1 import summarize
    summary = summarize(panel, outcomes)
    counts = {}
    for method in ('B0', 'B1'):
        rows = [v for v in outcomes if v['method'] == method]
        legal = 0
        for row in rows:
            score = row.get('score') or {}
            legal += bool(row['technical_valid'] is True and row['reset_ready'] is True
                and not row['hard_safety_stop'] and score.get('standing_passes') is True
                and score.get('unsafe_samples') == 0
                and score.get('maximum_object_lift_m', 0) >= .020
                and score.get('longest_contiguous_grasp_lift_hold_s', 0) >= .1-1e-9)
        counts[method] = dict(summary['methods'][method], legal_hold_progress_count=legal)
    return dict(complete=summary['complete'], methods=counts)


def evaluate(evidence):
    """Pure scientific rule. A partial panel can never authorize continuation."""
    c = contract()
    need(evidence.get('contract') == c, 'FROZEN_ENDPOINT_DECISION_CONTRACT')
    step = evidence.get('global_step')
    need(type(step) is int and step in c['endpoints'], 'DECLARED_FORMAL_ENDPOINT')
    need(evidence['native_panel']['B1_global_step']==step
        and evidence['native_panel']['policies']['B1']['package']['sha256']==evidence['package_sha256'],
        'NATIVE_PANEL_BOUND_TO_THE_ACTUAL_CANDIDATE_ENDPOINT_PACKAGE')
    current = native_counts(evidence['native_panel'], evidence['native_outcomes'])
    if not current['complete']:
        return dict(action=c['partial_action'], allows_execution=False,
                    reason='COMMON_DEV_PARTIAL_OR_NOT_RUN', next_endpoint=None)
    need(all(current['methods'][m]['requested'] == current['methods'][m]['attempted'] == 40
             and current['methods'][m]['unrun'] == 0 for m in ('B0', 'B1')),
         'ALL_EIGHTY_ACTUAL_DEV_REQUESTS_NOT_READY_SUBSET')
    before, after = evidence['open_before'], evidence['open_after']
    for split in ('TRAIN', 'DEV'):
        a, b = before[split], after[split]
        expected_step = 0 if step == 2000 else (2000 if step == 4000 else 4000)
        need(a['global_step'] == expected_step and b['global_step'] == step
            and a['actual_unique_probes'] == b['actual_unique_probes'] == (40 if split == 'TRAIN' else 20)
            and a['matched_observation_noise_sha256'] == b['matched_observation_noise_sha256']
            and all(a[k] == b[k] == evidence[k] for k in
                ('generation_id', 'dataset_sha256', 'statistics_sha256')), 'MATCHED_FORMAL_ENDPOINT_PROBE_PAIR')
        for side in (a,b):
            need(all(type(side['phases'][phase][h][ch]) in (int,float)
                and math.isfinite(side['phases'][phase][h][ch]) and side['phases'][phase][h][ch]>=0
                for phase in PHASES for h in ('H40','first') for ch in CHANNELS),
                'FINITE_ALL_PHASE_CHANNEL_ERRORS')
            expected_key={h:{ch:float(np.mean([side['phases'][p][h][ch] for p in KEY_PHASES]))
                for ch in CHANNELS}for h in ('H40','first')}
            need(side['key']==expected_key,'KEY_CHANNELS_RECOMPUTED_FROM_FIXED_PHASES')
    baseline = current['methods']['B0']
    drift = True
    if step > 2000:
        need(evidence['previous_native_panel']['B1_global_step']==expected_step,
             'PREVIOUS_NATIVE_PANEL_EXACT_PRECEDING_ENDPOINT')
        previous = native_counts(evidence['previous_native_panel'],evidence['previous_native_outcomes'])
        need(previous['complete'] is True, 'PREVIOUS_COMPLETE_CANDIDATE_DEV_NOT_PARTIAL')
        baseline = previous['methods']['B1']
        drift = abs(current['methods']['B0']['task_successes']-previous['methods']['B0']['task_successes']) <= 4
    candidate = current['methods']['B1']
    tolerances = c['maximum_absolute_channel_degradation']
    channels = all(after['DEV']['phases'][phase][h][ch] <=
        before['DEV']['phases'][phase][h][ch] + tolerances[ch]
        for phase in KEY_PHASES for h in ('H40', 'first') for ch in CHANNELS)
    train_before = before['TRAIN']['key']['H40']['XYZ_m']
    train_after = after['TRAIN']['key']['H40']['XYZ_m']
    train_fit = train_after <= .001 or train_before-train_after >= max(.1*train_before, .0001)
    native_gain = candidate['task_successes']-baseline['task_successes']
    legal_gain = candidate['legal_hold_progress_count']-baseline['legal_hold_progress_count']
    dev_before = before['DEV']['key']['H40']['XYZ_m']
    dev_after = after['DEV']['key']['H40']['XYZ_m']
    dev_improves = dev_before-dev_after >= max(.1*dev_before, .0001)
    safe = all(current['methods'][m]['hard_safety_stop_conditions'] ==
        current['methods'][m]['technical_failure_conditions'] == 0 for m in ('B0', 'B1'))
    guards = dict(complete_common_DEV=True, safety_and_technical=safe, TRAIN_fits_or_improves=train_fit,
        key_DEV_channels=channels, native_success_not_lower=native_gain >= 0, common_B0_drift=drift)
    progress = dict(native_success_gain4=native_gain >= 4,
        DEV_XYZ_and_legal_hold_gain4=dev_improves and legal_gain >= 4)
    allowed = all(guards.values()) and any(progress.values())
    action, next_endpoint = 'NEW_COVERAGE_OR_CAUSAL_OBSERVATION_DIAGNOSIS', None
    if not safe:
        action = 'PRESERVE_SAFETY_OR_TECHNICAL_BLOCKER_AND_REPAIR'
    elif not drift:
        action = 'COMMON_BASELINE_DRIFT_DIAGNOSIS_NO_PROMOTION'
    elif not train_fit:
        action = 'TRAIN_LABEL_PROCESSOR_GRADIENT_OR_EXPOSURE_DIAGNOSIS'
    elif all(guards.values()) and candidate['task_successes'] >= 32 and native_gain >= 4:
        action = 'FREEZE_FOR_SEED_CONFIRMATION_BEFORE_TEST'
    elif allowed and step < 8000:
        action, next_endpoint = 'CONTINUE_SAME_GENERATION', (4000 if step == 2000 else 8000)
    return dict(action=action, allows_execution=next_endpoint is not None, next_endpoint=next_endpoint,
        guards=guards, progress_OR=progress, native_success_gain=native_gain, legal_hold_count_gain=legal_gain,
        baseline_kind='COMMON_B0' if step == 2000 else 'PREVIOUS_COMPLETE_B1',
        independent_statistical_significance_established=False)


def validate_continuation(decision, spec):
    """A free-standing allows_execution boolean cannot admit a longer job."""
    from .task_learning_training_profile_v1 import descriptor as is_descriptor
    need(is_descriptor(decision.get('scientific_evidence'))and is_descriptor(decision.get('decision_contract')),
         'ACTUAL_SCIENTIFIC_EVIDENCE_AND_FROZEN_RULE_NOT_BOOLEAN_PERMISSION')
    evidence = load(decision['scientific_evidence'])
    frozen = load(decision['decision_contract'])
    need(frozen == contract() == evidence.get('contract'), 'PINNED_PREDECLARED_SCIENTIFIC_RULE')
    result = evaluate(evidence)
    need(decision.get('scientific_result') == result and result['allows_execution'] is True
        and decision.get('action') == result['action'] == 'CONTINUE_SAME_GENERATION'
        and result['next_endpoint'] == decision.get('next_endpoint') == spec['recipe']['endpoint']
        and evidence['global_step'] == spec['start_step']
        and all(decision.get(k) == evidence[k] == spec[k] for k in ('generation_id',))
        and decision.get('dataset_sha256') == evidence['dataset_sha256'] == spec['recipe']['dataset_sha256']
        and evidence['statistics_sha256'] == spec['recipe']['statistics_sha256']
        and decision.get('checkpoint_sha256') == evidence['checkpoint_sha256'] == spec['resume']['sha256'],
        'RECOMPUTED_COMPLETE_ENDPOINT_DECISION_AND_EXACT_RESUME_CHECKPOINT')
    return result


def validate_confirmation(decision,spec):
    """Confirmation seeds need the first seed's complete scientific evidence."""
    from .task_learning_training_profile_v1 import descriptor as is_descriptor
    need(is_descriptor(decision.get('scientific_evidence'))and is_descriptor(decision.get('decision_contract')),
        'ACTUAL_FIRST_SEED_EVIDENCE_NOT_BOOLEAN_PERMISSION')
    evidence=load(decision['scientific_evidence']);frozen=load(decision['decision_contract'])
    result=evaluate(evidence)
    need(frozen==contract()==evidence['contract']and evidence.get('training_seed')==20261109
        and result['action']=='FREEZE_FOR_SEED_CONFIRMATION_BEFORE_TEST'
        and decision.get('scientific_result')==result and decision.get('first_seed')==20261109
        and decision.get('first_seed_generation_id')==evidence['generation_id']
        and decision.get('dataset_sha256')==evidence['dataset_sha256']==spec['recipe']['dataset_sha256']
        and evidence['statistics_sha256']==spec['recipe']['statistics_sha256']
        and decision.get('allowed_confirmation_seeds')==[20261110,20261111]
        and spec['recipe']['seed']in decision['allowed_confirmation_seeds'],
        'COMPLETE_FIRST_SEED_GAIN_BEFORE_DECLARED_CONFIRMATION_SEEDS')
    return result
