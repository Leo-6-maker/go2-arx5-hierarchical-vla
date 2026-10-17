"""Compare saved fixed TRAIN draws only; never generate or score a rollout."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.pi0292.eval_contract import error_metrics, gate_action, summarize_records
from scripts.pi0294.native_dataset import sha

PRECISION = dict(mode='ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST',
                 cuda_matmul_allow_tf32=False, bf16_autocast=True)


def load_batch(folder, manifest, expected_weights, observations):
    folder = Path(folder)
    path = folder / 'ON_EXPERT_RESULTS.json'
    if folder.is_symlink() or path.is_symlink():
        raise PermissionError('NO_SAVED_PROBE_RESULT_SYMLINK')
    raw = path.read_bytes()
    value = json.loads(raw)
    windows = manifest['windows']
    ids = [r['id'] for r in windows]
    if (manifest['TRAIN_only'] is not True or manifest['count'] != len(windows)
            or not 0 < len(ids) <= 96 or len(set(ids)) != len(ids)
            or any(r['split'] != 'TRAIN' for r in windows)
            or value['complete'] is not True
            or value['status'] != 'ACTUAL_NATIVE_TRAIN_PROBES_NOT_TASK_SUCCESS'):
        raise PermissionError('COMPLETE_FIXED_TRAIN_PROBES_ONLY')
    records = value['records']
    by_id = {r['id']: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(ids) or set(observations) != set(ids):
        raise ValueError('EXACT_UNIQUE_FIXED_PROBE_MEMBERSHIP')
    rows, signatures = [], {}
    for window in windows:
        identifier = window['id']
        if Path(identifier).name != identifier or identifier in ['.', '..']:
            raise PermissionError('EXACT_PROBE_BASENAME')
        record = by_id[identifier]
        if any(record[k] != window[k] for k in ['episode_id', 'task', 'anchor', 'split']):
            raise ValueError('FIXED_PROBE_WINDOW_IDENTITY')
        precision = record['inference_precision']
        if (any(precision.get(k) != v for k, v in PRECISION.items())
                or record['seed'] != window['noise_seed']):
            raise ValueError('FIXED_PROBE_SEED_AND_EFFECTIVE_PRECISION')
        arrays = folder / (identifier + '.npz')
        if (arrays.is_symlink() or not arrays.resolve().is_relative_to(folder.resolve())
                or sha(arrays) != record['arrays_sha256']):
            raise ValueError('SAVED_PROBE_ARRAY_HASH_OR_PATH')
        with np.load(arrays, allow_pickle=False) as saved:
            required = {'physical_H40', 'expert_absolute10', 'state', 'noise_seed',
                        'noise_fingerprint_json', 'inference_precision_json',
                        'observation_fingerprint', 'checkpoint_files_fingerprint'}
            if not required.issubset(saved.files):
                raise ValueError('SAVED_PROBE_INPUT_FINGERPRINT_FIELDS_NOT_RECORDED')
            prediction, expert, state = [saved[k].copy() for k in ['physical_H40', 'expert_absolute10', 'state']]
            noise = json.loads(saved['noise_fingerprint_json'].item())
            stored_precision = json.loads(saved['inference_precision_json'].item())
            observation = saved['observation_fingerprint'].item()
            weights = saved['checkpoint_files_fingerprint'].item()
            seed = int(saved['noise_seed'].item())
        if (prediction.shape != (40, 10) or expert.shape != (40, 10)
                or state.shape != (10,) or not np.isfinite(expert).all()
                or not np.isfinite(state).all()):
            raise ValueError('FIXED_EXPERT_STATE_AND_ACTION_SHAPES')
        if (hashlib.sha256(state.tobytes()).hexdigest() != window['state_sha256']
                or hashlib.sha256(expert.tobytes()).hexdigest() != window['absolute_H40_sha256']
                or observation != observations[identifier] or weights != expected_weights
                or seed != window['noise_seed'] or noise != record['noise']
                or stored_precision != precision):
            raise ValueError('SAVED_PROBE_INPUT_NOISE_WEIGHT_OR_METADATA_IDENTITY')
        errors = error_metrics(prediction, expert, state)
        if errors != record['errors']:
            raise ValueError('INDEPENDENT_PROBE_METRICS_DO_NOT_RECOMPUTE')
        if gate_action(prediction, state, window['anchor'] * .05, identifier) != record['gate']:
            raise ValueError('INDEPENDENT_TRAJECTORY_GATE_DOES_NOT_RECOMPUTE')
        # Group by true episode, not by the many overlapping probe-window IDs.
        rows.append(dict(record, id=record['episode_id'], probe_id=identifier,
                         stage=window['stage'], errors=errors))
        signatures[identifier] = dict(observation=observation, noise=noise, seed=seed)
        if sha(arrays) != record['arrays_sha256']:
            raise ValueError('SAVED_PROBE_ARRAY_CHANGED_DURING_READ')
    if path.read_bytes() != raw:
        raise ValueError('SAVED_PROBE_RESULT_CHANGED_DURING_READ')
    return dict(rows=rows, signatures=signatures, result_sha256=hashlib.sha256(raw).hexdigest())


def summarize_batch(batch):
    first = {r['probe_id']: r for r in batch['rows']}
    if not first or len(first) != len(batch['rows']) or set(first) != set(batch['signatures']):
        raise ValueError('EXACT_UNIQUE_PROBE_SUMMARY_MEMBERSHIP')
    groups = {'ALL': list(first)}
    for task in sorted({r['task'] for r in first.values()}):
        groups[task] = [i for i, r in first.items() if r['task'] == task]
        for stage in sorted({first[i]['stage'] for i in groups[task]}):
            groups[task + ':' + stage] = [i for i in groups[task] if first[i]['stage'] == stage]
    return {name:summarize_records([first[i] for i in identifiers])['TRAIN']
            for name, identifiers in groups.items()}


def compare_batches(parent, candidate):
    if parent['signatures'] != candidate['signatures']:
        raise ValueError('PAIRED_PROBE_OBSERVATION_SEED_OR_NOISE_CHANGED')
    first = {r['probe_id']: r for r in parent['rows']}
    second = {r['probe_id']: r for r in candidate['rows']}
    if (not first or len(first) != len(parent['rows']) or len(second) != len(candidate['rows'])
            or set(first) != set(second) or set(first) != set(parent['signatures'])):
        raise ValueError('EXACT_PAIRED_PROBE_MEMBERSHIP')
    for identifier in first:
        if any(first[identifier][k] != second[identifier][k]
               for k in ['episode_id', 'task', 'anchor', 'split', 'stage']):
            raise ValueError('PAIRED_PROBE_CONTEXT_CHANGED')
    a, b = summarize_batch(parent), summarize_batch(candidate)
    summaries = {name:dict(parent=a[name], candidate=b[name]) for name in a}
    return dict(status='PASS_PAIRED_SAVED_TRAIN_PROBES_IDENTICAL_INPUTS_NOISE_AND_RECOMPUTED_METRICS',
        windows=len(first), distinct_episodes=len({r['episode_id'] for r in first.values()}),
        task_windows=dict(Counter(r['task'] for r in first.values())), summaries=summaries,
        parent_result_sha256=parent['result_sha256'], candidate_result_sha256=candidate['result_sha256'],
        metric_macro='MEAN_OF_WITHIN_TRUE_EPISODE_POOLED_WINDOW_POINT_STATISTICS',
        gate='TRAJECTORY_GATE_ONLY_NOT_ORIGINAL_S1_COLLISION_SHADOW',
        rejected_or_nonfinite_predictions_excluded=False,
        overlapping_windows_are_independent_episodes=False,
        stored_TF32_training_entry_flag_is_effective_inference_precision=False,
        closed_loop_or_generalization_or_causal_correction_gain_proven=False,
        new_model_queries=0, new_physics=0)
