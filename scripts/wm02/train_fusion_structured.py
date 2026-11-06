"""Fit a finite NumPy structured-response study; freeze both arms before DEV."""

import argparse
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np

from go2arx5_vla.world_model.fusion_structured import (
    ARMS, RIDGE_GRID, StructuredResponse, validate_cohort,
)
from go2arx5_vla.world_model.storage import digest, write_json


PROFILE = 'wm02-fusion-structured-offline-cpu'
KIND = 'FUSION_STRUCTURED_RESPONSE'
FROZEN_TRAIN_MIN_GAIN_MM = .08124448952310876
ORIGINAL_TASK_TOLERANCE_M = .00018584456642062617


def record_status(output, progress=None, completed=None):
    """Observer snapshots replace atomically; sealed artifacts remain exclusive."""
    from scripts.pi0292.durable import write
    if progress is not None:
        write(Path(output) / 'PROGRESS.private.json', progress)
    if completed is not None:
        write(Path(output) / 'COMPLETED.private.json', completed)


def _npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _finite_scope(spec):
    if (time.time() >= spec['deadline_epoch']
            or (Path(spec['registered_root']) / 'STOP.json').exists()):
        raise TimeoutError('FINITE_STRUCTURED_ENDPOINT_OR_STOP')


def _validate_inputs(spec):
    """Authenticate the existing replay seal, not raw simulator evidence again."""
    expected = dict(profile=PROFILE, kind=KIND, device='cpu', physics=False,
                    numeric_mode='NUMPY_FLOAT64', arms=list(ARMS),
                    min_lopo_gain_mm=FROZEN_TRAIN_MIN_GAIN_MM,
                    GR00T_updates=0, WBC_changes=False)
    if (any(spec.get(name) != value for name, value in expected.items())
            or spec.get('physics') is not False or spec.get('WBC_changes') is not False
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('FROZEN_NUMPY_CPU_TWO_ARM_TRAIN_ONLY_RECIPE_REQUIRED')
    for key in ('replay', 'teacher_metadata', 'training_replay_result',
                'task', 'train_mean_response_model'):
        if digest(spec[key]) != spec[key + '_sha256']:
            raise ValueError('SEALED_STRUCTURED_INPUT_CHANGED:' + key)
    report_path = Path(spec['training_replay_result'])
    report = json.loads(report_path.read_text())
    lease = json.loads((report_path.parent.parent / 'control/LEASE.json').read_text())
    if (lease.get('status') != 'CLOSED'
            or report.get('schema') != 'WM02_FUSION_REAL_RESPONSE_REPLAY_V1'
            or report.get('eligible_windows') != 48
            or report.get('real_post_choice_transitions') != 768
            or Counter(report.get('physical_parents', {}).values()) != dict(TRAIN=4, DEV=2)
            or report.get('excluded') != [] or report.get('TEST_read') is not False
            or report.get('expert_action_labels') is not False
            or report.get('replay_sha256') != spec['replay_sha256']):
        raise PermissionError('ACTUAL_COMPLETE_SEALED_BALANCED_RESPONSE_REPLAY_REQUIRED')
    with np.load(spec['replay'], allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    meta = json.loads(Path(spec['teacher_metadata']).read_text())
    splits = validate_cohort(data, meta)
    physical = dict(zip(map(str, data['parent']), map(str, data['split'])))
    identities = report.get('identities', [])
    observed = {(str(data['run_id'][i]), str(data['parent'][i]), str(data['split'][i]))
                for i in range(48)}
    recorded = {(row['run'], row['parent'], row['split']) for row in identities}
    if (physical != report['physical_parents'] or len(identities) != 48
            or len(recorded) != 48 or recorded != observed):
        raise ValueError('REPLAY_REPORT_RUN_PARENT_SPLIT_IDENTITIES_DIFFER')

    # Recompute only TRAIN summaries to bind the frozen threshold source to
    # exactly this replay. No DEV future labels are formed here.
    from go2arx5_vla.world_model.pilot2_mean_rule import fit as fit_rule, SCHEMA
    from go2arx5_vla.world_model.pilot2_evaluation import minimum_practical_gain
    task = json.loads(Path(spec['task']).read_text())
    rule = json.loads(Path(spec['train_mean_response_model']).read_text())
    train = splits['TRAIN']
    parents = sorted(set(map(str, data['parent'][train])))
    if (rule.get('schema') != SCHEMA or rule.get('fit_split') != 'TRAIN'
            or rule.get('task_sha256') != spec['task_sha256']
            or rule.get('episodes') != 32 or rule.get('restarts_per_action') != 2
            or set(rule.get('physical_parents', [])) != set(parents)
            or rule.get('exact_fork_oracle') is not False
            or rule.get('rule_fit_changes_task_or_score') is not False
            or task.get('success_tolerance_m') != ORIGINAL_TASK_TOLERANCE_M):
        raise ValueError('SAME_FROZEN_TRAIN_RULE_AND_ORIGINAL_TASK_REQUIRED')
    records = [dict(parent=str(data['parent'][i]), split='TRAIN',
                    run=str(data['run_id'][i]), action=int(data['observed_action'][i]),
                    repeat=int(data['repeat_index'][i]),
                    response_xy_m=(data['state'][i, -1, 86:88]
                                   - data['state'][i, 8, 86:88]).tolist()) for i in train]
    reconstructed = fit_rule(records, rule['physical_parents'], spec['task_sha256'])
    for name in ('mean_response_xy_m', 'same_action_restart_gap_m'):
        if not np.allclose(rule[name], reconstructed[name], atol=1e-12, rtol=0):
            raise ValueError('FROZEN_TRAIN_RULE_DIFFERS_FROM_REPLAY:' + name)
    if (set(rule.get('parent_mean_response_xy_m', {})) != set(parents)
            or any(not np.allclose(rule['parent_mean_response_xy_m'][parent],
                                   reconstructed['parent_mean_response_xy_m'][parent],
                                   atol=1e-12, rtol=0) for parent in parents)):
        raise ValueError('FROZEN_TRAIN_PARENT_RESPONSES_DIFFER_FROM_REPLAY')
    gain = minimum_practical_gain(rule, task)
    if not np.isclose(gain['minimum_practical_improvement_m'] * 1000.,
                      FROZEN_TRAIN_MIN_GAIN_MM, atol=1e-12, rtol=0):
        raise ValueError('FROZEN_POSITIVE_TRAIN_GAIN_SOURCE_DIFFERS')
    return data, meta, splits, gain


def _truth(data, ids):
    """Explicit staging seam: DEV rows may be passed only after both-arm freeze."""
    result = np.asarray(data['state'][ids, 9:, 86:89], float)
    if result.shape != (len(ids), 16, 3) or not np.isfinite(result).all():
        raise ValueError('FINITE_OBSERVED_H16_TCP_EVALUATION_TARGETS_REQUIRED')
    return result


def _metrics(prediction, truth):
    error = (np.asarray(prediction) - np.asarray(truth)) * 1000.
    if error.shape != truth.shape or not np.isfinite(error).all():
        raise ValueError('FINITE_MATCHED_TCP_PREDICTION_AND_TRUTH_REQUIRED')
    return dict(all_H16_XYZ_RMSE_mm=float(np.sqrt(np.mean(error ** 2))),
                all_H16_XY_RMSE_mm=float(np.sqrt(np.mean(error[..., :2] ** 2))),
                horizons={str(h): dict(
                    XYZ_RMSE_mm=float(np.sqrt(np.mean(error[:, h-1] ** 2))),
                    XY_RMSE_mm=float(np.sqrt(np.mean(error[:, h-1, :2] ** 2))),
                    max_XYZ_error_mm=float(np.max(np.linalg.norm(error[:, h-1], axis=-1))))
                    for h in (1, 4, 8, 16)})


def _evaluation(prediction, truth, data, ids):
    parents, actions = data['parent'][ids], data['observed_action'][ids]
    return dict(pooled=_metrics(prediction, truth),
                by_parent={str(parent): _metrics(prediction[parents == parent], truth[parents == parent])
                           for parent in sorted(set(parents))},
                by_action={str(action): _metrics(prediction[actions == action], truth[actions == action])
                           for action in range(4)})


def _effect(prediction, hold_prediction, data, ids):
    pairs, prefix = [], []
    for local, row in enumerate(ids):
        if data['observed_action'][row] == 0:
            continue
        holds = ids[(data['parent'][ids] == data['parent'][row])
                    & (data['observed_action'][ids] == 0)]
        actual = (data['state'][row, 9:, 86:89] - data['state'][row, 8:9, 86:89]
                  - (data['state'][holds, 9:, 86:89] - data['state'][holds, 8:9, 86:89]).mean(0))
        pairs.append(prediction[local] - hold_prediction[local] - actual)
        prefix.extend(np.linalg.norm(data['state'][holds, 8, 86:89]
                                     - data['state'][row, 8, 86:89], axis=-1))
    error = np.asarray(pairs) * 1000.
    return dict(independent_restart_pairs=len(pairs), exact_forks=False,
                prefix_mismatch_includes_restart_noise=True,
                max_prefix_TCP_mismatch_mm=float(np.max(prefix) * 1000.),
                all_H16_XYZ_action_delta_RMSE_mm=float(np.sqrt(np.mean(error ** 2))),
                horizons={str(h): dict(XYZ_action_delta_RMSE_mm=float(np.sqrt(np.mean(error[:, h-1] ** 2))))
                          for h in (1, 4, 8, 16)})


def _verify_reload(original, restored, data, train):
    before, after = original.arrays(), restored.arrays()
    if before.keys() != after.keys() or any(not np.array_equal(value, after[name])
                                           for name, value in before.items()):
        raise ValueError('EXACT_STRUCTURED_ARRAY_SERIALIZATION_DIFFERS')
    prediction = original.candidate_forecasts(data, train)
    reloaded = restored.candidate_forecasts(data, train)
    error = float(np.max(np.abs(prediction - reloaded)))
    if error != 0.:
        raise ValueError('EXACT_TRAIN_FOUR_CANDIDATE_RELOAD_DIFFERS')
    return dict(array_serialization_exact=True, TRAIN_candidate_reload_max_error_m=error,
                optimizer_restore='NOT_APPLICABLE_CLOSED_FORM_NO_OPTIMIZER')


def run(spec, output):
    output = Path(output)
    start, cpu = time.monotonic(), time.process_time()
    _finite_scope(spec)
    data, meta, splits, gain_source = _validate_inputs(spec)
    train = splits['TRAIN']
    record_status(output, progress=dict(stage='VALIDATED_SEALED_TRAIN_INPUT', DEV_evaluated=False))
    models, fitted = {}, []
    for arm in ARMS:
        _finite_scope(spec)
        directory = output / arm
        directory.mkdir(exist_ok=False)
        model = StructuredResponse(meta, arm).fit(data, min_lopo_gain_mm=FROZEN_TRAIN_MIN_GAIN_MM)
        path = directory / 'STRUCTURED_NUMPY.npz'
        _npz(path, **model.arrays())
        restored = StructuredResponse.load(path)
        restore = _verify_reload(model, restored, data, train)
        models[arm] = restored
        folds = len(restored.selection_report['folds'])
        solves = folds * sum(np.isfinite(ridge) for ridge in RIDGE_GRID) + int(np.isfinite(restored.ridge))
        record = dict(arm=arm, model_path=str(path), numpy_sha256=digest(path),
                      selected_ridge=str(restored.ridge), optimizer_updates=0,
                      completed_model_fits=1, completed_LOPO_parent_fits=folds,
                      completed_static_curve_fits=folds+1,
                      completed_normalizer_fits=folds+1,
                      completed_closed_form_ridge_solves=int(solves),
                      solve_accounting='COMPLETED_LOPO_FOLDS_X_FIVE_FINITE_RIDGES_PLUS_FINAL_IF_FINITE',
                      restore=restore, selection=restored.selection_report,
                      normalization_split='OWN_ARM_TRAIN_ONLY',
                      static_base=('TRAIN_HOLD_PLUS_ACTION_MINUS_HOLD' if arm == ARMS[0]
                                   else 'EQUAL_ACTION_TRAIN_MARGINAL'),
                      candidate_removed_from_features_and_base=arm == ARMS[1])
        fitted.append(record)
        write_json(directory / 'TRAIN_FIT.private.json', record)
        record_status(output, progress=dict(stage='ARM_TRAIN_FIT_RELOADED', arm=arm,
                                           DEV_evaluated=False), completed=fitted)

    # The immutable boundary below is written before any DEV truth, predictions
    # or baseline scores are constructed. A failed second fit never reaches DEV.
    _finite_scope(spec)
    freeze_path = output / 'TRAIN_FROZEN.private.json'
    freeze = dict(schema='WM02_FUSION_STRUCTURED_TRAIN_FREEZE_V1', arms=fitted,
                  source_sha=spec['source_sha'], numeric_mode='NUMPY_FLOAT64',
                  min_lopo_gain_mm=FROZEN_TRAIN_MIN_GAIN_MM,
                  gain_source=gain_source, replay_sha256=spec['replay_sha256'],
                  metadata_sha256=spec['teacher_metadata_sha256'],
                  train_mean_response_model_sha256=spec['train_mean_response_model_sha256'],
                  task_sha256=spec['task_sha256'], both_arms_exactly_reloaded=True,
                  chosen_before_DEV=True, DEV_evaluated=False)
    write_json(freeze_path, freeze)
    freeze_sha = digest(freeze_path)
    record_status(output, progress=dict(stage='BOTH_ARMS_FROZEN_BEFORE_DEV',
                                       freeze_sha256=freeze_sha, DEV_evaluated=False))

    parameters = models[ARMS[0]].parameters
    curves = parameters['hold_curve_m'][None] + parameters['action_response_m']
    curves_path = output / 'TRAIN_MEAN_RESPONSE_CURVES.npz'
    _npz(curves_path, curves_m=curves, fit_split=np.asarray('TRAIN'),
         marginal_curve_m=parameters['marginal_curve_m'])
    baseline_metrics, results = {}, []
    evaluations = {arm: {} for arm in ARMS}
    effects = {arm: {} for arm in ARMS}
    for split, ids in splits.items():
        _finite_scope(spec)
        truth = _truth(data, ids)
        anchor = data['state'][ids, 8:9, 86:89]
        baseline_predictions = dict(
            TRAIN_MEAN_ACTION_RESPONSE=anchor + curves[data['observed_action'][ids]],
            TRAIN_MARGINAL_RESPONSE=anchor + parameters['marginal_curve_m'],
            HOLD_CURRENT=np.repeat(anchor, 16, axis=1),
            CONSTANT_VELOCITY=anchor + np.arange(1, 17)[None, :, None]
            * (anchor - data['state'][ids, 7:8, 86:89]))
        for name, prediction in baseline_predictions.items():
            baseline_metrics.setdefault(name, {})[split] = _evaluation(prediction, truth, data, ids)
        for arm in ARMS:
            prediction = models[arm].predict(data, ids)
            hold = models[arm].predict(data, ids, candidate_actions=0)
            evaluations[arm][split] = _evaluation(prediction, truth, data, ids)
            effects[arm][split] = _effect(prediction, hold, data, ids)
            _npz(output / arm / (split + '_PREDICTIONS.private.npz'), ids=ids,
                 prediction=prediction, hold_prediction=hold, truth=truth)
    for fit in fitted:
        own_static = ('TRAIN_MEAN_ACTION_RESPONSE' if fit['arm'] == ARMS[0]
                      else 'TRAIN_MARGINAL_RESPONSE')
        result = dict(fit, absolute=evaluations[fit['arm']], action_effect=effects[fit['arm']],
                      own_static_comparator=own_static,
                      static_minus_model_pooled_RMSE_mm={split:
                          baseline_metrics[own_static][split]['pooled']['all_H16_XYZ_RMSE_mm']
                          - evaluations[fit['arm']][split]['pooled']['all_H16_XYZ_RMSE_mm']
                          for split in splits},
                      strong_action_static_minus_model_pooled_RMSE_mm={split:
                          baseline_metrics['TRAIN_MEAN_ACTION_RESPONSE'][split]['pooled']['all_H16_XYZ_RMSE_mm']
                          - evaluations[fit['arm']][split]['pooled']['all_H16_XYZ_RMSE_mm']
                          for split in splits})
        results.append(result)
        write_json(output / fit['arm'] / 'RESULT.private.json', result)
    write_json(output / 'BASELINES.private.json', baseline_metrics)
    result = dict(
        schema='WM02_FUSION_STRUCTURED_STUDY_RESULT_V1',
        status='CLOSED_FORM_FIT_EXACTLY_RELOADED_THEN_DEV_EVALUATED',
        source_sha=spec['source_sha'], replay_sha256=spec['replay_sha256'],
        training_replay_result_sha256=spec['training_replay_result_sha256'],
        metadata_sha256=spec['teacher_metadata_sha256'], task_sha256=spec['task_sha256'],
        train_mean_response_model_sha256=spec['train_mean_response_model_sha256'],
        TRAIN_mean_curves_sha256=digest(curves_path), TRAIN_freeze_sha256=freeze_sha,
        min_lopo_gain_mm=FROZEN_TRAIN_MIN_GAIN_MM, gain_source=gain_source,
        results=results, baseline_metrics=baseline_metrics,
        model_selection='WHOLE_PARENT_TRAIN_LOPO_ONLY', numeric_mode='NUMPY_FLOAT64',
        actual_TRAIN_windows=len(train), actual_DEV_windows=len(splits['DEV']),
        completed_model_fits=len(results),
        completed_closed_form_ridge_solves=sum(row['completed_closed_form_ridge_solves'] for row in results),
        optimizer_updates=0, neural_training=False, Actor_updates=0, GR00T_updates=0,
        native_integrations=0, GPU_calls=0, TEST_read=False,
        DEV_used_for_fit_normalization_or_selection=False,
        exact_hidden_state_forks=False, expert_action_labels=False,
        output='WORLD_TCP_XYZ_H16_ONLY_NOT_COMPLETE_STATE',
        CALIBRATION_CONTROL_GAIN='NOT_EVALUATED_REQUIRES_NEW_FROZEN_NATIVE_COMPARISON',
        TASK_TRANSFER_GAIN=False, CPU_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        wall_s=time.monotonic()-start, CPU_s=time.process_time()-cpu)
    write_json(output / 'RESULT.private.json', result)
    record_status(output, progress=dict(stage='DEV_EVALUATED', DEV_evaluated=True,
                                       freeze_sha256=freeze_sha), completed=results)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.output.name, PROFILE, args.output) as spec:
        print(json.dumps(run(spec, args.output), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
