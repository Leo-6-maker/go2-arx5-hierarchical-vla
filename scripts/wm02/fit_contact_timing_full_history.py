"""Fit and freeze the independent full927 contact-event baseline before exposed DEV.

Run only through a fresh original-owner CPU admission. This entry reconstructs
a TRAIN-only view from actual episode evidence and performs closed-form ridge
fits with no optimizer updates. It never edits historical models or metrics. Future inputs mean
selected/predeclared program conditioning, not installed H40/consumed actions.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from go2arx5_vla.world_model.contact_timing_full_history import (
    ARMS, BLIND_SOURCE, FEATURE_RECIPE, HISTORY_DIMENSION, HORIZONS, INFORMATION, RIDGE,
    ContactTimingFullHistory, contact_replay,
)
from go2arx5_vla.world_model.pilot2_supported import PLAN
from go2arx5_vla.world_model.storage import digest, write_json


PROFILE = 'wm02-fusion-contact-full-history-offline-cpu'
KIND = 'FUSION_CONTACT_TIMING_FULL_HISTORY_BASELINE'
RECIPE = 'WM02_CONTACT_TIMING_FULL927_MATCHED_TRAIN_RIDGE_V1'


def _finite(spec):
    if (time.time() >= spec['deadline_epoch']
            or (Path(spec['registered_root']) / 'STOP.json').exists()
            or spec.get('campaign_root') and (Path(spec['campaign_root']) / 'STOP.json').exists()):
        raise TimeoutError('FINITE_CONTACT_TIMING_CPU_ENDPOINT_OR_STOP')


def _pinned(path, pin):
    if digest(path) != pin:
        raise ValueError('SEALED_CONTACT_TIMING_INPUT_CHANGED:' + str(path))


def _bundle(item):
    _pinned(item['path'], item['sha256'])
    value = json.loads(gzip.decompress(Path(item['path']).read_bytes()))
    if (value.get('schema') != 'WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1'
            or value.get('plan_sha') != PLAN
            or value.get('complete_MPC_WBC_hidden_state') is not False
            or value.get('free_rollout_qualified') is not False):
        raise ValueError('ACTUAL_PARTIAL_CONTACT_BUNDLE_NOT_FULL_STATE_REQUIRED')
    return value['episodes']


def _training_inputs(spec):
    """Only metadata and TRAIN episode files are read before the freeze.

    The manifest is a subset of existing native evidence descriptors, not a
    full bundle containing DEV labels. Its hash and membership are preregistered.
    """
    _pinned(spec['training_manifest'], spec['training_manifest_sha256'])
    manifest = json.loads(Path(spec['training_manifest']).read_text())
    if manifest.get('schema') != 'WM02_PILOT2_SUPPORTED_BUNDLE_MANIFEST_V1' or manifest.get('plan_sha') != PLAN:
        raise ValueError('PINNED_TRAIN_ONLY_CONTACT_EVIDENCE_MANIFEST_REQUIRED')
    identities, parents = [], set()
    for row in manifest['episodes']:
        episode_path = Path(row['run']) / 'EPISODE_MANIFEST.private.json'
        _pinned(episode_path, row['episode_sha256'])
        episode = json.loads(episode_path.read_text())
        if (episode.get('split') != 'TRAIN'
                or episode.get('schema') != 'WM02_PILOT2_SUPPORTED_EPISODE_V1'
                or episode.get('purpose') != 'SUPPORTED_CONTACT_COLLECTION'
                or episode.get('command_program') not in ('CLOSE_KEEP', 'CLOSE_RELEASE', 'OPEN_ONLY')
                or episode.get('close_start_atom') not in (56, 64)):
            raise ValueError('ORIGINAL_TRAIN_ONLY_SELECTED_PROGRAMS_NO72_FIT_REQUIRED')
        run = Path(row['run']).name
        identities.append(dict(run_id=run, parent=episode['parent'], split='TRAIN'))
        parents.add(episode['parent'])
    if (len(identities) != spec['expected_TRAIN_episodes']
            or len(parents) != spec['expected_TRAIN_parents']
            or len({row['run_id'] for row in identities}) != len(identities)):
        raise ValueError('EXACT_PREREGISTERED_UNIQUE_TRAIN_EPISODES_AND_PARENTS_REQUIRED')
    return manifest, identities


def _measure(probability, data, ids):
    """Reuse the original event evaluator, without pretending to predict TCP."""
    from go2arx5_vla.world_model.pilot2_contact_model import metrics
    from go2arx5_vla.world_model.pilot2_contact_timing import transition_timing
    values = metrics(np.zeros_like(data['delta_mm']), probability, data, ids)
    for horizon in values.values():
        for stratum in horizon.values():
            stratum.pop('TCP_delta_RMSE_mm', None)
            stratum.pop('object_delta_RMSE_mm', None)
    return dict(event_forecast_metrics=values, grid_transition_timing=transition_timing(probability, data, ids))


def _strata(episodes, data, fit_runs):
    byrun = {ep['run_id']: ep for ep in episodes}
    if len(byrun) != len(episodes):
        raise ValueError('UNIQUE_EVALUATION_EPISODES_REQUIRED')
    runs = [identity['run_id'] for identity in data['identities']]
    result = {
        'TRAIN_FIT_EPISODES': np.array([i for i, run in enumerate(runs) if run in fit_runs], int),
        'TRAIN_TIMING_HOLDOUT_EXPOSED': np.array([i for i, run in enumerate(runs)
            if byrun[run]['split'] == 'TRAIN' and run not in fit_runs], int),
        'DEV_EXPOSED': np.flatnonzero(data['split'] == 'DEV'),
    }
    for parent in sorted({ep['parent'] for ep in episodes}):
        result['PARENT/' + parent] = np.array([i for i, run in enumerate(runs) if byrun[run]['parent'] == parent], int)
    for split in ('TRAIN', 'DEV'):
        for close in (56, 64, 72):
            result[split + '/CLOSE_' + str(close)] = np.array([i for i, run in enumerate(runs)
                if byrun[run]['split'] == split and byrun[run]['episode']['close_start_atom'] == close], int)
    return result


def _frozen_fir(spec, reconstructed, identities, evaluation, fit_runs):
    """Read the old, unchanged FIR only after the new durable TRAIN freeze.

    Bind the original frozen TRAIN identities and model hashes, not just a
    caller's claim that the old fit used the same data. No old fitting occurs.
    The old blind input is memory.opening, explicitly different from this new
    variant's anchor executed-reference blind input.
    """
    from go2arx5_vla.world_model.contact_timing_baseline import ContactTimingFIR
    from go2arx5_vla.world_model.contact_timing_baseline import FEATURE_RECIPE as old_recipe
    descriptor = spec['frozen_fir']
    _pinned(descriptor['freeze']['path'], descriptor['freeze']['sha256'])
    freeze = json.loads(Path(descriptor['freeze']['path']).read_text())
    if (freeze.get('schema') != 'WM02_CONTACT_TIMING_TRAIN_FREEZE_V1'
            or freeze.get('feature_recipe') != old_recipe
            or freeze.get('ridge') != RIDGE or freeze.get('information') != INFORMATION
            or freeze.get('training_manifest_sha256') != spec['training_manifest_sha256']
            or freeze.get('TRAIN_replay_sha256') != reconstructed['bundle_sha256']
            or freeze.get('identities') != identities
            or freeze.get('normalization') != 'OWN_ARM_VALID_TRAIN_ONLY_PER_HORIZON'
            or freeze.get('DEV_read') is not False or freeze.get('TRAIN72_read') is not False):
        raise ValueError('OLD_FIR_MUST_HAVE_THE_EXACT_SAME_FROZEN_TRAIN_MANIFEST_AND_REPLAY')
    expected = {v['arm']: v['numpy_sha256'] for v in freeze['arms']}
    supplied = descriptor['models']
    if (len(supplied) != len(ARMS) or len(freeze['arms']) != len(ARMS) or len(expected) != len(ARMS)
            or {v['arm'] for v in supplied} != set(ARMS) or set(expected) != set(ARMS)):
        raise ValueError('BOTH_ORIGINAL_FROZEN_FIR_ARMS_REQUIRED')
    values = []
    for item in supplied:
        _finite(spec)
        _pinned(item['path'], item['sha256'])
        if item['sha256'] != expected[item['arm']]:
            raise ValueError('OLD_FIR_MODEL_MUST_MATCH_ORIGINAL_TRAIN_FREEZE')
        model = ContactTimingFIR.load(item['path'])
        if model.arm != item['arm']:
            raise ValueError('ORIGINAL_FIR_ARM_IDENTITY_CHANGED')
        value = dict(arm=model.arm, numpy_sha256=item['sha256'], source_sha=freeze['source_sha'],
            freeze_sha256=descriptor['freeze']['sha256'], feature_recipe=old_recipe,
            training_manifest_sha256=freeze['training_manifest_sha256'],
            training_exposure_matches_baseline=True, observed_history_features=90,
            blind_source='INTERNAL_MEMORY_OPENING_NOT_ANCHOR_EXECUTED_REFERENCE', metrics={})
        for name, (episodes, actual) in evaluation.items():
            probability = model.predict(actual['history'], actual['plan'])
            value['metrics'][name] = {label: _measure(probability, actual, ids)
                for label, ids in _strata(episodes, actual, fit_runs).items()}
        values.append(value)
    return values


def run(spec, output):
    started, cpu = time.monotonic(), time.process_time()
    expected = dict(profile=PROFILE, kind=KIND, recipe=RECIPE, device='cpu', physics=False,
                    numeric_mode='NUMPY_FLOAT64', arms=list(ARMS), ridge=RIDGE,
                    information=INFORMATION, GR00T_updates=0, WBC_changes=False,
                    feature_recipe=FEATURE_RECIPE, observed_history_features=HISTORY_DIMENSION,
                    blind_source=BLIND_SOURCE)
    if (any(spec.get(k) != v for k, v in expected.items())
            or spec.get('physics') is not False or spec.get('WBC_changes') is not False
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise PermissionError('FINITE_SAME_INFORMATION_CPU_CONTACT_TIMING_RECIPE_REQUIRED')
    _finite(spec)
    output = Path(output)
    manifest, identities = _training_inputs(spec)
    from scripts.wm02.build_pilot2_contact import build
    reconstructed = build(manifest, output / 'TRAIN_REPLAY')
    training = _bundle(dict(path=str(output / 'TRAIN_REPLAY/CONTACT_BUNDLE.private.json.gz'),
                            sha256=reconstructed['bundle_sha256']))
    if {ep['run_id'] for ep in training} != {row['run_id'] for row in identities}:
        raise ValueError('TRAIN_ONLY_RECONSTRUCTION_IDENTITIES_DIFFER')
    data = contact_replay(training)
    fitted, models = [], {}
    for arm in ARMS:
        _finite(spec)
        model = ContactTimingFullHistory(arm).fit(data)
        path = output / (arm + '.npz')
        model.save(path)
        restored = ContactTimingFullHistory.load(path)
        arrays = restored.arrays()
        if (any(not np.array_equal(value, arrays[name]) for name, value in model.arrays().items())
                or not np.array_equal(model.predict(data),
                                      restored.predict(data))):
            raise ValueError('EXACT_CONTACT_TIMING_ARRAY_AND_PREDICTION_RELOAD_REQUIRED')
        models[arm] = restored
        fitted.append(dict(arm=arm, numpy_sha256=digest(path), exact_reload=True,
            completed_model_fits=1, closed_form_head_solves=len(HORIZONS), optimizer_updates=0,
            horizon_label_counts={str(h): int(restored.heads[h]['train_labels']) for h in HORIZONS}))
    frequencies = np.stack([data['events'][data['valid'][:, j], j].mean(0) for j in range(len(HORIZONS))])
    freeze_path = output / 'TRAIN_FROZEN.private.json'
    write_json(freeze_path, dict(schema='WM02_CONTACT_TIMING_FULL927_TRAIN_FREEZE_V1', source_sha=spec['source_sha'],
        training_manifest_sha256=spec['training_manifest_sha256'], TRAIN_replay_sha256=reconstructed['bundle_sha256'],
        identities=identities, TRAIN_parents=sorted({v['parent'] for v in identities}), arms=fitted,
        ridge=RIDGE, feature_recipe=FEATURE_RECIPE, information=INFORMATION,
        observed_history_features=HISTORY_DIMENSION, blind_source=BLIND_SOURCE,
        normalization='OWN_ARM_VALID_TRAIN_ONLY_PER_HORIZON', TRAIN_event_frequency=frequencies.tolist(),
        chosen_before_new_evaluation_read=True, DEV_read=False, TRAIN72_read=False,
        optimizer_updates=0, native_integrations=0))
    freeze_sha = digest(freeze_path)
    # No evaluation/matched bundle or comparator weights are opened above this
    # durable boundary. Their file identities were fixed in the admitted spec.
    _finite(spec)
    matched = _bundle(spec['matched_training_package'])
    contact_replay(matched)  # Authenticate partitions/knowledge, after freeze.
    matched_train = {ep['run_id']: ep['parent'] for ep in matched if ep['split'] == 'TRAIN'}
    fit_runs = {row['run_id'] for row in identities}
    if matched_train != {row['run_id']: row['parent'] for row in identities}:
        raise ValueError('BASELINE_MUST_MATCH_FROZEN_NEURAL_PACKAGE_TRAIN_EXPOSURE')
    provenance = ('native_result_sha256', 'score_sha256', 'source_sha',
                  'native_binary_sha256', 'scene_sha256')
    matched_by_run = {ep['run_id']: ep for ep in matched}
    if any(any(ep.get(key) != matched_by_run[ep['run_id']].get(key) for key in provenance)
           for ep in training):
        raise ValueError('MATCHED_TRAIN_NATIVE_DATA_PROVENANCE_DIFFERS')
    evaluation = {}
    names = [item['name'] for item in spec['evaluation_bundles']]
    if not names or len(set(names)) != len(names):
        raise ValueError('UNIQUE_PREREGISTERED_CONTACT_EVALUATION_COHORTS_REQUIRED')
    for item in spec['evaluation_bundles']:
        _finite(spec)
        episodes = _bundle(item)
        evaluation[item['name']] = (episodes, contact_replay(episodes))
    rows, comparison = {}, []
    for name, (episodes, actual) in evaluation.items():
        strata = _strata(episodes, actual, fit_runs)
        result = dict(bundle_sha256=next(v['sha256'] for v in spec['evaluation_bundles'] if v['name'] == name),
            episodes=len(episodes), anchors=len(actual['history']), valid_forecasts=int(actual['valid'].sum()),
            independent_parents=len({ep['parent'] for ep in episodes}), models={}, baselines={})
        for arm, model in models.items():
            _finite(spec)
            raw = model.predict(actual, return_unclipped=True)
            probability = np.clip(raw, 0, 1)
            valid = np.broadcast_to(actual['valid'][..., None], raw.shape)
            result['models'][arm] = dict(probability_link='EXPLICIT_CLIP_LINEAR_PROBABILITIES_ONLY_NOT_ACTIONS',
                raw_probabilities_outside_unit_interval=int(np.sum(((raw < 0) | (raw > 1)) & valid)),
                evaluated_event_probabilities=int(valid.sum()),
                strata={label: _measure(probability, actual, ids) for label, ids in strata.items()})
        for baseline, probability in (
                ('PERSISTENCE', np.repeat(actual['current_events'][:, None], len(HORIZONS), axis=1)),
                ('TRAIN_EVENT_FREQUENCY', np.broadcast_to(frequencies, actual['events'].shape))):
            result['baselines'][baseline] = {label: _measure(probability, actual, ids) for label, ids in strata.items()}
        rows[name] = result
    from go2arx5_vla.world_model.pilot2_contact_numpy import predict
    model_names = [model['name'] for model in spec.get('frozen_models', [])]
    if len(model_names) != len(set(model_names)):
        raise ValueError('UNIQUE_PREREGISTERED_FROZEN_CONTACT_MODEL_NAMES_REQUIRED')
    for item in spec.get('frozen_models', []):
        _finite(spec)
        if item['arm'] not in ARMS:
            raise ValueError('EXISTING_FROZEN_CONTACT_ARM_REQUIRED')
        _pinned(item['path'], item['sha256'])
        _pinned(item['checkpoint'], item['checkpoint_sha256'])
        with np.load(item['path'], allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        if ('prefix_mask' not in arrays or not np.array_equal(arrays['forecast_horizons'], HORIZONS)):
            raise ValueError('FROZEN_HORIZON_PREFIX_CONTACT_COMPARATORS_REQUIRED')
        value = dict(name=item['name'], arm=item['arm'], numpy_sha256=item['sha256'],
            checkpoint_sha256=item['checkpoint_sha256'],
            training_bundle_sha256=item['training_bundle_sha256'],
            training_exposure_matches_baseline=item['training_bundle_sha256'] == spec['matched_training_package']['sha256'],
            metrics={})
        for name, (episodes, actual) in evaluation.items():
            future = actual['plan' if item['arm'] == ARMS[0] else 'blind_plan']
            raw = predict(arrays, np.c_[actual['history'], future])
            probability = 1 / (1 + np.exp(-np.clip(raw[..., 6:], -50, 50)))
            value['metrics'][name] = {label: _measure(probability, actual, ids)
                for label, ids in _strata(episodes, actual, fit_runs).items()}
        comparison.append(value)
    fir_comparison = _frozen_fir(spec, reconstructed, identities, evaluation, fit_runs)
    if digest(freeze_path) != freeze_sha:
        raise ValueError('CONTACT_TIMING_TRAIN_FREEZE_CHANGED_DURING_EVALUATION')
    result = dict(schema='WM02_CONTACT_TIMING_FULL927_BASELINE_ACTUAL_V1',
        status='FULL927_CONTACT_EVENT_BASELINES_FROZEN_RELOADED_AND_EXPOSED_DEV_EVALUATED',
        source_sha=spec['source_sha'], freeze_sha256=freeze_sha, arms=fitted,
        matched_training_package_sha256=spec['matched_training_package']['sha256'],
        TRAIN_episode_count=len(training), TRAIN_parent_count=len({ep['parent'] for ep in training}),
        TRAIN_anchors=len(data['history']), information=INFORMATION,
        recipe=RECIPE, feature_recipe=FEATURE_RECIPE,
        observed_history_features=HISTORY_DIMENSION, unchanged_original_neural_history=True,
        blind_source=BLIND_SOURCE, raw_anchor_executed_reference_binding_verified=True,
        horizon_feature_counts={str(h): HISTORY_DIMENSION + h for h in HORIZONS},
        normalization='OWN_ARM_VALID_TRAIN_ONLY_PER_HORIZON', ridge=RIDGE,
        architecture_objective_or_fitted_normalizer_matches_neural=False,
        future_program_is_not_installed_or_consumed=True, INSTALLED_REFERENCE_ONLY='NOT_RUN',
        nine_actual_history_rows=True, future_prefix_invariant_by_construction=True,
        phase_clock_parent_goal_and_close_start_are_not_features=True,
        native_events_are_privileged_measurements_as_in_original_WM=True,
        completed_model_fits=len(fitted), closed_form_head_solves=len(fitted) * len(HORIZONS),
        main_optimizer_updates=0, restore_optimizer_probes=0, exact_array_and_prediction_reloads=len(fitted),
        native_integrations=0, new_real_transitions=0, GR00T_updates=0, Actor_updates=0,
        free_rollout_qualified=False, native_control_gain_evaluated=False,
        TEST_read=False, new_independent_TEST=False, all_evaluation_retrospective=True,
        old_exposed_DEV_and_TRAIN72_are_development_diagnostics=True,
        results=rows, frozen_neural_comparators=comparison, frozen_FIR90_comparators=fir_comparison,
        CPU_s=time.process_time() - cpu, wall_s=time.monotonic() - started,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(output / 'RESULT.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.output.name, PROFILE, args.output) as spec:
        result = run(spec, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'source_sha', 'completed_model_fits',
        'closed_form_head_solves', 'main_optimizer_updates', 'native_integrations', 'CPU_s', 'wall_s')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
