"""Finite CPU-only fit of the frozen 72-run trajectory response experiment.

Authenticate TRAIN first, fit/save/reload/verify/freeze, then read DEV sources.
No model/processor query, optimizer, physics, dispatch or scope registration.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from go2arx5_vla.world_model.fusion_scope import validate as validate_fusion_scope
from go2arx5_vla.world_model.pilot1_storage import local_path
from go2arx5_vla.world_model.storage import digest, write_json
from go2arx5_vla.world_model.fusion_trajectory_response_contract import (
    PHASE, RECIPE as DATA_RECIPE, canonical_sha256, need, pulse_basis, validate_episode, validate_panel,
)
from go2arx5_vla.world_model.fusion_trajectory_response_replay import build_replay, validate_cohort
from go2arx5_vla.world_model.fusion_trajectory_response import (
    MODELS, RIDGE, TrajectoryResponseStudy, forecast_metrics, mask_future_correction,
    ordinary_baselines, train_noise_threshold,
)

PROFILE = 'wm02-fusion-trajectory-response-offline-cpu'
KIND = 'FUSION_TRAJECTORY_RESPONSE_OFFLINE'
SITE = 'CORE_FUSION_TRAJECTORY_RESPONSE_CPU'
RECIPE = 'WM02_TRAJECTORY_RESPONSE_TRAIN_FREEZE_THEN_DEV_CPU_V1'
MANIFEST_SCHEMA = 'WM02_TRAJECTORY_RESPONSE_SOURCE_MANIFEST_V1'
ENTRY_FILE = 'scripts/wm02/fit_fusion_trajectory_response.py'
MODEL_FILES = (
    'src/go2arx5_vla/world_model/fusion_trajectory_response_contract.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_response_replay.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_response.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_adapter.py',
    'src/go2arx5_vla/trajectory/so3.py',
)
ARTIFACTS = ('execution_spec', 'episode_manifest', 'native_result', 'collection_report',
             'decision', 'lease', 'worker_exit')
PREDICTORS = ('STATIC',) + MODELS + ('SELECTED',)


def _finite(spec):
    if (time.time() >= spec['deadline_epoch']
            or (Path(spec['registered_root']) / 'STOP.json').exists()
            or (Path(spec['campaign_root']) / 'STOP.json').exists()
            or any((Path(spec[name]).parent / 'STOP.json').exists()
                   for name in ('fusion_binding', 'pilot1_binding') if name in spec)
            or ('_fit_wall_started' in spec and time.monotonic() - spec['_fit_wall_started'] >= spec['wall_limit_s'])):
        raise TimeoutError('FINITE_TRAJECTORY_RESPONSE_CPU_ENDPOINT_OR_STOP')


def _scope(spec, output):
    """Explicit reuse: generic runtime does not know this new CPU site yet."""
    expected = dict(profile=PROFILE, kind=KIND, execution_site=SITE, recipe=RECIPE,
                    device='cpu', physics=False, numeric_mode='NUMPY_FLOAT64', ridge=RIDGE,
                    GR00T_updates=0, Actor_updates=0, WBC_changes=False,
                    information='PREDECLARED_PROGRAM_CONDITIONAL')
    if (any(spec.get(k) != v for k, v in expected.items())
            or spec.get('physics') is not False or spec.get('WBC_changes') is not False
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise PermissionError('EXACT_NEW_CPU_RESPONSE_RECIPE_NO_CUDA_OR_PHYSICS_REQUIRED')
    need(np.isfinite(spec['noise_fraction']) and spec['noise_fraction'] > 0
         and np.isfinite(spec['noise_floor_mm']) and spec['noise_floor_mm'] > 0,
         'POSITIVE_SPEC_PREDECLARED_NEW_TRAIN_NOISE_RULE')
    need(isinstance(spec['source_sha'], str) and len(spec['source_sha']) == 40
         and all(c in '0123456789abcdef' for c in spec['source_sha'])
         and np.isfinite(spec['deadline_epoch'])
         and np.isfinite(spec['wall_limit_s']) and spec['wall_limit_s'] > 0,
         'FULL_FIT_SOURCE_AND_FINITE_WALL_ENDPOINT')
    root = Path(spec['registered_root']).resolve(strict=True)
    need(Path(output).resolve(strict=True).parent == root and Path(output).name == 'fit-01'
         and not Path(output).is_symlink(), 'EXACT_ADMITTED_FIT_01_OUTPUT')
    entry = Path(spec['source_root']) / ENTRY_FILE
    need(Path(__file__).resolve() == entry.resolve(strict=True)
         and spec['source_files'].get(ENTRY_FILE) == digest(entry), 'EXACT_FIT_ENTRY_CAPSULE_AND_SOURCE_PIN')
    validate_fusion_scope(spec)
    _finite(spec)
    for name in ('source_root', 'registered_root', 'python', 'data', 'cache', 'logs', 'tmp',
                 'panel', 'source_manifest', 'raw_action'):
        local_path(spec, spec[name])
    local_path(spec, output)


def _bytes(spec, descriptor, io, partition):
    _finite(spec)
    need(set(descriptor) == {'path', 'sha256'}, 'EXACT_PINNED_ARTIFACT_DESCRIPTOR')
    path = Path(descriptor['path'])
    local_path(spec, path)
    raw = path.read_bytes()
    io[partition]['files'] += 1
    io[partition]['logical_bytes_read'] += len(raw)
    need(hashlib.sha256(raw).hexdigest() == descriptor['sha256'], 'SOURCE_ARTIFACT_SHA256:' + str(path))
    _finite(spec)
    return raw


def _json(spec, descriptor, io, partition):
    return json.loads(_bytes(spec, descriptor, io, partition))


def _spec_descriptor(spec, name):
    return dict(path=spec[name], sha256=spec[name + '_sha256'])


def _preflight(spec, io):
    panel = validate_panel(_json(spec, _spec_descriptor(spec, 'panel'), io, 'COMMON'))
    manifest = _json(spec, _spec_descriptor(spec, 'source_manifest'), io, 'COMMON')
    need(manifest.get('schema') == MANIFEST_SCHEMA
         and manifest.get('panel_sha256') == canonical_sha256(panel)
         and manifest.get('collector_source_sha') == panel['source_sha'], 'PINNED_SOURCE_MANIFEST_AND_PANEL')
    pins = manifest['frozen_model_source_files']
    need(set(pins) == set(MODEL_FILES), 'EXACT_FROZEN_NUMPY_DEPENDENCY_FILES')
    for name in MODEL_FILES:
        need(spec['source_files'].get(name) == pins[name]
             and digest(Path(spec['source_root']) / name) == pins[name], 'FROZEN_MODEL_DEPENDENCY_CHANGED:' + name)
    entries = manifest['episodes']
    need(len(entries) == 72, 'EXACT_72_DECLARED_PHYSICAL_SOURCE_RUNS')
    seen, cells, endpoints = set(), Counter(), set()
    for entry in entries:
        need(set(entry) == {'episode', *ARTIFACTS}, 'EXACT_SOURCE_ENTRY_KEYS')
        episode = validate_episode(entry['episode'], panel)
        run = Path(episode['run'])
        need(run.is_absolute() and '..' not in run.parts and str(run) not in seen,
             'UNIQUE_ABSOLUTE_PHYSICAL_RUN_ROOTS')
        seen.add(str(run)); endpoints.add(episode['planned_end_atom'])
        cells[(episode['parent'], episode['context_id'], episode['theta_level'], episode['restart_index'])] += 1
        for key in ARTIFACTS:
            item = entry[key]
            path = Path(item['path'])
            need(set(item) == {'path', 'sha256'} and isinstance(item['sha256'], str)
                 and len(item['sha256']) == 64 and all(c in '0123456789abcdef' for c in item['sha256'])
                 and path.is_absolute() and '..' not in path.parts and path.is_relative_to(run),
                 'LEXICALLY_RUN_BOUND_ARTIFACT_DESCRIPTOR')
    expected = Counter((parent, context, theta, restart)
        for parent in panel['physical_parents'] for context in ('C0', 'C1')
        for theta in (-1, 0, 1) for restart in (0, 1))
    need(cells == expected and len(endpoints) == 1, 'COMPLETE_BALANCED_CELLS_AND_COMMON_PROGRAM_CLOCK')
    need(spec['raw_action_sha256'] == panel['query_pins']['action'], 'ORIGINAL_FROZEN_RAW_ACTION_PIN')
    raw_bytes = _bytes(spec, _spec_descriptor(spec, 'raw_action'), io, 'COMMON')
    with np.load(BytesIO(raw_bytes), allow_pickle=False) as archive:
        raw_H40 = archive['physical_H40'].copy()
    need(raw_H40.shape == (40, 10) and raw_H40.dtype.kind == 'f' and np.isfinite(raw_H40).all(),
         'ORIGINAL_ALREADY_DECODED_RAW_H40')
    return panel, manifest, raw_H40


def _load_record(spec, entry, panel, manifest, raw_H40, io, partition):
    """Open this partition's files only; DEV calls occur after durable freeze."""
    root = Path(entry['episode']['run']).resolve(strict=True)
    for key in ARTIFACTS:
        need(Path(entry[key]['path']).resolve(strict=True).is_relative_to(root), 'REAL_RUN_BOUND_ARTIFACT')
    value = {key: _json(spec, entry[key], io, partition) for key in ARTIFACTS}
    episode, collect = value['episode_manifest'], value['execution_spec']
    native, report, decision = value['native_result'], value['collection_report'], value['decision']
    lease, worker = value['lease'], value['worker_exit']
    need(episode == entry['episode'] and collect.get('registered_root') == str(root)
         and collect.get('phase') == native.get('phase') == PHASE
         and collect.get('source_sha') == native.get('source_sha') == panel['source_sha']
         and collect.get('episode_manifest_sha256') == entry['episode_manifest']['sha256']
         and collect.get('trajectory_response_panel_sha256') == spec['panel_sha256'],
         'ACTUAL_COLLECTOR_SOURCE_EPISODE_AND_PANEL_JOIN')
    need(all(collect.get('source_files', {}).get(name) == manifest['frozen_model_source_files'][name]
             for name in MODEL_FILES), 'COLLECTED_WITH_SAME_FROZEN_NUMPY_DEPENDENCIES')
    need(lease.get('status') == 'CLOSED' and lease.get('run_uuid') == root.name
         and lease.get('spec_sha256') == entry['execution_spec']['sha256']
         and lease.get('worker_exit_code') == worker.get('exit_code') == 0,
         'ACTUAL_CLOSED_ZERO_EXIT_COLLECTOR_LEASE')
    need(native.get('reference_qualification') == report
         and report.get('decision_record', {}).get('sha256') == entry['decision']['sha256']
         and decision.get('episode') == episode,
         'NATIVE_REPORT_AND_ACTUAL_SEALED_DECISION_JOIN')
    pins = {name: entry[key]['sha256'] for name, key in (
        ('native_result_sha256', 'native_result'), ('report_sha256', 'collection_report'),
        ('episode_sha256', 'episode_manifest'), ('decision_sha256', 'decision'))}
    pins['raw_action_sha256'] = spec['raw_action_sha256']
    return dict(episode=episode, native=native, report=report, decision=decision,
                raw_H40=raw_H40, pins=pins)


def _partition(spec, entries, panel, manifest, raw_H40, io, partition, completed):
    chunks = []
    for entry in entries:
        _finite(spec)
        record = _load_record(spec, entry, panel, manifest, raw_H40, io, partition)
        data, _ = build_replay([record], panel, require_complete=False)
        chunks.append(data)
        completed.append(entry['episode']['run'])
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def _fit_layout(training, entries):
    """Internal shape/role scaffolding. Never write this as real DEV replay."""
    required = ('history_state', 'past_reference', 'future_TCP', 'nominal_H40', 'corrected_H40',
                'correction_basis_H40', 'source_age_s', 'ttl_remaining_s', 'complete_live_atoms',
                'parent', 'split', 'run_id', 'context_id', 'theta_level', 'restart_index', 'source_record_sha256')
    lookup = {str(run): i for i, run in enumerate(training['run_id'])}
    rows = []
    for entry in entries:
        episode = entry['episode']
        if episode['split'] == 'TRAIN':
            row = {key: training[key][lookup[episode['run']]] for key in required}
        else:
            # Only declared design information, not a measured or fitted sample.
            nominal = np.zeros((40, 10)); nominal[:, 3:9] = [1, 0, 0, 0, 1, 0]
            basis = pulse_basis(np.arange(40))
            corrected = nominal.copy(); corrected[:, 0] += episode['theta_level'] * .0005 * basis
            row = dict(history_state=np.full((9, 106), np.nan), past_reference=np.full((8, 17), np.nan),
                future_TCP=np.full((16, 3), np.nan), nominal_H40=nominal, corrected_H40=corrected,
                correction_basis_H40=basis, source_age_s=np.nan, ttl_remaining_s=np.nan,
                complete_live_atoms=np.nan, parent=episode['parent'], split='DEV', run_id=episode['run'],
                context_id=episode['context_id'], theta_level=episode['theta_level'],
                restart_index=episode['restart_index'], source_record_sha256='DECLARED_UNREAD_DEV_SOURCE')
        rows.append(row)
    return {key: np.stack([row[key] for row in rows]) for key in required}


def _actual_layout(training, dev, entries):
    sources = {str(run): (partition, i) for partition in (training, dev)
               for i, run in enumerate(partition['run_id'])}
    return {key: np.stack([sources[e['episode']['run']][0][key][sources[e['episode']['run']][1]]
                           for e in entries]) for key in training}


@contextmanager
def _count_solves(spec, counts):
    """Transparent process-local instrumentation; original solver restored always."""
    original = np.linalg.solve
    def counted(*args, **kwargs):
        _finite(spec)
        counts['attempted'] += 1
        result = original(*args, **kwargs)
        counts['completed'] += 1
        _finite(spec)
        return result
    np.linalg.solve = counted
    try:
        yield
    finally:
        np.linalg.solve = original


def _npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _progress(output, stage, **values):
    from scripts.pi0292.durable import write
    write(Path(output) / 'PROGRESS.private.json', dict(stage=stage, **values))


def _metrics(prediction, truth, data, ids):
    return dict(pooled=forecast_metrics(prediction, truth),
        by_parent={str(parent): forecast_metrics(prediction[data['parent'][ids] == parent], truth[data['parent'][ids] == parent])
                   for parent in sorted(set(data['parent'][ids]))},
        by_context={context: forecast_metrics(prediction[data['context_id'][ids] == context], truth[data['context_id'][ids] == context])
                    for context in ('C0', 'C1')},
        by_theta={str(theta): forecast_metrics(prediction[data['theta_level'][ids] == theta], truth[data['theta_level'][ids] == theta])
                  for theta in (-1, 0, 1)})


def _evaluate(model, data, ids):
    truth = data['future_TCP'][ids]
    predictions = {name: model.predict(data, ids, model=name) for name in PREDICTORS}
    predictions.update(ordinary_baselines(data, ids))
    current = data['history_state'][ids, -1:, 86:89]
    truth_step = np.diff(np.concatenate((current, truth), axis=1), axis=1)
    results = {}
    for name, prediction in predictions.items():
        step = np.diff(np.concatenate((current, prediction), axis=1), axis=1)
        results[name] = dict(absolute=_metrics(prediction, truth, data, ids),
                             temporal_step=_metrics(step, truth_step, data, ids))
    blind = model.candidate_forecasts(data, ids, model='CORRECTION_BLIND')
    need(np.array_equal(blind[:, 0], blind[:, 1]) and np.array_equal(blind[:, 1], blind[:, 2])
         and np.array_equal(blind[:, 0], model.predict(mask_future_correction(data), ids, model='CORRECTION_BLIND')),
         'EXACT_BLIND_CANDIDATE_AND_CORRECTED_PATH_INVARIANCE')
    return results, predictions


def run(spec, output):
    spec = dict(spec, _fit_wall_started=time.monotonic())
    output = Path(output)
    _scope(spec, output)
    started, cpu = time.monotonic(), time.process_time()
    counts = dict(attempted=0, completed=0)
    io = {key: dict(files=0, logical_bytes_read=0) for key in ('COMMON', 'TRAIN', 'DEV')}
    completed = dict(TRAIN=[], DEV=[])
    stage = 'ADMITTED_CPU_INPUT_PREFLIGHT'
    write_json(output / 'RUN_ATTEMPT.private.json', dict(schema='WM02_TRAJECTORY_RESPONSE_FIT_ATTEMPT_V1',
        fit_source_sha=spec['source_sha'], source_manifest_sha256=spec['source_manifest_sha256']))
    try:
        panel, manifest, raw = _preflight(spec, io)
        entries = manifest['episodes']
        train_entries = [e for e in entries if e['episode']['split'] == 'TRAIN']
        dev_entries = [e for e in entries if e['episode']['split'] == 'DEV']
        stage = 'AUTHENTICATING_TRAIN_ONLY'
        training = _partition(spec, train_entries, panel, manifest, raw, io, 'TRAIN', completed['TRAIN'])
        _finite(spec)
        _npz(output / 'TRAIN_REPLAY.private.npz', **training)
        layout = _fit_layout(training, entries)
        splits = validate_cohort(layout, panel)
        need(len(completed['TRAIN']) == 48 and io['DEV']['files'] == 0, '48_TRAIN_ONLY_SOURCES_AUTHENTICATED')
        threshold = train_noise_threshold(layout, panel, fraction=spec['noise_fraction'], floor_mm=spec['noise_floor_mm'])
        stage = 'FITTING_ONLY_AUTHENTICATED_TRAIN'
        _progress(output, stage, actual_TRAIN_runs=48, DEV_source_files_read=0)
        with _count_solves(spec, counts):
            original = TrajectoryResponseStudy(panel).fit(layout,
                min_lopo_gain_mm=threshold['min_lopo_gain_mm'], threshold_source=threshold)
        need(counts['completed'] == original.report['closed_form_ridge_solves'], 'ACTUAL_SOLVE_COUNT_MATCHES_FROZEN_MODEL')
        _finite(spec)
        model_path = output / 'TRAJECTORY_RESPONSE_NUMPY.npz'
        original.save(model_path)
        restored = TrajectoryResponseStudy.load(model_path)
        restore = original.verify_reload(restored, layout)
        candidates = {name: restored.candidate_forecasts(layout, splits['TRAIN'], model=name) for name in PREDICTORS}
        _npz(output / 'TRAIN_CANDIDATE_PREDICTIONS.private.npz', ids=splits['TRAIN'],
             run_id=layout['run_id'][splits['TRAIN']], **candidates)
        stage = 'WRITING_TRAIN_FREEZE_BEFORE_ANY_DEV_SOURCE_READ'
        _scope(spec, output)
        model_sha = digest(model_path)
        freeze = dict(schema='WM02_TRAJECTORY_RESPONSE_TRAIN_FREEZE_V1',
            provenance_state='AUTHENTICATED_TRAIN48_DEV_DECLARED24_UNREAD',
            collector_source_sha=panel['source_sha'], fit_source_sha=spec['source_sha'],
            source_manifest_sha256=spec['source_manifest_sha256'], panel_file_sha256=spec['panel_sha256'],
            panel_canonical_sha256=canonical_sha256(panel), frozen_model_source_files=manifest['frozen_model_source_files'],
            authenticated_TRAIN_runs=completed['TRAIN'], declared_DEV_runs=[e['episode']['run'] for e in dev_entries],
            DEV_source_files_read=0, DEV_source_identities_authenticated=False,
            numeric_mode='NUMPY_FLOAT64', numpy_version=np.__version__,
            model_report_count_semantics='DECLARED_PANEL_LAYOUT_COUNTS_NOT_DEV_SOURCE_AUTHENTICATION',
            model_sha256=model_sha, exact_restore=restore, threshold_source=threshold,
            actual_numpy_solves=counts.copy(), model_report=restored.report,
            placeholder_layout_written_as_real_replay=False, optimizer_updates=0,
            neural_training=False, TEST_read=False, selected_before_DEV=True)
        freeze_path = output / 'TRAIN_FROZEN.private.json'
        write_json(freeze_path, freeze)
        _fsync_directory(output)
        freeze_sha = digest(freeze_path)
        _progress(output, 'TRAIN_FROZEN', freeze_sha256=freeze_sha, DEV_source_files_read=0)
        stage = 'AUTHENTICATING_DEV_AFTER_DURABLE_TRAIN_FREEZE'
        _scope(spec, output)
        need(digest(freeze_path) == freeze_sha and digest(model_path) == model_sha, 'IMMUTABLE_TRAIN_FREEZE_AND_MODEL')
        dev = _partition(spec, dev_entries, panel, manifest, raw, io, 'DEV', completed['DEV'])
        data = _actual_layout(training, dev, entries)
        real_splits = validate_cohort(data, panel)
        need(len(completed['DEV']) == 24, 'ALL_24_REAL_DEV_SOURCES_AUTHENTICATED')
        need(np.array_equal(layout['run_id'][splits['TRAIN']], data['run_id'][real_splits['TRAIN']])
             and all(np.array_equal(candidates[name], restored.candidate_forecasts(data, real_splits['TRAIN'], model=name))
                     for name in PREDICTORS), 'REAL72_REPLAY_TRAIN_IDENTITY_AND_FROZEN_FORECAST_PARITY')
        _finite(spec)
        _npz(output / 'ACTUAL_72_REPLAY.private.npz', **data)
        stage = 'EVALUATING_FROZEN_MODELS_NO_REFIT'
        results = {}
        for split, ids in real_splits.items():
            _finite(spec)
            results[split], prediction = _evaluate(restored, data, ids)
            _npz(output / (split + '_PREDICTIONS.private.npz'), ids=ids, run_id=data['run_id'][ids],
                 truth=data['future_TCP'][ids], **prediction)
        need(digest(freeze_path) == freeze_sha and digest(model_path) == model_sha, 'FROZEN_ARTIFACTS_CHANGED_DURING_EVALUATION')
        _progress(output, 'COMPLETED', actual_numpy_solves=counts, actual_TRAIN_runs=48, actual_DEV_runs=24)
        _finite(spec)
        result = dict(schema='WM02_TRAJECTORY_RESPONSE_CPU_FIT_ACTUAL_V1',
            status='ACTUAL72_AUTHENTICATED_TRAIN_FROZEN_EXACTLY_RELOADED_THEN_DEV_EVALUATED',
            collector_source_sha=panel['source_sha'], fit_source_sha=spec['source_sha'],
            source_manifest_sha256=spec['source_manifest_sha256'], panel_sha256=spec['panel_sha256'],
            model_sha256=model_sha, TRAIN_freeze_sha256=freeze_sha,
            actual_TRAIN_runs=48, actual_DEV_runs=24, actual_TRAIN_parents=4, actual_DEV_parents=2,
            one_predeclared_probe_per_run=True, actual_numpy_solves=counts, optimizer_updates=0,
            completed_regressors=4, portable_numpy_study_file_loads=1, exact_restore=restore, model_report=restored.report,
            threshold_source=threshold, results=results, input_io=io,
            input_io_accounting='LOGICAL_PINNED_SCIENCE_INPUT_BYTES_EXCLUDES_CODE_VALIDATION_AND_OUTPUT_RELOADS',
            DEV_used_for_fit_normalization_selection=False, neural_model_loads=0, GR00T_queries=0,
            GR00T_updates=0, Actor_updates=0, native_integrations=0, TEST_read=False,
            exact_hidden_state_forks=False, forecast_conditioning='PREDECLARED_PROGRAM_CONDITIONAL',
            CONTROL_GAIN='NOT_EVALUATED_REQUIRES_NEW_FROZEN_NATIVE_COMPARISON', TASK_TRANSFER_GAIN=False,
            wall_s=time.monotonic()-started, CPU_s=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output / 'RESULT.private.json', result)
        return result
    except BaseException as error:
        failure = dict(schema='WM02_TRAJECTORY_RESPONSE_CPU_FIT_FAILURE_V1', stage=stage,
            exception_type=type(error).__name__, reason=str(error), fit_source_sha=spec['source_sha'],
            completed_authenticated_runs=completed, actual_numpy_solves=counts, optimizer_updates=0,
            input_io=io, wall_s=time.monotonic()-started, CPU_s=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artifacts_retained=True, automatic_retry=False, successful_ACTUAL72_result=False)
        write_json(output / 'FAILURE.private.json', failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.output.name, PROFILE, args.output, physics=False, gpu=False) as spec:
        result = run(spec, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'collector_source_sha', 'fit_source_sha',
        'actual_TRAIN_runs', 'actual_DEV_runs', 'actual_numpy_solves', 'optimizer_updates', 'wall_s', 'CPU_s')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
