"""Synthetic tiny-file CLI checks only; never actual72 fitting or site effects."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from go2arx5_vla.world_model.storage import digest, jsonable
from go2arx5_vla.world_model.fusion_trajectory_response import TrajectoryResponseStudy
from scripts.wm02 import fit_fusion_trajectory_response as cli
from tests.world_model.test_fusion_trajectory_response import panel, record


def save(path, value):
    path.write_text(json.dumps(jsonable(value), separators=(',', ':'), allow_nan=False))
    return dict(path=str(path), sha256=digest(path))


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    """Synthetic receipts keep original ABI and exact independent cell identities."""
    base = tmp_path_factory.mktemp('SYNTHETIC_ONLY_TRAJECTORY_RESPONSE_CLI')
    p = panel()
    raw = record(p)['raw_H40']
    raw_path = base / 'SYNTHETIC_RAW_ACTION.npz'
    np.savez_compressed(raw_path, physical_H40=raw)
    p['query_pins']['action'] = digest(raw_path)
    panel_path = base / 'PANEL.json'
    panel_desc = save(panel_path, p)
    source = Path(cli.__file__).resolve().parents[2]
    dependencies = {name: digest(source / name) for name in cli.MODEL_FILES}
    entries = []
    for parent in p['physical_parents']:
        for context in ('C0', 'C1'):
            for theta in (-1, 0, 1):
                for restart in (0, 1):
                    rec = record(p, parent, context, theta, restart)
                    run = base / ('SYNTHETIC_' + parent + '_' + context + '_' + str(theta) + '_' + str(restart))
                    run.mkdir()
                    ep = rec['episode']; ep['run'] = str(run); ep['planned_end_atom'] = 40
                    decision, report, native = rec['decision'], rec['report'], rec['native']
                    for key in ('full_nominal_reference_path', 'full_corrected_reference_path', 'desired_path'):
                        decision[key] = decision[key][:41].copy()
                    decision['episode'] = ep
                    corrected = decision['full_corrected_reference_path']
                    report['windows'] = report['windows'][:96]
                    for index, row in enumerate(report['windows'][48:]):
                        origin = 7 * (index // 7)
                        row['full_final_installed_H40'] = corrected[np.minimum(origin + np.arange(40), 40)].copy()
                        row['actual_installed_consumed_EEF10_BASE7'][:10] = corrected[min(index, 40)]
                    report['complete_task_atoms'] = report['solver_target_confirmed_atoms'] = report['gripper_confirmed_atoms'] = 96
                    native['actual_target_integrations'] = report['windows'][-1]['native_end_step']
                    native['source_sha'] = p['source_sha']
                    ep_desc = save(run / 'EPISODE_MANIFEST.private.json', ep)
                    decision_desc = save(run / 'TRAJECTORY_RESPONSE_DECISION.private.json', decision)
                    report['decision_record'] = decision_desc.copy()
                    report_desc = save(run / 'TRAJECTORY_RESPONSE.private.json', report)
                    native['reference_qualification'] = report
                    native_desc = save(run / 'RESULT.json', native)
                    collect = dict(registered_root=str(run), phase=cli.PHASE, source_sha=p['source_sha'],
                        episode_manifest_sha256=ep_desc['sha256'], trajectory_response_panel_sha256=panel_desc['sha256'],
                        source_files=dependencies)
                    spec_desc = save(run / 'EXECUTION_SPEC.json', collect)
                    lease_desc = save(run / 'LEASE.json', dict(status='CLOSED', run_uuid=run.name,
                        spec_sha256=spec_desc['sha256'], worker_exit_code=0))
                    worker_desc = save(run / 'WORKER_EXIT.private.json', dict(exit_code=0))
                    entries.append(dict(episode=copy.deepcopy(ep), execution_spec=spec_desc, episode_manifest=ep_desc,
                        native_result=native_desc, collection_report=report_desc, decision=decision_desc,
                        lease=lease_desc, worker_exit=worker_desc))
    manifest = dict(schema=cli.MANIFEST_SCHEMA, panel_sha256=cli.canonical_sha256(p),
        collector_source_sha=p['source_sha'], frozen_model_source_files=dependencies, episodes=entries)
    manifest_path = base / 'SOURCE_MANIFEST.json'; manifest_desc = save(manifest_path, manifest)
    spec = dict(profile=cli.PROFILE, kind=cli.KIND, execution_site=cli.SITE, recipe=cli.RECIPE,
        device='cpu', physics=False, numeric_mode='NUMPY_FLOAT64', ridge=1., GR00T_updates=0, Actor_updates=0,
        WBC_changes=False, information='PREDECLARED_PROGRAM_CONDITIONAL',
        noise_fraction=.25, noise_floor_mm=.000001, source_sha='f' * 40,
        source_root=str(source), source_files={**dependencies, cli.ENTRY_FILE: digest(Path(cli.__file__))}, python=sys.executable,
        data=str(base), cache=str(base), logs=str(base), tmp=str(base), campaign_root=str(base),
        panel=panel_desc['path'], panel_sha256=panel_desc['sha256'],
        source_manifest=manifest_desc['path'], source_manifest_sha256=manifest_desc['sha256'],
        raw_action=str(raw_path), raw_action_sha256=digest(raw_path), deadline_epoch=time.time() + 3600,
        wall_limit_s=300.)
    return base, spec, manifest


@pytest.fixture
def admitted(monkeypatch):
    """Only the original external scope/disk seams are mocked in these fixtures."""
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    calls = []
    monkeypatch.setattr(cli, 'validate_fusion_scope', lambda spec: calls.append('scope'))
    monkeypatch.setattr(cli, 'local_path', lambda spec, path: calls.append(('path', str(path))) or Path(path))
    return calls


def worker_spec(inputs, tmp_path):
    _, spec, _ = inputs
    value = copy.deepcopy(spec)
    value['registered_root'] = str(tmp_path)
    output = tmp_path / 'fit-01'; output.mkdir()
    return value, output


def custom_manifest(spec, tmp_path, manifest):
    path = tmp_path / 'MODIFIED_SYNTHETIC_MANIFEST.json'
    descriptor = save(path, manifest)
    spec.update(source_manifest=descriptor['path'], source_manifest_sha256=descriptor['sha256'])


def test_actual_read_order_restore_freeze_and_source_versions_are_separate(inputs, admitted, tmp_path, monkeypatch):
    spec, output = worker_spec(inputs, tmp_path)
    original, sync = cli._bytes, cli._fsync_directory
    events = []
    def read_checked(spec, descriptor, io, partition):
        if partition == 'DEV':
            assert 'dir-fsync' in events
            freeze = json.loads((output / 'TRAIN_FROZEN.private.json').read_text())
            assert freeze['provenance_state'] == 'AUTHENTICATED_TRAIN48_DEV_DECLARED24_UNREAD'
            assert freeze['DEV_source_files_read'] == 0
            assert not freeze['DEV_source_identities_authenticated']
            assert freeze['exact_restore']['TRAIN_candidate_max_error_m'] == 0.
            events.append('DEV-read')
        return original(spec, descriptor, io, partition)
    def synced(path):
        sync(path); events.append('dir-fsync')
    monkeypatch.setattr(cli, '_bytes', read_checked)
    monkeypatch.setattr(cli, '_fsync_directory', synced)
    result = cli.run(spec, output)
    assert events[0] == 'dir-fsync' and events.count('DEV-read') == 24 * 7
    assert result['collector_source_sha'] != result['fit_source_sha']
    assert result['fit_source_sha'] == 'f' * 40
    assert result['actual_numpy_solves'] == dict(attempted=20, completed=20)
    assert result['optimizer_updates'] == result['neural_model_loads'] == result['native_integrations'] == 0
    assert (result['actual_TRAIN_runs'], result['actual_DEV_runs']) == (48, 24)
    assert set(result['results']) == {'TRAIN', 'DEV'}
    assert 'ADDITIVE_LAG_VELOCITY' in result['results']['DEV']
    assert result['TRAIN_freeze_sha256'] == digest(output / 'TRAIN_FROZEN.private.json')
    with np.load(output / 'TRAIN_REPLAY.private.npz', allow_pickle=False) as archive:
        assert len(archive['parent']) == 48 and set(archive['split']) == {'TRAIN'}
        assert np.isfinite(archive['future_TCP']).all()
    with np.load(output / 'ACTUAL_72_REPLAY.private.npz', allow_pickle=False) as archive:
        assert len(archive['parent']) == 72 and np.isfinite(archive['history_state']).all()
    assert result['wall_s'] > 0 and result['CPU_s'] >= 0
    assert len([v for v in admitted if v == 'scope']) >= 3
    with pytest.raises(FileExistsError): cli.run(spec, output)


def test_internal_DEV_layout_has_no_observed_DEV_values_and_is_never_written(inputs, admitted, tmp_path, monkeypatch):
    spec, output = worker_spec(inputs, tmp_path)
    fit = TrajectoryResponseStudy.fit
    def fail_after_audit(model, data, **kwargs):
        dev = data['split'] == 'DEV'
        assert dev.sum() == 24
        for key in ('history_state', 'past_reference', 'future_TCP', 'source_age_s', 'ttl_remaining_s'):
            assert np.isnan(data[key][dev]).all()
        assert np.isfinite(data['nominal_H40'][dev]).all()
        assert set(data['source_record_sha256'][dev]) == {'DECLARED_UNREAD_DEV_SOURCE'}
        raise RuntimeError('SYNTHETIC_STOP_AFTER_LAYOUT_AUDIT')
    monkeypatch.setattr(TrajectoryResponseStudy, 'fit', fail_after_audit)
    with pytest.raises(RuntimeError, match='LAYOUT_AUDIT'): cli.run(spec, output)
    assert not (output / 'ACTUAL_72_REPLAY.private.npz').exists()
    failure = json.loads((output / 'FAILURE.private.json').read_text())
    assert failure['input_io']['DEV']['files'] == 0
    assert failure['actual_numpy_solves'] == dict(attempted=0, completed=0)
    assert not (output / 'TRAIN_FROZEN.private.json').exists()


@pytest.mark.parametrize('bad', ['bad_TRAIN_SHA', 'cross_run_lease', 'nonzero_worker', 'incomplete72',
                               'changed_dependency', 'changed_collector_dependency', 'different_clock'])
def test_invalid_sources_fail_before_any_solve_or_DEV(inputs, admitted, tmp_path, monkeypatch, bad):
    spec, output = worker_spec(inputs, tmp_path)
    manifest = copy.deepcopy(inputs[2])
    if bad == 'bad_TRAIN_SHA': manifest['episodes'][0]['native_result']['sha256'] = '0' * 64
    elif bad == 'cross_run_lease': manifest['episodes'][0]['lease'] = manifest['episodes'][1]['lease'].copy()
    elif bad == 'nonzero_worker':
        original = cli._json
        def wrong_worker(spec, descriptor, io, partition):
            value = original(spec, descriptor, io, partition)
            return dict(value, exit_code=1) if descriptor['path'] == manifest['episodes'][0]['worker_exit']['path'] else value
        monkeypatch.setattr(cli, '_json', wrong_worker)
    elif bad == 'incomplete72': manifest['episodes'] = manifest['episodes'][:12]
    elif bad == 'changed_dependency': manifest['frozen_model_source_files'][cli.MODEL_FILES[0]] = '0' * 64
    elif bad == 'changed_collector_dependency':
        original = cli._json
        def changed_collect(spec, descriptor, io, partition):
            value = original(spec, descriptor, io, partition)
            if descriptor['path'] == manifest['episodes'][0]['execution_spec']['path']:
                value['source_files'][cli.MODEL_FILES[2]] = '0' * 64
            return value
        monkeypatch.setattr(cli, '_json', changed_collect)
    elif bad == 'different_clock': manifest['episodes'][-1]['episode']['planned_end_atom'] += 1
    custom_manifest(spec, tmp_path, manifest)
    with pytest.raises((ValueError, PermissionError)): cli.run(spec, output)
    failure = json.loads((output / 'FAILURE.private.json').read_text())
    assert failure['actual_numpy_solves'] == dict(attempted=0, completed=0)
    assert failure['input_io']['DEV']['files'] == 0
    assert not (output / 'TRAIN_FROZEN.private.json').exists()


def test_partial_numpy_solves_are_actual_and_original_solver_restored(inputs, admitted, tmp_path, monkeypatch):
    spec, output = worker_spec(inputs, tmp_path)
    actual = np.linalg.solve
    calls = []
    def fail_fourth(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4: raise np.linalg.LinAlgError('SYNTHETIC_SOLVE_FAILURE')
        return actual(*args, **kwargs)
    monkeypatch.setattr(np.linalg, 'solve', fail_fourth)
    with pytest.raises(np.linalg.LinAlgError, match='SOLVE_FAILURE'): cli.run(spec, output)
    assert np.linalg.solve is fail_fourth
    failure = json.loads((output / 'FAILURE.private.json').read_text())
    assert failure['actual_numpy_solves'] == dict(attempted=4, completed=3)
    assert failure['input_io']['DEV']['files'] == 0 and failure['optimizer_updates'] == 0


@pytest.mark.parametrize('failure_stage', ['save', 'reload', 'exact', 'freeze_fsync', 'STOP_after_fit'])
def test_freeze_failures_never_touch_DEV(inputs, admitted, tmp_path, monkeypatch, failure_stage):
    spec, output = worker_spec(inputs, tmp_path)
    def fail(*args, **kwargs): raise RuntimeError('SYNTHETIC_' + failure_stage)
    if failure_stage == 'save': monkeypatch.setattr(TrajectoryResponseStudy, 'save', fail)
    elif failure_stage == 'reload': monkeypatch.setattr(TrajectoryResponseStudy, 'load', fail)
    elif failure_stage == 'exact': monkeypatch.setattr(TrajectoryResponseStudy, 'verify_reload', fail)
    elif failure_stage == 'freeze_fsync': monkeypatch.setattr(cli, '_fsync_directory', fail)
    else:
        fit = TrajectoryResponseStudy.fit
        def stopping(model, *args, **kwargs):
            result = fit(model, *args, **kwargs)
            (tmp_path / 'STOP.json').write_text('{}')
            return result
        monkeypatch.setattr(TrajectoryResponseStudy, 'fit', stopping)
    with pytest.raises((RuntimeError, TimeoutError)): cli.run(spec, output)
    failure = json.loads((output / 'FAILURE.private.json').read_text())
    assert failure['input_io']['DEV']['files'] == 0
    assert failure['actual_numpy_solves'] == dict(attempted=20, completed=20)
    assert not (output / 'RESULT.private.json').exists()


def test_bad_DEV_preserves_frozen_TRAIN_and_never_reports_complete72(inputs, admitted, tmp_path):
    spec, output = worker_spec(inputs, tmp_path)
    manifest = copy.deepcopy(inputs[2])
    first_dev = next(entry for entry in manifest['episodes'] if entry['episode']['split'] == 'DEV')
    first_dev['native_result']['sha256'] = '0' * 64
    custom_manifest(spec, tmp_path, manifest)
    with pytest.raises(ValueError, match='SHA256'): cli.run(spec, output)
    freeze = json.loads((output / 'TRAIN_FROZEN.private.json').read_text())
    assert freeze['model_sha256'] == digest(output / 'TRAJECTORY_RESPONSE_NUMPY.npz')
    assert freeze['DEV_source_files_read'] == 0
    assert not (output / 'RESULT.private.json').exists()
    assert not (output / 'ACTUAL_72_REPLAY.private.npz').exists()
    failure = json.loads((output / 'FAILURE.private.json').read_text())
    assert failure['actual_numpy_solves']['completed'] == 20
    assert failure['successful_ACTUAL72_result'] is False


def test_manifest_mixed_role_order_keeps_saved_prediction_ids_aligned(inputs, admitted, tmp_path):
    spec, output = worker_spec(inputs, tmp_path)
    manifest = copy.deepcopy(inputs[2])
    rng = np.random.default_rng(8324)
    manifest['episodes'] = [manifest['episodes'][i] for i in rng.permutation(72)]
    custom_manifest(spec, tmp_path, manifest)
    cli.run(spec, output)
    with np.load(output / 'ACTUAL_72_REPLAY.private.npz', allow_pickle=False) as replay, \
            np.load(output / 'TRAIN_CANDIDATE_PREDICTIONS.private.npz', allow_pickle=False) as prediction:
        np.testing.assert_array_equal(replay['run_id'][prediction['ids']], prediction['run_id'])
        assert replay['run_id'].tolist() == [entry['episode']['run'] for entry in manifest['episodes']]


def test_STOP_after_evaluation_cannot_emit_a_success_result(inputs, admitted, tmp_path, monkeypatch):
    spec, output = worker_spec(inputs, tmp_path)
    original = cli._evaluate
    def evaluate_and_stop(model, data, ids):
        result = original(model, data, ids)
        if np.all(data['split'][ids] == 'DEV'):
            (tmp_path / 'STOP.json').write_text('{}')
        return result
    monkeypatch.setattr(cli, '_evaluate', evaluate_and_stop)
    with pytest.raises(TimeoutError): cli.run(spec, output)
    failure = json.loads((output / 'FAILURE.private.json').read_text())
    assert failure['actual_numpy_solves'] == dict(attempted=20, completed=20)
    assert (output / 'TRAIN_FROZEN.private.json').exists()
    assert not (output / 'RESULT.private.json').exists()


def test_completion_progress_failure_cannot_leave_a_success_result(inputs, admitted, tmp_path, monkeypatch):
    spec, output = worker_spec(inputs, tmp_path)
    original = cli._progress
    def fail_completion(path, stage, **values):
        if stage == 'COMPLETED':
            raise OSError('SYNTHETIC_COMPLETION_PROGRESS_FAILURE')
        return original(path, stage, **values)
    monkeypatch.setattr(cli, '_progress', fail_completion)
    with pytest.raises(OSError, match='COMPLETION_PROGRESS'): cli.run(spec, output)
    failure = json.loads((output / 'FAILURE.private.json').read_text())
    assert failure['actual_numpy_solves'] == dict(attempted=20, completed=20)
    assert (output / 'TRAIN_FROZEN.private.json').exists()
    assert not (output / 'RESULT.private.json').exists()


@pytest.mark.parametrize('kind', ['wrong_profile', 'cuda_visible', 'scope_rejected', 'disk_rejected'])
def test_no_generic_site_or_original_scope_path_bypass(inputs, admitted, tmp_path, monkeypatch, kind):
    spec, output = worker_spec(inputs, tmp_path)
    if kind == 'wrong_profile': spec['profile'] = 'wm02-fusion-structured-offline-cpu'
    elif kind == 'cuda_visible': monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    elif kind == 'scope_rejected': monkeypatch.setattr(cli, 'validate_fusion_scope', lambda spec: (_ for _ in ()).throw(PermissionError('scope')))
    else: monkeypatch.setattr(cli, 'local_path', lambda *args: (_ for _ in ()).throw(PermissionError('disk')))
    with pytest.raises(PermissionError): cli.run(spec, output)
    assert not (output / 'RUN_ATTEMPT.private.json').exists()


def test_main_reuses_original_admitted_run_explicit_CPU_flags(monkeypatch, tmp_path):
    import go2arx5_vla.world_model.runtime as runtime
    captured = {}
    @contextmanager
    def admitted(path, run_id, profile, output, **kwargs):
        captured.update(profile=profile, run_id=run_id, kwargs=kwargs)
        yield {'fixture': 'SYNTHETIC_ONLY'}
    monkeypatch.setattr(runtime, 'admitted_run', admitted)
    monkeypatch.setattr(cli, 'run', lambda spec, output: dict(status='SYNTHETIC_TEST_ONLY', collector_source_sha='a'*40,
        fit_source_sha='f'*40, actual_TRAIN_runs=48, actual_DEV_runs=24, actual_numpy_solves={},
        optimizer_updates=0, wall_s=0., CPU_s=0.))
    assert cli.main(['--spec', str(tmp_path / 'fixture.json'), '--output', str(tmp_path / 'fit-01')]) == 0
    assert captured['profile'] == cli.PROFILE and captured['kwargs'] == dict(physics=False, gpu=False)
