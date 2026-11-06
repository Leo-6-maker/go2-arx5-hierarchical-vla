"""Synthetic runtime admission only; no source replay, fit, native or GPU work."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from scripts.pi0292 import durable
from scripts.wm02 import fit_fusion_trajectory_response as cli
from go2arx5_vla.world_model import fusion_scope, pilot1_storage, runtime
from go2arx5_vla.world_model.storage import digest


@pytest.fixture
def admitted_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'SYNTHETIC_RESPONSE_CPU_WORKER'
    root.mkdir()
    source = Path(cli.__file__).resolve().parents[2]
    for name in ('PANEL.json', 'SOURCE_MANIFEST.json', 'RAW_ACTION.npz'):
        (root / name).write_text('SYNTHETIC_ADMISSION_ONLY_NOT_REPLAY')
    spec = dict(profile=cli.PROFILE, kind=cli.KIND, execution_site=cli.SITE,
        host=os.uname().nodename, uid=os.getuid(), source_root=str(source), source_sha='a' * 40,
        source_files={cli.ENTRY_FILE: digest(Path(cli.__file__))}, registered_root=str(root),
        role='SYNTHETIC_FRESH_RESPONSE_CPU', registry_snapshot=str(root / 'REGISTRY.json'),
        lease=str(root / 'LEASE.json'), cpus=[min(os.sched_getaffinity(0))],
        deadline_epoch=time.time() + 600, wall_limit_s=300., device='cpu', physics=False,
        recipe=cli.RECIPE, numeric_mode='NUMPY_FLOAT64', ridge=cli.RIDGE,
        GR00T_updates=0, Actor_updates=0, WBC_changes=False,
        information='PREDECLARED_PROGRAM_CONDITIONAL', noise_fraction=.25, noise_floor_mm=.001,
        python=sys.executable, data=str(root), cache=str(root), logs=str(root), tmp=str(root),
        panel=str(root / 'PANEL.json'), source_manifest=str(root / 'SOURCE_MANIFEST.json'),
        raw_action=str(root / 'RAW_ACTION.npz'), campaign_root=str(root))
    spec_path = root / 'EXECUTION_SPEC.json'
    events = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    def original_scope(value):
        events.append(('original_scope', (root / 'fit-01').exists()))
        return {'synthetic_external_scope': True}
    def original_local_path(value, path):
        events.append(('local_path', str(path)))
        return Path(path)
    monkeypatch.setattr(fusion_scope, 'validate', original_scope)
    monkeypatch.setattr(cli, 'validate_fusion_scope', original_scope)
    monkeypatch.setattr(pilot1_storage, 'local_path', original_local_path)
    monkeypatch.setattr(cli, 'local_path', original_local_path)
    class Owner:
        def __init__(self, path): events.append(('owner', str(path)))
        def close(self): events.append(('owner_closed', True))
    monkeypatch.setattr(durable, 'OwnerLock', Owner)
    monkeypatch.setattr(runtime.os, 'sched_setaffinity', lambda *args: events.append(('affinity', args)))
    def seal():
        spec_path.write_text(json.dumps(spec))
        (root / 'REGISTRY.json').write_text(json.dumps(dict(entries=[dict(
            role=spec['role'], path=str(root), source_sha=spec['source_sha'])])))
        (root / 'LEASE.json').write_text(json.dumps(dict(status='ACTIVE', run_uuid=root.name,
            spec_sha256=digest(spec_path))))
    seal()
    return spec, spec_path, root / 'fit-01', events, seal


def test_dedicated_cpu_runtime_calls_original_scope_paths_and_full_CLI_before_yield(admitted_fixture):
    spec, path, output, events, _ = admitted_fixture
    with runtime.admitted_run(path, 'fit-01', cli.PROFILE, output, physics=False, gpu=False) as admitted:
        assert admitted == spec
        assert (output / 'STARTED.json').exists()
        assert [value for key, value in events if key == 'original_scope'] == [False, True]
        observed = {value for key, value in events if key == 'local_path'}
        for name in ('source_root', 'registered_root', 'python', 'data', 'cache', 'logs', 'tmp',
                     'panel', 'source_manifest', 'raw_action'):
            assert spec[name] in observed
        assert str(output) in observed
        assert not (output / 'RUN_ATTEMPT.private.json').exists()
    assert events[-1] == ('owner_closed', True)


@pytest.mark.parametrize('fault', ['unknown_site', 'missing_site', 'other_profile', 'other_kind',
                                  'physics_arg', 'physics_spec', 'gpu_arg', 'visible_cuda', 'unset_cuda'])
def test_identity_cannot_fall_through_unknown_site_or_enable_compute(admitted_fixture, monkeypatch, fault):
    spec, path, output, events, seal = admitted_fixture
    flags = dict(physics=False, gpu=False)
    if fault == 'unknown_site': spec['execution_site'] = 'CORE_UNRECOGNIZED_CPU'
    elif fault == 'missing_site': del spec['execution_site']
    elif fault == 'other_profile': spec['profile'] = 'GENERIC_CPU'
    elif fault == 'other_kind': spec['kind'] = 'ARX_RESIDUAL'
    elif fault == 'physics_arg': flags['physics'] = True
    elif fault == 'physics_spec': spec['physics'] = True
    elif fault == 'gpu_arg': flags['gpu'] = True
    elif fault == 'visible_cuda': monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    else: monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    seal()
    with pytest.raises(PermissionError):
        with runtime.admitted_run(path, 'fit-01', spec['profile'], output, **flags):
            pytest.fail('invalid response identity was admitted')
    assert not output.exists()
    assert not any(key in ('owner', 'affinity') for key, _ in events)


@pytest.mark.parametrize('fault', ['original_scope', 'original_disk'])
def test_original_scope_or_disk_failure_precedes_owner_output_and_caller(admitted_fixture, monkeypatch, fault):
    _, path, output, events, _ = admitted_fixture
    def reject(*args): raise PermissionError('SYNTHETIC_ORIGINAL_' + fault)
    if fault == 'original_scope': monkeypatch.setattr(fusion_scope, 'validate', reject)
    else: monkeypatch.setattr(pilot1_storage, 'local_path', reject)
    with pytest.raises(PermissionError, match='SYNTHETIC_ORIGINAL'):
        with runtime.admitted_run(path, 'fit-01', cli.PROFILE, output):
            pytest.fail('original owner/disk rejection was bypassed')
    assert not output.exists()
    assert not any(key in ('owner', 'affinity') for key, _ in events)


@pytest.mark.parametrize('fault', ['recipe', 'numeric_mode', 'device', 'noise_floor', 'entry_pin'])
def test_full_frozen_CLI_scientific_validator_precedes_STARTED_and_yield(admitted_fixture, fault):
    spec, path, output, events, seal = admitted_fixture
    if fault == 'recipe': spec['recipe'] = 'OLD_RECIPE'
    elif fault == 'numeric_mode': spec['numeric_mode'] = 'FLOAT32'
    elif fault == 'device': spec['device'] = 'cuda'
    elif fault == 'noise_floor': spec['noise_floor_mm'] = 0.
    else: spec['source_files'][cli.ENTRY_FILE] = '0' * 64
    seal()
    with pytest.raises((PermissionError, ValueError)):
        with runtime.admitted_run(path, 'fit-01', cli.PROFILE, output):
            pytest.fail('scientific admission was bypassed')
    assert not (output / 'STARTED.json').exists()
    assert not (output / 'RUN_ATTEMPT.private.json').exists()
    if output.exists():
        assert events[-1] == ('owner_closed', True)
