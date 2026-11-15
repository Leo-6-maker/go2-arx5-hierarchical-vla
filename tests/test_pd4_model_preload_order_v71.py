"""Only the typed PD4 startup seam changes; no simulated score is produced."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from go2arx5_vla.world_model import pd4_student_v71 as pd4
from go2arx5_vla.world_model import task_learning_student_batch_v1 as batch
from scripts.wm02.sla_pd4_student_queue_v71 import host_capacity_admission


def setup(tmp_path):
    native, neural, stage, campaign = [tmp_path / p for p in ('native', 'gpu', 'stage', 'campaign')]
    for p in (native, neural, stage, campaign, neural / 'fit-01'):
        p.mkdir()
    (neural / 'EXECUTION_SPEC.json').write_text('{}')
    policy = {'arm': 'PD4_EXECUTED_REF', 'package': {'sha256': 'a' * 64}}
    values = dict(native=dict(task_learning_role=pd4.PROTOCOL, registered_root=str(native),
        campaign_root=str(campaign), deadline_epoch=100., **{pd4.KEY: {}}),
        gpu=dict(core_local_protocol=pd4.PROTOCOL, registered_root=str(neural),
        target_gr00t_policy=policy, deadline_epoch=100., **{pd4.KEY: {}}))
    loaded = dict(actual_SLA_parent_loads=1, optimizer_loaded=False, teacher_loaded=False,
                  training_cache_used=False, deployed_LoRA_retained=True)
    return values, stage, neural, loaded


def test_waits_for_exact_load_without_starting_native_or_query(tmp_path, monkeypatch):
    values, stage, neural, loaded = setup(tmp_path)
    sleeps = []
    monkeypatch.setattr(batch.time, 'time', lambda: 1.)
    def load(seconds):
        sleeps.append(seconds)
        (neural / 'fit-01/MODEL_LOAD_INTENT.private.json').write_text(json.dumps(
            dict(policy=values['gpu']['target_gr00t_policy'], retries=False)))
        (neural / 'fit-01/MODEL_LOADED.private.json').write_text(json.dumps(loaded))
    monkeypatch.setattr(batch.time, 'sleep', load)
    batch.await_pd4_model_before_native(values, SimpleNamespace(poll=lambda: None), stage, 3)
    assert sleeps == [.5]
    native = Path(values['native']['registered_root'])
    receipt = json.loads((native / 'SLA_MODEL_READY_BEFORE_NATIVE.private.json').read_text())
    assert receipt['actual_parent_loads'] == 1 and receipt['actual_queries'] == receipt['physical_requests'] == 0
    assert not (native / 'SUPERVISOR.private.json').exists()
    assert not (neural / 'fit-01/queries').exists()
    progress = json.loads((stage / 'PROGRESS.private.json').read_text())
    assert progress['completed'] == progress['physical_requests'] == 3


def test_failed_gpu_cannot_start_native(tmp_path, monkeypatch):
    values, stage, _, _ = setup(tmp_path)
    monkeypatch.setattr(batch.time, 'time', lambda: 1.)
    with pytest.raises(ValueError, match='GPU_EXIT_BEFORE_NATIVE_START'):
        batch.await_pd4_model_before_native(values, SimpleNamespace(poll=lambda: 1), stage, 0)
    assert not (Path(values['native']['registered_root']) / 'SUPERVISOR.private.json').exists()


@pytest.mark.parametrize('change', ['role', 'receipt', 'query', 'deadline'])
def test_rejects_other_role_wrong_receipt_prior_query_and_expired_deadline(tmp_path, monkeypatch, change):
    values, stage, neural, loaded = setup(tmp_path)
    monkeypatch.setattr(batch.time, 'time', lambda: 1.)
    (neural / 'fit-01/MODEL_LOAD_INTENT.private.json').write_text(json.dumps(
        dict(policy=values['gpu']['target_gr00t_policy'], retries=False)))
    if change == 'role':
        values['native']['task_learning_role'] = 'OTHER_ROLE'
    elif change == 'receipt':
        loaded['actual_SLA_parent_loads'] = 2
    elif change == 'query':
        (neural / 'fit-01/queries').mkdir()
    else:
        values['native']['deadline_epoch'] = 0.
    (neural / 'fit-01/MODEL_LOADED.private.json').write_text(json.dumps(loaded))
    with pytest.raises((ValueError, TimeoutError)):
        batch.await_pd4_model_before_native(values, SimpleNamespace(poll=lambda: None), stage, 0)


def test_host_admission_default_and_measured_serial_startup():
    assert host_capacity_admission({}) == 32*1024**3
    spec=dict(minimum_host_available_bytes=24*1024**3,startup_schedule='SLA_MODEL_READY_BEFORE_NATIVE_V71',
        host_capacity_evidence=dict(schema='SLA_CLOSED_CORE_MEASURED_HOST_CAPACITY_V71',rows=[
            dict(unit='sla-pd4-matched500-core-v71-20261115-01-pd4_executed_ref',
                result_sha256='31f8223b1afeaadfeef4002f35e16a16af52dcd122174539a0c5103c8416ad8c',MainPID='0',SubState='exited',peak_bytes=15233007616),
            dict(unit='sla-pd4-student-train4-v71-20261115-04',
                result_sha256='cf8fb33818cb950fcd2d27c6097837a61a89de7d1ca4c38423cdf0a94039dc3d',MainPID='0',SubState='failed',peak_bytes=10936266752)]))
    assert host_capacity_admission(spec) == 24*1024**3
    spec['startup_schedule']='CONCURRENT_ORIGINAL_V71'
    with pytest.raises(PermissionError):host_capacity_admission(spec)


def test_host_admission_rejects_missing_live_or_unbounded_capacity_evidence():
    with pytest.raises(PermissionError):host_capacity_admission(dict(minimum_host_available_bytes=24*1024**3))
    with pytest.raises(PermissionError):host_capacity_admission(dict(minimum_host_available_bytes=16*1024**3))
