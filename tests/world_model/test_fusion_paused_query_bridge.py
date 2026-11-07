"""Registered two-scope seam doubles; never launches a worker or a world."""
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from scripts.pi0292.relay_protocol import unseal
from scripts.wm02 import infer_fusion_paused_vla as cli
from go2arx5_vla.world_model import fusion_paused_query_bridge as b
from go2arx5_vla.world_model import fusion_paused_inference as p
from go2arx5_vla.world_model import fusion_scope
from go2arx5_vla.world_model.fusion_observation import transform_hash
from go2arx5_vla.world_model.storage import digest, write_json
from tests.world_model.test_fusion_paused_inference import make_spec, rewrite_spec, install_fake_draw


def setup_pair(tmp_path, monkeypatch):
    spec, adapter, _, events = make_spec(tmp_path,monkeypatch,capture_now=False)
    native_root, gpu_root = Path(adapter.spec['registered_root']), Path(spec['registered_root'])
    pair_path = tmp_path/'QUERY_PAIR.private.json'
    pair = dict(schema=b.SCHEMA,pair_uuid='22345678-1234-4234-8234-123456789abc',query_uuid=spec['query_uuid'],
        native_root=str(native_root),native_run_id='native-01',native_boundary_step=3703,capture_index=0,
        gpu_root=str(gpu_root),task_sha256=spec['paused_task_sha256'],scene_sha256=adapter.fixture_binding.scene_sha256,
        frame_map_sha256=adapter.fixture_binding.expected_transform_sha256,seed=spec['seed'],
        checkpoint_manifest_sha256=p.CHECKPOINT,source_sha=spec['source_sha'],deadline_epoch=spec['deadline_epoch'],
        max_queries=1,native_integrations_by_bridge=0)
    write_json(pair_path,pair)
    common = dict(owner='EXISTING_LOCAL_PC',host='SYNTHETIC_TEST',uid=2009,
        registry=str(tmp_path/'REGISTRY.private.json'),pilot1_binding=str(tmp_path/'P1/BOUND.json'),
        pilot1_binding_sha256='b'*64,paused_query_pair=str(pair_path),
        paused_query_pair_sha256=digest(pair_path))
    spec.update(common,role='FRESH_SYNTHETIC_GPU',snapshot_binding_mode=b.MODE,
        recorded_observation_sha256=None,snapshot_wait_limit_s=40,lease=str(gpu_root/'control/LEASE.json'))
    producer = dict(adapter.spec,**common)
    producer.update(physics=True,max_ticks=3703,source_sha=spec['source_sha'],source_root=spec['source_root'],
        role='FRESH_SYNTHETIC_NATIVE',lease=str(native_root/'control/LEASE.json'),
        paused_query_gpu_spec=str(gpu_root/'EXECUTION_SPEC.json'),
        paused_target_task_sha256=spec['paused_task_sha256'],deadline_epoch=spec['deadline_epoch']+1,
        paused_frame_map_sha256=pair['frame_map_sha256'],
        campaign_root=spec['campaign_root'])
    (native_root/'EXECUTION_SPEC.json').write_text(json.dumps(producer))
    adapter.spec = producer
    quota = (adapter.output/'quota.binding').read_text().splitlines()
    quota[4] = '3703'
    (adapter.output/'quota.binding').write_text('\n'.join(quota)+'\n')
    adapter.fixture_binding = replace(adapter.fixture_binding,target_runtime_sha256=digest(native_root/'EXECUTION_SPEC.json'))
    spec.update(producer_native_spec=str(native_root/'EXECUTION_SPEC.json'),
                producer_native_spec_sha256=digest(native_root/'EXECUTION_SPEC.json'))
    rewrite_spec(spec)
    entries = []
    for item in (producer,spec):
        root = Path(item['registered_root']); (root/'control').mkdir()
        write_json(item['lease'],dict(status='ACTIVE',spec_sha256=digest(root/'EXECUTION_SPEC.json'),run_uuid=root.name))
        entries.append(dict(path=str(root),role=item['role'],source_sha=item['source_sha'],owner=item['owner'],local_lease=item['lease']))
    write_json(spec['registry'],dict(entries=entries))
    # Only external admission is doubled; local file, pair, lease, registry and
    # actual pause/clock/packet/capture/return bindings stay real.
    calls = []
    monkeypatch.setattr(fusion_scope,'validate',lambda value:calls.append(value['registered_root']))
    return spec, adapter, pair, events, calls


def close_gpu(spec, *, code=0):
    root = Path(spec['registered_root'])
    write_json(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=None))
    lease = json.loads(Path(spec['lease']).read_text())
    lease.update(status='CLOSED',worker_exit_code=code)
    Path(spec['lease']).write_text(json.dumps(lease))


def test_preregistered_pair_validates_before_snapshot_exists_without_circular_sha(tmp_path,monkeypatch):
    spec,adapter,pair,events,calls = setup_pair(tmp_path,monkeypatch)
    p.validate_spec(spec)
    assert not Path(spec['recorded_observation']).exists()
    checked,producer = b.validate_pair_spec(spec)
    assert checked == pair and producer == adapter.spec
    assert 'recorded_observation_sha256' not in producer
    assert 'producer_native_spec_sha256' not in pair
    assert events.saves == []
    proof = b.preflight_pair(Path(spec['registered_root'])/'EXECUTION_SPEC.json',adapter.spec)
    assert proof['snapshot_exists'] is False and not proof['native_execution_authorized']
    assert proof['native_boundary_step'] == 3703
    json.dumps(proof,allow_nan=False)


def test_native_capture_single_query_return_checks_original_closed_worker_and_actual_pause(tmp_path,monkeypatch):
    spec,adapter,pair,native_events,calls = setup_pair(tmp_path,monkeypatch)
    draw = install_fake_draw(monkeypatch)
    def execute_registered_job_once(seconds):
        assert seconds == .05
        ready = Path(spec['recorded_observation']).parent/'PAIR_SNAPSHOT_READY.private.json'
        assert ready.is_file()
        output = Path(spec['registered_root'])/'fit-01'; output.mkdir()
        cli.infer(spec,output)
        close_gpu(spec)
    monkeypatch.setattr(b.time,'sleep',execute_registered_job_once)
    result = b.capture_and_wait_query(adapter,binding=adapter.fixture_binding,task=adapter.fixture_task,
        task_sha256=spec['paused_task_sha256'],gpu_spec_path=Path(spec['registered_root'])/'EXECUTION_SPEC.json')
    assert draw.queries == draw.loads == 1 and native_events.saves == [(0,3703)]
    assert result['native_integrations_by_bridge'] == 0 and not result['native_execution_authorized']
    assert not result['online_action_execution']
    assert result['query_result']['model_result_checked'] is True
    assert calls and len(calls) >= 3
    json.dumps(result,allow_nan=False)
    body = unseal(json.loads(Path(spec['recorded_observation']).read_text()))
    assert body['native_step'] == 3703 and body['native_time_s'] == 3703*.002
    assert json.loads((adapter.output/'pause_ack.json').read_text())['clock_paused'] is True
    with pytest.raises((FileExistsError,ValueError)):
        b.capture_and_wait_query(adapter,binding=adapter.fixture_binding,task=adapter.fixture_task,
            task_sha256=spec['paused_task_sha256'],gpu_spec_path=Path(spec['registered_root'])/'EXECUTION_SPEC.json')
    assert draw.queries == 1


@pytest.mark.parametrize('fault',['producer_sha','pair_bytes','seed','boundary','frame','task','root','snapshot','owner','lease','registry'])
def test_bad_pair_or_original_registration_rejects_before_capture(tmp_path,monkeypatch,fault):
    spec,adapter,pair,events,_ = setup_pair(tmp_path,monkeypatch)
    if fault == 'producer_sha': spec['producer_native_spec_sha256'] = 'e'*64
    elif fault == 'pair_bytes': Path(spec['paused_query_pair']).write_text('{}')
    elif fault == 'seed': spec['seed'] += 1
    elif fault == 'boundary':
        pair['native_boundary_step'] = 3728
        Path(spec['paused_query_pair']).write_text(json.dumps(pair))
        spec['paused_query_pair_sha256'] = digest(spec['paused_query_pair'])
    elif fault == 'frame':
        changed = adapter.fixture_binding.T_source_from_target.copy(); changed[0,3] += .001
        adapter.fixture_binding = replace(adapter.fixture_binding,T_source_from_target=changed,
                                         expected_transform_sha256=transform_hash(changed))
    elif fault == 'task': spec['paused_task_sha256'] = 'e'*64
    elif fault == 'root': spec['registered_root'] = str(tmp_path/'other-gpu')
    elif fault == 'snapshot': spec['recorded_observation'] = str(tmp_path/'other.json')
    elif fault == 'owner': spec['owner'] = 'OTHER_OWNER'
    elif fault == 'lease':
        lease = json.loads(Path(spec['lease']).read_text()); lease['status'] = 'CLOSED'
        Path(spec['lease']).write_text(json.dumps(lease))
    elif fault == 'registry': Path(spec['registry']).write_text('{"entries":[]}')
    # Retain the original GPU file hash/lease when testing an in-memory forged
    # callback input; the function always rereads its fixed immutable file.
    if fault in ('producer_sha','seed','boundary','task','root','snapshot','owner'):
        (Path(pair['gpu_root'])/'EXECUTION_SPEC.json').write_text(json.dumps(spec))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):
        b.capture_and_wait_query(adapter,binding=adapter.fixture_binding,task=adapter.fixture_task,
            task_sha256=spec['paused_task_sha256'],gpu_spec_path=Path(pair['gpu_root'])/'EXECUTION_SPEC.json')
    assert events.saves == []


@pytest.mark.parametrize('fault',['capture_failed','producer_closed','stop','timeout'])
def test_gpu_wait_terminates_without_model_or_query_on_failed_missing_capture(tmp_path,monkeypatch,fault):
    spec,adapter,pair,events,_ = setup_pair(tmp_path,monkeypatch)
    draw = install_fake_draw(monkeypatch)
    if fault == 'capture_failed':
        folder = Path(spec['recorded_observation']).parent; folder.mkdir(parents=True)
        (folder/'CAPTURE_FAILED.private.json').write_text('{}')
    elif fault == 'producer_closed':
        path = Path(adapter.spec['lease']); lease = json.loads(path.read_text()); lease['status']='CLOSED'
        path.write_text(json.dumps(lease))
    elif fault == 'stop': (Path(spec['registered_root'])/'STOP.json').write_text('{}')
    else:
        spec['snapshot_wait_limit_s'] = .001
        monkeypatch.setattr(b.time,'sleep',lambda seconds:None)
    output = Path(spec['registered_root'])/'fit-01'; output.mkdir()
    with pytest.raises(ValueError): cli.infer(spec,output)
    assert draw.loads == draw.queries == 0 and events.saves == []
    assert (output/'FAILED.private.json').is_file()


def test_changed_native_packet_while_waiting_rejects_without_install_or_requery(tmp_path,monkeypatch):
    spec,adapter,pair,events,_ = setup_pair(tmp_path,monkeypatch)
    def advance_illegally(seconds):
        (adapter.output/'step_quota').write_text('3728\n')
    monkeypatch.setattr(b.time,'sleep',advance_illegally)
    with pytest.raises(ValueError,match='PAUSE_ACK_AND_EXACT_CURRENT_STEP_QUOTA'):
        b.capture_and_wait_query(adapter,binding=adapter.fixture_binding,task=adapter.fixture_task,
            task_sha256=spec['paused_task_sha256'],gpu_spec_path=Path(spec['registered_root'])/'EXECUTION_SPEC.json')
    assert events.saves == [(0,3703)]
    assert not (Path(spec['registered_root'])/'fit-01/GENERATION_INTENT.private.json').exists()
