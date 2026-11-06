"""One-draw and captured-file tests; no real model, CUDA, renderer or physics."""
import copy
import json
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal
from scripts.wm02 import infer_fusion_paused_vla as cli
from go2arx5_vla.world_model import fusion_paused_inference as p
from go2arx5_vla.world_model.storage import digest, write_json
from tests.world_model.test_fusion_paused_observation import capture_fixture, capture, QUERY

PROJECT = Path(__file__).resolve().parents[2]


def make_spec(tmp_path, monkeypatch, *, capture_now=True):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    adapter.fixture_binding, adapter.fixture_task = binding, task
    record = capture(adapter, binding, task) if capture_now else None
    capsule = tmp_path/'capsule'
    for name in p.SOURCE_MEMBERS:
        target = capsule/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT/name).read_bytes())
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'processor').mkdir()
    (checkpoint/'model.safetensors').write_bytes(b'FAKE MOCK WEIGHTS NEVER LOAD A REAL MODEL')
    (checkpoint/'processor/config.json').write_text('{"mock": true}')
    files = {name: digest(checkpoint/name) for name in ('model.safetensors','processor/config.json')}
    inventory = document_hash(files)
    monkeypatch.setattr(p, 'CHECKPOINT_INVENTORY', inventory)
    monkeypatch.setattr(cli, 'CHECKPOINT_INVENTORY', inventory)
    gpu_root = tmp_path/'gpu'
    gpu_root.mkdir()
    path = adapter.output/'PAUSED_OBSERVATIONS'/QUERY/'PAUSED_OBSERVATION.private.json'
    spec = dict(kind=p.KIND, profile=p.PROFILE, execution_site='CORE_FUSION_NEURAL',
        inference_mode=p.MODE, observation_mode=p.INPUT_MODE, physics=False, device='cuda', max_queries=1,
        numeric_mode='ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST', GR00T_updates=0, WM_updates=0, Actor_updates=0,
        checkpoint_manifest_sha256=p.CHECKPOINT, checkpoint_files=files, checkpoint=str(checkpoint),
        checkpoint_parameter_bytes=8, allocator_limit_bytes=16,
        GPU_uuid='GPU-845348f1-1511-dc19-0877-cc97658e3edd', source_sha='a'*40,
        seed=366103, query_uuid=QUERY, recorded_observation=str(path),
        recorded_observation_sha256=digest(path) if capture_now else 'e'*64,
        paused_task_sha256=document_hash(task), snapshot_binding_mode=p.BINDING_MODE,
        source_root=str(capsule), source_files={name:digest(capsule/name) for name in p.SOURCE_MEMBERS},
        registered_root=str(gpu_root), campaign_root=str(tmp_path/'campaign'), deadline_epoch=time.time()+60,
        neural_bindings=dict(source='MOCK-OFFICIAL', cosmos_hub='MOCK-CACHE'),
        wall_limit_s=50)
    write_json(gpu_root/'EXECUTION_SPEC.json', spec)
    return spec, adapter, record, events


def rewrite_spec(spec):
    (Path(spec['registered_root'])/'EXECUTION_SPEC.json').write_text(json.dumps(spec))


def fake_torch():
    class Event:
        def __init__(self, **kw): assert kw == dict(enable_timing=True)
        def record(self): pass
        def elapsed_time(self, other): return 12.
    return SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        cuda=SimpleNamespace(Event=Event, synchronize=lambda:None, max_memory_allocated=lambda:32,
                             max_memory_reserved=lambda:64))


def install_fake_draw(monkeypatch, *, failure=False):
    events = SimpleNamespace(loads=0, queries=0, payloads=[], processor_decode_calls=0)
    instance = SimpleNamespace(identity=None, processor=SimpleNamespace())
    def loader(spec, output):
        events.loads += 1
        instance.identity = dict(checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'], files=spec['checkpoint_files'])
        return instance, fake_torch()
    monkeypatch.setattr(cli, '_load_frozen', loader)
    from scripts.pi0294.resident import ResidentModel
    def generate(self, payload, seed):
        assert self is instance
        events.queries += 1
        events.payloads.append(copy.deepcopy(payload))
        assert seed == 366103
        if failure: raise RuntimeError('MOCK_DRAW_FAILURE_DO_NOT_RETRY')
        # This is the existing generation primitive's decoded result. The entry
        # must preserve every value, including normalized coordinates outside 1.
        physical = np.arange(400,dtype=np.float64).reshape(40,10)/1000
        normalized = np.full((1,40,132),1.25,dtype=np.float32)
        return dict(physical_H40=physical, normalized=normalized,
            noise=dict(shape=[1,40,132],dtype='torch.float32',sha256='1'*64),
            identity=self.identity,extra_correction=False,physics_integrations=0,
            normalized_outside_unit_components=400)
    monkeypatch.setattr(ResidentModel, 'generate', generate)
    return events


def test_reads_every_actual_capture_member_and_preserves_sidecar(tmp_path, monkeypatch):
    spec, adapter, record, events = make_spec(tmp_path, monkeypatch)
    payload, identity, returned = p.read_payload(spec)
    assert returned == record
    assert identity['files'] == unseal(record)['files']
    assert identity['world']['native_output'] == str(adapter.output)
    assert identity['snapshot_binding_mode'] == p.BINDING_MODE
    assert identity['mapped_payload_sha256'] == document_hash(payload)
    assert 'payload' not in identity['source_policy_view']
    assert not identity['online_closed_loop'] and not identity['native_execution_authorized']
    assert identity['native_caller_must_verify_still_paused']
    assert events.saves == [(0,3703)]


@pytest.mark.parametrize('fault', ['png','calibration','rawstate','nested','missing','envelope','task','query','clock','source'])
def test_rejects_missing_or_changed_bytes_and_sealed_cross_binding(tmp_path, monkeypatch, fault):
    spec, _, record, _ = make_spec(tmp_path, monkeypatch)
    path = Path(spec['recorded_observation'])
    folder = path.parent
    body = unseal(record)
    if fault in ('png','calibration','rawstate','nested','missing'):
        name = {'png':'images/000000-body_rgb.png', 'calibration':'images/000000-calibration.json',
                'rawstate':'STATE.private.bin', 'nested':'BOUND_SHADOW_ABI.private.json', 'missing':'images/000000-wrist_rgb.png'}[fault]
        if fault == 'missing': (folder/name).unlink()
        else: (folder/name).write_bytes(b'TAMPERED')
    elif fault == 'envelope': path.write_text('{}')
    elif fault == 'task': spec['paused_task_sha256'] = 'e'*64
    elif fault == 'query': spec['query_uuid'] = '12345678-1234-4234-8234-123456789abd'
    elif fault == 'clock':
        body['native_step'] += 25
        path.write_text(json.dumps(sealed(body)))
        spec['recorded_observation_sha256'] = digest(path)
    else: (Path(spec['source_root'])/p.SOURCE_MEMBERS[0]).write_text('CHANGED')
    with pytest.raises((ValueError,FileNotFoundError,KeyError)): p.read_payload(spec)


def test_uses_original_generate_once_retains_unclipped_raw_and_requires_actual_pause_after_return(tmp_path, monkeypatch):
    spec, adapter, _, native_events = make_spec(tmp_path, monkeypatch)
    events = install_fake_draw(monkeypatch)
    output = Path(spec['registered_root'])/'fit-01'
    output.mkdir()
    result = cli.infer(spec, output)
    assert events.loads == events.queries == 1
    assert result['noise']['sha256'] == '1'*64
    assert result['numeric_mode'] == spec['numeric_mode']
    assert result['precision']['cuda_matmul_allow_tf32'] is False
    assert not result['online_closed_loop'] and not result['native_execution_authorized']
    assert result['model_cold_load_wall_s'] >= 0 and result['query_wall_s'] >= 0
    assert result['simulated_time_advanced_s'] == 0 and result['reference_TTL_rebased'] is False
    assert result['returned_action_is_live_reference'] is False
    with np.load(output/'REAL_VLA_ACTION.npz',allow_pickle=False) as raw:
        assert raw['physical_H40'].shape == (40,10) and raw['physical_H40'][-1,-1] == .399
        assert np.all(raw['normalized'] == 1.25)
    checked = p.verify_paused_query_return(adapter,spec,output/'RESULT.private.json')
    assert checked['model_result_checked'] and not checked['native_execution_authorized']
    native_events.pid += 1
    with pytest.raises(ValueError,match='SAME_ACTUAL_OWNED_SIMULATOR'):
        p.verify_paused_query_return(adapter,spec,output/'RESULT.private.json')
    with pytest.raises(FileExistsError): cli.infer(spec,output)
    assert events.queries == 1


def test_draw_failure_is_retained_and_never_retried(tmp_path, monkeypatch):
    spec, _, _, _ = make_spec(tmp_path, monkeypatch)
    events = install_fake_draw(monkeypatch,failure=True)
    output = Path(spec['registered_root'])/'fit-01'
    output.mkdir()
    with pytest.raises(RuntimeError,match='DO_NOT_RETRY'): cli.infer(spec,output)
    failure = json.loads((output/'FAILED.private.json').read_text())
    assert failure['generation_invocations'] == 1 and failure['retry_authorized'] is False
    assert failure['stage'] == 'ONE_GENERATION' and failure['native_integrations'] == 0
    with pytest.raises(FileExistsError): cli.infer(spec,output)
    assert events.loads == events.queries == 1


def test_modified_weight_fails_before_load_or_query(tmp_path, monkeypatch):
    spec, _, _, _ = make_spec(tmp_path, monkeypatch)
    events = install_fake_draw(monkeypatch)
    (Path(spec['checkpoint'])/'model.safetensors').write_bytes(b'CHANGED')
    output = Path(spec['registered_root'])/'fit-01'
    output.mkdir()
    with pytest.raises(ValueError,match='PROCESSOR_BYTES_CHANGED'): cli.infer(spec,output)
    assert events.loads == events.queries == 0


@pytest.mark.parametrize('fault',['none','uuid','devices','headroom','bf16','bytes','loading','steps','category'])
def test_loader_exact_saved_processor_fp32_fourstep_uuid_and_full_loading_info(tmp_path, monkeypatch, fault):
    spec, _, _, _ = make_spec(tmp_path, monkeypatch)
    from scripts.pi0292 import train_adapter
    from scripts.pi0294 import local_backbone
    called = []
    monkeypatch.setattr(train_adapter,'official_import',lambda value:called.append(('official',value)))
    monkeypatch.setattr(local_backbone,'local_model_name',lambda *args:'MOCK-SAVED-ALIAS')
    torch = fake_torch()
    torch.float32 = 'torch.float32'
    torch.cuda.device_count = lambda:2 if fault == 'devices' else 1
    total = 2*1024**3
    torch.cuda.get_device_properties = lambda n:SimpleNamespace(
        uuid='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad' if fault == 'uuid' else spec['GPU_uuid'],total_memory=total)
    torch.cuda.mem_get_info = lambda n:(spec['allocator_limit_bytes']+1024**3 if fault=='headroom' else total,total)
    torch.cuda.set_per_process_memory_fraction = lambda value,n:called.append(('allocator',value,n))
    torch.cuda.reset_peak_memory_stats = lambda:None
    monkeypatch.setitem(sys.modules,'torch',torch)
    parameter = SimpleNamespace(numel=lambda:2,element_size=lambda:4,
        dtype='torch.bfloat16' if fault == 'bf16' else torch.float32,requires_grad=False)
    if fault == 'bytes': spec['checkpoint_parameter_bytes'] = 9
    model = SimpleNamespace(parameters=lambda:[parameter],requires_grad_=lambda value:None,
                            action_head=SimpleNamespace(num_inference_timesteps=1 if fault == 'steps' else 4))
    model.to = lambda device:model
    model.eval = lambda:model
    def load_model(path, **kw):
        called.append(('model',path,kw))
        return model,dict(missing_keys=['weight'] if fault == 'loading' else [],
                          unexpected_keys=[],mismatched_keys=[],error_msgs=[])
    processor = SimpleNamespace(embodiment_id_mapping={'new_embodiment':9 if fault == 'category' else 10},eval=lambda:None)
    def load_processor(path, **kw):
        called.append(('processor',path,kw))
        return processor
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load_model)))
    import types
    for name in ['gr00t','gr00t.model','gr00t.model.gr00t_n1d7','gr00t.model.gr00t_n1d7.gr00t_n1d7']:
        module = types.ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules,name,module)
        if '.' in name:
            parent, leaf = name.rsplit('.',1)
            setattr(sys.modules[parent],leaf,module)
    monkeypatch.setitem(sys.modules,'gr00t.model.gr00t_n1d7.processing_gr00t_n1d7',
        SimpleNamespace(Gr00tN1d7Processor=SimpleNamespace(from_pretrained=load_processor)))
    output = tmp_path/'loader'; output.mkdir()
    if fault != 'none':
        with pytest.raises(ValueError): cli._load_frozen(spec,output)
        return
    instance, returned = cli._load_frozen(spec,output)
    assert returned is torch and instance.processor is processor
    model_call = next(row for row in called if row[0]=='model')
    assert model_call[2]['load_bf16'] is False and model_call[2]['backbone_trainable_params_fp32'] is True
    assert model_call[2]['output_loading_info'] is True
    processor_call = next(row for row in called if row[0]=='processor')
    assert processor_call[1] == Path(spec['checkpoint'])/'processor'
    assert processor_call[2]['transformers_loading_kwargs']['revision'] == '9ce19a195e423419c349abfc86fd07178b230561'
    assert ('allocator',spec['allocator_limit_bytes']/total,0) in called
    assert (output/'MODEL_LOADING_INFO.private.json').exists()
