from copy import deepcopy
import pytest
from scripts.pi0294.multitask.book_gpu_capacity import GIB, REQUEST, entry_requirement


def test_all_existing_training_and_dev_keep_original_24gib_entry():
    assert entry_requirement({}) == 24*GIB
    assert entry_requirement(dict(kind='V7_BOOK_M32_FINITE_LEARNING',recipe=dict(max_steps=4096))) == 24*GIB


def test_finite_native_capacity_test_does_not_admit_model_or_silently_reduce_training():
    m=dict(kind='V7_BOOK_M32_FINITE_LEARNING',GPU_capacity_qualification=deepcopy(REQUEST),
        recipe=dict(max_steps=10,capacity_qualification_only=True,limited_DiT_blocks=2,TRAIN_visits=320))
    assert entry_requirement(m)==13*GIB
    for key,value in [('max_steps',4096),('limited_DiT_blocks',0),('TRAIN_visits',10),('capacity_qualification_only',False)]:
        bad=deepcopy(m);bad['recipe'][key]=value
        with pytest.raises(PermissionError):entry_requirement(bad)
    bad=deepcopy(m);bad['GPU_capacity_qualification']['microbatch']=2
    with pytest.raises(PermissionError):entry_requirement(bad)


def test_small_gpu_inference_permission_cannot_admit_training_physics_or_unbounded_queries():
    from scripts.pi0294.multitask.book_gpu_capacity import INFERENCE_REQUEST
    from scripts.pi0294.multitask.book_phase_diagnostic import KIND
    m=dict(kind=KIND,GPU_inference_qualification=deepcopy(INFERENCE_REQUEST),limits=dict(
        model_loads=1,optimizer_updates=0,physics_integrations=0,generation_queries=96,RGB_pairs=0,source_constructors=0))
    assert entry_requirement(m)==13*GIB
    for key,value in [('optimizer_updates',10),('physics_integrations',1),('generation_queries',97),('source_constructors',1)]:
        bad=deepcopy(m);bad['limits'][key]=value
        with pytest.raises(PermissionError):entry_requirement(bad)
    for key,value in [('kind','V7_BOOK_M32_FINITE_LEARNING'),('GPU_capacity_qualification',deepcopy(REQUEST))]:
        bad=deepcopy(m);bad[key]=value
        with pytest.raises(PermissionError):entry_requirement(bad)
    bad=deepcopy(m);bad['GPU_inference_qualification']['minimum_headroom_bytes']=0
    with pytest.raises(PermissionError):entry_requirement(bad)


def test_runtime_capacity_rejects_the_first_below_one_gib_sample():
    from types import SimpleNamespace
    from scripts.pi0294.multitask.book_gpu_capacity import runtime_inference_capacity
    free = [GIB, GIB-1]
    cuda = SimpleNamespace(synchronize=lambda: None, mem_get_info=lambda: (free.pop(0), 16*GIB),
        max_memory_allocated=lambda: 8*GIB, max_memory_reserved=lambda: 9*GIB)
    state = runtime_inference_capacity(SimpleNamespace(cuda=cuda), dict(minimum_free_bytes=2*GIB, measurements=0), 1.)
    assert state['capacity_pass'] and state['measurements'] == 1
    state = runtime_inference_capacity(SimpleNamespace(cuda=cuda), state, 2.)
    assert not state['capacity_pass'] and state['minimum_free_bytes'] == GIB-1


def test_dev_inference_needs_a_closed_native96_on_the_same_checkpoint_and_gpu(tmp_path):
    import hashlib
    import json
    from scripts.pi0294.multitask.book_gpu_capacity import INFERENCE_REQUEST
    from scripts.pi0294.multitask.book_phase_diagnostic import KIND
    q = dict(kind=KIND, GPU_uuid='GPU1', delta_checkpoint_manifest_sha256='CKPT',
        GPU_inference_qualification=deepcopy(INFERENCE_REQUEST), limits=dict(model_loads=1,
        optimizer_updates=0, physics_integrations=0, generation_queries=96, RGB_pairs=0, source_constructors=0))
    result = dict(status='COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC', error=None,
        model_load_attempts=1, generation_attempts=96, completed_generations=96, optimizer_updates=0,
        physics_integrations=0, formal_DEV_conditions=0, evaluated_checkpoint_manifest_sha256='CKPT')
    capacity = dict(capacity_pass=True, completed_generations=96, minimum_free_bytes=2*GIB, measurements=[{}]*97)
    lease = dict(status='CLOSED')

    def manifest():
        pins = {}
        for name, value in [('RUN_MANIFEST.json', q), ('RESULT.json', result),
                ('GPU_INFERENCE_CAPACITY.json', capacity), ('control/LEASE.json', lease)]:
            if name == 'control/LEASE.json':
                value = dict(lease, manifest_sha256=pins['RUN_MANIFEST.json'], result_sha256=pins['RESULT.json'])
            path = tmp_path/name
            path.parent.mkdir(exist_ok=True)
            body = json.dumps(value).encode()
            path.write_bytes(body)
            pins[name] = hashlib.sha256(body).hexdigest()
        return dict(kind='V7_BOOK_FROZEN_DEV10', GPU_uuid='GPU1', delta_checkpoint_manifest_sha256='CKPT',
            limits=dict(model_loads=1, optimizer_updates=0, physics_integrations=185000,
                generation_queries=800, RGB_pairs=6420, source_constructors=10),
            GPU_inference_admission=dict(schema='N17_NATIVE96_CERTIFIED_DEV10_INFERENCE_V1',
                qualification_root=str(tmp_path), files=pins))

    m = manifest()
    assert entry_requirement(m) == 13*GIB
    for key, value in [('GPU_uuid', 'GPU0'), ('delta_checkpoint_manifest_sha256', 'OTHER'),
            ('kind', 'V7_BOOK_M32_FINITE_LEARNING')]:
        bad = deepcopy(m)
        bad[key] = value
        with pytest.raises(PermissionError):
            entry_requirement(bad)
    for key, value in [('optimizer_updates', 1), ('generation_queries', 801), ('source_constructors', 11)]:
        bad = deepcopy(m)
        bad['limits'][key] = value
        with pytest.raises(PermissionError):
            entry_requirement(bad)
    for key, value in [('status', 'ACTIVE')]:
        lease[key] = value
        with pytest.raises(PermissionError):
            entry_requirement(manifest())
    lease['status'] = 'CLOSED'
    capacity['minimum_free_bytes'] = GIB-1
    with pytest.raises(PermissionError):
        entry_requirement(manifest())
    capacity['minimum_free_bytes'] = 2*GIB
    m = manifest()
    (tmp_path/'RESULT.json').write_text('{}')
    with pytest.raises(PermissionError, match='CERTIFICATE_CHANGED'):
        entry_requirement(m)
