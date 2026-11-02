import hashlib
import json
import ast
from pathlib import Path

import pytest

from scripts.pi0294.night10.native_outcome import diagnostic_outcome, known_failure


def put(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def journal(root):
    return [dict(MESSAGE='Started '+root.name+'.service', __REALTIME_TIMESTAMP='1000000'),
        dict(CPU_USAGE_NSEC='2300000000', __REALTIME_TIMESTAMP='4000000')]


def diagnostic(root, *, loads=1, attempts=1, complete=1):
    manifest = dict(run_uuid=root.name, kind='V7_BOOK_PHASE_MATCHED_TRAIN32_DIAGNOSTIC',
        limits=dict(generation_queries=96, optimizer_updates=0))
    pin = put(root, 'RUN_MANIFEST.json', manifest)
    put(root, 'control/LEASE.json', dict(status='ACTIVE', manifest_sha256=pin))
    for i in range(attempts):
        put(root, f'{i:02}.INTENT.json', dict(seed=i))
    for i in range(complete):
        put(root, f'{i:02}.RESULT.json', dict(arrays_sha256='preserved'))
    result = dict(status='FAILED_MATCHED_TRAIN32_PHASE_DIAGNOSTIC',
        error='PermissionError: EXACT_INFERENCE_CAPACITY_REQUEST', model_load_attempts=loads,
        generation_attempts=attempts, completed_generations=complete, optimizer_updates=0,
        physics_integrations=0, formal_DEV_conditions=0)
    put(root, 'RESULT.json', result)
    return dict(manifest_sha256=pin), result


def known(root, staged, **kw):
    return known_failure(root, staged, main_pid=kw.pop('main_pid', 0),
        journal=kw.pop('journal', journal(root)),
        canonical=kw.pop('canonical', dict(unknown_effects=[], pending_updates=[], pending_queries=[])), **kw)


def test_real_partial_diagnostic_shape_closes_exact_actual_effects_without_success(tmp_path):
    staged, result = diagnostic(tmp_path)
    out = known(tmp_path, staged)
    assert out['actual']['model_loads'] == out['actual']['generation_queries'] == 1
    assert out['actual']['completed_generations'] == 1
    assert out['actual']['CPU_s'] == 2.3 and out['actual']['GPU_allocated_s_upper'] == 3
    assert out['actual']['all_costs_retained'] and not out['actual']['scientific_success']
    assert out['scientific_result'] == 'INCOMPLETE_DIAGNOSTIC'
    assert out['files']['RESULT.json'] == hashlib.sha256((tmp_path/'RESULT.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('case', ['live', 'unknown_write', 'bad_identity', 'missing_usage', 'bad_counts'])
def test_actual_unknowns_are_not_closed_or_retried(tmp_path, case):
    staged, result = diagnostic(tmp_path)
    kw = {}
    if case == 'live': kw['main_pid'] = 42
    if case == 'unknown_write': kw['canonical'] = dict(unknown_effects=['write'], pending_updates=[], pending_queries=[])
    if case == 'bad_identity': staged['manifest_sha256'] = 'changed'
    if case == 'missing_usage': kw['journal'] = []
    if case == 'bad_counts':
        result['completed_generations'] = 2
        put(tmp_path, 'RESULT.json', result)
    with pytest.raises(PermissionError): known(tmp_path, staged, **kw)


def test_zero_model_resource_refusal_is_closed_but_not_inference_qualification(tmp_path):
    staged, result = diagnostic(tmp_path, loads=0, attempts=0, complete=0)
    known(tmp_path, staged)
    outcome = diagnostic_outcome(result, closed=True)
    assert outcome['lifecycle'] == 'TERMINAL' and outcome['native_effects_closed']
    assert outcome['scientific_result'] == 'RESOURCE_ENTRY_REFUSAL'
    assert not outcome['scientific_success'] and not outcome['inference_capacity_pass']


def test_complete_generation_counts_need_actual_closed_positive_capacity():
    result = dict(status='COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC', error=None,
        model_load_attempts=1, generation_attempts=96, completed_generations=96,
        optimizer_updates=0, physics_integrations=0)
    capacity = dict(capacity_pass=True, completed_generations=96, minimum_free_bytes=2*1024**3,
        measurements=[{}]*97)
    assert diagnostic_outcome(result, capacity, closed=True)['scientific_success']
    assert not diagnostic_outcome(result, closed=True)['scientific_success']
    assert not diagnostic_outcome(result, dict(capacity, completed_generations=95), closed=True)['scientific_success']
    assert not diagnostic_outcome(result, dict(capacity, capacity_pass=False), closed=True)['scientific_success']
    with pytest.raises(PermissionError): diagnostic_outcome(result, capacity, closed=False)


def test_stopped_dev_with_physics_is_never_closed_by_model_only_failure_path(tmp_path):
    manifest = dict(run_uuid=tmp_path.name, kind='V7_BOOK_FROZEN_DEV10',
        limits=dict(generation_queries=800, optimizer_updates=0))
    pin = put(tmp_path, 'RUN_MANIFEST.json', manifest)
    put(tmp_path, 'control/LEASE.json', dict(status='ACTIVE', manifest_sha256=pin))
    result = dict(status='STOPPED_PARTIAL_DEV10', error='RuntimeError: JOINT_LIMIT',
        conditions=[dict(valid_model_condition=True)], valid_conditions=1,
        model_queries=56, integrations=13634, RGB_pairs=447, model_load_attempts=1,
        optimizer_updates=0)
    put(tmp_path, 'RESULT.json', result)
    with pytest.raises(PermissionError): known(tmp_path, dict(manifest_sha256=pin))


def test_pre_physics_evaluation_failure_uses_same_known_failure_path(tmp_path):
    manifest = dict(run_uuid=tmp_path.name, kind='V7_BOOK_FROZEN_DEV10',
        limits=dict(generation_queries=800, optimizer_updates=0))
    pin = put(tmp_path, 'RUN_MANIFEST.json', manifest)
    put(tmp_path, 'control/LEASE.json', dict(status='ACTIVE', manifest_sha256=pin))
    put(tmp_path, 'RESULT.json', dict(status='STOPPED_PARTIAL_DEV10',
        error='PermissionError: BOOK_DEV10_EXPLICIT_NATIVE_GPU_CAPACITY_ENTRY',
        conditions=[], valid_conditions=0, model_queries=0, integrations=0, RGB_pairs=0,
        model_load_attempts=0, optimizer_updates=0))
    result = known(tmp_path, dict(manifest_sha256=pin))
    assert result['actual']['unrun_DEV_conditions'] == 10
    assert result['actual']['valid_DEV_conditions'] == 0


def test_partial_training_retains_actual_updates_and_costs(tmp_path):
    manifest = dict(run_uuid=tmp_path.name, kind='V7_BOOK_M32_FINITE_LEARNING',
        limits=dict(generation_queries=32, optimizer_updates=4096))
    pin = put(tmp_path, 'RUN_MANIFEST.json', manifest)
    put(tmp_path, 'control/LEASE.json', dict(status='ACTIVE', manifest_sha256=pin))
    put(tmp_path, 'RESULT.json', dict(status='FAILED_BOOK_FINITE_OPTIMIZATION',
        error='RuntimeError: optimizer failure', model_load_attempts=1,
        generation_attempts=16, completed_generations=16, optimizer_updates=122,
        TRAIN_visits=3904, physics_integrations=0))
    result = known(tmp_path, dict(manifest_sha256=pin))
    assert result['actual']['optimizer_updates'] == 122
    assert result['actual']['TRAIN_window_visits'] == 3904
    assert result['scientific_result'] == 'FAILED_OPTIMIZATION'


@pytest.mark.parametrize('complete', [False, True])
def test_actual_gpu1_wrapper_does_not_confuse_closed_refusal_with_success(tmp_path, complete):
    # Exercise the actual outer wrapper, not a second implementation of its rule.
    source = Path(__file__).parents[1]/'scripts/pi0294/night10/native_bindings.py'
    tree = ast.parse(source.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu1')
    folder = tmp_path/'GPU1';folder.mkdir()
    native = tmp_path/'native';native.mkdir()
    for name in ('STARTED.json','STAGED.json','FINAL_INDEPENDENT.json','PREPARED.json'):
        put(folder, name, {})
    put(folder, 'DECISION.json', dict(status='ADMITTED_FINITE_GPU1_INFERENCE_QUALIFICATION_ONLY'))
    put(native, 'control/LEASE.json', dict(status='CLOSED'))
    put(native, 'RESULT.json', dict(status='COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC'if complete
        else'FAILED_MATCHED_TRAIN32_PHASE_DIAGNOSTIC', error=None if complete else
        'PermissionError: BOOK_DEV10_EXPLICIT_NATIVE_GPU_CAPACITY_ENTRY',
        model_load_attempts=1 if complete else 0, generation_attempts=96 if complete else 0,
        completed_generations=96 if complete else 0, optimizer_updates=0, physics_integrations=0))
    if complete:
        put(native, 'GPU_INFERENCE_CAPACITY.json', dict(capacity_pass=True,
            completed_generations=96, minimum_free_bytes=2*1024**3, measurements=[{}]*97))
    ns=dict(P=tmp_path,read=lambda p:json.loads(Path(p).read_bytes()),
        control=lambda *a:None,check_staged=lambda _:({},native),wait=lambda _:None)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    result=ns['gpu1']()
    assert result['native_effects_closed'] and result['scientific_success'] is complete
    assert result['status']==('COMPLETE_GPU1_MODEL_DIAGNOSTIC_CLOSED'if complete
        else'FAILED_GPU1_DIAGNOSTIC_CLOSED')
