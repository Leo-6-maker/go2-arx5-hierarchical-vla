"""Comparison integrity and attribution tests; these are not GPU qualification."""
import json

import numpy as np
import pytest

from go2arx5_vla.world_model import task_learning_tensor_compare_v1 as compare
from go2arx5_vla.world_model.storage import digest


def write_pin(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=digest(path))


def make_trace(folder):
    folder.mkdir()
    records = []
    for query in range(4):
        for module in ('backbone', 'vlln', 'vl_self_attention', 'state_encoder',
                       'action_encoder', 'denoiser', 'action_decoder'):
            for call in range(4 if module in ('action_encoder', 'denoiser', 'action_decoder') else 1):
                for stage in ('input', 'output'):
                    path = folder / f'{query}-{module}-{call}-{stage}.npz'
                    value = np.array([query + 0.25, -1.0], dtype=np.float32)
                    np.savez(path, tensor0000=value)
                    records.append(dict(query_index=query, module=module, call=call, stage=stage,
                        file=path.name, sha256=digest(path), tensor_bytes=value.nbytes,
                        structure=[dict(key='tensor0000', path=['features'], dtype='torch.float32', shape=[2])],
                        active_allow_tf32=False, active_allow_bf16_reduced_precision_reduction=True))
    manifest = dict(mode=compare.MODE, diagnostic_only=True, complete=True, failure=None,
                    original_query_count=40, traced_query_count=4,
                    tensor_bytes=sum(r['tensor_bytes'] for r in records),
                    numeric_flags_changed=False, tensors_mutated=False, package_qualified=False,
                    reference_result=dict(sha256='0' * 64), records=records)
    queries = [dict(episode='episode', split='TRAIN', anchor=i, seed=100+i, command='grasp',
                    noise=dict(sha256=str(i)), phase_audit_only='CLOSE',
                    model_identity=dict(global_step=2000, dataset_sha256='data', statistics_sha256='stats',
                        parent_manifest_sha256='parent', candidate_package=dict(path=str(folder), sha256='package')))
               for i in range(40)]
    return (write_pin(folder / 'TRACE_MANIFEST.private.json', manifest),
            write_pin(folder / 'RESULT.json', dict(actual_VLA_queries=40, records=queries)))


def mutate(pin, fn):
    from pathlib import Path
    path = Path(pin['path']); value = json.loads(path.read_text()); fn(value)
    path.write_text(json.dumps(value)); pin['sha256'] = digest(path)


def test_same_traces_and_different_host_paths_are_equal(tmp_path):
    a, b = make_trace(tmp_path/'a'), make_trace(tmp_path/'b')
    result = compare.compare(*a, *b)
    assert result['compared_boundaries'] == result['bit_exact_boundaries'] == 128
    assert result['matched_query_identities'] == 40
    assert result['first_non_bit_exact_boundary'] is None
    assert result['package_qualified'] is False and result['model_loads'] == 0


def test_first_difference_identifies_query_module_and_tensor_path(tmp_path):
    a, b = make_trace(tmp_path/'a'), make_trace(tmp_path/'b')
    member = tmp_path/'b/0-vlln-0-output.npz'
    np.savez(member, tensor0000=np.array([0.5, -1.0], dtype=np.float32))
    mutate(b[0], lambda m: next(r for r in m['records'] if r['file'] == member.name).update(sha256=digest(member)))
    result = compare.compare(*a, *b)
    assert result['bit_exact_boundaries'] == 127
    assert result['first_non_bit_exact_boundary'] == dict(query_index=0, module='vlln', stage='output',
                                                        call=0, differing_paths=[['features']])
    row = next(r for r in result['boundary_comparisons'] if r['module']=='vlln' and r['query_index']==0 and r['stage']=='output')
    assert row['tensors'][0]['maximum_absolute_finite_difference'] == 0.25


@pytest.mark.parametrize('field', ['noise', 'dataset', 'package', 'precision'])
def test_mismatched_inputs_or_model_refuse_hardware_attribution(tmp_path, field):
    a, b = make_trace(tmp_path/'a'), make_trace(tmp_path/'b')
    def change(v):
        row = v['records'][12]
        if field == 'noise': row['noise']['sha256'] = 'different'
        elif field == 'dataset': row['model_identity']['dataset_sha256'] = 'different'
        elif field == 'package': row['model_identity']['candidate_package']['sha256'] = 'different'
        else: row['model_identity']['inference_precision_diagnostic'] = 'FP32_NO_TF32_FIXED40_DIAGNOSTIC_V1'
    mutate(b[1], change)
    with pytest.raises(ValueError, match='all40 observation/noise/model/data'):
        compare.compare(*a, *b)


def test_changed_tensor_without_new_pinned_manifest_is_rejected(tmp_path):
    a, b = make_trace(tmp_path/'a'), make_trace(tmp_path/'b')
    member = tmp_path/'b/0-backbone-0-input.npz'
    np.savez(member, tensor0000=np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match='member SHA256'):
        compare.compare(*a, *b)


@pytest.mark.parametrize('bad', ['incomplete', 'backend', 'path', 'shape', 'order'])
def test_incomplete_or_incompatible_traces_are_rejected(tmp_path, bad):
    a, b = make_trace(tmp_path/'a'), make_trace(tmp_path/'b')
    def change(m):
        if bad == 'incomplete': m['records'].pop()
        elif bad == 'backend': m['records'][0]['active_allow_tf32'] = True
        elif bad == 'path': m['records'][0]['file'] = '../outside.npz'
        elif bad == 'shape': m['records'][0]['structure'][0]['shape'] = [1, 2]
        else: m['records'][0], m['records'][1] = m['records'][1], m['records'][0]
    mutate(b[0], change)
    with pytest.raises(ValueError): compare.compare(*a, *b)


def test_nonfinite_patterns_and_signed_zero_are_explicit():
    a = np.array([np.nan, np.inf, -np.inf, 0., 1.], dtype=np.float32)
    b = np.array([np.nan, np.inf, -np.inf, -0., 1.], dtype=np.float32)
    diff = compare.tensor_difference(a, b)
    assert diff['nonfinite_patterns_equal'] is True
    assert diff['numeric_equal'] is True and diff['bit_exact'] is False
    assert diff['maximum_absolute_finite_difference'] == diff['finite_RMSE'] == 0.
    b[1] = -np.inf
    diff = compare.tensor_difference(a, b)
    assert diff['nonfinite_patterns_equal'] is False and diff['changed_elements'] == 1
    assert diff['maximum_absolute_finite_difference'] == 0.


def test_metadata_pin_mismatch_rejected(tmp_path):
    a, b = make_trace(tmp_path/'a'), make_trace(tmp_path/'b')
    b[0]['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='metadata SHA256'):
        compare.compare(*a, *b)
