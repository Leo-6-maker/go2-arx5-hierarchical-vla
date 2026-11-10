"""Scientific comparison integrity tests, without a model or GPU."""
from pathlib import Path

import numpy as np
import pytest

from go2arx5_vla.world_model import task_learning_fp32_compare_v1 as comp
from go2arx5_vla.world_model.storage import digest
from test_task_learning_tensor_compare_v1 import make_trace, mutate, write_pin


def fixture(folder):
    manifest, probes = make_trace(folder)
    def add_samples(value):
        for row in value['records']:
            row['model_identity']['inference_precision_diagnostic'] = comp.MODE
            name = f"sample-{row['anchor']}.npz"; file = folder/name
            np.savez(file, physical_H40=np.zeros((40, 10), dtype=np.float32),
                     normalized=np.zeros((40, 32), dtype=np.float32),
                     actual_state10=np.zeros(10, dtype=np.float32),
                     expert_policy_H40=np.zeros((40, 10), dtype=np.float32),
                     noise_seed=np.asarray(row['seed'], dtype=np.int64))
            row.update(sample_file=name, sha256=digest(file))
    mutate(probes, add_samples)
    precision = write_pin(folder/'precision.json', dict(mode=comp.MODE, diagnostic_only=True,
        original_queries=40, failure=None, original_flags_restored=True,
        original_flags=dict(matmul=False, cudnn=True), final_flags=dict(matmul=False, cudnn=True),
        original_method_restored=True, weights_changed=False, precision_changed=True, package_qualified=False,
        observations=[dict(query_index=i, cuda_autocast_enabled=False,
            action_prediction_dtype='torch.float32', matmul_allow_tf32=False, cudnn_allow_tf32=False)
            for i in range(40)]))
    return dict(manifest=manifest, probes=probes, precision=precision, samples=str(folder))


def pair(tmp_path):
    return {side+'_'+key: value for side in ('reference', 'candidate')
            for key, value in fixture(tmp_path/side).items()}


def change_sample(args, fn):
    folder = Path(args['candidate_samples']); file = folder/'sample-2.npz'
    with np.load(file, allow_pickle=False) as old:
        arrays = {key: old[key] for key in old.files}
    fn(arrays); np.savez(file, **arrays)
    mutate(args['candidate_probes'], lambda v: v['records'][2].update(sha256=digest(file)))


def test_complete_matching_fp32_is_diagnostic_without_qualification(tmp_path):
    result = comp.compare(**pair(tmp_path))
    assert result['within_original_component_tolerances'] is True
    assert result['byte_exact_query_counts'] == dict(physical_H40=40, normalized=40)
    assert result['boundary_comparison']['bit_exact_boundaries'] == 128
    assert result['matched_query_identities'] == 40
    assert result['model_loads'] == result['native_integrations'] == 0
    assert not result['package_qualified'] and not result['qualification_issued']


def test_failed_original_tolerance_returns_negative_evidence_instead_of_promoting(tmp_path):
    args = pair(tmp_path)
    change_sample(args, lambda a: a['physical_H40'].__setitem__((0, 9), 0.002))
    result = comp.compare(**args)
    assert result['queries_over_original_tolerances'] == dict(XYZ_m=0, rot6d_component=0,
                                                            gripper=1, normalized_component=0)
    assert result['within_original_component_tolerances'] is False
    assert result['original_cross_site_tolerances']['gripper'] == 0.001
    assert result['qualification_issued'] is False


@pytest.mark.parametrize('bad', ['bf16', 'partial', 'TF32', 'restoration', 'weight_change'])
def test_missing_or_inconsistent_precision_evidence_is_rejected(tmp_path, bad):
    args = pair(tmp_path)
    if bad == 'bf16':
        mutate(args['candidate_probes'], lambda v: v['records'][10]['model_identity'].pop('inference_precision_diagnostic'))
    else:
        def change(v):
            if bad == 'partial': v['observations'].pop()
            elif bad == 'TF32': v['observations'][10]['cudnn_allow_tf32'] = True
            elif bad == 'restoration': v['final_flags']['cudnn'] = False
            else: v['weights_changed'] = True
        mutate(args['candidate_precision'], change)
    with pytest.raises(ValueError): comp.compare(**args)


@pytest.mark.parametrize('bad', ['expert', 'nonfinite', 'unhashed_member', 'BF16_activation'])
def test_other_changes_cannot_be_misattributed_to_device_arithmetic(tmp_path, bad):
    args = pair(tmp_path)
    if bad == 'expert':
        change_sample(args, lambda a: a['expert_policy_H40'].__setitem__((0, 0), 1.))
    elif bad == 'nonfinite':
        change_sample(args, lambda a: a['normalized'].__setitem__((0, 0), np.nan))
    elif bad == 'unhashed_member':
        (Path(args['candidate_samples'])/'sample-2.npz').write_bytes(b'corrupted')
    else:
        mutate(args['candidate_manifest'], lambda v: v['records'][1]['structure'][0].update(dtype='torch.bfloat16'))
        mutate(args['reference_manifest'], lambda v: v['records'][1]['structure'][0].update(dtype='torch.bfloat16'))
    with pytest.raises(ValueError): comp.compare(**args)
