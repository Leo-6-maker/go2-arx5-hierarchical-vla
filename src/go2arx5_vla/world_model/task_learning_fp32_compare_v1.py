"""Compare two completed FP32 diagnostic artifacts; never qualify a policy."""
from pathlib import Path

import numpy as np

from .storage import digest
from .task_learning_fp32_diagnostic_v1 import MODE
from .task_learning_tensor_compare_v1 import compare as compare_boundaries
from .task_learning_tensor_compare_v1 import pinned_json, query_identity, require
from .target_gr00t_inference_v1 import CROSS_SITE_ATOL


def receipt(pin):
    _, value = pinned_json(pin)
    require(value['mode'] == MODE and value['diagnostic_only'] is True
            and value['original_queries'] == 40 and value['failure'] is None
            and value['original_flags_restored'] is True
            and value['original_flags'] == value['final_flags']
            and value['original_method_restored'] is True
            and value['weights_changed'] is False and value['precision_changed'] is True
            and value['package_qualified'] is False, 'complete FP32 diagnostic receipt')
    rows = value['observations']
    require(len(rows) == 40 and [r['query_index'] for r in rows] == list(range(40))
            and all(r['cuda_autocast_enabled'] is False
                    and r['action_prediction_dtype'] == 'torch.float32'
                    and r['matmul_allow_tf32'] is False and r['cudnn_allow_tf32'] is False
                    for r in rows), 'all40 actual FP32 observations without TF32')


def sample(folder, row):
    name = row['sample_file']
    require(Path(name).name == name and name.endswith('.npz'), 'FP32 sample basename')
    file = Path(folder)/name
    require(not file.is_symlink() and file.is_file() and file.stat().st_size < 1024**2
            and digest(file) == row['sha256'], 'pinned FP32 prediction member')
    return file


def compare(*, reference_manifest, reference_probes, reference_precision, reference_samples,
            candidate_manifest, candidate_probes, candidate_precision, candidate_samples):
    for pin in (reference_precision, candidate_precision):
        receipt(pin)
    _, a = pinned_json(reference_probes); _, b = pinned_json(candidate_probes)
    require(a['actual_VLA_queries'] == b['actual_VLA_queries'] == 40
            and len(a['records']) == len(b['records']) == 40
            and all(r['model_identity'].get('inference_precision_diagnostic') == MODE
                    for v in (a, b) for r in v['records']), 'both original40 models explicitly FP32')
    require([query_identity(r) for r in a['records']] == [query_identity(r) for r in b['records']],
            'all40 same-precision observation/noise/model/data identities')
    # The outer capture itself remains transparent; the receipt declares the
    # separate precision intervention. Never reinterpret it as original BF16.
    boundaries = compare_boundaries(reference_manifest, reference_probes,
                                   candidate_manifest, candidate_probes)
    dtypes = set()
    for pin in (reference_manifest, candidate_manifest):
        _, manifest = pinned_json(pin)
        dtypes.update(s['dtype'] for r in manifest['records'] if r['stage'] == 'output'
                      for s in r['structure'] if 'dtype' in s)
    require('torch.float32' in dtypes and dtypes <= {'torch.float32', 'torch.bool'},
            'captured FP32 module output dtypes')
    maximum = {k: 0. for k in CROSS_SITE_ATOL}; counts = {k: 0 for k in CROSS_SITE_ATOL}
    exact = {k: 0 for k in ('physical_H40', 'normalized')}
    for x, y in zip(a['records'], b['records']):
        with np.load(sample(reference_samples, x), allow_pickle=False) as aa, \
                np.load(sample(candidate_samples, y), allow_pickle=False) as bb:
            for key in ('actual_state10', 'expert_policy_H40', 'noise_seed'):
                u, v = aa[key], bb[key]
                require(u.shape == v.shape and u.dtype == v.dtype and u.tobytes() == v.tobytes(),
                        'byte-exact actual states, experts and noise seeds')
            for key in exact:
                u, v = aa[key], bb[key]
                require(u.shape == v.shape and u.dtype == v.dtype and u.dtype.kind == 'f'
                        and u.size > 0 and np.isfinite(u).all() and np.isfinite(v).all(),
                        'finite matching FP32 prediction shape and storage dtype')
                exact[key] += int(u.tobytes() == v.tobytes())
            require(aa['physical_H40'].shape == (40, 10), 'original physical H40 shape')
            delta = np.abs(aa['physical_H40'].astype(np.float64)-bb['physical_H40'].astype(np.float64))
            values = dict(XYZ_m=float(delta[:, :3].max()),
                          rot6d_component=float(delta[:, 3:9].max()), gripper=float(delta[:, 9].max()),
                          normalized_component=float(np.abs(aa['normalized'].astype(np.float64)
                                                            -bb['normalized'].astype(np.float64)).max()))
            for key, value in values.items():
                maximum[key] = max(maximum[key], value)
                counts[key] += int(value > CROSS_SITE_ATOL[key])
    return dict(status='ACTUAL_MATCHED_FP32_DIAGNOSTIC_COMPARISON', diagnostic_only=True,
                mode=MODE, reference_precision_sha256=reference_precision['sha256'],
                candidate_precision_sha256=candidate_precision['sha256'],
                matched_query_identities=40, all_states_experts_noise_byte_exact=True,
                captured_output_dtypes=sorted(dtypes), boundary_comparison=boundaries,
                original_cross_site_tolerances=CROSS_SITE_ATOL, maximum_absolute_error=maximum,
                queries_over_original_tolerances=counts, byte_exact_query_counts=exact,
                within_original_component_tolerances=not any(counts.values()),
                model_loads=0, optimizer_updates=0, native_integrations=0,
                package_qualified=False, qualification_issued=False,
                interpretation='A same-precision diagnostic comparison. Any ordinary runtime adoption '
                               'requires a separately versioned qualification; this does not replace '
                               'the failed original BF16 gate or establish native task success.')
