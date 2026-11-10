"""Read-only comparison of pinned private traces; never qualifies a policy."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .storage import digest
from .task_learning_tensor_trace_v1 import MODE, MAX_BYTES, QUERIES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(pin):
    path = Path(pin['path'])
    require(path.stat().st_size < 32 * 1024**2, 'bounded JSON metadata')
    require(digest(path) == pin['sha256'], 'pinned metadata SHA256')
    return path, json.loads(path.read_bytes())


def boundary_key(row):
    return tuple(row[k] for k in ('query_index', 'module', 'stage', 'call'))


def query_identity(row):
    model = row['model_identity']
    return dict(episode=row['episode'], split=row['split'], anchor=row['anchor'],
                seed=row['seed'], command=row['command'], noise=row['noise'],
                phase=row['phase_audit_only'], global_step=model['global_step'],
                dataset=model['dataset_sha256'], statistics=model['statistics_sha256'],
                parent=model['parent_manifest_sha256'], package=model['candidate_package']['sha256'])


def load_trace(manifest_pin, probes_pin):
    manifest_path, manifest = pinned_json(manifest_pin)
    _, probes = pinned_json(probes_pin)
    require(manifest['mode'] == MODE and manifest['diagnostic_only'] is True
            and manifest['complete'] is True and manifest['failure'] is None
            and manifest['original_query_count'] == probes['actual_VLA_queries'] == 40
            and len(probes['records']) == 40 and manifest['traced_query_count'] == QUERIES
            and manifest['numeric_flags_changed'] is False
            and manifest['tensors_mutated'] is False and manifest['package_qualified'] is False,
            'complete original40 diagnostic trace only')
    rows = manifest['records']
    keys = [boundary_key(row) for row in rows]
    require(len(set(keys)) == len(keys), 'unique trace boundaries')
    expected = Counter({(q, module, stage): count
        for q in range(QUERIES)
        for module, count in dict(backbone=1, vlln=1, vl_self_attention=1,
                                 state_encoder=1, action_encoder=4,
                                 denoiser=4, action_decoder=4).items()
        for stage in ('input', 'output')})
    require(Counter((r['query_index'], r['module'], r['stage']) for r in rows) == expected,
            'complete four-query seven-module boundaries')
    for (q, module, stage), count in expected.items():
        require({r['call'] for r in rows if boundary_key(r)[:3] == (q, module, stage)}
                == set(range(count)), 'complete call indices')
    require(0 < manifest['tensor_bytes'] == sum(r['tensor_bytes'] for r in rows) <= MAX_BYTES,
            'bounded complete tensor bytes')
    for row in rows:
        require(row['active_allow_tf32'] is False
                and row['active_allow_bf16_reduced_precision_reduction'] is True,
                'original backend flags throughout capture')
        name = row['file']
        require(Path(name).name == name and name.endswith('.npz'), 'trace member basename')
        path = manifest_path.parent / name
        require(not path.is_symlink() and path.is_file()
                and path.stat().st_size <= 256 * 1024**2, 'bounded regular trace member')
        require(digest(path) == row['sha256'], 'trace member SHA256')
    identities = [query_identity(r) for r in probes['records']]
    require(len({(r['episode'], r['anchor'], r['seed']) for r in identities}) == 40,
            'unique original query identities')
    require(all(r['global_step'] == 2000 and r['split'] == 'TRAIN' for r in identities),
            'first2000 sealed TRAIN probes only')
    return manifest_path.parent, manifest, identities


def tensor_difference(a, b):
    require(a.shape == b.shape and a.dtype == b.dtype and a.dtype.kind in 'biuf',
            'identical numeric tensor shape and storage dtype')
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    same_special = bool(np.array_equal(finite_a, finite_b)
        and np.array_equal(np.isnan(a), np.isnan(b))
        and np.array_equal(np.isposinf(a), np.isposinf(b))
        and np.array_equal(np.isneginf(a), np.isneginf(b)))
    common = finite_a & finite_b
    delta = a[common].astype(np.float64) - b[common].astype(np.float64)
    equal = bool(np.array_equal(a, b, equal_nan=True))
    return dict(bit_exact=a.tobytes() == b.tobytes(), numeric_equal=equal,
                nonfinite_patterns_equal=same_special,
                finite_pairs=int(common.sum()), elements=int(a.size),
                changed_elements=int(np.count_nonzero((a != b) & ~(np.isnan(a) & np.isnan(b)))),
                maximum_absolute_finite_difference=float(np.max(np.abs(delta))) if delta.size else None,
                finite_RMSE=float(np.sqrt(np.mean(delta**2))) if delta.size else None)


def compare(reference_manifest, reference_probes, candidate_manifest, candidate_probes):
    """Compare corresponding tensors in actual call order, without inference."""
    left, a, queries_a = load_trace(reference_manifest, reference_probes)
    right, b, queries_b = load_trace(candidate_manifest, candidate_probes)
    require(queries_a == queries_b, 'all40 observation/noise/model/data identities match')
    require(a['reference_result']['sha256'] == b['reference_result']['sha256'],
            'same original Core40 trace reference')
    require([boundary_key(r) for r in a['records']] == [boundary_key(r) for r in b['records']],
            'identical actual boundary call order')
    rows = []
    for x, y in zip(a['records'], b['records']):
        require(x['structure'] == y['structure'], 'identical tensor and scalar structure')
        tensors = []
        with np.load(left / x['file'], allow_pickle=False) as aa, np.load(right / y['file'], allow_pickle=False) as bb:
            expected = {s['key'] for s in x['structure'] if 'key' in s}
            require(set(aa.files) == set(bb.files) == expected, 'complete tensor members')
            sizes = [0, 0]
            for s in x['structure']:
                if 'key' not in s:
                    continue
                k = s['key']; u, v = aa[k], bb[k]
                require(list(u.shape) == list(v.shape) == s['shape'], 'declared tensor shape')
                for i, array in enumerate((u, v)):
                    sizes[i] += array.nbytes
                tensors.append(dict(path=s['path'], source_dtype=s['dtype'],
                                    **tensor_difference(u, v)))
            require(sizes == [x['tensor_bytes'], y['tensor_bytes']], 'declared complete tensor byte counts')
        rows.append(dict(query_index=x['query_index'], module=x['module'], stage=x['stage'],
                         call=x['call'], all_bit_exact=all(t['bit_exact'] for t in tensors),
                         tensors=tensors))
    first = next((dict(query_index=r['query_index'], module=r['module'], stage=r['stage'],
                       call=r['call'], differing_paths=[t['path'] for t in r['tensors'] if not t['bit_exact']])
                  for r in rows if not r['all_bit_exact']), None)
    first_by_query = {str(q): next((dict(module=r['module'], stage=r['stage'], call=r['call'])
        for r in rows if r['query_index'] == q and not r['all_bit_exact']), None) for q in range(QUERIES)}
    return dict(status='ACTUAL_PINNED_INTERMEDIATE_TENSOR_COMPARISON', diagnostic_only=True,
                reference_manifest_sha256=reference_manifest['sha256'],
                candidate_manifest_sha256=candidate_manifest['sha256'],
                reference_probes_sha256=reference_probes['sha256'],
                candidate_probes_sha256=candidate_probes['sha256'],
                matched_query_identities=40, compared_boundaries=len(rows),
                bit_exact_boundaries=sum(r['all_bit_exact'] for r in rows),
                first_non_bit_exact_boundary=first, first_non_bit_exact_by_query=first_by_query,
                boundary_comparisons=rows, model_loads=0, optimizer_updates=0,
                native_integrations=0, package_qualified=False,
                causality='First observed module boundary difference; internal kernel cause is not established.')
