"""Read-only first-query backbone comparison, without a model or qualification."""
from pathlib import Path

import numpy as np

from .storage import digest
from .task_learning_backbone_trace_v1 import MODE, MAX_BYTES
from .task_learning_tensor_compare_v1 import pinned_json, require, boundary_key, tensor_difference


def load(pin):
    path, value = pinned_json(pin)
    require(value['mode'] == MODE and value['diagnostic_only'] is True
            and value['complete'] is True and value['failure'] is None
            and value['original_query_count'] == 40 and value['traced_query_count'] == 1
            and value['numeric_flags_changed'] is False and value['tensors_mutated'] is False
            and value['package_qualified'] is False, 'complete first-query backbone trace')
    rows = value['records']; keys = [boundary_key(r) for r in rows]
    require(len(keys) == len(set(keys)) and keys[0] == (0, 'constants', 'load', 0),
            'unique ordered boundaries starting with constants')
    require(0 < value['tensor_bytes'] == sum(r['tensor_bytes'] for r in rows) <= MAX_BYTES,
            'bounded complete backbone trace bytes')
    for row in rows:
        require(row['query_index'] == row['call'] == 0 and row['active_allow_tf32'] is False
                and row['active_bf16_reduced_precision_reduction'] is True,
                'only first query with original numerical flags')
        name = row['file']; require(Path(name).name == name and name.endswith('.npz'), 'member basename')
        f = path.parent / name
        require(f.is_file() and not f.is_symlink() and f.stat().st_size < 256 * 1024**2
                and digest(f) == row['sha256'], 'pinned regular backbone member')
    return path.parent, value


def compare(reference_manifest, candidate_manifest):
    left, a = load(reference_manifest); right, b = load(candidate_manifest)
    require(a['candidate_package_sha256'] == b['candidate_package_sha256'], 'same formal candidate package')
    require(a['runtime']['HF_Qwen_source_sha256'] == b['runtime']['HF_Qwen_source_sha256'],
            'same actual Qwen implementation')
    require([boundary_key(r) for r in a['records']] == [boundary_key(r) for r in b['records']],
            'same actual backbone boundary order')
    comparisons = []
    for x, y in zip(a['records'], b['records']):
        require(x['structure'] == y['structure'], 'same tensor/scalar/cache metadata structure')
        tensors = []
        with np.load(left/x['file'], allow_pickle=False) as aa, np.load(right/y['file'], allow_pickle=False) as bb:
            expected = {s['key'] for s in x['structure'] if 'key' in s}
            require(set(aa.files) == set(bb.files) == expected, 'complete backbone tensor members')
            sizes = [0, 0]
            for s in x['structure']:
                if 'key' not in s: continue
                u, v = aa[s['key']], bb[s['key']]
                require(list(u.shape) == list(v.shape) == s['shape'], 'declared backbone tensor shape')
                sizes[0] += u.nbytes; sizes[1] += v.nbytes
                tensors.append(dict(path=s['path'], source_dtype=s['dtype'], **tensor_difference(u, v)))
            require(sizes == [x['tensor_bytes'], y['tensor_bytes']], 'declared backbone byte counts')
        comparisons.append(dict(query_index=0, module=x['module'], stage=x['stage'], call=0,
            all_bit_exact=all(t['bit_exact'] for t in tensors), tensors=tensors))
    first = next((r for r in comparisons if not r['all_bit_exact']), None)
    return dict(status='ACTUAL_PINNED_FROZEN_BACKBONE_BOUNDARY_COMPARISON', diagnostic_only=True,
                reference_manifest_sha256=reference_manifest['sha256'],
                candidate_manifest_sha256=candidate_manifest['sha256'],
                compared_boundaries=len(comparisons),
                bit_exact_boundaries=sum(r['all_bit_exact'] for r in comparisons),
                first_non_bit_exact_boundary=first, boundary_comparisons=comparisons,
                model_loads=0, optimizer_updates=0, native_integrations=0, package_qualified=False,
                interpretation='First observed saved boundary; cache KV tensors are not recorded. '
                               'A differing operator output does not identify its unique internal cause.')
