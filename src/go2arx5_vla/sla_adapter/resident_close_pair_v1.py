"""Cold serving of matched500 adapters with one shared immutable parent.

No model load or native dispatch happens on import. Ordinary RGB/reference
qualification is still required for this new serving identity.
"""
import json
from pathlib import Path
import numpy as np
from .checkpoint_v1 import sha

METHODS = {'CLOSE_FM_CONTROL500', 'CLOSE_TIME500'}


def shared_contract(one, two):
    for key in ('frozen_fingerprint', 'parent_host_cast_v7', 'parent_manifest',
                'tokenizer', 'inference_route', 'derived_statistics_sha256'):
        if one['bindings'].get(key) != two['bindings'].get(key):
            raise ValueError('MATCHED_CLOSE_RESIDENT_PARENT_AND_ROUTE:' + key)
    for key in ('statistics_sha256', 'tokenizer_sha256', 'parent_manifest'):
        if one[key] != two[key]:
            raise ValueError('MATCHED_CLOSE_RESIDENT_ASSET:' + key)


def providers(spec):
    import jax
    from .provider_close_v1 import Provider as CloseProvider
    from .provider_time_v1 import Provider as TimeProvider
    from .provider_v1 import active_rows
    if set(spec['policies']) != METHODS:
        raise ValueError('EXACT_MATCHED_CLOSE500_POLICY_PAIR')
    first = spec['policies']['CLOSE_FM_CONTROL500']['package']
    one = CloseProvider(first['path'], manifest_sha256=first['sha256'],
        parent_params=spec['checkpoint'], parent_manifest_path=spec['checkpoint_manifest_path'])
    if one.manifest['bindings']['gripper_lambda'] != 0:
        raise ValueError('ORIGINAL_PURE_FM_CONTROL_REQUIRED')
    second = spec['policies']['CLOSE_TIME500']['package']; root = Path(second['path'])
    if sha(root / 'MANIFEST.private.json') != second['sha256']:
        raise ValueError('FROZEN_TIME_RESIDENT_PACKAGE')
    manifest = json.loads((root / 'MANIFEST.private.json').read_bytes())
    two = TimeProvider.__new__(TimeProvider)
    two.__dict__ = one.__dict__.copy()
    two.package = root; two.manifest = manifest
    two.validate_package_contract(manifest); shared_contract(one.manifest, manifest)
    if manifest['bindings']['gripper_lambda'] != 0:
        raise ValueError('REPRESENTATION_ONLY_NO_AUXILIARY_OBJECTIVE')
    for filename, key in (('STATISTICS.json', 'statistics_sha256'),
                          ('paligemma_tokenizer.model', 'tokenizer_sha256')):
        if sha(root / filename) != manifest[key]:
            raise ValueError('TIME_RESIDENT_ASSET_CHANGED')
    two.validate_statistics_metadata(json.loads((root / 'STATISTICS.json').read_bytes()))
    rows, structure = active_rows(dict(arrays=manifest['adapter_arrays']), one.active)
    leaves = []
    for row in rows:
        path = root / row['file']
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != row['sha256']:
            raise ValueError('TIME_RESIDENT_ADAPTER_BYTES_CHANGED')
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']:
            raise ValueError('TIME_RESIDENT_ADAPTER_SHAPE_OR_DTYPE')
        leaves.append(jax.device_put(value))
    two.active, = jax.tree_util.tree_unflatten(structure, leaves)
    return dict(CLOSE_FM_CONTROL500=one, CLOSE_TIME500=two)
