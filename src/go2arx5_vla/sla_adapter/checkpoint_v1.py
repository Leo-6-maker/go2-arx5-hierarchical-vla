"""Complete PEFT/Adam/RNG checkpoints with template-checked array restoration.

Frozen dense weights remain the separately byte-verified immutable parent.
No pickle, dropped adapters, implicit optimizer reconstruction or cache export.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, tree, *, step, cursor, bindings):
    import jax
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    records = []
    for index, (key, value) in enumerate(leaves):
        a = np.asarray(value)
        file = path/f'{index:04d}.npy'
        np.save(file, a, allow_pickle=False)
        records.append(dict(key=jax.tree_util.keystr(key), file=file.name, shape=list(a.shape),
                            dtype=str(a.dtype), sha256=sha(file)))
    complete = dict(schema='SLA_COMPLETE_PEFT_STATE_V1', step=int(step), cursor=int(cursor),
                    arrays=records, bindings=bindings, training_activations_saved=False,
                    deployment_requires_parent_and_retains_all_lora=True)
    with (path/'COMPLETE.json').open('x') as f:
        json.dump(complete, f, indent=2, allow_nan=False)
    return dict(path=str(path), sha256=sha(path/'COMPLETE.json'))


def restore(path, template, *, bindings):
    import jax
    path = Path(path)
    manifest = json.loads((path/'COMPLETE.json').read_bytes())
    if manifest.get('schema') != 'SLA_COMPLETE_PEFT_STATE_V1' or manifest['bindings'] != bindings:
        raise ValueError('SLA_CHECKPOINT_PARENT_RECIPE_DATA_BINDING_MISMATCH')
    leaves, structure = jax.tree_util.tree_flatten_with_path(template)
    if len(leaves) != len(manifest['arrays']):
        raise ValueError('SLA_COMPLETE_OPTIMIZER_AND_RNG_TREE_REQUIRED')
    values = []
    for (key, reference), row in zip(leaves, manifest['arrays']):
        file = path/row['file']
        if (not file.resolve().is_relative_to(path.resolve()) or sha(file) != row['sha256']
                or jax.tree_util.keystr(key) != row['key']):
            raise ValueError('SLA_CHECKPOINT_ARRAY_IDENTITY_MISMATCH')
        a = np.load(file, allow_pickle=False)
        if list(a.shape) != row['shape'] or a.shape != reference.shape or str(a.dtype) != row['dtype'] or a.dtype != reference.dtype:
            raise ValueError('SLA_CHECKPOINT_ARRAY_SHAPE_OR_DTYPE_MISMATCH')
        values.append(jax.device_put(a))
    if manifest['cursor'] != manifest['step']*32:
        raise ValueError('SLA_RESTORE_MUST_BE_AT_COMPLETE_EFFECTIVE_BATCH_BOUNDARY')
    return jax.tree_util.tree_unflatten(structure, values), manifest['step'], manifest['cursor']


def compare(left, right):
    """CPU-only comparison of two completed states, including Adam and RNG."""
    left, right = Path(left), Path(right)
    a, b = [json.loads((p/'COMPLETE.json').read_bytes()) for p in (left, right)]
    if any(a[k] != b[k] for k in ('schema', 'step', 'cursor', 'bindings')) or len(a['arrays']) != len(b['arrays']):
        raise ValueError('INDEPENDENT_RESTORE_STATE_IDENTITIES_DIFFER')
    differences = []
    for x, y in zip(a['arrays'], b['arrays']):
        if any(x[k] != y[k] for k in ('key', 'shape', 'dtype')):
            raise ValueError('INDEPENDENT_RESTORE_TREE_DIFFERS')
        for root, row in ((left, x), (right, y)):
            if sha(root/row['file']) != row['sha256']:
                raise ValueError('SAVED_STATE_ARRAY_CHANGED')
        if x['sha256'] != y['sha256']:
            v, w = np.load(left/x['file'], allow_pickle=False), np.load(right/y['file'], allow_pickle=False)
            differences.append(dict(key=x['key'], max_abs=float(np.abs(v.astype(np.float64)-w.astype(np.float64)).max())))
    return dict(bit_exact=not differences, arrays=len(a['arrays']), step=a['step'], cursor=a['cursor'], differences=differences)
