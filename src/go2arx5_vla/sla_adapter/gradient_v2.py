"""Finite unclipped merged-gradient evidence; never a training checkpoint."""
import json
from pathlib import Path
import numpy as np

from .checkpoint_v1 import sha


def save(path, gradient, *, step, cursor, bindings, stochastic_inputs, indices):
    import jax
    path = Path(path)
    path.mkdir(exist_ok=False)
    rows = []
    for index, (key, value) in enumerate(jax.tree_util.tree_flatten_with_path((gradient,))[0]):
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError('NONFINITE_MERGED_GRADIENT')
        file = path / f'{index:04d}.npy'
        np.save(file, array, allow_pickle=False)
        rows.append(dict(key=jax.tree_util.keystr(key), file=file.name, shape=list(array.shape),
                         dtype=str(array.dtype), sha256=sha(file),
                         norm=float(np.linalg.norm(array.astype(np.float64)))))
    manifest = dict(schema='SLA_UNCLIPPED_EFFECTIVE32_GRADIENT_V2', step=step, cursor=cursor,
                    bindings=bindings, stochastic_inputs=stochastic_inputs, indices=indices,
                    arrays=rows, effective_batch=32, gradient_clip_applied=False,
                    checkpoint_or_optimizer_state=False)
    (path / 'MANIFEST.private.json').write_text(json.dumps(manifest, indent=2))
    return dict(path=str(path / 'MANIFEST.private.json'), sha256=sha(path / 'MANIFEST.private.json'))


def compare(left, right, *, left_loss, right_loss):
    a, b = [json.loads(Path(value).read_bytes()) for value in (left, right)]
    for value in (a, b):
        if (value['schema'] != 'SLA_UNCLIPPED_EFFECTIVE32_GRADIENT_V2'
                or value['step'] != 0 or value['cursor'] != 0
                or value['effective_batch'] != 32 or value['gradient_clip_applied']):
            raise ValueError('SAME_INITIAL_UNCLIPPED_EFFECTIVE32_GRADIENTS_REQUIRED')
    for key in (set(a['bindings']) | set(b['bindings'])) - {'microbatch', 'accumulation_steps'}:
        if a['bindings'].get(key) != b['bindings'].get(key):
            raise ValueError('MERGED_GRADIENT_SCIENTIFIC_INPUTS_DIFFER:' + key)
    if a['indices'] != b['indices'] or a['stochastic_inputs'] != b['stochastic_inputs'] or len(a['arrays']) != len(b['arrays']):
        raise ValueError('MERGED_GRADIENT_DATA_OR_RANDOM_INPUTS_DIFFER')
    dot = norm_a = norm_b = difference = 0.
    records = []
    for x, y in zip(a['arrays'], b['arrays']):
        if any(x[key] != y[key] for key in ('key', 'shape', 'dtype')):
            raise ValueError('MERGED_GRADIENT_TREE_DIFFERS')
        values = []
        for root, row in ((Path(left).parent, x), (Path(right).parent, y)):
            file = root / row['file']
            if not file.resolve().is_relative_to(root.resolve()) or sha(file) != row['sha256']:
                raise ValueError('MERGED_GRADIENT_ARRAY_CHANGED')
            array = np.load(file, allow_pickle=False)
            if list(array.shape) != row['shape'] or str(array.dtype) != row['dtype'] or not np.isfinite(array).all():
                raise ValueError('MERGED_GRADIENT_ARRAY_INVALID')
            values.append(array.astype(np.float64).reshape(-1))
        u, v = values
        aa, bb, ab, dd = float(u @ u), float(v @ v), float(u @ v), float((u-v) @ (u-v))
        norm_a += aa; norm_b += bb; dot += ab; difference += dd
        records.append(dict(key=x['key'], left_norm=float(np.sqrt(aa)), right_norm=float(np.sqrt(bb)),
                            both_exact_zero=aa == bb == 0, small_norm=max(aa, bb) <= 1e-24,
                            relative_L2=float(np.sqrt(dd / aa)) if aa else None))
    if not norm_a or not norm_b:
        raise ValueError('ACTUAL_MODEL_MERGED_GRADIENT_MUST_BE_NONZERO')
    cosine = float(dot / np.sqrt(norm_a * norm_b))
    relative = float(np.sqrt(difference / norm_a))
    loss_pass = bool(np.isclose(right_loss, left_loss, rtol=1e-3, atol=1e-5))
    return dict(passes=loss_pass and cosine >= .999 and relative <= .02,
                loss_rtol=1e-3, loss_atol=1e-5, left_loss=left_loss, right_loss=right_loss,
                loss_pass=loss_pass, merged_gradient_cosine=cosine, merged_gradient_relative_L2=relative,
                records=records, small_norm_cutoff=1e-12, optimizer_updates_during_comparison=0)
