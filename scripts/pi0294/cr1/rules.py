"""Predeclared CR1 lineage split and single PAD5 range, independent of outcomes."""
import copy
import hashlib
import inspect
import json

import numpy as np

AUTH = 'e7593fdde645b2f22fc4d06431f98c2da180f855'
CANDIDATE = 'R1_TRAIN_ENVELOPE_PAD5'
SEED = 'CR1-e7593fdd-SPLIT-V1'
TRACE_ANCHORS = (0, 1, 208, 209, 210, 300, 599, 600)
CONTROL_ANCHORS = (0, 80, 160, 209, 300, 600)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def split_groups(rows):
    """Merge geometric/reference siblings transitively before joint-task stratification."""
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('CR1_UNIQUE_NONEMPTY_MAIN_PARENTS')
    links = {r['id']: r['id'] for r in rows}

    def find(key):
        while links[key] != key:
            links[key] = links[links[key]]
            key = links[key]
        return key

    seen = {}
    for row in sorted(rows, key=lambda r: r['id']):
        if row['split'] != 'TRAIN' or row['nominal_clearance_m'] != .004:
            raise PermissionError('CR1_MAIN_ONLY_ORIGINAL_4MM_TRAIN')
        for kind, key in (('group', row['scene_group']), ('geometry', row['geometry_sha256']),
                          ('generator', row['generator_identity'])):
            if not key:
                raise ValueError('CR1_EXPLICIT_LINEAGE_REQUIRED')
            prior = seen.setdefault((kind, key), row['id'])
            left, right = sorted((find(row['id']), find(prior)))
            links[right] = left
    groups = {}
    for row in rows:
        groups.setdefault(find(row['id']), []).append(row)
    strata = {}
    for members in groups.values():
        lineage = digest(sorted(r['id'] for r in members))
        # Task-set strata preserve paired groups without separating their tasks.
        stratum = json.dumps([sorted({r['task'] for r in members}),
                              sorted({r['direction'] for r in members}),
                              sorted({r['family'] for r in members})], separators=(',', ':'))
        strata.setdefault(stratum, []).append((lineage, members))
    output, counts = [], []
    for stratum, items in sorted(strata.items()):
        items.sort(key=lambda item: (digest([SEED, stratum, item[0]]), item[0]))
        diagnostic = min(len(items)-1, max(1, (len(items)+2)//4)) if len(items) >= 2 else 0
        counts.append(dict(stratum=json.loads(stratum), groups=len(items),
                           diagnostic_groups=diagnostic, singleton_fit_only=len(items) == 1))
        for index, (lineage, members) in enumerate(items):
            for row in members:
                output.append(dict(row, lineage=lineage,
                                   role='DIAGNOSTIC' if index < diagnostic else 'FIT'))
    return sorted(output, key=lambda r: r['id']), counts


def isclose_defaults():
    signature = inspect.signature(np.isclose)
    return {key: float(signature.parameters[key].default) for key in ('rtol', 'atol')}


class Envelope:
    def __init__(self, shape):
        self.shape = tuple(shape)
        if self.shape not in ((9,), (40, 9)):
            raise ValueError('CR1_PRESERVE_SUPPORTED_ORIGINAL_STATISTIC_UNIT_SHAPE')
        self.minimum = np.full(self.shape, np.inf)
        self.maximum = np.full(self.shape, -np.inf)
        self.windows = 0

    def add(self, relative):
        relative = np.asarray(relative)
        if relative.shape != (40, 9) or not np.isfinite(relative).all():
            raise ValueError('CR1_FINITE_OFFICIAL_RELATIVE_H40_EEF9')
        low = relative.min(axis=0) if len(self.shape) == 1 else relative
        high = relative.max(axis=0) if len(self.shape) == 1 else relative
        self.minimum = np.minimum(self.minimum, low)
        self.maximum = np.maximum(self.maximum, high)
        self.windows += 1


def candidate_statistics(statistics, effective, envelope):
    original = statistics['new_embodiment']['relative_action']['eef_9d']
    a0, b0 = (np.asarray(effective[key]) for key in ('min', 'max'))
    if (not envelope.windows or a0.shape != envelope.shape or b0.shape != a0.shape
            or not np.isfinite(a0).all() or not np.isfinite(b0).all() or np.any(a0 > b0)
            or any(not np.array_equal(np.asarray(original[k]), effective[k]) for k in ('min', 'max'))):
        raise ValueError('CR1_NONEMPTY_FIT_EXACT_EFFECTIVE_ORIGINAL_RANGES')
    a, b = np.minimum(a0, envelope.minimum), np.maximum(b0, envelope.maximum)
    s = np.maximum(1., np.maximum(np.abs(a), np.abs(b)))
    defaults = isclose_defaults()
    padding = np.maximum.reduce((.05*(b-a), 2*(defaults['atol']+defaults['rtol']*s),
                                 16*np.finfo(np.float32).eps*s))
    a1, b1 = a-padding, b+padding
    if not np.isfinite(a1).all() or not np.isfinite(b1).all() or np.isclose(a1, b1).any():
        raise ValueError('CR1_SINGLE_CANDIDATE_NONFINITE_OR_NEAR_CONSTANT_NO_PADDING_SEARCH')
    result = copy.deepcopy(statistics)
    changed = result['new_embodiment']['relative_action']['eef_9d']
    changed.update(min=a1.tolist(), max=b1.tolist())
    check = copy.deepcopy(result)
    check['new_embodiment']['relative_action']['eef_9d'] = copy.deepcopy(original)
    if check != statistics:
        raise AssertionError('CR1_ONLY_ACTIVE_RELATIVE_EEF9_MINMAX_MAY_CHANGE')
    span0, span1 = b0-a0, b1-a1
    near = np.isclose(b0, a0)
    ratio = np.divide(span1, span0, out=np.ones_like(span0), where=~near)
    return result, dict(candidate=CANDIDATE, formula='TRAIN_ENVELOPE_PAD5_NO_SEARCH',
        fit_windows=envelope.windows, fit_min=envelope.minimum.tolist(), fit_max=envelope.maximum.tolist(),
        original_min=a0.tolist(), original_max=b0.tolist(), candidate_min=a1.tolist(), candidate_max=b1.tolist(),
        padding=padding.tolist(), numpy_isclose_defaults=defaults, float32_eps=float(np.finfo(np.float32).eps),
        original_isclose_mask=near.tolist(), candidate_isclose_mask=np.isclose(a1, b1).tolist(),
        original_unit_shape=list(a0.shape), absolute_span_increase=(span1-span0).tolist(),
        span_ratio_flat=[None if n else float(r) for n, r in zip(near.flat, ratio.flat)],
        signal_scale_ratio_flat=[None if n else float(1/r) for n, r in zip(near.flat, ratio.flat)],
        same_normalized_output_physical_shift_at_z={str(z):(a1+(z+1)*span1/2-(a0+(z+1)*span0/2)).tolist()
                                                   for z in (-1, 0, 1)},
        inactive_subtrees_inherited=True, inactive_fields_reestimated=False,
        production_compatible_with_old_weights='NOT_TESTED_NOT_AUTHORIZED', padding_searches=0)
