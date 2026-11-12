"""Exact S1/S2 common recipe; no launcher or inherited GR00T qualification."""
import hashlib
import math
import numpy as np

SEED = 20261112
UPSTREAM_SHA = '405ca8879c032a05f0a778adc7d86f4e221c5eba'
PLAN_SHA = '5ff216ce5399d1bfb78bd7bbf968951d1e4b23f8'


def learning_rates():
    return [5e-5*j/25 if j <= 25 else 1e-5+.5*(5e-5-1e-5)*(1+math.cos(math.pi*(j-25)/475))
            for j in range(1, 501)]


def lr_sha256():
    return hashlib.sha256(''.join(v.hex()+'\n' for v in learning_rates()).encode('ascii')).hexdigest()


def visits(manifest):
    episodes = manifest['episodes']
    if len(episodes) != 100 or any(e['split'] != 'TRAIN' for e in episodes):
        raise ValueError('SEALED_TRAIN100_ONLY')
    unique = [(e['id'], i) for e in episodes for i in e['supervised_anchors']]
    if len(unique) != len(set(unique)) or len(unique) != 1450:
        raise ValueError('EXACT_1450_UNIQUE_EXPERT_WINDOWS')
    lookup = {v: i for i, v in enumerate(unique)}
    exposure = []
    for e in episodes:
        if len(e['optimization_anchors']) != 32:
            raise ValueError('ORIGINAL_32_EXPOSURES_PER_EPISODE')
        exposure.extend(lookup[e['id'], a] for a in e['optimization_anchors'])
    if set(exposure) != set(range(1450)):
        raise ValueError('EVERY_EXPERT_WINDOW_RETAINED')
    exposure = np.asarray(exposure, dtype='<i8')
    return np.concatenate([exposure[np.random.Generator(np.random.PCG64(SEED+cycle)).permutation(3200)]
                           for cycle in range(5)])


def trainable(path, *, align=False):
    """Positive allowlist over actual flattened NNX paths, checked by inventory."""
    parts = tuple(str(p) for p in path)
    if parts and parts[0] in ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out'):
        return True
    if align and parts and parts[0] == 'align_projector':
        return True
    return len(parts) >= 3 and parts[:2] == ('PaliGemma', 'llm') and any('lora' in p for p in parts[2:])
