"""Fixed W2 proposals. Never selects a new seed/scene after observing a score."""
import numpy as np

ARMS = ('unchanged', 'gradient', 'negative_gradient', 'random_matched_norm', 'action_blind')
ACTION_SCALE = np.asarray([.01]*3+[.05]*3+[1.]).reshape(1,7)


def bounded(raw):
    raw = np.asarray(raw, dtype=float).reshape(8,7).copy()
    for sl, bound in ((slice(0,3), .01), (slice(3,6), .05)):
        norm = np.linalg.norm(raw[:,sl], axis=1, keepdims=True)
        raw[:,sl] *= np.minimum(1., bound/np.maximum(norm,1e-12))
    raw[:,6] = np.clip(raw[:,6],0,1)
    return raw


def proposals(base, gradient_normalized, mean_action, *, seed):
    """One draw; project once, then shrink convex feasible deltas to equal norm.

    Norm is in declared primitive units (1cm, .05rad, unit gripper). The three
    gradient/negative/random arms share their actual feasible norm, even when
    gripper/translation bounds are active. Zero feasible norm remains a tie.
    """
    base = np.asarray(base,dtype=float).reshape(8,7)
    if not np.allclose(base,bounded(base),rtol=0,atol=1e-10):
        raise ValueError('Original command outside primitive; retain as a rejection')
    g = np.asarray(gradient_normalized,dtype=float).reshape(8,7)
    random = np.random.default_rng(seed).normal(size=(8,7))
    directions = [g,-g,random]
    deltas = []
    for direction in directions:
        norm = np.linalg.norm(direction)
        proposed = bounded(base+.05*ACTION_SCALE*direction/max(norm,1e-12))
        deltas.append((proposed-base)/ACTION_SCALE)
    norms = [np.linalg.norm(d)for d in deltas]
    common = min(norms)
    result = {'unchanged':base.copy()}
    for name,delta,norm in zip(ARMS[1:4],deltas,norms):
        result[name] = base+ACTION_SCALE*delta*(common/max(norm,1e-12))
    direction = np.asarray(mean_action).reshape(8,7)-base
    result['action_blind'] = bounded(base+.05*direction/max(np.linalg.norm(direction/ACTION_SCALE),1e-12))
    return result,dict(matched_normalized_norm=float(common), zero_feasible_direction=bool(common<1e-12),
                       random_seed=seed, redraws=0)


def cluster_interval(values, scenes, *, seed=1103, draws=2000):
    groups = sorted(set(scenes))
    if not groups:
        raise ValueError('No declared scene denominator')
    means = np.asarray([np.mean([v for v,s in zip(values,scenes)if s==g])for g in groups])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0,len(groups),(draws,len(groups)))].mean(1)
    return dict(mean=float(means.mean()), ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                scene_clusters=len(groups), states=len(values))
