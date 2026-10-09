"""Hard limits come from the loaded asset, never the native reward soft envelope."""
import numpy as np


def hard_limit_report(names, positions, properties):
    q = np.asarray(positions, dtype=float)
    lower = np.asarray(properties['lower'], dtype=float)
    upper = np.asarray(properties['upper'], dtype=float)
    if q.shape != lower.shape or upper.shape != lower.shape or len(names) != q.size:
        raise ValueError('Hard limit dimensions disagree')
    if not np.isfinite(np.r_[q, lower, upper]).all() or np.any(lower > upper):
        raise ValueError('Invalid hard limit state')
    penetration = np.maximum(np.maximum(lower - q, q - upper), 0)
    return dict(hard_limit=bool(np.any(penetration > 0)), positions=q.tolist(),
                lower=lower.tolist(), upper=upper.tolist(),
                penetration=penetration.tolist(),
                offending=[names[i] for i in np.flatnonzero(penetration > 0)],
                limit_source='ACTUAL_LOADED_ACTOR_DOF_PROPERTIES_UNSHRUNK')
