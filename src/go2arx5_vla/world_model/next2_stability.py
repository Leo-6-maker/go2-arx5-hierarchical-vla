"""Exact free ARX companion operator at a fixed goal; not a control certificate."""
import numpy as np

def companion(model):
    n=model.dimension
    # Row-vector map from raw current/previous states to raw following state.
    k=model.weight[:-1]/model.feature_scale[:,None]*model.scale[None,:]
    current=np.eye(n)+(k[:n]+k[n:2*n])/model.scale[:,None]
    previous=-k[n:2*n]/model.scale[:,None]
    for j in range(3):
        current[model.layout['tcp_position_m'].start+j]-=100*k[2*n+j]
        current[model.layout['qpos'].start+j]-=100*k[2*n+10+j]
    return np.block([[current.T,previous.T],[np.eye(n),np.zeros((n,n))]])

def inspect(model):
    a=companion(model);eigenvalues=np.linalg.eigvals(a)
    return dict(companion_dimension=len(a),spectral_radius=float(np.abs(eigenvalues).max()),largest_eigenvalue_magnitudes=sorted(np.abs(eigenvalues).tolist(),reverse=True)[:8],largest_singular_value=float(np.linalg.svd(a,compute_uv=False)[0]),interpretation='Fixed-goal state+previous-state free recurrence. Large radius predicts error amplification; this is not closed-loop WBC stability or a global model safety guarantee.')
