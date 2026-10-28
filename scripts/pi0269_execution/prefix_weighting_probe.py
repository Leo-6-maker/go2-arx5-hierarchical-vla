"""Constructed CPU arithmetic fixtures, NOT a model or project trajectory replay.

Shows a small full-H40 RMSE can coexist with backwards executed-prefix motion.
Checks the proposed temporal reducers without changing any model/runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def weights(kind: str, transitions: bool = False) -> np.ndarray:
    n, prefix = (39, 8) if transitions else (40, 9)
    if kind == 'UNIFORM':
        return np.full(n, 1.0 / n, dtype=np.float64)
    if kind != 'PREFIX80':
        raise ValueError('Unknown temporal weighting contract')
    return np.r_[np.full(prefix, .8 / prefix), np.full(n-prefix, .2 / (n-prefix))]


def reduce_errors(errors, valid, kind: str, transitions: bool = False) -> float:
    e = np.asarray(errors, dtype=np.float64)
    m = np.asarray(valid)
    w = weights(kind, transitions)
    if e.shape != w.shape or m.shape != w.shape or m.dtype != np.bool_:
        raise ValueError('Expected one horizon vector and a Boolean validity mask')
    if not np.isfinite(e).all() or (e < 0).any() or not m.any():
        raise ValueError('Invalid squared errors or empty support')
    wm = w * m
    return float(np.dot(e, wm) / wm.sum())


def main():
    checks = []
    for transitions in (False, True):
        n, p = (39, 8) if transitions else (40, 9)
        valid = np.ones(n, dtype=bool)
        wu, wp = weights('UNIFORM', transitions), weights('PREFIX80', transitions)
        assert (wp > 0).all() and np.isclose(wp.sum(), 1.)
        assert np.isclose(wp[:p].sum(), .8) and np.isclose(wp[p:].sum(), .2)
        checks.append(f'positive_unit_mass_and_80_20_{n}')
        assert np.isclose(reduce_errors(np.ones(n)*7,valid,'UNIFORM',transitions),
                          reduce_errors(np.ones(n)*7,valid,'PREFIX80',transitions))
        assert reduce_errors(np.zeros(n),valid,'PREFIX80',transitions) == 0.
        checks.append(f'constant_error_scale_and_zero_optimum_{n}')
        e=np.arange(n,dtype=float); mask=valid.copy(); mask[0]=False
        original=reduce_errors(e,mask,'PREFIX80',transitions); e[0]=1e9
        assert reduce_errors(e,mask,'PREFIX80',transitions)==original
        checks.append(f'mask_not_repurposed_{n}')
    t=np.arange(40)*.05
    expert=np.c_[.02*t,np.zeros((40,2))]
    prediction=expert.copy()
    prediction[1:9,0] *= -1
    squared=np.sum((prediction-expert)**2,axis=1)
    rmse=float(np.sqrt(squared.mean()))
    assert rmse < .005 and expert[8,0] == .008 and prediction[8,0] == -.008
    checks.append('low_window_rmse_with_reverse_execution_constructed')
    peak=float(np.linalg.norm(np.diff(prediction,axis=0),axis=1).max()/.05)
    # Only a mathematical speed check; NOT the project's SafetyGate.
    assert peak < 2.
    checks.append('constructed_linear_speed_below_2_without_gate_claim')
    late=np.zeros(40);late[32:40]=squared[1:9]
    valid=np.ones(40,dtype=bool)
    assert np.isclose(reduce_errors(squared,valid,'UNIFORM'),reduce_errors(late,valid,'UNIFORM'))
    assert reduce_errors(squared,valid,'PREFIX80') > reduce_errors(late,valid,'PREFIX80')
    checks.append('temporal_location_not_global_weight_scaling')
    for bad in ('BAD',):
        try: weights(bad)
        except ValueError: pass
        else: raise AssertionError('invalid contract accepted')
    try: reduce_errors(np.zeros(40),np.zeros(40,dtype=bool),'PREFIX80')
    except ValueError: pass
    else: raise AssertionError('empty support accepted')
    checks.append('invalid_contract_and_empty_support_rejected')
    return dict(status='PASS_CONSTRUCTED_CPU_TEMPORAL_FIXTURES',checks=checks,
                source_scope='Synthetic arithmetic only; no original trajectory, learned forward/backward, renderer, WBC or gate execution.',
                prefix_grid=dict(H=40,dt_s=.05,last_executed_index=8,execution_s=.4),
                weights=dict(uniform_prefix_point_mass=9/40,prefix80_point_mass=.8,
                             prefix_point_weight_multiplier=(.8/9)/(1/40),
                             tail_point_weight_multiplier=(.2/31)/(1/40),
                             uniform_prefix_transition_mass=8/39),
                constructed_reverse=dict(full_h40_xyz_rmse_m=rmse,
                    expert_executed_endpoint_m=expert[8].tolist(),
                    prediction_executed_endpoint_m=prediction[8].tolist(),
                    prefix_endpoint_error_m=float(np.linalg.norm(prediction[8]-expert[8])),
                    linear_speed_peak_mps=peak,
                    uniform_position_term=reduce_errors(squared/.05**2,valid,'UNIFORM'),
                    prefix80_position_term=reduce_errors(squared/.05**2,valid,'PREFIX80')),
                interpretation='Equal perfect-fit minima; temporal reweighting does not prove stability or identify the cause of actual PI0268 failure.',
                model_queries=0,optimizer_updates=0,renderer_calls=0,physics_calls=0,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result=main(); a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
