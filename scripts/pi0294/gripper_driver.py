"""One declared task-independent finger motor intervention; original cap stays."""
from __future__ import annotations
import math
import numpy as np


def force(raw, velocity, bias, spec):
    raw,velocity,bias=(np.asarray(x,dtype=float) for x in (raw,velocity,bias))
    if any(x.shape!=(2,) or not np.isfinite(x).all() for x in (raw,velocity,bias)):
        raise ValueError('FINITE_NATIVE_TWO_FINGER_INPUT_REQUIRED')
    if spec['kind']=='VELOCITY_FEEDBACK':
        expected=2*math.sqrt(400.*spec['inertia_diagonal_bound_kg'])-3.
        kd=float(spec['additional_damping_N_s_m'])
        if kd<=0 or not math.isclose(kd,expected,rel_tol=0,abs_tol=1e-12):
            raise ValueError('FROZEN_MASS_DERIVED_DISSIPATIVE_GAIN_REQUIRED')
        extra=-kd*velocity
        if np.any(extra*velocity>1e-12):raise ValueError('DISSIPATIVE_TERM_CANNOT_INJECT_POWER')
    elif spec['kind']=='BIAS_FEEDFORWARD':
        extra=bias
    else:
        raise ValueError('ONLY_TWO_PREDECLARED_SINGLE_MECHANISMS_ALLOWED')
    return raw+extra


def bind(engine,spec):
    from scripts.pi0291_arx5_data.config import PARAMETERS as P
    if (not spec['frozen_before_stress'] or not spec['evidence_sha256'] or P['finger_kp']!=400.
            or P['finger_force_max_N']!=8. or P['finger_target_speed_m_s']!=.04
            or engine.model.opt.timestep!=.002
            or not np.all(engine.model.actuator_gear[engine.finger_ctrl,0]==1.)):
        raise ValueError('ORIGINAL_DRIVER_CAP_SPEED_GAIN_DT_AND_UNIT_GEAR_REQUIRED')
    engine.finger_driver_version=spec['version']
    engine.finger_driver=lambda e,raw: force(raw,e.data.qvel[e.finger_v],e.data.qfrc_bias[e.finger_v],spec)
    return engine


def self_check():
    spec=dict(kind='VELOCITY_FEEDBACK',inertia_diagonal_bound_kg=.091402,
              additional_damping_N_s_m=2*math.sqrt(400*.091402)-3)
    raw=np.array([.5,-.5]);velocity=np.array([.1,-.1])
    actual=force(raw,velocity,[0,0],spec)
    assert np.all((actual-raw)*velocity<=0)
    assert np.array_equal(force(raw,[0,0],[999,-999],spec),raw), 'velocity-only does not add bias'
    assert np.all(np.abs(np.clip(force([100,-100],velocity,[0,0],spec),-8,8))<=8)
    try:force(raw,velocity,[0,0],dict(spec,additional_damping_N_s_m=-1))
    except ValueError:pass
    else:raise AssertionError('negative damping accepted')
    assert np.array_equal(force(raw,velocity,[.01,-.02],dict(kind='BIAS_FEEDFORWARD')),raw+[.01,-.02])
    print('PASS_CPU_SINGLE_MECHANISM_POWER_GAIN_AND_ORIGINAL_FORCE_CAP')


if __name__=='__main__':self_check()
