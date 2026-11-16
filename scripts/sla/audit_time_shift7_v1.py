"""Read saved trajectories only; keep all96 original conditions and failures."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha


def analyze(control_file, candidate_file):
    control = json.loads(Path(control_file).read_bytes()); candidate = json.loads(Path(candidate_file).read_bytes())
    if len(control) != 96 or len(candidate) != 96:
        raise ValueError('ORIGINAL_FIXED48_TIMES_NOISE2_REQUIRED')
    records = []; lift = []; failures = []
    for a, b in zip(control, candidate):
        if (a['index'], a['noise'], a['phase']) != (b['index'], b['noise'], b['phase']):
            raise ValueError('MATCHED_WINDOW_NOISE_PHASE_REQUIRED')
        conditions = {}
        for name, row in (('original_time500', a), ('shift7_500', b)):
            output = row['output']
            if sha(output['path']) != output['sha256']:
                raise ValueError('SEALED_PREDICTION_CHANGED')
            with np.load(output['path'], allow_pickle=False) as z:
                x = z['physical_H40'].copy()
            velocity = np.diff(x[:, :3], axis=0) / .05
            acceleration = np.linalg.norm(np.diff(np.vstack((np.zeros(3), velocity, np.zeros(3))), axis=0) / .05, axis=1)
            saved = row['envelope']['paths']['nominal']['channels']['linear_acceleration']
            if not np.isclose(acceleration.max(), saved['peak_value'], rtol=1e-8, atol=1e-9):
                raise ValueError('ORIGINAL_ACCELERATION_ARITHMETIC_REQUIRED')
            conditions[name] = dict(original_full_H40_pass=row['envelope']['original_full_H40_pass'],
                linear_acceleration_peak_m_s2=float(acceleration.max()), peak_k=int(acceleration.argmax()),
                fixed_k_m_s2={str(k):float(acceleration[k]) for k in (6, 7, 13, 14, 20, 21, 26, 27, 33, 34)},
                first_original_failure=row['envelope']['first_temporal_exceedance'])
        r = dict(index=a['index'], noise=a['noise'], phase=a['phase'], conditions=conditions)
        records.append(r)
        if a['phase'] == 'LIFT': lift.append(r)
        if not b['envelope']['original_full_H40_pass']: failures.append(r)
    if len(lift) != 20: raise ValueError('ALL20_FIXED_LIFT_PREDICTIONS_REQUIRED')
    summary = {}
    for name in ('original_time500', 'shift7_500'):
        summary[name] = dict(LIFT_count=20, LIFT_pass=sum(r['conditions'][name]['original_full_H40_pass'] for r in lift),
            LIFT_peak_k_counts=dict(Counter(str(r['conditions'][name]['peak_k']) for r in lift)),
            LIFT_acceleration_at_fixed_k={str(k):dict(maximum=max(r['conditions'][name]['fixed_k_m_s2'][str(k)] for r in lift),
                mean=float(np.mean([r['conditions'][name]['fixed_k_m_s2'][str(k)] for r in lift]))) for k in (6, 13, 20, 26, 33)})
    return dict(schema='SLA_MATCHED_SHIFT7_FIXED96_BOUNDARY_CURVATURE_ACTUAL_V1', records=records, summary=summary,
        candidate_failed=len(failures), candidate_failure_phases=dict(Counter(r['phase'] for r in failures)),
        candidate_first_channels=dict(Counter(r['conditions']['shift7_500']['first_original_failure']['channel'] for r in failures)),
        parameter_block_shift=7, old_boundaries=[14, 27], new_boundaries=[7, 21, 34],
        boundary_coupling_causality_not_automatically_proven=True, original_scores_changed=False,
        new_model_calls=0, new_updates=0, new_native_requests=0, new_integrations=0)
