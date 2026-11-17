"""Read-only source diagnostics; no data admission or deployment context.

The width and EEF envelope calculations are explicitly conditional on the
upstream numeric interpretation. They cannot establish command provenance.
"""
import hashlib
import json
from pathlib import Path
import time


def temporal_alignment(action, state, lags=range(-3, 4)):
    import numpy as np
    action, state = np.asarray(action), np.asarray(state)
    if action.shape != state.shape or action.ndim < 1:
        raise ValueError('MATCHING_RECORDED_ARRAYS_REQUIRED')
    rows = []
    for lag in lags:
        if abs(lag) >= len(action):
            continue
        a = action[max(-lag, 0):len(action)-max(lag, 0)]
        s = state[max(lag, 0):len(state)-max(-lag, 0)]
        equal = np.all((a == s).reshape(len(a), -1), axis=1)
        rows.append(dict(state_index_minus_action_index=lag, compared_rows=len(a),
            bit_value_equal_rows=int(equal.sum()), all_rows_equal=bool(equal.all()),
            maximum_absolute_difference=float(np.max(abs(a-s)))))
    return rows


def necessary_width_error_bounds(width, context_width, dt=.05, vmax=.04):
    """Lower bounds for ANY monotone and speed-bounded reconstruction.

    Context is fixed; errors are allowed only on the 40 reconstructed outputs.
    Thus a context/output violation has its full error, whereas two output
    samples can share a violation equally. No actual-command clock is created.
    """
    import numpy as np
    w = np.asarray(width, dtype=float)
    if w.shape != (40,) or not np.isfinite(w).all() or not np.isfinite(context_width):
        raise ValueError('FINITE_FULL_NUMERIC_WINDOW_REQUIRED')
    decrease = float(np.max(np.maximum.accumulate(w)-w))
    increase = float(np.max(w-np.minimum.accumulate(w)))
    inc = max(0., context_width-float(w.min()), decrease/2)
    dec = max(0., float(w.max())-context_width, increase/2)
    times = dt*(1+np.arange(40))
    speed = max(0., float(np.max(abs(w-context_width)-vmax*times)))
    for k in range(1, len(w)):
        speed = max(speed, float(np.max(abs(w[k:]-w[:-k])-vmax*dt*k))/2)
    return dict(monotonic_max_error_lower_bound_m=min(inc, dec),
        speed_max_error_lower_bound_m=speed,
        combined_max_error_lower_bound_m=max(min(inc, dec), speed))


def numeric_probe(path, frequency):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as source:
        fields = {g+'/'+arm+'_'+tail:np.asarray(source[g+'/'+arm+'_'+tail][()])
            for g in ('action','state') for arm in ('left','right')
            for tail in ('arm_joint_states','ee_poses','ee_joint_states')}
    lengths = {len(a) for a in fields.values()}
    if len(lengths) != 1 or not 0 < frequency < 1000:
        raise ValueError('SAME_LENGTH_FIELDS_AND_RECORDED_POSITIVE_RATE_REQUIRED')
    count = lengths.pop()
    fingerprint = hashlib.sha256(b''.join(k.encode()+fields[k].tobytes() for k in sorted(fields))).hexdigest()
    shifts = {arm+'_'+tail:temporal_alignment(fields['action/'+arm+'_'+tail], fields['state/'+arm+'_'+tail])
        for arm in ('left','right') for tail in ('arm_joint_states','ee_poses','ee_joint_states')}
    frame_t = np.arange(count)/frequency
    # Only nominal, half-second query anchors with a preceding source-grid
    # sample and a complete [query, query+1.95] future window are inspected.
    queries = np.arange(.5, frame_t[-1]-1.95+1e-10, .5)
    output = []
    for arm in ('left','right'):
        pose = fields['action/'+arm+'_ee_poses']
        grip = fields['action/'+arm+'_ee_joint_states'].reshape(-1)
        for q in queries:
            grid = q+.05*np.arange(40)
            for method in ('ZOH','LINEAR_SENSITIVITY_ONLY'):
                times = np.r_[q-.05, grid]
                if method == 'ZOH':
                    # Small roundoff tolerance only for exactly coincident
                    # nominal frame indices; never use a future frame.
                    indices = np.floor(times*frequency+1e-9).astype(int)
                    sampled = grip[indices]
                    p = pose[indices[1:], :3]
                else:
                    sampled = np.interp(times, frame_t, grip)
                    p = np.stack([np.interp(grid, frame_t, pose[:, k]) for k in range(3)], axis=1)
                # Conditional retargeting, not a measured source width claim.
                widths = .002+.040*sampled
                bound = necessary_width_error_bounds(widths[1:], widths[0])
                velocity = np.diff(p, axis=0)/.05
                acceleration = np.linalg.norm(np.diff(np.vstack([np.zeros(3),velocity,np.zeros(3)]),axis=0)/.05,axis=1)
                difference = np.diff(sampled[1:])
                direction = ('MIXED' if difference.min() < -1e-8 and difference.max() > 1e-8 else
                    'DECREASING' if difference.min() < -1e-8 else 'INCREASING' if difference.max() > 1e-8 else 'HOLD')
                output.append(dict(arm=arm,query_nominal_s=float(q),method=method,
                    normalized_range=[float(sampled.min()),float(sampled.max())],
                    normalized_direction=direction,range_exceeds_0_1=bool(sampled.min()<0 or sampled.max()>1),
                    equivalent_width_range_m=float(np.ptp(widths[1:])),
                    necessary_max_error_gate_failed=bound['combined_max_error_lower_bound_m']>.0002+1e-12,
                    **bound,conditional_EEF_head_acceleration=float(acceleration[0]),
                    conditional_EEF_tail_acceleration=float(acceleration[-1]),
                    conditional_EEF_interior_acceleration=float(acceleration[1:-1].max()),
                    conditional_EEF_any_target_envelope_failure=bool(acceleration.max()>.25+1e-12)))
    return dict(recorded_frames=count,nominal_duration_s=float(frame_t[-1]),numeric_fingerprint=fingerprint,
        temporal_alignment=shifts,windows=output,
        excluded_nominal_query_anchors_missing_context=1,
        future_window_points=40,grid_step_s=.05,first_point_offset_s=0.,
        context_semantics='SOURCE_NOMINAL_GRID_ONLY_NOT_ACTUAL_EMITTED_REFERENCE',
        resampling_admission='UNDECIDED_PENDING_COMMAND_PRODUCER',
        source_EEF_frame_TCP_calibration='NOT_ESTABLISHED_BY_NUMERIC_PROBE',
        model_generations=0,native_requests=0,optimizer_updates=0)


def inspect(root):
    from .robodojo_source_audit_v1 import sha, read
    root = Path(root)
    result = read(root/'RESULT.private.json')
    if result['exit_code'] or result['profiled_episodes'] != 12:
        raise ValueError('COMPLETE_FIXED_FIRST12_REQUIRED_NO_REDOWNLOAD')
    rows = []
    for reference in result['results']:
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('PROFILE_IDENTITY_CHANGED')
        record = read(reference['path'])
        if sha(record['transfer']['path']) != record['source']['sha256']:
            raise ValueError('RAW_PIN_CHANGED')
        row = numeric_probe(record['transfer']['path'], record['profile']['frequency'])
        rows.append(dict(source=record['source'],profile_sha256=reference['sha256'],**row))
    return dict(schema='SLA_ROBODOJO_FIRST12_NUMERIC_READONLY_DIAGNOSTICS',records=rows,
        source_native_retained=True,source_canonical_qualified=False,
        command_provenance='UNKNOWN_NOT_INFERRED_FROM_TEMPORAL_CORRELATION',
        no_previous_actual_emitted_context_created=True,model_generations=0,native_requests=0,
        optimizer_updates=0,optimization_visits=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    result = inspect(args.root)
    result.update(source_sha=args.source_sha, wall_s=time.monotonic()-started,
        CPU_s=time.process_time()-cpu, epoch=time.time())
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result_sha256=hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        episodes=len(result['records']), wall_s=result['wall_s'], CPU_s=result['CPU_s'])))
