"""Read-only saved-array replay and fixed quality summaries for PI0260-R3."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pi0260_r3_eval import R3, card, read, sha, source


def metrics(predicted, expert):
    from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix
    predicted, expert = np.asarray(predicted, np.float64), np.asarray(expert, np.float64)
    error = np.linalg.norm(predicted[:, :3] - expert[:, :3], axis=1)
    rotations = rot6d_rows_to_matrix(predicted[:, 3:9])
    reference = rot6d_rows_to_matrix(expert[:, 3:9])
    angle = geodesic_angle(rotations, reference)
    return dict(xyz_rmse_m=float(np.sqrt(np.mean(error**2))), so3_rmse_rad=float(np.sqrt(np.mean(angle**2))),
                gripper_mae=float(np.mean(np.abs(predicted[:, 9]-expert[:, 9]))),
                first_xyz_error_m=float(error[0]), endpoint_xyz_error_m=float(error[-1]),
                max_linear_speed_mps=float(np.max(np.linalg.norm(np.diff(predicted[:, :3], axis=0), axis=1))/.05),
                max_angular_speed_radps=float(np.max(geodesic_angle(rotations[:-1], rotations[1:]))/.05))


def replay(point):
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    plan = card()
    arm, step = point.split('_')
    fixed = [r for r in plan['rows'] if r['arm'] == arm and r['step'] == int(step)]
    assert len(fixed) == 440
    motion = read(R3 / 'motion2_eval_cases.json')['cases']
    original = read(R3 / 'O64_eval_cases.json')['rows']
    noises = {}
    for name in ('B0', 'B1'):
        with np.load(R3 / 'noise' / (name + '.npz'), allow_pickle=False) as values:
            noises[name] = values['float32'].copy()
    groups, semantic = defaultdict(list), {}
    repeated = defaultdict(list)
    for row in fixed:
        cell_id = f"{row['cohort']}-{row['case_index']:03d}-{row['noise']}"
        cell = R3 / 'eval' / point / 'cells' / cell_id
        record = read(cell / 'record.json')
        assert record['status'] == 'VALID' and record['point'] == point and record['cell_id'] == cell_id
        assert record['eval_card_sha256'] == sha(R3 / 'S4_FIXED_EVAL_CARD.json')
        tensor = cell / 'tensors.npz'
        assert sha(tensor) == record['tensor_sha256']
        packet, expert, case = source(row, motion, original)
        assert case == record['source']
        with np.load(tensor, allow_pickle=False) as saved:
            assert np.array_equal(saved['expert'], expert) and np.array_equal(saved['state'], packet['state_v1'])
            assert np.array_equal(saved['noise'], noises[row['noise']])
            assert saved['raw'].shape == (1, 40, 132) and saved['decoded'].shape == (1, 40, 10)
            assert saved['trace_x'].shape == saved['trace_v'].shape == (4, 1, 40, 132)
            predicted = saved['decoded'][0].copy()
            raw = saved['raw'].copy()
        observed = metrics(predicted, expert)
        assert all(abs(observed[k] - record['metrics'][k]) <= 1e-9 for k in observed), cell_id
        state = packet['state_v1']
        orientation = matrix_to_quat(rot6d_rows_to_matrix(state[None, 3:9]))[0]
        gate = TrajectorySafetyGateV1().evaluate(selector._build_chunk(predicted[None], chunk_id='pi0260-r3-' + cell_id),
                                                 now=0., achieved_position=state[:3], achieved_orientation=orientation)
        assert gate.decision.value == record['gate']['decision']
        assert [x.code for x in gate.issues] == [x['code'] for x in record['gate']['issues']]
        task = case.get('task', 'O64')
        groups[task, row['noise'], case.get('kind', 'O64')].append(dict(**observed, reject=gate.decision.value == 'REJECT'))
        repeated[row['noise'], record['condition_sha256']].append(raw)
        if case.get('kind') in ('SEMANTIC', 'PARAPHRASE'):
            semantic[task, row['noise'], case['kind'], case['family'], case['command']] = predicted
    assert sum(len(x) for x in groups.values()) == 440
    summary = {}
    for key, values in sorted(groups.items()):
        xyz = np.asarray([x['xyz_rmse_m'] for x in values])
        so3 = np.asarray([x['so3_rmse_rad'] for x in values])
        summary['/'.join(key)] = dict(n=len(values), median_xyz_m=float(np.median(xyz)),
                                     p95_xyz_m=float(np.percentile(xyz, 95)), median_so3_rad=float(np.median(so3)),
                                     reject_count=sum(x['reject'] for x in values))
    for task in ('M1', 'M3'):
        for noise in ('B0', 'B1'):
            group = summary[f'{task}/{noise}/KIN-VAL']
            assert group['n'] == 16
            group['quality_pass'] = (group['median_xyz_m'] <= .020 and group['p95_xyz_m'] <= .035
                                     and group['median_so3_rad'] <= .050 and group['reject_count'] == 0)
    consistency = [float(np.max(np.abs(x - values[0]))) for values in repeated.values() if len(values) > 1 for x in values[1:]]
    result = dict(status='PASS_INDEPENDENT_CPU_SAVED_ARRAY_REPLAY', point=point, cells=440,
                  groups=summary, repeated_condition_max_abs=max(consistency, default=0.),
                  semantic_predictions=len(semantic), model_queries=0, selected_checkpoint='NONE')
    output = R3 / 'eval' / point / 'cpu_replay.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=result['status'], point=point, rows=440, sha256=sha(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', choices=('Base_0', 'Base_6000', 'G_0', 'G_6000'))
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    if args.demo:
        exact = np.zeros((40, 10), np.float32)
        exact[:, 3] = exact[:, 7] = 1.
        assert all(value == 0. for value in metrics(exact, exact).values())
        print('PASS_R3_INDEPENDENT_CPU_METRIC_DEMO')
    else:
        assert args.point
        replay(args.point)
