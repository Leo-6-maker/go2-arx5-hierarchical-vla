"""Compare nominal H40, receiver endpoint and measured TCP at sealed P10 boundaries."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.world_model.storage import digest
from scripts.wm02.audit_d4_native_cases import read


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite recorded translation')
    if not len(values):
        return None
    return dict(min=float(values.min()), p50=float(np.quantile(values, .5)),
                p95=float(np.quantile(values, .95)), max=float(values.max()))


def observe(case_file):
    spec = read(case_file)
    case_root = case_file.parent
    lease = read(case_root/'control/LEASE.json')
    if lease['status'] != 'CLOSED' or lease['spec_sha256'] != digest(case_file):
        raise ValueError('An original sealed case is required')
    result_file = case_root/'RESULT.private.json'
    result = read(result_file)
    native = case_root/'native-01'
    commands = {}
    with (native/'COMMANDS.private.jsonl').open() as stream:
        for row in map(json.loads,stream):
            if row['phase'] == 'v6-g7-original-h40':
                if row['query'] in commands:
                    raise ValueError('Duplicate H40 command event')
                commands[row['query']] = row
    samples = {}
    with (native/'MOBILE_SAMPLES.private.jsonl').open() as stream:
        for row in map(json.loads,stream):
            if row['native_start_step'] >= 5703:
                if row['native_start_step'] in samples:
                    raise ValueError('Duplicate native sample start clock')
                samples[row['native_start_step']] = row

    frame = np.asarray(spec['policy_from_world'], dtype=float)
    world = np.linalg.inv(frame)
    links = []
    checked_samples = set()
    queries = 0
    for path in sorted((native/'g7-queries').glob('*/QUERY_ACTUAL.private.json')):
        query = read(path)
        action = query['action']
        if digest(action['path']) != action['sha256']:
            raise ValueError('An original H40 action is required')
        queries += 1
        intervals = query.get('confirmed_intervals', [])
        if len(intervals) != query['confirmed_atoms']:
            raise ValueError('Incomplete confirmed interval ledger')
        if not intervals:
            continue
        event = commands[query['query']]
        with np.load(action['path'], allow_pickle=False) as data:
            raw = data['physical_H40']
            if event['action_sha256'] != action['sha256'] or not np.array_equal(raw,event['raw_policy_H40']):
                raise ValueError('Command event and archived model action differ')
            mapped = raw[:, :3] @ world[:3, :3].T + world[:3, 3]
        if not np.allclose(mapped,np.asarray(event['mapped_target_H40'])[:, :3],rtol=0,atol=1e-10):
            raise ValueError('Policy and world nominal actions are inconsistent')
        for interval in intervals:
            start = interval['native_start_step']
            end = interval['native_end_step']
            index = interval['atom']
            row = samples[start]
            accepted = interval['accepted_boundary']
            if (row['native_end_step'] != end or end != start + 25
                    or start != query['native_start_step'] + 25*index
                    or row['installed_reference_sha256'] != interval['wire_sha256']
                    or accepted['accepted_wire_sha256'] != interval['wire_sha256']
                    or not accepted['exact_receiver_knot']
                    or abs(accepted['receiver_knot_time_s'] - end*.002) > 1e-8
                    or start in checked_samples):
                raise ValueError('Native sample and acknowledged P10 interval differ')
            checked_samples.add(start)
            nominal = mapped[index+1]
            installed = np.asarray(row['target_endpoint_EEF10'][:3])
            receiver = np.asarray(accepted['accepted_pose7'][:3])
            actual = np.asarray(row['actual_TCP'])[:3, 3]
            target = np.asarray(row['object_pose'])[:3, 3]
            distance = lambda a, b: float(np.linalg.norm(a-b))*1000.
            links.append(dict(reanchor_mm=distance(nominal,installed), tracking_mm=distance(actual,receiver),
                              installed_vs_receiver_mm=distance(installed,receiver),
                              actual_vs_nominal_mm=distance(actual,nominal),
                              nominal_to_object_mm=distance(nominal,target),
                              receiver_to_object_mm=distance(receiver,target),
                              actual_to_object_mm=distance(actual,target),
                              receiver_minus_nominal_target_distance_mm=(distance(receiver,target)
                                                                        - distance(nominal,target)),
                              actual_minus_nominal_target_distance_mm=(distance(actual,target)
                                                                      - distance(nominal,target))))
    if queries != result['model_generations']:
        raise ValueError('Recorded model generation denominator changed')
    return dict(case=case_root.name, arm=spec['arm'], family=spec['mobile_condition']['family'],
                native_result_sha256=digest(result_file), native_ticks=result['actual_native_integrations'],
                model_generations=queries, matched_atoms=len(links),
                native_rows_outside_confirmed_atoms=len(samples)-len(checked_samples),
                metrics={key:summarize([row[key] for row in links]) for key in links[0]} if links else {},
                selected_MPC_reference='UNKNOWN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent',type=Path)
    parser.add_argument('spec_sha256')
    parser.add_argument('--count',type=int,required=True)
    args = parser.parse_args()
    path = args.parent/'EXECUTION_SPEC.json'
    if digest(path) != args.spec_sha256 or not 1 <= args.count <= 24:
        raise ValueError('A bounded original GRASP24 panel is required')
    spec = read(path)
    if spec['evaluation_stage'] != 'D5_GRASP24':
        raise ValueError('Expected the GRASP24 experiment')
    cases = [observe(Path(file)) for file in spec['cases'][:args.count]]
    print(json.dumps(dict(spec_sha256=args.spec_sha256,source_sha=spec['source_sha'],cases=cases,
                          new_model_generations=0,new_native_requests=0,new_native_ticks=0,optimizer_updates=0),
                     allow_nan=False))


if __name__ == '__main__':
    main()
