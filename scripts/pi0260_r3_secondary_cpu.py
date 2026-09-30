"""Saved-array-only R3 ambiguity and task breakdown; no model calls."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def summarize(values):
    array = np.asarray(values, np.float64)
    return dict(n=len(array), median_xyz_m=float(np.median(array)),
                p95_xyz_m=float(np.percentile(array, 95)))


def main(root):
    audit = read(root / 'condition_audit.json')
    alias_keys = {split: {row['key'] for row in audit['partitions'][split]['aliases']}
                  for split in ('TRAIN', 'KIN-VAL')}
    result = dict(schema='PI0260_R3_SAVED_ARRAY_SECONDARY_V1',
                  status='PASS_SAVED_ARRAY_SECONDARY',
                  condition_audit_sha256=sha(root / 'condition_audit.json'),
                  interpretation='Descriptive fixed-case diagnostics only; primary labels, gates, and denominators unchanged.',
                  points={}, model_queries=0, physics_steps=0)
    for point in ('Base_0', 'Base_6000', 'G_0', 'G_6000'):
        folder = root / 'eval' / point
        cpu_path, semantic_path = folder / 'cpu_replay.json', folder / 'semantic_replay.json'
        cpu, semantic = read(cpu_path), read(semantic_path)
        assert cpu['status'] == 'PASS_INDEPENDENT_CPU_SAVED_ARRAY_REPLAY' and cpu['cells'] == 440
        assert semantic['status'] == 'PASS_SAVED_ARRAY_SEMANTIC_REFERENCE_MARGIN_REPLAY'
        assert len(semantic['rows']) == 96 and cpu['repeated_condition_max_abs'] == 0
        rows = []
        floor_groups = defaultdict(list)
        for cell in sorted((folder / 'cells').iterdir()):
            if not cell.is_dir():
                continue
            record = read(cell / 'record.json')
            assert record['status'] == 'VALID' and record['point'] == point
            if not cell.name.startswith('Motion2-'):
                continue
            source = record['source']
            split = source['kind']
            row = dict(task=source['task'], noise=record['noise_name'], kind=split,
                       family=source['sequence'].split('-M1-')[0].split('-M3-')[0],
                       quartile=source.get('quartile'), condition=record['condition_sha256'],
                       xyz=float(record['metrics']['xyz_rmse_m']),
                       so3=float(record['metrics']['so3_rmse_rad']),
                       gate=record['gate']['decision'])
            if split in alias_keys:
                row['ambiguity'] = 'alias' if row['condition'] in alias_keys[split] else 'singleton'
            rows.append(row)
            if split == 'KIN-VAL':
                tensor = cell / 'tensors.npz'
                assert sha(tensor) == record['tensor_sha256']
                with np.load(tensor, allow_pickle=False) as saved:
                    reference = saved['expert'][:, :3].astype(np.float64)
                    predicted = saved['decoded'][0, :, :3].astype(np.float64)
                floor_groups[(row['task'], row['noise'], row['condition'])].append((reference, predicted))
        assert len(rows) == 312
        breakdown = defaultdict(list)
        families = defaultdict(list)
        for row in rows:
            if row['kind'] in ('TRAIN', 'KIN-VAL'):
                breakdown[(row['task'], row['noise'], row['kind'], row['ambiguity'])].append(row['xyz'])
                families[(row['task'], row['noise'], row['kind'], row['family'])].append(row['xyz'])
                if row['quartile'] is not None:
                    breakdown[(row['task'], row['noise'], row['kind'], 'quartile' + str(row['quartile']))].append(row['xyz'])
        floor = defaultdict(lambda: dict(rows=0, aliased_rows=0, groups=0, mse=0., floor=0., max_same_noise_abs=0.))
        for (task, noise, _), items in floor_groups.items():
            references = np.stack([item[0] for item in items])
            predictions = np.stack([item[1] for item in items])
            same = float(np.max(np.abs(predictions - predictions[0])))
            assert same <= 1e-6
            mean_reference = references.mean(axis=0)
            group_floor = float(np.mean(np.sum((references - mean_reference) ** 2, axis=-1)))
            group_mse = float(np.mean(np.sum((predictions - references) ** 2, axis=-1)))
            assert group_mse + 1e-10 >= group_floor
            bucket = floor[(task, noise)]
            bucket['rows'] += len(items)
            bucket['aliased_rows'] += len(items) if len(items) > 1 else 0
            bucket['groups'] += 1
            bucket['mse'] += len(items) * group_mse
            bucket['floor'] += len(items) * group_floor
            bucket['max_same_noise_abs'] = max(bucket['max_same_noise_abs'], same)
        floor_summary = {}
        for key, bucket in floor.items():
            n = bucket['rows']
            floor_summary['/'.join(key)] = dict(n=n, aliased_rows=bucket['aliased_rows'],
                                               groups=bucket['groups'],
                                               model_mean_squared_xyz_m2=bucket['mse'] / n,
                                               empirical_reference_floor_m2=bucket['floor'] / n,
                                               max_same_noise_abs_m=bucket['max_same_noise_abs'])
            assert n == 16
        margins = defaultdict(list)
        for row in semantic['rows']:
            margins[(row['task'], row['kind'], row['noise'], str(row['command']))].append(float(row['margin_m']))
        margin_summary = {'/'.join(key): dict(n=len(values), correct=sum(value > 0 for value in values),
                                              median_margin_m=float(np.median(values)))
                          for key, values in margins.items()}
        family_summary = {'/'.join(key): summarize(values) for key, values in families.items()}
        result['points'][point] = dict(cpu_sha256=sha(cpu_path), semantic_sha256=sha(semantic_path),
                                       ambiguity_phase={'/'.join(key): summarize(values) for key, values in breakdown.items()},
                                       families=family_summary, semantic_command_margins=margin_summary,
                                       kin_val_squared_xyz_vs_floor=floor_summary)
    output = root / 'eval' / 'SECONDARY_SAVED_ARRAY_REPORT.json'
    assert not output.exists()
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], sha256=sha(output), points=list(result['points']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root)
