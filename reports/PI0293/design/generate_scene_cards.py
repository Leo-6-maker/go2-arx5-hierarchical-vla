"""PI0293 deterministic scene-group design; no simulator/model execution.

The output is a proposal, not a reachability or physics qualification. All three
language tasks in a group must share its initial geometry, cameras and state.
Run and seal this generator on the owner machine before formal collection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
from pathlib import Path

DOMAIN = 'PI0293-MULTITASK-SCENE-V1'
FIELDS = ('group_id', 'split', 'batch', 'family', 'seed_u64', 'source_x_mm',
          'source_y_mm', 'cube_yaw_mrad', 'north_x_mm', 'north_y_mm',
          'south_x_mm', 'south_y_mm', 'blue_side', 'table_gray_milli',
          'light_scale_milli', 'cam_dx_mm', 'cam_dy_mm', 'cam_dz_mm')
SPLITS = (('TRAIN', 48), ('DEV', 12), ('CONFIRM', 12))


def integer(group: str, key: str, low: int, high: int) -> int:
    if low > high:
        raise ValueError('Invalid integer interval')
    digest = hashlib.sha256(f'{DOMAIN}/{group}/{key}'.encode('utf-8')).digest()
    return low + int.from_bytes(digest[:8], 'big') % (high - low + 1)


def records() -> list[dict]:
    rows = []
    for split, count in SPLITS:
        for index in range(count):
            group = f'{split}-G{index:03d}'
            family = f'F{index % 4}'
            wide = family != 'F0'
            x = integer(group, 'source_x', 430 if wide else 438, 500 if wide else 462)
            rows.append(dict(
                group_id=group, split=split,
                batch=('A' if index < 24 else 'B') if split == 'TRAIN' else split,
                family=family,
                seed_u64=str(integer(group, 'physics_seed', 0, 2**64-1)),
                source_x_mm=x, source_y_mm=integer(group, 'source_y', -10, 10),
                cube_yaw_mrad=integer(group, 'yaw', -250 if wide else -100,
                                     250 if wide else 100),
                north_x_mm=x+integer(group, 'north_dx', -10, 10),
                north_y_mm=integer(group, 'north_y', 135, 150),
                south_x_mm=x+integer(group, 'south_dx', -10, 10),
                south_y_mm=-integer(group, 'south_y', 135, 150),
                blue_side='north' if (index // 4) % 2 == 0 else 'south',
                table_gray_milli=integer(group, 'table_gray', 550, 780)
                    if family == 'F2' else 690,
                light_scale_milli=integer(group, 'light', 800, 1150)
                    if family == 'F2' else 1000,
                cam_dx_mm=integer(group, 'cam_x', -30, 30) if family == 'F3' else 0,
                cam_dy_mm=integer(group, 'cam_y', -30, 30) if family == 'F3' else 0,
                cam_dz_mm=integer(group, 'cam_z', -20, 20) if family == 'F3' else 0))
    return rows


def validate(rows: list[dict]) -> None:
    if len(rows) != 72 or len({r['group_id'] for r in rows}) != 72:
        raise ValueError('Scene group cardinality/identity mismatch')
    geometries = set()
    for row in rows:
        if not 0 <= int(row['seed_u64']) < 2**64:
            raise ValueError('Invalid uint64 seed')
        if row['blue_side'] not in ('north', 'south'):
            raise ValueError('Invalid target-color assignment')
        for side in ('north', 'south'):
            gx, gy = row[f'{side}_x_mm'], row[f'{side}_y_mm']
            if not (330 <= gx-80 and gx+80 <= 670 and -240 <= gy-80 and gy+80 <= 240):
                raise ValueError(f'Target outside table: {row["group_id"]}')
            dx, dy = gx-row['source_x_mm'], gy-row['source_y_mm']
            if dx*dx+dy*dy < 125**2:
                raise ValueError('Insufficient source-target distance')
        if row['north_y_mm']-row['south_y_mm'] <= 160:
            raise ValueError('Overlapping target zones')
        key = tuple(row[k] for k in FIELDS[5:])
        if key in geometries:
            raise ValueError('Duplicate full scene across groups')
        geometries.add(key)
    for split, n in SPLITS:
        if sum(r['split'] == split for r in rows) != n:
            raise ValueError('Split cardinality mismatch')
        for family in ('F0', 'F1', 'F2', 'F3'):
            if sum(r['split'] == split and r['family'] == family for r in rows) != n//4:
                raise ValueError('Family cardinality mismatch')
    for family in ('F0', 'F1', 'F2', 'F3'):
        train = [r for r in rows if r['split'] == 'TRAIN' and r['family'] == family]
        if sum(r['blue_side'] == 'north' for r in train) != 6:
            raise ValueError('Training color/side assignment unbalanced')
        a = [r for r in train if r['batch'] == 'A']
        if len(a) != 6 or sum(r['blue_side'] == 'north' for r in a) != 3:
            raise ValueError('Drop-A color/side assignment unbalanced')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = records()
    validate(rows)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode('utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        handle.write(payload)
    print(f'PROPOSAL_ONLY groups={len(rows)} sha256={hashlib.sha256(payload).hexdigest()}')
    print('Static bounds checked. Reachability, renderer and physical teacher are UNTESTED.')


if __name__ == '__main__':
    main()
