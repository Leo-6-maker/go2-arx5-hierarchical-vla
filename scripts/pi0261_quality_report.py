"""Render the frozen donor/C/P verdict from independent saved-array CPU receipts."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'reports/PI0261'
POINTS = ('Donor', 'C2000', 'P2000')


def read(name):
    path = ROOT / name
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    data, hashes = {}, {}
    for point in POINTS:
        name = 'DONOR_CPU_REPLAY.json' if point == 'Donor' else f'{point.upper()}_CPU_REPLAY.json'
        data[point], hashes[point] = read(name)
        assert (data[point]['status'], data[point]['point'], data[point]['rows']) == (
            'PASS_CPU_REPLAY', point, 680)
        assert data[point]['selected_checkpoint'] == data[point]['deployment_selected_checkpoint'] == 'NONE'

    rows = []
    for point in POINTS:
        for task in ('M1', 'M3'):
            for noise in ('B0', 'B1'):
                item = data[point]['numeric'][f'{task}/{noise}/KIN-VAL']
                rows.append((point, task, noise, item['n'],
                             f"{1000 * item['median_xyz_m']:.1f}",
                             f"{1000 * item['p95_xyz_m']:.1f}",
                             f"{item['median_so3_rad']:.4f}", item['reject_count'],
                             'PASS' if item['quality_pass'] else 'FAIL'))
    numeric = table(('Point', 'Task', 'Noise', 'N', 'median XYZ mm', 'p95 XYZ mm',
                     'median SO3 rad', 'REJECT', 'Gate'), rows)

    rows = []
    for point in POINTS:
        for task in ('M1', 'M3'):
            for kind in ('SEMANTIC', 'PARAPHRASE'):
                for noise in ('B0', 'B1'):
                    item = data[point]['semantic'][f'{task}/{kind}/{noise}']
                    assert item['n'] == 12 and sum(item['per_command'].values()) == item['correct']
                    rows.append((point, task, kind.lower(), noise,
                                 f"{item['correct']}/12", item['per_command']['0'],
                                 item['per_command']['1'], item['full_correct_pairs'],
                                 item['ties'], 'PASS' if item['pass_10_of_12'] else 'FAIL'))
    semantic = table(('Point', 'Task', 'Wording', 'Noise', 'Correct', 'c0/6', 'c1/6',
                      'both-correct families/6', 'Ties', 'Gate'), rows)

    rows = []
    for point in POINTS:
        for task in ('M1', 'M3'):
            for noise in ('B0', 'B1'):
                item = data[point]['train_entry'][f'{task}/{noise}']
                rows.append((point, task, noise, f"{item['correct']}/60",
                             f"{item['full_correct_pairs']}/30", item['ties']))
    train = table(('Point', 'Task', 'Noise', 'TRAIN correct', 'both-correct families', 'Ties'), rows)

    rows = []
    for point in POINTS:
        for noise in ('B0', 'B1'):
            item = data[point]['numeric'][f'O64/{noise}/O64']
            rows.append((point, noise, item['n'], f"{1000 * item['median_xyz_m']:.1f}",
                         f"{1000 * item['p95_xyz_m']:.1f}",
                         f"{item['median_so3_rad']:.4f}", item['reject_count']))
    o64 = table(('Point', 'Noise', 'N', 'median XYZ mm', 'p95 XYZ mm',
                 'median SO3 rad', 'REJECT'), rows)

    rows = []
    for point in POINTS:
        for task in ('M1', 'M3'):
            values = [row['instruction_output_xyz_rmse_m'] for row in data[point]['matched_input_effect']
                      if row['task'] == task]
            train_values = [row['instruction_output_xyz_rmse_m'] for row in data[point]['train_pair_effect']
                            if row['task'] == task]
            rows.append((point, task, len(values), f'{1000 * statistics.median(values):.2f}',
                         f'{1000 * statistics.median(train_values):.2f}'))
    effect = table(('Point', 'Task', 'VAL matched pairs', 'VAL output difference median mm',
                    'TRAIN output difference median mm'), rows)

    qualified = [point for point in ('C2000', 'P2000') if data[point]['research_qualified']]
    candidate = 'C2000' if 'C2000' in qualified else ('P2000' if 'P2000' in qualified else 'NONE')
    lines = [
        '# PI0261 fixed donor/C/P offline quality', '',
        'All values are from the frozen 2040 logical-row card and independent saved-array CPU replays. '
        'The original 440 donor rows were reused. This is kinematic rendered reference evaluation, '
        'with no physics, WBC, TEST, DEV-F, or hardware result.', '',
        f'**Research-qualified synthetic candidate:** `{candidate}`. '
        '**Formal selected/deployment checkpoint:** `NONE` / `NONE`.', '',
        '## Original KIN-VAL numeric gates', '', numeric, '',
        'Original gate: median XYZ <=20 mm, p95 XYZ <=35 mm, median SO3 <=.050 rad, zero REJECT.', '',
        '## Original instruction and paraphrase gates', '', semantic, '',
        'Each task/wording/noise group requires >=10/12. Both commands use their original opposite references; '
        'ties count as incorrect.', '',
        '## TRAIN common-start entry panel', '', train, '',
        'TRAIN results measure the intervention support. They are not independent generalization evidence.', '',
        '## O64 disclosed regression panel', '', o64, '',
        'O64 is reported under the synthetic-domain processor; it is not a renewed historical qualification.', '',
        '## Same-observation instruction-swap response', '', effect, '',
        'Output difference alone is not correctness; the primary own/opposite reference margin appears '
        'in the per-family saved-array rows.', '',
        '## Reproducible row-level evidence', '',
        '- `SEMANTIC_BY_FAMILY.csv`: every point, family, command, wording and noise, with own/opposite '
        'distance, signed margin and correctness.',
        '- `MATCHED_INPUT_EFFECT.csv`: same-observation/noise output differences and reference separation.',
        '- The three `*_CPU_REPLAY.json` files retain all 680 rows per point, TRAIN entry rows, group '
        'metrics, and the original gate computations.', '',
        'CPU receipt SHA256: ' + ', '.join(f'{point} `{hashes[point]}`' for point in POINTS) + '.', '',
        'The semantic and paraphrase families informed PI0261 design. These are development validation '
        'results, not blind generalization or physical task success.', '',
    ]
    report = ROOT / 'MODEL_QUALITY_REPORT.md'
    assert not report.exists()
    report.write_text('\n'.join(lines))

    for name, key, fields in (
        ('SEMANTIC_BY_FAMILY.csv', 'semantic_rows',
         ('point', 'task', 'family', 'kind', 'noise', 'command', 'own_m', 'opposite_m',
          'margin_m', 'correct', 'reference_separation_m')),
        ('MATCHED_INPUT_EFFECT.csv', 'matched_input_effect',
         ('point', 'task', 'family', 'noise', 'instruction_output_xyz_rmse_m',
          'empty_vs_command0_xyz_rmse_m', 'empty_vs_command1_xyz_rmse_m',
          'reference_separation_m')),
    ):
        path = ROOT / name
        assert not path.exists()
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for point in POINTS:
                for row in data[point][key]:
                    writer.writerow({field: (point if field == 'point' else row[field]) for field in fields})
    print(json.dumps(dict(status='PASS_PI0261_FIXED_QUALITY_REPORT', candidate=candidate,
                          receipt_sha256=hashes)))


if __name__ == '__main__':
    main()
