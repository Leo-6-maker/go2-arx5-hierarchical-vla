"""Preliminary offline-only CPU analysis; reuse the frozen numeric helpers."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import sys
from pathlib import Path
sys.path.insert(0, '/home/liuyu-nfs/Projects/go2-arx5/PI0269/code')
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
def forbidden(*args, **kwargs):
    raise RuntimeError('OFFLINE_SAVED_ARRAYS_ONLY_NO_NEURAL_OR_OPTIMIZER_CALL')
torch.nn.Module.__call__ = forbidden
torch.optim.Optimizer.__init__ = forbidden
import numpy as np
import report as fixed
from pi0264_freeze_eval import source

fixed.admitted()
card = fixed.read(fixed.ROOT / 'design/FIXED_OFFLINE_CARD.json')
card_sha = fixed.sha(fixed.ROOT / 'design/FIXED_OFFLINE_CARD.json')
donor_card_sha = fixed.sha(fixed.BASE / 'PI0268/R1/design/FIXED_OFFLINE_CARD.json')
donor_members = {r['path']: r for r in fixed.read(fixed.ROOT/'evidence/DONOR_EXACT_REUSE.json')['members']}
out = fixed.ROOT / 'evidence/offline-only'
out.mkdir(exist_ok=True)
rows, checks, labels = [], [], {}
for point in fixed.POINTS:
    root = fixed.OLD_RAW if point == 'ROT2048' else fixed.RAW
    for domain, count in (('fit', 224), ('factor', 304), ('legacy', 440)):
        expected = {r['uid']: r for r in card[domain + '_rows']}
        paths = sorted((root / 'offline' / point / domain).glob('*/record.json'))
        assert len(paths) == count == len(expected)
        for path in paths:
            rec = fixed.read(path)
            row = rec['row']
            assert rec['status'] == 'VALID' and rec['point'] == point
            assert row == expected[row['uid']]
            assert rec['card_sha256'] == (donor_card_sha if point == 'ROT2048' else card_sha)
            if point == 'ROT2048':
                retained = donor_members[str(path)]
                assert fixed.sha(path) == retained['sha256'] and rec['tensor_sha256'] == retained['tensor_sha256']
            assert fixed.sha(rec['tensor']) == rec['tensor_sha256']
            with np.load(rec['tensor'], allow_pickle=False) as z:
                decoded, state = z['decoded'][0].astype(float), z['state'].copy()
            key = (domain, row['uid'])
            if key not in labels:
                if domain == 'legacy':
                    _, expert = source(row)
                else:
                    assert fixed.sha(row['input']) == row['input_sha256']
                    with np.load(row['input'], allow_pickle=False) as z:
                        expert = z['raw_action'].copy()
                labels[key] = expert.astype(float)
            expert = labels[key]
            metrics = rec['metrics']
            difference = None
            if metrics is not None:
                independent = fixed.independent_metrics(decoded, expert)
                difference = max(abs(independent[k] - metrics[k]) for k in independent)
                assert difference <= 1e-6, (point, domain, row['uid'], difference)
            checks.append(dict(point=point, domain=domain, uid=row['uid'], record_sha256=fixed.sha(path),
                               tensor_sha256=rec['tensor_sha256'], independent_max_abs=difference))
            transform = row.get('transform', {})
            rows.append(dict(point=point, domain=domain, uid=row['uid'], gate=rec['gate']['decision'],
                numerical_failure=metrics is None, task=row.get('task'), command=row.get('command'),
                family=row.get('family'), variant=row.get('variant'), anchor=row.get('anchor'),
                factor_kind=transform.get('kind'), factor_axis=transform.get('axis'), factor_sign=transform.get('sign'),
                noise=row.get('noise') or (Path(row['noise_file']).name + ':' + row['noise_name']),
                source_row=fixed.json.dumps(row, sort_keys=True), **(metrics or {}),
                **fixed.prefix_metrics(decoded, expert, state)))
assert len(rows) == 2904
index = {(r['point'], r['domain'], r['uid']): r for r in rows}
panels, subgroups, contrast = {}, [], []
for point in fixed.POINTS:
    panels[point] = {}
    for domain in ('fit', 'factor', 'legacy'):
        own = [r for r in rows if r['point'] == point and r['domain'] == domain]
        panels[point][domain] = dict(**fixed.summary(own),
            median_prefix_xyz_rmse_m=float(np.median([r['prefix_xyz_rmse_m'] for r in own])),
            median_suffix_xyz_rmse_m=float(np.median([r['suffix_xyz_rmse_m'] for r in own])))
        axes = ('task', 'command', 'variant', 'noise') if domain == 'fit' else (
            ('task', 'command', 'factor_kind', 'factor_axis', 'factor_sign') if domain == 'factor' else ('noise',))
        for group in sorted({tuple(r[k] for k in axes) for r in own}, key=str):
            selected = [r for r in own if tuple(r[k] for k in axes) == group]
            subgroups.append(dict(point=point, domain=domain, **dict(zip(axes, group)), **fixed.summary(selected),
                median_prefix_xyz_rmse_m=float(np.median([r['prefix_xyz_rmse_m'] for r in selected])),
                median_prefix_so3_rmse_rad=float(np.median([r['prefix_so3_rmse_rad'] for r in selected]))))
for task in ('M1', 'M3'):
    own = [r for r in rows if r['point'] == 'UNIFORM4096' and r['domain'] == 'fit' and r['task'] == task]
    deltas = []
    for before in own:
        after = index['PREFIX4096', 'fit', before['uid']]
        assert before['source_row'] == after['source_row']
        a, b = before['prefix_xyz_rmse_m'], after['prefix_xyz_rmse_m']
        deltas.append(b-a if np.isfinite(a) and np.isfinite(b) else None)
    delta = float(np.median(deltas)) if all(v is not None for v in deltas) else None
    contrast.append(dict(task=task, rows=len(own), paired_median_delta_prefix_xyz_rmse_m=delta,
                         PASS=delta is not None and delta < 0))
reject_ok = all(panels['PREFIX4096'][d]['REJECT'] <= panels['UNIFORM4096'][d]['REJECT'] for d in ('factor', 'legacy'))
fixed.csv_write(out / 'OFFLINE2904_PRELIMINARY.csv', rows)
fixed.csv_write(out / 'OFFLINE_SUBGROUPS_PRELIMINARY.csv', subgroups)
result = dict(status='OFFLINE_ONLY_PRELIMINARY_MAIN_COMPONENT_AND_FINAL_VERDICT_PENDING',
    rows=2904, reused_donor=968, new_offline_outputs=1936, panels=panels,
    temporal_fit_contrast=contrast, no_factor_legacy_refusal_increase=reject_ok,
    temporal_contrast_PASS=all(r['PASS'] for r in contrast) and reject_ok,
    qualification_verdict='PENDING_FIXED_MAIN_AND_COMPONENT_EVALUATION',
    numeric_checks=checks, model_queries_by_this_analysis=0, gradient_queries=0, render_calls=0, physics_steps=0,
    native_Euler_and_codec_replay='PENDING_FULL_FINAL_REPORT',
    source_sha256={'analysis':fixed.sha(Path(__file__)), 'fixed_report':fixed.sha(fixed.ROOT/'code/report.py')},
    selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
fixed.write(out / 'OFFLINE_ONLY_PRELIMINARY.json', result)
print(fixed.json.dumps({k:v for k,v in result.items() if k != 'numeric_checks'}), flush=True)
