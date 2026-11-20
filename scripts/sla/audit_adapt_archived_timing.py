"""Read fixed ADAPT0/500 predictions and audit closure timing without generating actions."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha


PANEL_SHA256 = '3136404c06f890a9dd830b2cf5f40e487ef436ede16e805e14ffc0cb06b5eefd'


def arrays(row):
    path = Path(row['output']['path'])
    if sha(path) != row['output']['sha256']:
        raise ValueError('Sealed prediction archive changed')
    with np.load(path, allow_pickle=False) as data:
        normalized, decoded = data['normalized'], data['decoded']
    for key, value in (('normalized_sha256', normalized), ('decoded_sha256', decoded)):
        if bytes_sha(value) != row[key] or not np.isfinite(value).all():
            raise ValueError('Sealed prediction array changed')
    return decoded


def transitions(before, actions):
    width = np.r_[before, .002+.040*np.asarray(actions)[:,9]]
    closed = width < .022
    return np.flatnonzero(~closed[:-1] & closed[1:]), np.flatnonzero(closed[:-1] & ~closed[1:])


def first(indices):
    return int(indices[0]) if len(indices) else None


def timing(expert, predicted):
    if predicted is None:
        return 'missing', None
    difference = predicted-expert
    if difference < -2:
        category = 'early'
    elif difference > 2:
        category = 'late'
    else:
        category = 'timely_P10' if predicted < 10 else 'within_100ms_suffix'
    return category, 50*difference


def guard(metrics):
    envelope = metrics['envelope']
    h40 = envelope['original_full_H40_pass']
    seam = not metrics['gripper_seam']['original_execution_speed_exceeded']
    return dict(h40=h40, seam=seam, both=h40 and seam,
                first_exceedance=envelope['first_temporal_exceedance'],
                seam_speed_m_s=metrics['gripper_seam']['first_seam_speed_m_s'])


def evaluate_phase(result, panel):
    rows = []
    for observation, fixed in zip(result['rows']['target_phase'], panel['phase_train'], strict=True):
        if any(observation[key] != value for key, value in fixed.items()):
            raise ValueError('Frozen phase row changed')
        context = panel['target_contexts'][str(observation['index'])]
        before = context['previous_command']['width_m']
        physical = arrays(observation)
        if physical.shape != (40,10):
            raise ValueError('Expected complete physical H40')
        closes, opens = transitions(before,physical)
        phase = observation['metrics']['phase']
        expert, predicted = phase['expert_first_crossing_index'], first(closes)
        if predicted != phase['predicted_first_crossing_index']:
            raise ValueError('Archived action and reported crossing disagree')
        gates = guard(observation['metrics'])
        if gates['both'] != observation['metrics']['H40_and_seam_pass']:
            raise ValueError('Archived guard components disagree')
        row = dict(identity=observation['identity'], index=observation['index'],
                   bucket=observation['phase_bucket'], output_sha256=observation['output']['sha256'],
                   previous_command=context['previous_command'], expert_close=expert,
                   predicted_close=predicted, predicted_open=first(opens), gates=gates)
        if row['bucket'] == 'P10_DUE_CLOSE':
            category, difference = timing(expert,predicted)
            row.update(timing=category,signed_error_ms=difference,
                       timely_and_gates=category == 'timely_P10' and gates['both'])
            if row['timely_and_gates'] != phase['timely_P10_close_and_original_gates']:
                raise ValueError('Archived timing and guard result disagree')
        rows.append(row)
    return rows


def distribution(rows):
    counts = dict(Counter(row['timing'] for row in rows if 'timing' in row))
    failures = Counter()
    for row in rows:
        gates = row['gates']
        if 'timing' in row:
            failures[f"{row['timing']}|H40_{gates['h40']}|seam_{gates['seam']}"] += 1
        if gates['first_exceedance'] is not None:
            first_gate = gates['first_exceedance']
            key = first_gate.get('k',first_gate.get('first_exceedance_k'))
            if key is not None:
                failures['H40_first_prefix' if key < 10 else 'H40_first_suffix'] += 1
    return dict(windows=len(rows),episodes=len({r['identity']['episode_id'] for r in rows}),
                categories=counts, intersection=dict(failures),
                h40=sum(row['gates']['h40'] for row in rows),seam=sum(row['gates']['seam'] for row in rows),
                both=sum(row['gates']['both'] for row in rows),
                P10_close=sum(r['predicted_close'] is not None and r['predicted_close'] < 10 for r in rows),
                suffix_close=sum(r['predicted_close'] is not None and r['predicted_close'] >= 10 for r in rows),
                P10_open=sum(r['predicted_open'] is not None and r['predicted_open'] < 10 for r in rows),
                timely_joint=sum(row.get('timely_and_gates',False) for row in rows))


def mean(values):
    return float(np.mean(values))


def source_summary(result, domain):
    rows = result['rows'][domain]
    for row in rows:
        arrays(row)
    def summarize(selected):
        metrics = [row['metrics'] for row in selected]
        report = dict(windows=len(selected), episodes=len({r['identity']['episode'] for r in selected}),
                      normalized_active_mae=mean([m['normalized_active_mae'] for m in metrics]))
        if domain == 'libero':
            report.update(normalized_zero_mae=mean([m['normalized_zero_command_mae'] for m in metrics]),
                native_channel_mae=np.mean([m['native_command_channel_mae'] for m in metrics],axis=0).tolist())
        else:
            errors = [m['recorded_reference'] for m in metrics]
            report.update(normalized_persistence_mae=mean([e['persistence']['normalized_active14_MAE'] for e in errors]),
                          dynamic_windows=sum(m['dynamic'] for m in metrics))
        return report
    groups = sorted({row['group'] for row in rows})
    return dict(all=summarize(rows), per_task={g:summarize([r for r in rows if r['group']==g]) for g in groups})


def audit_labels(spec, panel, reports):
    from go2arx5_vla.sla_adapter.adapt5k_contract import due_close
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze

    def unused_images(*args, **kwargs):
        raise RuntimeError('This audit does not read camera frames')

    data = spec['target_D1']
    target = AdaptTargetDataset(data['legacy'],data['captures'],resize_with_pad=unused_images)
    target.bind_state(data['statistics'])
    labels = []
    paired = zip(panel['phase_train'],*(report['rows'] for report in reports),strict=True)
    for fixed, phase0, phase500 in paired:
        index = fixed['index']
        episode, anchor = target.rows[index]
        identity = target.identity(index)
        context = panel['target_contexts'][str(index)]
        previous = target.reference(index)
        if (previous != context['previous_command'] or identity != fixed['identity']
                or abs(previous['query_native_s']-.002*identity['native_step']) > 1e-8):
            raise ValueError('Original target clock, reference or window changed')
        archive = target.labels[episode['id']]
        action = archive['nominal_policy_H40'][anchor]
        state = archive['state_eef10'][anchor]
        close, opened = transitions(previous['width_m'],action)
        if first(close) != phase0['expert_close'] or first(close) != phase500['expert_close']:
            raise ValueError('Archived expert crossing disagrees with sealed source label')
        consumed = archive['consumed_mask'][anchor]
        if not np.array_equal(consumed,np.arange(40)<10):
            raise ValueError('Label does not carry the original confirmed P10')
        if fixed['phase_bucket'] == 'P10_DUE_CLOSE' and not due_close(action,consumed,previous['width_m']):
            raise ValueError('Expert due window is inconsistent with the label and P10')
        envelope = analyze(action,state,np.asarray(context['policy_from_world']),np.asarray(context['reference']))
        seam = label_margin_report(action,previous)
        labels.append(dict(index=index,bucket=fixed['phase_bucket'],episode=episode['id'],
                           family=episode.get('family'),task=episode['task'],phase=str(archive['phase'][anchor]),
                           expert_open=first(opened),expert_close=first(close),
                           expert_due=due_close(action,consumed,previous['width_m']),
                           label_range_m=[float(v) for v in (.002+.040*action[:,9].min(),.002+.040*action[:,9].max())],
                           gates=guard(dict(envelope=envelope,gripper_seam=seam))))
    return labels


def audit(paths, expert_spec=None):
    results = [json.loads(p.read_bytes()) for p in paths]
    if ([r['phase'] for r in results] != [0,500] or results[0]['panel'] != results[1]['panel']
            or results[0]['panel']['sha256'] != PANEL_SHA256
            or any(r['exit_code'] or r['model_generations'] != 376 or r['old_TEST_access'] for r in results)):
        raise ValueError('Expected the two fixed successfully sealed ADAPT0/500 offline panels')
    panel = read_pin(results[0]['panel'])
    reports = []
    for result in results:
        rows = evaluate_phase(result,panel)
        groups = sorted({r['bucket'] for r in rows})
        reports.append(dict(phase=result['phase'],checkpoint_sha256=result['checkpoint']['sha256'],
            target_phase=distribution(rows),
            by_bucket={group:distribution([row for row in rows if row['bucket']==group]) for group in groups},
            source={domain:source_summary(result,domain) for domain in ('robodojo','libero')},rows=rows))
    report = dict(panel_sha256=PANEL_SHA256,formal_result_sha256=[sha(p) for p in paths],
                  new_updates=0,new_visits=0,new_generations=0,native_requests=0,native_ticks=0,phases=reports)
    if expert_spec is not None:
        spec = json.loads(expert_spec.read_bytes())
        if spec['panel'] != results[0]['panel'] or spec['checkpoint'] != results[0]['checkpoint']:
            raise ValueError('Expert reader does not match the sealed evaluation identity')
        report['expert_spec_sha256'] = sha(expert_spec)
        report['expert_labels'] = audit_labels(spec,panel,reports)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert-spec',type=Path)
    parser.add_argument('formal',nargs=2,type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.formal,args.expert_spec),allow_nan=False))
