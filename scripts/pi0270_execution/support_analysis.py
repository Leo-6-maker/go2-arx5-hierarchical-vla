"""Summarize already-closed label/codec support; no new label or model call."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path
from admission import ROOT, read, sha, write

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def values(rows, field):
    observed = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
    finite = [x for x in observed if np.isfinite(x)]
    return dict(rows=len(rows), observed=len(observed), unobserved=len(rows)-len(observed),
                nonfinite=len(observed)-len(finite), median=float(np.median(finite)) if finite else None,
                maximum=float(np.max(finite)) if finite else None)


def main():
    os.nice(19)
    proposals = read(ROOT/'design/LABEL_PROPOSAL_CARD.json')['proposals']; assert len(proposals) == 6784
    card = read(ROOT/'design/COLLECTION_CARD.json')
    families = list(dict.fromkeys(r['family'] for r in card['cases']))
    closed = [f for f in families if (ROOT/'evidence'/('SUPPORT_CPU_'+f+'.json')).exists()]
    eligible = []; sources = []; exclusions = Counter(); cells = defaultdict(set); groups = defaultdict(list)
    pending = 0; excluded = 0; ranges = []
    for proposal in proposals:
        path = RAW/'codec'/proposal['source_lane']/proposal['source_episode_key']/f"{proposal['query_index']:03d}.json"
        if proposal['source_lane'] == 'original' and proposal['family'] not in closed or not path.exists():
            pending += 1; continue
        row = read(path); sources.append(dict(uid=proposal['uid'], path=str(path), sha256=sha(path)))
        source = row['source']
        assert (source['family'],source['task'],source['command']) == (proposal['family'],proposal['task'],proposal['command'])
        if not row['eligible']:
            excluded += 1; exclusions.update(row.get('reasons', row['source'].get('reasons', []))); continue
        assert row['roundtrip']['PASS'] and row['train_eval_bit_identity'] and row['collector_full_condition_bit_identity']
        cells[proposal['family'], proposal['task'], proposal['command']].add(row['processed_state_sha256'])
        diagnostic = source['diagnostic']; eligible.append(row); ranges.append(row['normalized_ranges'])
        groups[proposal['source_lane'], proposal['task'], source['backend'], bool(source['teacher']['terminal_hold'])].append(diagnostic)
    assert len(sources) + pending == 6784 and len(eligible) + excluded == len(sources)
    coverage = [dict(family=f, task=t, command=c, eligible_distinct_states=len(cells[f,t,c]),
                     current_floor_met=len(cells[f,t,c]) >= 8, final_admission_not_claimed=True)
                for f in families for t in ('M1','M3') for c in (0,1)]
    summary = []
    for (lane, task, backend, terminal), rows in groups.items():
        directions = [r['directional_cosine'] for r in rows if isinstance(r.get('directional_cosine'), (int,float)) and np.isfinite(r['directional_cosine'])]
        summary.append(dict(lane=lane, task=task, backend=backend, projected_terminal=terminal, eligible_proposals=len(rows),
             direction_defined=len(directions), opposed_direction=sum(x < 0 for x in directions),
             zero_teacher_prefix_displacement=sum(bool(r.get('hold_case')) for r in rows),
             **{k:values(rows,k) for k in ('first_position_bias_m','prefix_xyz_rmse_m','prefix_endpoint_error_m',
                    'terminal_current_error_m','terminal_teacher_prefix_error_m','terminal_predicted_prefix_error_m')}))
    initial = [r['source'] for r in eligible]
    report = dict(status='SAVED_CLOSED_SUPPORT_DIAGNOSIS_NO_NEW_QUERIES', total_fixed_proposals=6784,
          observed_closed_proposals=len(sources), pending_CPU_records=pending, eligible=len(eligible), excluded_or_missing=excluded,
          exclusion_reasons=dict(exclusions), closed_families=closed, coverage=coverage, by_lane_task_backend=summary,
          normalized_ranges={k:float(fn([r[k] for r in ranges])) for k,fn in
              [('state_min',min),('state_max',max),('action_min',min),('action_max',max)]} if ranges else {},
          first_teacher_pose={k:values(initial,k) for k in ('first_pose_position_error_m','first_pose_rotation_matrix_abs_error')},
          projected_terminal_proposals=sum(bool(r['source']['teacher']['terminal_hold']) for r in eligible),
          zero_teacher_prefix_displacement=sum(bool(r['source']['diagnostic']['hold_case']) for r in eligible),
          terminal_projection_is_not_zero_action=True, no_performance_based_exclusions=True,
          source_records=sources, source_sha256=sha(__file__), model_queries=0, teacher_calls=0, optimizer_updates=0,
          renderer_calls=0, statistics_refits=0, describes_collector_not_AGG_training_or_quality=True,
          selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    destination = ROOT/'evidence'/('SUPPORT_MECHANISM_FAMILIES_'+str(len(closed))+'.json')
    write(destination, report)
    print(report['status'], report['observed_closed_proposals'], report['eligible'], report['normalized_ranges'], flush=True)


if __name__ == '__main__': main()
