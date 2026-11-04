"""Read-only analysis of frozen paired data/predictions; no model, tuning, or physics."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from .replay import SLICES


def position_pairs(real, predicted):
    rows = []
    for j, k in itertools.combinations(range(len(real)), 2):
        truth, estimate = real[j] - real[k], predicted[j] - predicted[k]
        size, magnitude = float(np.linalg.norm(truth)), float(np.linalg.norm(estimate))
        dot = float(truth @ estimate)
        rows.append(dict(real_m=size, predicted_m=magnitude,
                         error_squared_m2=float(np.sum((truth-estimate)**2)),
                         informative=size > .001, tied=magnitude <= 1e-12,
                         positive=dot > 0, cosine=dot/(size*magnitude) if size*magnitude > 1e-12 else None))
    informative = [r for r in rows if r['informative']]
    cosine = [r['cosine'] for r in informative if r['cosine'] is not None]
    return dict(pairs=len(rows), informative=len(informative), flat=len(rows)-len(informative),
                positive_direction=sum(r['positive'] for r in informative),
                predicted_ties_informative=sum(r['tied'] for r in informative),
                mean_cosine_informative=float(np.mean(cosine)) if cosine else None,
                vector_difference_RMSE_m=float(np.sqrt(np.mean([r['error_squared_m2'] for r in rows]))),
                zero_difference_baseline_RMSE_m=float(np.sqrt(np.mean([r['real_m']**2 for r in rows]))))


def progress_ranking(pairs):
    # Keep ties separate: an action-blind model cannot choose between identical inputs.
    informative = [p for p in pairs if abs(p['real_progress_difference']) > 1e-5]
    return dict(declared_pairs=len(pairs), real_progress_informative=len(informative),
                real_progress_flat=len(pairs)-len(informative),
                correct_sign=sum(p['real_progress_difference']*p['predicted_progress_difference'] > 0
                                 for p in informative if abs(p['predicted_progress_difference']) > 1e-9),
                predicted_ties=sum(abs(p['predicted_progress_difference']) <= 1e-9 for p in informative))


def oracle_opportunities(pairs):
    groups = {}
    for p in pairs:
        groups.setdefault((p['scene'], p['anchor']), []).append(p)
    rows = []
    for (scene, anchor), group in groups.items():
        # Absolute offsets cancel; reconstruct each score relative to first candidate hold.
        values = {'hold': 0.}
        values.update({p['second']: -p['real_progress_difference'] for p in group if p['first'] == 'hold'})
        if set(values) != {'hold', 'nominal', 'left', 'right', 'grip_timing'}:
            raise ValueError('Incomplete five-candidate denominator')
        improvement = max(values.values()) - values['nominal']
        rows.append(dict(scene=scene, anchor=anchor, oracle_improvement_vs_nominal=improvement,
                         opportunity_above_fixed_1e_5=improvement > 1e-5))
    return dict(anchors=len(rows), improvable_vs_nominal=sum(x['opportunity_above_fixed_1e_5'] for x in rows),
                maximum_improvement=max(x['oracle_improvement_vs_nominal'] for x in rows), rows=rows)


def analyze(replay_path, predictions_path, result_path):
    result = json.loads(Path(result_path).read_text())
    data_pin = hashlib.sha256(Path(replay_path).read_bytes()).hexdigest()
    if data_pin != result['data_sha256']:
        raise ValueError('Frozen evaluated replay hash differs')
    with np.load(replay_path, allow_pickle=False) as f:
        data = {k: f[k] for k in ('state','events','initial_event','valid','scene','anchor','partition','candidate','raw')}
    with np.load(predictions_path, allow_pickle=False) as f:
        predictions = {k: f[k] for k in ('conditioned','no_future_action','shifted_future','raw')}
    selected = (data['partition']=='ONCE_HOLDOUT') & (data['valid'][:,8:].sum(-1)==16)
    hold = {k: v[selected] for k,v in data.items()}
    if not np.array_equal(hold['raw'].astype(np.float32), predictions.pop('raw')):
        raise ValueError('Saved prediction proposal ordering differs from replay')
    if any(x.shape != (len(hold['state']),16,50) for x in predictions.values()):
        raise ValueError('Saved prediction state schema differs')
    effects = {}
    for scene in sorted(set(hold['scene'])):
        for anchor in sorted(set(hold['anchor'])):
            ids = np.flatnonzero((hold['scene']==scene)&(hold['anchor']==anchor))
            real = hold['state'][ids,24,SLICES['object_position']]
            effects[f'{int(scene)}/{anchor}'] = {name: position_pairs(real, x[ids,-1,SLICES['object_position']])
                                                for name,x in predictions.items()}
    coverage = {}
    for partition in ('FIT','DIAGNOSTIC_DEV','ONCE_HOLDOUT'):
        coverage[partition] = {}
        for anchor in sorted(set(data['anchor'])):
            ids = (data['partition']==partition)&(data['anchor']==anchor)
            event = data['events'][ids,8:,2]>.5
            before = np.concatenate((data['initial_event'][ids,None,2]>.5,event[:,:-1]),axis=1)
            valid = data['valid'][ids,8:].astype(bool)
            coverage[partition][str(anchor)] = dict(branches=int(ids.sum()),
                grasp_establishment_transitions=int((event & ~before & valid).sum()),
                grasp_loss_or_release_transitions=int((~event & before & valid).sum()),
                grasp_positive_transitions=int((event & valid).sum()),
                valid_transitions=int(valid.sum()))
    evaluation=result['evaluation']
    shifted_pairs=[]
    for scene in sorted(set(hold['scene'])):
        for anchor in sorted(set(hold['anchor'])):
            ids=np.flatnonzero((hold['scene']==scene)&(hold['anchor']==anchor))
            if not np.array_equal(predictions['conditioned'][np.roll(ids,1)], predictions['shifted_future'][ids]):
                raise ValueError('Cannot derive shifted ranking from frozen same-history candidate permutation')
            names=list(hold['candidate'][ids]);previous={name:names[(i-1)%len(names)] for i,name in enumerate(names)}
            pairs=[p for p in evaluation['paired_conditioned']['pairs'] if p['scene']==scene and p['anchor']==anchor]
            scores={'hold':0.}
            scores.update({p['second']:-p['predicted_progress_difference'] for p in pairs if p['first']=='hold'})
            shifted_pairs.extend(dict(p,predicted_progress_difference=scores[previous[p['first']]]-scores[previous[p['second']]]) for p in pairs)
    ranking={name:progress_ranking(evaluation[name]['pairs']) for name in ('paired_conditioned','paired_no_future_action')}
    ranking['future_only_shift_derived_verified_permutation']=progress_ranking(shifted_pairs)
    event_pairs={}
    for name in ('paired_conditioned','paired_no_future_action'):
        pairs=evaluation[name]['pairs'];informative=[p for p in pairs if abs(p['true_grasp_difference'])>.5]
        event_pairs[name]=dict(declared_pairs=len(pairs),different_true_grasp_pairs=len(informative),
            grasp_difference_RMSE=float(np.sqrt(np.mean([(p['predicted_grasp_difference']-p['true_grasp_difference'])**2 for p in pairs]))),
            correct_sign_informative=sum(p['true_grasp_difference']*p['predicted_grasp_difference']>0 for p in informative),
            maximum_predicted_grasp_contrast=float(max(abs(p['predicted_grasp_difference']) for p in pairs)))
    return dict(purpose='FROZEN_PAIRED_CONTACT_POSTHOC_ANALYSIS_NO_REFIT',data_sha256=data_pin,
        predictions_sha256=hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest(),
        evaluated_model_source_sha=result['source_sha'],heldout_physical_parents=len(set(hold['scene'])),
        effects_by_parent_and_anchor=effects,contact_coverage=coverage,
        ranking=ranking,event_pairs=event_pairs,
        oracle=oracle_opportunities(evaluation['paired_conditioned']['pairs']),
        new_physics=0,new_model_updates=0,live_model_advice_executions=0,
        interpretation='Descriptive four-parent results. Oracle opportunity is audited after evaluation, not a changed gate. No formal reward, task-success or target claim.')
