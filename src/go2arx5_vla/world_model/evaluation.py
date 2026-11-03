"""Held-out physical-unit errors; offline strata never enter model inputs."""
import numpy as np
import torch
from .features import FIELDS, SLICES, EVENTS
from .model import rotation_distance


def _summary(values, mask):
    values = np.asarray(values)[mask]
    if not len(values):
        return dict(count=0, mean=None, p95=None)
    return dict(count=len(values), mean=float(np.mean(values)), p95=float(np.percentile(values, 95)))


def _reliability(probabilities, labels, mask):
    probabilities, labels = np.asarray(probabilities)[mask], np.asarray(labels)[mask]
    bins = []
    for index in range(10):
        selected = (probabilities >= index/10) & (probabilities < (index+1)/10)
        if index == 9:
            selected |= probabilities == 1
        if selected.any():
            bins.append(dict(lower=index/10, upper=(index+1)/10, count=int(selected.sum()),
                probability_mean=float(probabilities[selected].mean()), frequency=float(labels[selected].mean())))
    count = len(labels)
    return dict(count=count, positives=int(labels.sum()),
        brier=float(np.mean((probabilities-labels)**2)) if count else None, bins=bins,
        calibration_status='NO_POSITIVES_NOT_FAILURE_CALIBRATED' if not labels.any() else
            ('NO_NEGATIVES_NOT_CALIBRATED' if labels.all() else 'HELD_OUT_RELIABILITY_DESCRIPTIVE_ONLY'))


def evaluate_predictions(model, replay, *, device='cpu', limit=None):
    """Use all fixed CALIB/DEV windows by default, with no optimizer or fit.

    Norms are per-field, in native physical units. Mixed linear/angular vectors
    additionally expose component MAE. Event strata use the *observed first
    target*, explicitly for retrospective analysis, never as a model input.
    Overlapping windows are descriptive samples, not independent CI units.
    """
    if replay.split not in ('CALIB', 'DEV'):
        raise ValueError('Held-out prediction evaluation requires CALIB or DEV')
    count = len(replay.keys) if limit is None else min(len(replay.keys), limit)
    if count <= 0:
        raise ValueError('At least one held-out window is required')
    summaries, event_reports, disagreements = {}, {}, {}
    shuffle = np.random.default_rng(110302).permutation(count)
    stages = np.asarray(replay.stages[:count])
    first_events = np.asarray(replay.events[:count, 0])
    contact = first_events[:, :2].any(1)
    masks = dict(all=np.ones(count, dtype=bool), first_target_contact=contact,
                 first_target_no_contact=~contact,
                 any_target_terminal=np.asarray(replay.events[:count, :, -1]).any(1))
    for stage in ('approach', 'align', 'close', 'verify_grasp', 'lift', 'carry', 'lower',
                  'release', 'withdraw', 'hold', 'recovery'):
        masks['teacher_stage_'+stage] = stages == stage
    for stage in np.unique(stages):
        masks.setdefault('teacher_stage_'+str(stage), stages == stage)
    for name in ('model', 'action_shuffle', 'persistence'):
        errors, components, event_values, spread = ({1:{}, 4:{}} for _ in range(4))
        for start in range(0, count, 64):
            end = min(start+64, count)
            states = torch.as_tensor(replay.states[start:end], device=device)
            action_ids = shuffle[start:end] if name == 'action_shuffle' else np.arange(start, end)
            actions = torch.as_tensor(replay.actions[action_ids], device=device)
            goals = torch.as_tensor(replay.goals[start:end], device=device)
            targets = torch.as_tensor(replay.targets[start:end], device=device)
            with torch.no_grad():
                imagined = [states[:, 0].clone() for _ in model.members]
                for k in range(4):
                    if name != 'persistence':
                        outputs = [member(state, actions[:, k], goals, model.mean, model.scale)
                                   for member, state in zip(model.members, imagined)]
                        imagined = [output[0] for output in outputs]
                    if k+1 not in (1, 4):
                        continue
                    horizon = k+1
                    if name != 'persistence':
                        probabilities = torch.stack([torch.sigmoid(output[1]) for output in outputs]).mean(0)
                        for index, event in enumerate(EVENTS):
                            event_values[horizon].setdefault(event, []).extend(probabilities[:, index].cpu().tolist())
                    # Average member errors on SO(3), never average raw matrices.
                    for field, _, group in FIELDS:
                        if group == 'time':
                            continue
                        sl = SLICES[field]
                        if group == 'rotation':
                            value = torch.stack([rotation_distance(p[:, sl].reshape(-1,3,3),
                                targets[:, k, sl].reshape(-1,3,3)) for p in imagined]).mean(0)
                            uncertainty = torch.stack([rotation_distance(a[:, sl].reshape(-1,3,3),
                                b[:, sl].reshape(-1,3,3)) for i, a in enumerate(imagined)
                                for b in imagined[i+1:]]).mean(0)
                        else:
                            differences = torch.stack([p[:, sl]-targets[:, k, sl] for p in imagined])
                            value = torch.linalg.vector_norm(differences, dim=-1).mean(0)
                            component = differences.abs().mean(0)
                            components[horizon].setdefault(field, []).extend(component.cpu().tolist())
                            uncertainty = torch.stack([torch.linalg.vector_norm(a[:, sl]-b[:, sl], dim=-1)
                                for i, a in enumerate(imagined) for b in imagined[i+1:]]).mean(0)
                        errors[horizon].setdefault(field, []).extend(value.cpu().tolist())
                        spread[horizon].setdefault(field, []).extend(uncertainty.cpu().tolist())
        summaries[name] = {str(h):{stratum:{field:_summary(v, mask) for field,v in fields.items()}
                                  for stratum,mask in masks.items()} for h,fields in errors.items()}
        # Component MAE makes mixed qvel/controller/solver vectors inspectable.
        for horizon, fields in components.items():
            for stratum, mask in masks.items():
                if mask.any():
                    for field, values in fields.items():
                        summaries[name][str(horizon)][stratum][field]['component_MAE'] = np.asarray(values)[mask].mean(0).tolist()
        if name != 'persistence':
            event_reports[name] = {str(h):{stratum:{event:_reliability(v, replay.events[:count, h-1, index], mask)
                for index,(event,v) in enumerate(fields.items())} for stratum,mask in masks.items()}
                for h,fields in event_values.items()}
            disagreements[name] = {str(h):{field:_summary(v, masks['all']) for field,v in fields.items()}
                for h,fields in spread.items()}
    return dict(split=replay.split, windows=count, episodes=replay.episodes,
        prediction_errors=summaries, event_reliability=event_reports, ensemble_mean_pairwise_distance=disagreements,
        unique_macro_events_positive_counts=replay.event_counts,
        stratum_counts={key:int(value.sum()) for key,value in masks.items()},
        stratification='Teacher stage at first macro; observed first-target contact; any-target terminal. Offline analysis only.',
        terminal_calibration='UNCALIBRATED_NO_POSITIVES' if replay.event_counts[-1]==0 else 'SEE_HELD_OUT_RELIABILITY',
        angles_unit='rad', position_unit='m', mixed_vectors='native physical components, see field schema; component MAE supplied',
        persistence_events='NOT_AVAILABLE_WITHOUT_CURRENT_CONTACT_INPUTS',
        window_independence=False, normalization_updated=False, optimizer_updates=0, TEST_read=False)
