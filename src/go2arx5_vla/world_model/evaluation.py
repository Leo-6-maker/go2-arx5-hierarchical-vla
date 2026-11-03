"""Physical-unit prediction errors and fixed action-shuffle baselines."""
import numpy as np
import torch
from .features import FIELDS, SLICES
from .model import rotation_distance


def evaluate_predictions(model, replay, *, device='cpu', limit=None):
    count = len(replay.keys) if limit is None else min(len(replay.keys), limit)
    summaries = {}
    shuffle = np.random.default_rng(110302).permutation(count)
    for name in ('model', 'action_shuffle', 'persistence'):
        errors = {1:{}, 4:{}}
        events = []
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
                        imagined = [member(state, actions[:, k], goals, model.mean, model.scale)[0]
                                    for member, state in zip(model.members, imagined)]
                    if k+1 not in (1, 4):
                        continue
                    # Average member errors on SO(3), not a raw matrix average.
                    for field, _, group in FIELDS:
                        if group == 'time': continue
                        sl = SLICES[field]
                        if group == 'rotation':
                            value = torch.stack([rotation_distance(p[:, sl].reshape(-1,3,3),
                                targets[:, k, sl].reshape(-1,3,3)) for p in imagined]).mean(0)
                        else:
                            value = torch.stack([torch.linalg.vector_norm(p[:, sl]-targets[:, k, sl], dim=-1)
                                                 for p in imagined]).mean(0)
                        errors[k+1].setdefault(field, []).extend(value.cpu().tolist())
        summaries[name] = {str(h):{field:dict(mean=float(np.mean(v)), p95=float(np.percentile(v,95)))
                                  for field,v in fields.items()} for h,fields in errors.items()}
    return dict(split=replay.split, windows=count, prediction_errors=summaries,
                events_positive_counts=replay.event_counts,
                terminal_calibration='UNCALIBRATED_NO_POSITIVES' if replay.event_counts[-1]==0 else 'REQUIRES_RELIABILITY_REPORT',
                angles_unit='rad', position_unit='m', velocity_unit='native vector physical units')
