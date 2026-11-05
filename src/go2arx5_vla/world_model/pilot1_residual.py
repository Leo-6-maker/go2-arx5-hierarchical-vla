"""Local TCP residual over the frozen ARX, with a free residual recurrence.

The output is three world-coordinate TCP positions. It is not a complete robot
state or a contact model. No rotation, qpos or hidden WBC state is synthesized.
"""
import numpy as np
import torch
from torch import nn

from .pilot1_numpy import actor_features
from .target_arx import CausalARX


class TCPResidual(nn.Module):
    def __init__(self, dimension, hidden=64):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dimension))
        self.register_buffer('scale', torch.ones(dimension))
        self.context = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh())
        self.step = nn.Sequential(nn.Linear(hidden+10, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.step[-1].weight)
        nn.init.zeros_(self.step[-1].bias)

    def fit_normalizer(self, train_x):
        self.mean.copy_(train_x.mean(0))
        self.scale.copy_(train_x.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, x, base, observed, horizon=16):
        if not 1 <= horizon <= 16 or base.shape[1:] != (16, 3):
            raise ValueError('DECLARED_LOCAL_16_ATOM_TCP_FORECAST_REQUIRED')
        context = self.context((x-self.mean)/self.scale)
        residual = torch.zeros_like(observed)
        previous_delta = torch.zeros_like(residual)
        result = []
        for step in range(horizon):
            preceding = observed if step == 0 else base[:, step-1]
            base_delta = base[:, step]-preceding
            clock = torch.full_like(residual[:, :1], step/16.)
            inputs = torch.cat((context, residual/.001, previous_delta/.001, base_delta/.001, clock), -1)
            previous_delta = self.step(inputs)*.0001
            residual = residual+previous_delta
            result.append(base[:, step]+residual)
        return torch.stack(result, 1)


def inputs(teacher, data, meta, *, blind_choice=False, force_hold=False):
    """All inputs come from prefix [:9], past references and known candidate.

    Blind-choice removes only the newly chosen reference. Actual past reference
    and prefix observations remain identical. Forecasts are free ARX rollouts;
    no achieved future state is used to construct these inputs.
    """
    ids = np.arange(len(data['state']))
    chosen = data['goal'].copy()
    offset = chosen[:, :2]-data['history_goal'][:, 0, :2]
    if blind_choice or force_hold:
        chosen[:] = data['history_goal'][:, 0]
        offset = np.zeros_like(offset)
    sample = dict(data, goal=chosen)
    feature = actor_features(data, meta, ids, offset)
    forecast = teacher.predict(sample, ids)
    sl = teacher.layout['tcp_position_m']
    return dict(x=feature, base=forecast[:, :, sl], observed=data['state'][:, 8, sl],
                full_ARX_forecast=forecast)


def loss(prediction, truth, base):
    # Meter units, TRAIN-fixed millimeter scale. Smooth penalty is not clipping.
    return (((prediction-truth)/.001)**2).mean()+.01*(((prediction-base)/.001)**2).mean()


def errors(prediction, truth, ids):
    return {str(h): dict(XYZ_RMSE_mm=float(np.sqrt(np.mean((prediction[ids, h-1]-truth[ids, h-1])**2))*1000),
                        XY_RMSE_mm=float(np.sqrt(np.mean((prediction[ids, h-1, :2]-truth[ids, h-1, :2])**2))*1000),
                        max_XYZ_error_mm=float(np.max(np.linalg.norm(prediction[ids, h-1]-truth[ids, h-1], axis=-1)))*1000)
            for h in (1, 4, 8, 16)}


def response_errors(prediction, hold_prediction, data, meta, ids, layout):
    """Observed action-minus-HOLD displacements from independent same parents.

    This is not an exact fork, per-step intervention or a physical Jacobian.
    Model action-minus-HOLD is evaluated at the same observed prefix; measured
    HOLD is from separate native restarts. Their mismatch includes prefix noise.
    """
    sl = layout['tcp_position_m']
    action = data['goal'][:, :3]-data['history_goal'][:, 0, :3]
    selected = set(map(int, ids)); pairs = []
    for i in ids:
        if np.linalg.norm(action[i]) < 1e-8:
            continue
        parent = meta['episodes'][int(data['episode_index'][i])]['parent']
        hold_ids = [j for j in selected if meta['episodes'][int(data['episode_index'][j])]['parent'] == parent
                    and np.linalg.norm(action[j]) < 1e-8]
        if not hold_ids:
            continue
        observed = data['state'][i, 9:, sl]-data['state'][i, 8, sl]
        hold = (data['state'][hold_ids, 9:, sl]-data['state'][hold_ids, 8:9, sl]).mean(0)
        pairs.append(dict(predicted=prediction[i]-hold_prediction[i], actual=observed-hold,
                          parent=parent, prefix_TCP_mismatch_m=float(np.max(np.linalg.norm(
                              data['state'][hold_ids, 8, sl]-data['state'][i, 8, sl], axis=-1)))))
    return dict(independent_restart_pairs=len(pairs), exact_forks=False,
                max_prefix_TCP_mismatch_mm=None if not pairs else max(p['prefix_TCP_mismatch_m']for p in pairs)*1000,
                horizons={str(h): dict(XYZ_action_delta_RMSE_mm=float(np.sqrt(np.mean([
                    (p['predicted'][h-1]-p['actual'][h-1])**2 for p in pairs])))*1000)
                    for h in (1, 4, 8, 16)} if pairs else {})
