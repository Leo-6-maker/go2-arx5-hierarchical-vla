"""Four-action local XY reach distillation; teacher scores are labels, never inputs."""
import numpy as np
import torch
from torch import nn
from .target_arx import CausalARX

ACTION_NAMES = ('HOLD', 'PLUS_X_1MM', 'MINUS_X_1MM', 'PLUS_Y_1MM')
OFFSETS = np.asarray([[0., 0., 0.], [.001, 0., 0.], [-.001, 0., 0.], [0., .001, 0.]])


class TorchARX(nn.Module):
    def __init__(self, model, dtype=torch.float64):
        super().__init__()
        self.layout = model.layout
        for name, value in model.arrays().items():
            self.register_buffer(name, torch.as_tensor(value.copy(), dtype=dtype))

    def forward(self, prefix, goal, origin, age, remaining):
        current, previous = prefix[:, 8], prefix[:, 7]
        result = []
        sl = self.layout['tcp_position_m']
        root = self.layout['qpos'].start
        for step in range(16):
            action = torch.cat(((goal[:, :3] - current[:, sl]) / .01, goal[:, 3:10],
                                (goal[:, 10:13] - current[:, root:root+3]) / .01, goal[:, 13:]), dim=-1)
            known = (remaining > step).to(current.dtype)
            clocks = torch.stack(((origin + (8 + step) * .05) / 1.2, (age + step * .05) * known,
                                  known, torch.full_like(origin, step / 16.)), dim=-1)
            feature = torch.cat(((current - self.mean) / self.scale, (current - previous) / self.scale,
                                 action, clocks), dim=-1)
            design = torch.cat(((feature - self.feature_mean) / self.feature_scale,
                                torch.ones_like(origin[:, None])), dim=-1)
            following = current + design @ self.weight * self.scale
            previous, current = current, following
            result.append(current)
        return torch.stack(result, dim=1)


def candidate_predictions(model, data, ids):
    result = []
    for offset in OFFSETS:
        sample = dict(data)
        sample['goal'] = data['goal'].copy()
        sample['goal'][ids] = data['history_goal'][ids, 0]
        sample['goal'][ids, :3] += offset
        result.append(model.predict(sample, ids))
    return np.stack(result, axis=1)


def task_from_train(data, meta, ids):
    model = CausalARX(meta)
    sl = model.layout['tcp_position_m']
    action = data['goal'][ids, :3] - data['history_goal'][ids, 0, :3]
    delta = data['state'][ids, 24, sl][:, :2] - data['state'][ids, 8, sl][:, :2]
    means, counts = [], []
    for offset in OFFSETS:
        selected = np.all(np.isclose(action, offset, atol=1e-8, rtol=0), axis=1)
        if selected.sum() < 2:
            raise ValueError('REAL_TRAIN_RESPONSES_FOR_ALL_FOUR_ACTIONS_REQUIRED')
        means.append(delta[selected].mean(0)); counts.append(int(selected.sum()))
    hold = delta[np.all(np.abs(action) < 1e-8, axis=1)]
    noise = float(np.sqrt(np.mean((hold - hold.mean(0))**2)))
    return dict(schema='LOCAL_EEF_REACH_PILOT_V1', scored_coordinates='WORLD_XY_ONLY_Z_REPORTED_SEPARATELY',
                action_names=list(ACTION_NAMES), offsets_m=OFFSETS.tolist(), mean_response_xy_m=np.asarray(means).tolist(),
                goal_generator='TRAIN_MEAN_RESPONSE_CENTERS_PLUS_FIXED_UNIFORM_XY_JITTER_V1',
                goal_seed=36001, goals_per_center=64, jitter_m=.0002,
                success_tolerance_m=max(.00018, 3 * noise), TRAIN_HOLD_coordinate_noise_m=noise,
                nonhold_action_cost_m=.00002, teacher_temperature_m=max(.00005, noise),
                TRAIN_action_counts=counts, no_full_manipulation_claim=True)


def goal_offsets(task):
    rng = np.random.default_rng(task['goal_seed'])
    jitter = rng.uniform(-task['jitter_m'], task['jitter_m'], (4, task['goals_per_center'], 2))
    return (np.asarray(task['mean_response_xy_m'])[:, None] + jitter).reshape(-1, 2)


def robot_columns(meta):
    layout = CausalARX(meta).layout
    cube = [j for j in meta['joints'] if j['name'] == 'task_cube_free']
    if len(cube) != 1 or (cube[0]['qpos_address'], cube[0]['qvel_address']) != (27, 26):
        raise ValueError('ACTUAL_ROBOT_OBJECT_BOUNDARY_REQUIRED')
    columns = list(range(layout['qpos'].start, layout['qpos'].start + 27))
    columns += list(range(layout['qvel'].start, layout['qvel'].start + 26))
    for name in ('ctrl', 'tcp_position_m', 'tcp_rotation6d', 'finger_position_m'):
        columns += list(range(layout[name].start, layout[name].stop))
    return columns


def features(data, meta, ids, offsets_xy):
    # Future state, object pose, episode/parent IDs and teacher predictions are never read here.
    prefix = data['state'][ids, :9][:, :, robot_columns(meta)].reshape(len(ids), -1)
    past = data['history_goal'][ids, :8].reshape(len(ids), -1)
    age = data['source_age'][ids, 8:9]
    remaining = CausalARX.remaining(data, ids)[:, None]
    return np.concatenate((prefix, past, np.asarray(offsets_xy) / .001, age, remaining), axis=1)


def examples(model, data, meta, ids, task):
    goals = goal_offsets(task)
    n = len(goals)
    repeated = np.repeat(ids, n)
    offsets = np.tile(goals, (len(ids), 1))
    sl = model.layout['tcp_position_m']
    endpoint = candidate_predictions(model, data, ids)[:, :, -1, sl][:, :, :2]
    delta = endpoint - data['state'][ids, 8, sl][:, None, :2]
    error = np.linalg.norm(delta[:, None] - goals[None, :, None], axis=-1)
    score = -error - task['nonhold_action_cost_m'] * np.asarray([0, 1, 1, 1])[None, None]
    logits = score / task['teacher_temperature_m']
    logits -= logits.max(-1, keepdims=True)
    labels = np.exp(logits); labels /= labels.sum(-1, keepdims=True)
    return features(data, meta, repeated, offsets), labels.reshape(-1, 4), score.reshape(-1, 4)


class Actor(nn.Module):
    def __init__(self, dimension, hidden=128):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dimension))
        self.register_buffer('scale', torch.ones(dimension))
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, x):
        return self.net((x - self.mean) / self.scale)

    def fit_normalizer(self, x):
        self.mean.copy_(x.mean(0))
        self.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-4))


def numpy_logits(arrays, x):
    value = (np.asarray(x, dtype=np.float32) - arrays['mean']) / arrays['scale']
    for layer in (0, 2, 4):
        value = value @ arrays[f'w{layer}'].T + arrays[f'b{layer}']
        if layer < 4:
            value = np.maximum(value, 0.)
    return value


def export_arrays(actor):
    arrays = dict(mean=actor.mean.detach().cpu().numpy(), scale=actor.scale.detach().cpu().numpy())
    for layer in (0, 2, 4):
        arrays[f'w{layer}'] = actor.net[layer].weight.detach().cpu().numpy()
        arrays[f'b{layer}'] = actor.net[layer].bias.detach().cpu().numpy()
    return arrays
