"""TRAIN-only, scaled causal ARX reference baseline with recursive evaluation."""
import numpy as np


class CausalARX:
    def __init__(self, meta, ridge=1.):
        self.ridge = ridge
        self.layout = {}
        offset = 0
        for field in meta['state_fields']:
            self.layout[field['name']] = slice(offset, offset + field['size'])
            offset += field['size']
        self.dimension = offset

    def feature(self, current, previous, goal, origin, age, remaining, step):
        action = goal.copy()
        action[:, :3] = (action[:, :3] - current[:, self.layout['tcp_position_m']]) / .01
        root = self.layout['qpos'].start
        action[:, 10:13] = (action[:, 10:13] - current[:, root:root+3]) / .01
        known = (remaining > step).astype(float)
        elapsed = (origin + (8 + step) * .05) / 1.2
        clocks = np.stack((elapsed, (age + step*.05)*known, known,
                           np.full(len(current), step/16.)), -1)
        return np.concatenate(((current-self.mean)/self.scale,
                               (current-previous)/self.scale, action, clocks), -1)

    def fit(self, data, ids):
        # The caller supplies TRAIN rows only. No DEV normalization, feature
        # scaling, regression fit or regularization selection occurs here.
        if not len(ids):
            raise ValueError('ACTUAL_TRAIN_ROWS_REQUIRED')
        state = np.asarray(data['state'][ids], dtype=float)
        self.mean = state.mean((0, 1))
        self.scale = np.maximum(state.std((0, 1)), .001)
        goal = np.asarray(data['goal'][ids], dtype=float)
        origin = data.get('window_start_s', np.zeros(len(data['state'])))[ids]
        age = data['source_age'][ids, 8]
        remaining = self.remaining(data, ids)
        x = np.concatenate([self.feature(state[:, 8+j], state[:, 7+j], goal,
                                        origin, age, remaining, j) for j in range(16)])
        y = np.concatenate([(state[:, 9+j]-state[:, 8+j])/self.scale for j in range(16)])
        self.feature_mean = x.mean(0)
        self.feature_scale = np.maximum(x.std(0), 1e-6)
        z = self.design(x)
        penalty = np.eye(z.shape[1])*self.ridge
        penalty[-1, -1] = 0.
        self.weight = np.linalg.solve(z.T@z + penalty, z.T@y)
        return self

    def design(self, x):
        return np.concatenate(((x-self.feature_mean)/self.feature_scale,
                               np.ones((len(x), 1))), -1)

    @staticmethod
    def remaining(data, ids):
        if 'future_prefix_remaining' in data:
            return data['future_prefix_remaining'][ids]
        atom = data.get('window_start_atom', np.zeros(len(data['state'])))[ids]
        return 7-np.remainder(atom+8, 7)

    def predict(self, data, ids):
        # Deliberately copy only the observed prefix. Post-decision achieved
        # states and future measured reference ages never enter inference.
        prefix = np.asarray(data['state'][ids, :9], dtype=float)
        current, previous = prefix[:, 8].copy(), prefix[:, 7].copy()
        goal = np.asarray(data['goal'][ids], dtype=float)
        origin = data.get('window_start_s', np.zeros(len(data['state'])))[ids]
        age = data['source_age'][ids, 8]
        remaining = self.remaining(data, ids)
        result = []
        for step in range(16):
            feature = self.feature(current, previous, goal, origin, age, remaining, step)
            following = current + self.design(feature)@self.weight*self.scale
            previous, current = current, following
            result.append(current.copy())
        return np.stack(result, 1)

    def arrays(self):
        return {name: getattr(self, name) for name in
                ('mean', 'scale', 'feature_mean', 'feature_scale', 'weight')}
