"""Exact, checkpointable sample order for the controlled PILOT2 Actor study."""
from copy import deepcopy
import numpy as np

MODES = ('ORIGINAL_GROUPED_CYCLIC', 'SEEDED_GLOBAL_SHUFFLE')


class ExactSampler:
    def __init__(self, size, mode, seed):
        if type(size) is not int or size < 1 or mode not in MODES or type(seed) is not int or seed < 0:
            raise ValueError('EXPLICIT_SAMPLER_SIZE_MODE_SEED_REQUIRED')
        self.size, self.mode, self.seed = size, mode, seed
        self.generator = np.random.default_rng(seed)
        self.epoch = self.offset = self.draws = 0
        self.permutation = self._permutation()

    def _permutation(self):
        return (self.generator.permutation(self.size) if self.mode == MODES[1]
                else np.arange(self.size, dtype=np.int64))

    def take(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('POSITIVE_EXACT_BATCH_SIZE_REQUIRED')
        parts, left = [], count
        while left:
            if self.offset == self.size:
                self.epoch += 1
                self.offset = 0
                self.permutation = self._permutation()
            n = min(left, self.size-self.offset)
            parts.append(self.permutation[self.offset:self.offset+n].copy())
            self.offset += n
            self.draws += n
            left -= n
        return np.concatenate(parts)

    def exposure_counts(self):
        counts = np.full(self.size, self.epoch, dtype=np.int64)
        counts[self.permutation[:self.offset]] += 1
        return counts

    def state_dict(self):
        # Lists/ints/dicts remain safe with torch.load(weights_only=True).
        return dict(schema='WM02_EXACT_ACTOR_SAMPLER_V1', size=self.size,
                    mode=self.mode, seed=self.seed, epoch=self.epoch,
                    offset=self.offset, draws=self.draws,
                    permutation=self.permutation.tolist(),
                    generator=deepcopy(self.generator.bit_generator.state))

    def load_state_dict(self, state):
        if any(state.get(k) != v for k, v in dict(schema='WM02_EXACT_ACTOR_SAMPLER_V1',
                size=self.size, mode=self.mode, seed=self.seed).items()):
            raise ValueError('SAMPLER_RESTORE_IDENTITY_DIFFERS')
        epoch, offset, draws = (state[k] for k in ('epoch', 'offset', 'draws'))
        permutation = np.asarray(state['permutation'])
        if (any(type(v) is not int for v in (epoch, offset, draws)) or epoch < 0
                or not 0 <= offset <= self.size or draws != epoch*self.size+offset
                or permutation.dtype.kind not in 'iu' or permutation.shape != (self.size,)
                or not np.array_equal(np.sort(permutation), np.arange(self.size))
                or self.mode == MODES[0] and not np.array_equal(permutation, np.arange(self.size))):
            raise ValueError('SAMPLER_CHECKPOINT_NOT_COMPLETE_OR_VALID')
        generator = np.random.default_rng(self.seed)
        generator.bit_generator.state = deepcopy(state['generator'])
        self.generator = generator
        self.epoch, self.offset, self.draws = epoch, offset, draws
        self.permutation = permutation.astype(np.int64, copy=True)

