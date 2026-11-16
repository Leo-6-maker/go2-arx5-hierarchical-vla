"""Same pure-FM learner, C20 start and RNG; only the parameter-slot layout moves."""
import numpy as np
from .time_training_v1 import TimeDataset, TimeEngine
from .runtime_v1 import read_pin
from .checkpoint_v1 import sha
from .gripper_time_shift7_v1 import VERSION, SHIFT, SLOTS, relocate, decode
from .codec_v1 import Codec, pad32

RECIPE = 'SLA_TIME_PARAMETER_LAYOUT_SHIFT7_PARENT1000_NEW500_V1'


def physical(normalized, state, frame, statistics, reference):
    values = np.asarray(normalized)
    if values.shape != (1, 40, 32) or not np.isfinite(values).all():
        raise ValueError('FINITE_H40_32_SHIFT7_CONTAINER')
    active7 = statistics['actions'].unnormalize(values[0, :, :7]); active7[:, 6] = 0.
    result = Codec(frame).decode(state, pad32(active7))
    width, detail = decode(values[0, :, 6], reference,
        statistics['actions'].q01[6], statistics['actions'].q99[6])
    result[:, 9] = (width - .002) / .040
    return result, detail


class ShiftDataset(TimeDataset):
    def __init__(self, *args, shift_audit, **kwargs):
        super().__init__(*args, **kwargs)
        audit = read_pin(shift_audit)
        shifted = relocate(self.time_targets)
        import hashlib
        if (audit['schema'] != 'SLA_TIME_SHIFT7_ALL_C20_LAYOUT_AUDIT_V1'
                or not audit['passes'] or audit['shift'] != SHIFT
                or audit['dataset_manifest_sha256'] != self.manifest_sha
                or audit['statistics_sha256'] != self.manifest['statistics']['sha256']
                or audit['old_time_audit'] != kwargs['time_audit']
                or audit['shifted_target_array_sha256'] != hashlib.sha256(shifted.tobytes()).hexdigest()):
            raise ValueError('ACTUAL_ALL1376_SHIFT_ONLY_PERMUTATION_REQUIRED')
        self.time_targets = shifted


class ShiftEngine(TimeEngine):
    def dataset_for_spec(self, spec):
        from openpi_client.image_tools import resize_with_pad
        audit = read_pin(spec['target_audit'])
        if (spec['operation'] != RECIPE or spec['gripper_lambda'] != 0.
                or spec['microbatch'] != 4 or spec['planned_candidate_updates'] != 500
                or spec['new_optimizer_updates_budget'] != 500
                or spec['new_training_visits_budget'] != 16000
                or spec['training_matmul_precision'] != 'highest'
                or not audit['passes'] or audit['manifest_sha256'] != spec['dense_manifest_sha256']):
            raise ValueError('USER_AUTHORIZED_SHIFT_ONLY_EXACT500')
        return ShiftDataset(spec['dataset'], spec['dense_manifest_sha256'],
            resize_with_pad=resize_with_pad, max_cache_bytes=spec['frame_cache_bytes'],
            time_audit=spec['time_codec_audit'], shift_audit=spec['shift_layout_audit'])

    def __init__(self, spec, output):
        super().__init__(spec, output)
        self.bindings.update(recipe=RECIPE, codec_version=VERSION, parameter_slot_shift=SHIFT,
            time_parameter_slots=[[int(i) for i in group] for group in SLOTS],
            shift_layout_audit=spec['shift_layout_audit'])

    def generate(self, index, noise_id):
        import jax
        from .model_v1 import observation
        item = self.dataset[index]; obs, _ = observation(item, self.tokenizer)
        noise = jax.random.normal(jax.random.PRNGKey(2026111200 + noise_id), (1, 40, 32))
        self.counts['ordinary_RGB_generation_attempts'] += 1
        value = self.infer(self.active, self.frozen, obs, noise)
        jax.block_until_ready(value); value = np.asarray(value)
        state, *_ = self.dataset.raw(index); ep, _ = self.dataset.rows[index]
        trajectory, _ = physical(value, state, self.dataset.frames[ep['id']]['T_policy_from_mount'],
            self.dataset.statistics, self.dataset.reference(index))
        self.counts['ordinary_RGB_generations'] += 1
        return value, trajectory
