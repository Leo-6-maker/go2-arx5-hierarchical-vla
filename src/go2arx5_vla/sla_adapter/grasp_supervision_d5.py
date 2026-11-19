"""D5 full-generated mechanical width and time-increment supervision."""
import hashlib
from pathlib import Path

import numpy as np

from . import batch_v2, dense_g7_v6, multisource_d4
from .checkpoint_v1 import restore
from .provider_multitask_v6 import export_payload
from .runtime_v1 import read_pin

PLAN = 'cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb'
RECIPE = 'SLA_D5_FROM_HISTORICAL_B_STEP_ZERO_G7'
SCHEMA = 'SLA_D5_C20_G7_RGB_PACKAGE_V1'
ARMS = ('D5-W', 'D5-T')
HISTORICAL_INITIAL = '515e5200a2e756e7ddcf7e415c3f8d6177485ec897d6b345ec3f10b5751975e3'
STEP_ZERO = 'f9134574a0bf357d64852fc6d626eafb2e90d4de5b068d7d59354aa0969f0eaa'
VISIT_BANK = 'a151bfca90511a5fee70398aa50616472f989cbf41cdd899ca93d569a24040af'
DROPOUT = '84f34962c986e6eb046547940acdd49e0ef44b102ac00a844faa870cba765a7c'


def width_losses(predicted, expected, previous, *, xp):
    if predicted.shape != expected.shape or predicted.shape[-1] != 40 or previous.shape != predicted.shape[:-1]:
        raise ValueError('D5_REQUIRES_ALL_40_WIDTHS_AND_ONE_ACTUAL_PREVIOUS_COMMAND')
    width = xp.mean(((predicted-expected)/.040)**2)
    predicted_delta = xp.diff(xp.concatenate((previous[..., None], predicted), axis=-1), axis=-1)
    expected_delta = xp.diff(xp.concatenate((previous[..., None], expected), axis=-1), axis=-1)
    return width, xp.mean(((predicted_delta-expected_delta)/.002)**2)


def verify_history(spec):
    if spec['plan_commit'] != PLAN or spec['arm'] not in ARMS:
        raise ValueError('FIXED_D5_PLAN_AND_ARM_REQUIRED')
    pin = spec['historical_B_initial']
    if pin['sha256'] != HISTORICAL_INITIAL:
        raise ValueError('ORIGINAL_D4_B_STEP_ZERO_RECEIPT_REQUIRED')
    actual = read_pin(pin)
    initial = actual['checkpoint']
    receipt = read_pin(dict(path=str(Path(initial['path'])/'COMPLETE.json'), sha256=STEP_ZERO))
    if (actual['match']['checked_array_count'] != 87 or initial['sha256'] != STEP_ZERO
            or (receipt['step'], receipt['cursor'], len(receipt['arrays'])) != (0, 0, 87)
            or actual['visit_map_sha256'] != VISIT_BANK or receipt['bindings']['visit_map_sha256'] != VISIT_BANK):
        raise ValueError('D5_MATCHED_87_B_STEP_ZERO_AND_C20_SCHEDULE_REQUIRED')
    return actual, receipt


class Engine(multisource_d4.TransferEngine):
    recipe = RECIPE
    arms = ARMS
    source_arm = None
    auxiliary_metric = 'weighted_G7_XYZ_speed_and_D5_supervision'

    def __init__(self, spec, output):
        history, receipt = verify_history(spec)
        self.delta_weight = float(spec['arm'] == ARMS[1])
        super().__init__(spec, output)
        prior = receipt['bindings']
        comparable = ('tokenizer', 'target_audit', 'visit_map_sha256', 'lr_sha256', 'seed', 'frozen_fingerprint',
                      'upstream_sha', 'initial_function_rule', 'XLA_FLAGS', 'optimizer', 'microbatch',
                      'accumulation_steps', 'effective_batch', 'random_stream', 'sample_cursor',
                      'training_matmul_precision', 'aux_samples_per_update', 'full_sampling_steps',
                      'derived_statistics_sha256', 'dataset_manifest_sha256')
        if (any(self.bindings[k] != prior[k] for k in comparable)
                or self.bindings['parent_manifest']['sha256'] != prior['parent_manifest']['sha256']
                or hashlib.sha256(bytes(batch_v2.drop_state(i) for i in range(16000))).hexdigest() != DROPOUT):
            raise ValueError('D5_TRAINING_INPUTS_OR_OPTIMIZER_DIFFER_FROM_HISTORICAL_B')
        (self.active, self.opt_state, self.rng), self.step, self.cursor = restore(
            history['checkpoint']['path'], (self.active, self.opt_state, self.rng), bindings=prior)
        noise, flow = self.draw_bank(self.rng, np.arange(32, dtype=np.uint32))
        for key, value in (('initial_noise_sha256', noise), ('initial_flow_time_sha256', flow)):
            if hashlib.sha256(np.asarray(value).tobytes()).hexdigest() != history[key]:
                raise ValueError('D5_ACTUAL_INITIAL_RANDOM_BANK_MISMATCH:'+key)
        self.bindings.update(historical_B_step_zero_sha256=STEP_ZERO, physical_width_lambda=1.,
                             physical_delta_lambda=self.delta_weight)
        self.initial_match = dict(verified_model_Adam_RNG_arrays=87, historical_B_initial_sha256=HISTORICAL_INITIAL,
                                  step_zero_sha256=STEP_ZERO, visit_bank_sha256=VISIT_BANK, dropout_sha256=DROPOUT,
                                  first_noise_sha256=history['initial_noise_sha256'],
                                  first_flow_time_sha256=history['initial_flow_time_sha256'])

    def target_width_for_aux(self, index):
        episode, anchor = self.dataset.rows[index]
        targets = self.dataset.labels[episode['id']]['nominal_policy_H40'][anchor, :, 9]
        return .002 + .040 * targets

    def auxiliary_objective(self, normalized, low, high, previous_width, target_width, *, xp, sigmoid):
        original, detail = dense_g7_v6.full_objective(normalized, low, high, previous_width, xp=xp, sigmoid=sigmoid)
        latent = (normalized[..., 6]+1)*.5*(high[6]-low[6]+1e-6)+low[6]
        width, delta = width_losses(.002+.040*sigmoid(latent), target_width, previous_width, xp=xp)
        return original + width + self.delta_weight*delta, dict(detail, physical_width_loss=width,
                                                                physical_delta_loss=delta)


def package_contract(manifest):
    multisource_d4.package_contract(manifest, schema=SCHEMA, recipe=RECIPE, arms=ARMS, source_arm=None)
    binding = manifest['bindings']
    if (binding['goal_plan_sha'] != PLAN or binding['historical_B_step_zero_sha256'] != STEP_ZERO
            or binding['physical_width_lambda'] != 1. or binding['physical_delta_lambda'] != (binding['arm'] == ARMS[1])):
        raise ValueError('D5_ENDPOINT_REQUIRES_THE_PREDECLARED_HISTORICAL_START_AND_LOSS')


def export(package, checkpoint, *, engine):
    stats = engine.dataset.root/engine.dataset.manifest['statistics']['file']
    return export_payload(package, checkpoint, engine=engine, statistics=stats,
                          schema=SCHEMA, validate=package_contract, step=500, cursor=16000)


class Provider(multisource_d4.Provider):
    def validate_package_contract(self, manifest):
        package_contract(manifest)
