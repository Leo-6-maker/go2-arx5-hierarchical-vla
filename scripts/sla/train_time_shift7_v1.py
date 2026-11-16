"""Reuse the qualified500 worker and ordinaryFM gradient for one layout intervention."""
from scripts.sla import train_close_time_v1 as original
from go2arx5_vla.sla_adapter.time_shift7_training_v1 import ShiftEngine
from go2arx5_vla.sla_adapter.provider_time_shift7_v1 import export, Provider
from pathlib import Path
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha
from scripts.pi0292.durable import write


COMMON = ('parent_manifest','tokenizer','target_audit','visit_map_sha256','lr_sha256','seed',
    'frozen_fingerprint','upstream_sha','optimizer','initial_function_rule','XLA_FLAGS',
    'microbatch','accumulation_steps','effective_batch','random_stream','sample_cursor',
    'training_matmul_precision','parent_C20_package','parent_step','parent_visits',
    'dense_manifest_sha256','state_token_dropout','inference_route','gripper_lambda')


def initial_match(spec, root, complete):
    control = Path(spec['matched_control_root'])
    if sha(control/'RESULT.private.json') != spec['reused_control_result_sha256']:
        raise ValueError('SEALED_PRIMARY_TIME500_CONTROL_CHANGED')
    prior = read(control/'work/step0000/COMPLETE.json')
    if (len(prior['arrays']) != 87 or len(complete['arrays']) != 87
            or any(any(a[k] != b[k] for k in ('key','shape','dtype','sha256'))
                for a,b in zip(prior['arrays'],complete['arrays']))
            or any(prior['bindings'][k] != complete['bindings'][k] for k in COMMON)):
        raise ValueError('EXACT_TIME500_INITIAL_MODEL_ADAM_RNG_AND_RECIPE_MATCH')
    a,b = [read(p/'GRADIENT_ACTUAL.private.json') for p in (control,root)]
    if any(a[k] != b[k] for k in ('noise_bank_sha256','tau_bank_sha256','visit_map_sha256')):
        raise ValueError('MATCHED_FIRST_ACTUAL_NOISE_TIME_VISIT_BANK_REQUIRED')
    write(root/'INITIAL_CONTROL_MATCH_ACTUAL.private.json',dict(initial_arrays_bit_exact=87,
        common_bindings_bit_exact=True,first_actual_noise_time_bank_bit_exact=True,
        new_optimizer_updates=0,changed_variable='CYCLIC_PARAMETER_SLOT_SHIFT_PLUS7_ONLY'))


def train(spec):
    return original.train(spec, engine_class=ShiftEngine, export_package=export, initial_match_fn=initial_match)


def replay(spec):
    return original.replay(spec, provider_class=Provider)


if __name__ == '__main__':
    original.main(train_fn=train, replay_fn=replay, RGB_operation='--shift-rgb')
