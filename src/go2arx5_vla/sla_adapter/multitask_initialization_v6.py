"""Verify and transplant only the shared S1000 LoRA factors into M-SRC."""
from pathlib import Path

import numpy as np

from .checkpoint_v1 import sha
from .runtime_v1 import read_pin
from .source_training_v6 import RECIPE as SOURCE_RECIPE


def shared_key(key):
    return (key.startswith("[0]['PaliGemma']['llm']") and key.endswith('.value')
            and key.rsplit("']", 1)[0].endswith(('lora_a','lora_b')))


def sealed_source(campaign_pin, official_parent):
    campaign = read_pin(campaign_pin)
    gate = campaign['transfer_gate']
    if (campaign['exit_code'] or campaign['new_updates'] != 1000 or campaign['new_visits'] != 32000
            or campaign['new_native_requests'] or not gate['passed'] or not gate['finite'] or not gate['reloadable']
            or not gate['no_stationary_collapse_evidence'] or gate['DEV_dynamic_relative_error_reduction'] < .10
            or len(set(gate['improved_tasks'])) < 4):
        raise ValueError('SEALED_S1000_TRANSFER_GATE_REQUIRED_FOR_M_SRC')
    roles = {row['role']:row for row in campaign['records']}
    if set(roles) != {'S1000','RESTORE-S0','RESTORE-S1000'} or len(campaign['records']) != 3:
        raise ValueError('FIXED_SOURCE_AND_BOTH_RGB_RESTORES_REQUIRED')
    records = {}
    for role, row in roles.items():
        if row['result_exit'] or row['process_exit']:
            raise ValueError('SOURCE_CHILD_TERMINAL_FAILURE_PRECLUDES_TRANSFER')
        records[role] = read_pin(dict(path=row['result_path'], sha256=row['result_sha256']))
    for role in ('RESTORE-S0','RESTORE-S1000'):
        restore = records[role]
        if (restore['endpoint'] != role.removeprefix('RESTORE-') or not restore['passed']
                or not restore['full_restore']['bit_exact'] or len(restore['comparisons']) != 16
                or not all(row['bit_exact'] for row in restore['comparisons'])):
            raise ValueError('ACTUAL_INDEPENDENT_RGB16_SOURCE_RESTORE_REQUIRED')
    train = records['S1000']
    if train['new_updates'] != 1000 or train['new_visits'] != 32000 or not train['full_restore']['bit_exact']:
        raise ValueError('ACTUAL_S1000_MODEL_ADAM_RNG_RESTORE_REQUIRED')
    families = read_pin(train['shared_lora'])
    endpoint = train['last_checkpoint']
    root = Path(endpoint['path'])
    ckpt = read_pin(dict(path=str(root/'COMPLETE.json'), sha256=endpoint['sha256']))
    if (families['source_checkpoint'] != endpoint or ckpt['step'] != 1000 or ckpt['cursor'] != 32000
            or ckpt['bindings']['recipe'] != SOURCE_RECIPE
            or ckpt['bindings']['parent_manifest']['sha256'] != official_parent['sha256']):
        raise ValueError('SHARED_ROWS_MUST_BE_FROM_THE_ACTUAL_FIXED_SOURCE_ENDPOINT')
    expected = {row['key']:dict(row, path=str(root/row['file'])) for row in ckpt['arrays'] if shared_key(row['key'])}
    rows = families['arrays']
    if len(expected) != 20 or len(rows) != len(expected) or {row['key']:row for row in rows} != expected:
        raise ValueError('EXACT_TWENTY_SHARED_LORA_NAMES_SHAPES_AND_SHAS_REQUIRED')
    return rows, dict(source_campaign=campaign_pin, source_checkpoint=endpoint,
        source_lora_manifest=train['shared_lora'], DEV_dynamic_relative_error_reduction=gate['DEV_dynamic_relative_error_reduction'],
        source_updates=1000, source_visits=32000, source_IO_norm_optimizer_copied=False)


def transplant(active, source_rows):
    import jax

    leaves, structure = jax.tree_util.tree_flatten_with_path((active,))
    rows = {row['key']:row for row in source_rows}
    keys = {jax.tree_util.keystr(key) for key, _ in leaves if shared_key(jax.tree_util.keystr(key))}
    if len(leaves) != 28 or len(keys) != 20 or set(rows) != keys or len(rows) != len(source_rows):
        raise ValueError('SOURCE_AND_TARGET_SHARED_LORA_SETS_MUST_MATCH_EXACTLY')
    values = []
    target_io = []
    for path, value in leaves:
        key = jax.tree_util.keystr(path)
        if key in rows:
            row = rows[key]
            if (row['shape'] != list(value.shape) or row['dtype'] != str(value.dtype)
                    or sha(row['path']) != row['sha256']):
                raise ValueError('CANNOT_COPY_UNMATCHED_SHARED_LORA:'+key)
            array = np.load(row['path'], allow_pickle=False)
            if array.shape != value.shape or array.dtype != value.dtype or not np.isfinite(array).all():
                raise ValueError('CANNOT_COPY_INVALID_SHARED_LORA_BYTES:'+key)
            values.append(jax.device_put(array))
        else:
            values.append(value)
            target_io.append(key)
    (initialized,) = jax.tree_util.tree_unflatten(structure, values)
    return initialized, dict(shared_rows=source_rows, target_IO_original_keys=target_io)


def match_initial_checkpoints(base_pin, source_pin, source_rows, *, arms=('M-BASE', 'M-SRC')):
    """Prove identical target I/O, Adam and RNG at the two step-zero boundaries."""
    base, src = (read_pin(dict(path=str(Path(pin['path'])/'COMPLETE.json'), sha256=pin['sha256']))
                 for pin in (base_pin, source_pin))
    left, right = ({row['key']:row for row in value['arrays']} for value in (base,src))
    changed_keys = {row['key']:row for row in source_rows}
    common_bindings = ('parent_manifest','tokenizer','target_audit','visit_map_sha256',
                       'lr_sha256','seed','frozen_fingerprint','optimizer','initial_function_rule',
                       'recipe','codec_version','derived_statistics_sha256','full_XYZ_lambda','full_gripper_speed_lambda')
    if (any(value['step'] or value['cursor'] for value in (base,src))
            or set(left) != set(right) or len(left) != 87 or len(changed_keys) != 20
            or (base['bindings']['arm'], src['bindings']['arm']) != arms
            or base['bindings']['source_initialization'] is not None
            or any(base['bindings'][key] != src['bindings'][key] for key in common_bindings)
            or set(changed_keys) != {key for key in right if shared_key(key)}):
        raise ValueError('MATCHED_M_STEP_ZERO_STATE_AND_TRAINING_BINDINGS_REQUIRED')
    changed = []
    for key, row in right.items():
        expected = changed_keys.get(key, left[key])
        if any(row[field] != expected[field] for field in ('shape','dtype','sha256')):
            raise ValueError('M_INITIALIZATION_CHANGED_OUTSIDE_DECLARED_SOURCE_LORA:'+key)
        if row['sha256'] != left[key]['sha256']:
            changed.append(key)
    return dict(base=base_pin, source=source_pin, checked_array_count=len(left),
        shared_lora_array_count=len(changed_keys), nonshared_bit_exact=len(left)-len(changed_keys), changed_shared_keys=changed)
