"""Finite all1376 permutation and eight existing-input checks; no new model calls."""
import hashlib
from pathlib import Path
import time
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin, probe_indices
from go2arx5_vla.sla_adapter.time_training_v1 import TimeDataset, physical as original_physical
from go2arx5_vla.sla_adapter.time_shift7_training_v1 import physical
from go2arx5_vla.sla_adapter.gripper_time_shift7_v1 import SHIFT, relocate, decode
from go2arx5_vla.sla_adapter.gripper_time_v1 import decode as original_decode


def run(spec):
    from openpi_client.image_tools import resize_with_pad
    started = time.time(); cpu = time.process_time()
    if any(spec[k] for k in ('new_optimizer_updates_budget', 'new_physical_requests_budget', 'new_complete_generations_budget')):
        raise ValueError('SHIFT_PREPARATION_HAS_ZERO_MODEL_TRAINING_NATIVE_BUDGET')
    d = TimeDataset(spec['dataset'], spec['dense_manifest_sha256'], time_audit=spec['time_codec_audit'],
        resize_with_pad=resize_with_pad, max_cache_bytes=128*1024**2)
    old = d.time_targets; shifted = relocate(old)
    if len(old) != 1376 or not np.array_equal(np.roll(shifted, -SHIFT, axis=-1), old):
        raise ValueError('ALL1376_EXACT_PARAMETER_PERMUTATION')
    low, high = d.statistics['actions'].q01[6], d.statistics['actions'].q99[6]
    for index in range(len(d)):
        a, _ = original_decode(old[index], d.reference(index), low, high)
        b, _ = decode(shifted[index], d.reference(index), low, high)
        if not np.array_equal(a, b):
            raise ValueError('PERMUTATION_MUST_NOT_CHANGE_LABEL_PHYSICAL_TRAJECTORY')
    records = []
    for index in probe_indices(d):
        d.time_targets = old; before = d[index]
        d.time_targets = shifted; after = d[index]
        if (not np.array_equal(after['state'], before['state']) or after['prompt'] != before['prompt']
                or not np.array_equal(after['actions'][:, :6], before['actions'][:, :6])
                or not np.array_equal(after['actions'][:, 7:], before['actions'][:, 7:])
                or not np.array_equal(after['actions'][:, 6], relocate(before['actions'][:, 6]))):
            raise ValueError('ONLY_PARAMETER_SLOT_POSITIONS_MAY_CHANGE')
        for key in ('image', 'image_mask', 'image_padding_mask'):
            for name in after[key]:
                if not np.array_equal(after[key][name], before[key][name]):
                    raise ValueError('ORIGINAL_IMAGE_INPUT_CHANGED')
        state, *_ = d.raw(index); ep, _ = d.rows[index]; frame = d.frames[ep['id']]['T_policy_from_mount']
        a, _ = original_physical(before['actions'][None], state, frame, d.statistics, d.reference(index))
        b, _ = physical(after['actions'][None], state, frame, d.statistics, d.reference(index))
        if not np.array_equal(a, b):
            raise ValueError('FULL_LABEL_XYZ_SO3_WIDTH_DECODE_MUST_BE_BIT_EXACT')
        records.append(dict(index=index,RGB_state_prompt_other31_action_channels_bit_exact=True,
            parameter_slot_permutation_only=True,physical_H40_bit_exact=True))
    return dict(schema='SLA_TIME_SHIFT7_ALL_C20_LAYOUT_AUDIT_V1', exit_code=0, passes=True,
        shift=SHIFT, windows=len(d), dataset_manifest_sha256=d.manifest_sha,
        statistics_sha256=d.manifest['statistics']['sha256'], old_time_audit=spec['time_codec_audit'],
        shifted_target_array_sha256=hashlib.sha256(shifted.tobytes()).hexdigest(),
        all1376_label_physical_H40_bit_exact=True, parameter_repetitions=[14, 13, 13], records=records,
        original_data_or_stats_written=False, new_model_calls=0, new_updates=0, new_visits=0,
        new_native_requests=0, new_integrations=0, wall_s=time.time()-started, CPU_s=time.process_time()-cpu)


if __name__ == '__main__':
    import sys, json, traceback
    s = json.loads(Path(sys.argv[2]).read_bytes()); root = Path(s['registered_root'])
    try:
        value = run(s)
    except BaseException:
        value = dict(exit_code=1, passes=False, failure=traceback.format_exc(),
            new_updates=0, new_model_calls=0, new_native_requests=0)
    write(root/'RESULT.private.json', value)
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s, value['exit_code']); raise SystemExit(value['exit_code'])
