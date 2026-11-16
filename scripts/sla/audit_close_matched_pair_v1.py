"""CPU-only audit of actual paired states and committed input transcripts.

This does not run JAX, generate predictions, alter scores, or grant admission.
"""
import hashlib
import json
import math
from pathlib import Path
import time


RECIPE = 'SLA_CLOSE_PARENT1000_NEW500_MATCHED_V1'
ARMS = ('fm-control', 'grip-temporal')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def actual_log(path):
    # A live writer may be between write and flush. Only complete lines count.
    content = Path(path).read_bytes()
    return [json.loads(line) for line in content.rsplit(b'\n', 1)[0].splitlines()]


def input_record(row, ordinal):
    if row['step'] != ordinal + 1 or row['cursor'] != 32 * (ordinal + 1):
        raise ValueError('ACTUAL_INPUT_LOG_STEP_OR_CURSOR_GAP')
    if not math.isfinite(row['grad_norm_before_clip']) or row['grad_norm_before_clip'] <= 0:
        raise ValueError('ACTUAL_UPDATE_WITHOUT_FINITE_NONZERO_GRADIENT')
    micro = row['micro']
    if len(micro) != 8:
        raise ValueError('FIXED_MICRO4_EFFECTIVE32_REQUIRED')
    for index, part in enumerate(micro):
        if (part['first_absolute_cursor'] != ordinal * 32 + index * 4
                or len(part['indices']) != 4 or len(part['state_token_dropped']) != 4
                or any(type(x) is not bool for x in part['state_token_dropped'])):
            raise ValueError('ACTUAL_MICROBATCH_INPUT_TRANSCRIPT_INCOMPLETE')
    return {key: row[key] for key in ('step', 'cursor', 'used_lr', 'micro')}


def audit(root, *, require_complete=False):
    root = Path(root)
    started = time.time()
    initial = [read(root / arm / 'work/step0000/COMPLETE.json') for arm in ARMS]
    if any(v['step'] != 0 or v['cursor'] != 0 or len(v['arrays']) != 87 for v in initial):
        raise ValueError('COMMON_FRESH_COMPLETE_INITIAL_STATE_REQUIRED')
    common = []
    for coefficient, value in zip((0., 1.), initial):
        bindings = dict(value['bindings'])
        if bindings.pop('gripper_lambda') != coefficient or bindings['recipe'] != RECIPE:
            raise ValueError('EXACT_ONE_TEMPORAL_OBJECTIVE_PAIR_REQUIRED')
        common.append(bindings)
    if common[0] != common[1]:
        raise ValueError('SCIENTIFIC_BINDINGS_DIFFER_BEYOND_TEMPORAL_LAMBDA')
    for rows in zip(*(v['arrays'] for v in initial)):
        if any(rows[0][key] != rows[1][key] for key in ('key', 'shape', 'dtype', 'sha256')):
            raise ValueError('COMMON_INITIAL_MODEL_ADAM_RNG_BYTES_DIFFER')
        for arm, row in zip(ARMS, rows):
            folder = root / arm / 'work/step0000'
            path = folder / row['file']
            if not path.resolve().is_relative_to(folder.resolve()) or sha(path) != row['sha256']:
                raise ValueError('INITIAL_STATE_SAVED_BYTES_CHANGED')
    gradients = [read(root / arm / 'GRADIENT_ACTUAL.private.json') for arm in ARMS]
    for key in ('noise_bank_sha256', 'tau_bank_sha256', 'visit_map_sha256'):
        if gradients[0][key] != gradients[1][key]:
            raise ValueError('FIRST_REAL_NOISE_TIME_OR_VISIT_BANK_DIFFER')
    logs = [actual_log(root / arm / 'work/updates.private.jsonl') for arm in ARMS]
    streams = [[input_record(row, i) for i, row in enumerate(rows)] for rows in logs]
    count = min(map(len, streams))
    if not count or any(len(rows) > 500 for rows in streams):
        raise ValueError('FINITE_NEW500_ACTUAL_UPDATE_SCOPE')
    if streams[0][:count] != streams[1][:count]:
        raise ValueError('ACTUAL_VISIT_DROP_OR_LR_INPUT_PREFIX_DIFFER')
    endpoint = {}
    if require_complete:
        controller = read(root / 'RESULT.private.json')
        if (controller['exit_code'] != 0 or not controller['common_initial_arrays_bit_exact']
                or not controller['actual_input_stream_matched'] or [len(x) for x in logs] != [500, 500]):
            raise ValueError('COMPLETE_MATCHED_PAIR_ENDPOINT_NOT_PROVEN')
        for arm in ARMS:
            result = read(root / arm / 'RESULT.private.json')
            rgb = read(root / (arm + '-fresh-rgb') / 'RESULT.private.json')
            if (result['exit_code'] != 0 or result['new_updates'] != 500
                    or result['new_training_visits'] != 16000 or result['new_physical_requests'] != 0
                    or not result['full_restore']['bit_exact'] or result['full_restore']['arrays'] != 87
                    or rgb['exit_code'] != 0 or not rgb['all_bit_exact'] or rgb['ordinary_RGB_generations'] != 16):
                raise ValueError('REAL500_FULL_RESTORE_AND_ORDINARY_RGB16_REQUIRED')
            endpoint[arm] = dict(metrics=result['metrics'],training_result_sha256=sha(root / arm / 'RESULT.private.json'),
                fresh_RGB_result_sha256=sha(root / (arm + '-fresh-rgb') / 'RESULT.private.json'))
    digest = hashlib.sha256(json.dumps(streams[0][:count], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema='SLA_CLOSE_MATCHED_ACTUAL_INPUT_AUDIT_V1',complete_pair=require_complete,
        observed_epoch=time.time(),actual_updates={arm: len(rows) for arm, rows in zip(ARMS, logs)},
        actual_training_visits={arm: 32 * len(rows) for arm, rows in zip(ARMS, logs)},
        matched_actual_input_prefix_updates=count,matched_actual_input_prefix_visits=count * 32,
        common_model_Adam_RNG_arrays=87,initial_saved_arrays_byte_verified=True,
        scientific_bindings_equal_except_lambda=True,actual_visit_dropout_LR_prefix_equal=True,
        actual_prefix_stream_sha256=digest,first_real_noise_time_bank_sha_equal=True,
        remaining_noise_time_identity_evidence='SAME_BYTE_VERIFIED_INITIAL_RNG_FIXED32_PRODUCER_AND_ABSOLUTE_CURSOR_IN_COMMON_SEALED_SOURCE',
        all_step_noise_time_banks_individually_saved=False,
        common_bindings_sha256=hashlib.sha256(json.dumps(common[0], sort_keys=True).encode()).hexdigest(),
        initial_COMPLETE_sha256={arm:sha(root / arm / 'work/step0000/COMPLETE.json') for arm in ARMS},
        endpoints=endpoint,new_generations=0,new_optimizer_updates=0,new_physical_requests=0,
        original_scores_changed=False,independent_admission=False,goal_met=False,wall_s=time.time()-started)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.root, require_complete=args.require_complete)))
