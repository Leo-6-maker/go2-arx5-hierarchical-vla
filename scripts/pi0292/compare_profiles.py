"""CPU attribution of completed sealed B0/S2048/S4096 offline records.

Checks actual arrays and paired initial noise before comparing errors. Does not
select a checkpoint or authorize a physical run. Generation scripts stay sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np

from .durable import read, write
from .eval_contract import error_metrics, gate_action, summarize_records
from .train_adapter import sha

PROFILES = ('B0', 'S2048', 'S4096')
METRICS = ('XYZ_euclidean_m', 'SO3_rad', 'gripper_absolute')


def key(row):
    return row['id'], row['anchor'], row['noise_bank']


def scalar_summary(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    complete = len(finite) == len(values) and bool(values)
    return dict(expected=len(values), finite=len(finite), complete=complete,
        mean=float(np.mean(finite)) if complete else None,
        median=float(np.median(finite)) if complete else None,
        p95=float(np.percentile(finite, 95)) if complete else None)


def validate_records(probe, records):
    if probe['status'] != 'SEALED_BEFORE_MAIN' or probe['expert_rejects'] != 0:
        raise ValueError('SEALED_EXPERT_POSITIVE_CONTROL_REQUIRED')
    if probe['noise_banks'] != [29201, 29202] or probe['native_inference_steps'] != 4:
        raise ValueError('FROZEN_TWO_NOISE_FOUR_STEP_PROTOCOL_REQUIRED')
    expected = {(r['id'], r['anchor'], bank):r for r in probe['probes'] for bank in probe['noise_banks']}
    if len(expected) != len(probe['probes']) * 2 or len(expected) != probe['per_profile_queries']:
        raise ValueError('DUPLICATE_OR_INCOMPLETE_PROBE_PLAN')
    if set(records) != set(PROFILES):
        raise ValueError('ALL_THREE_REGISTERED_PROFILES_REQUIRED')
    indices = {}
    for profile in PROFILES:
        rows = records[profile]
        mapping = {key(row):row for row in rows}
        if len(mapping) != len(rows) or set(mapping) != set(expected):
            raise ValueError('PROFILE_QUERY_SET_INCOMPLETE_DUPLICATE_OR_EXTRA')
        for identity, row in mapping.items():
            planned = expected[identity]
            if any(row.get(field) != planned[field] for field in
                   ('id', 'anchor', 'split', 'stages', 'samples_sha256', 'numeric_sha256', 'noise_seeds')):
                raise ValueError('PROBE_SOURCE_OR_STAGE_BINDING_CHANGED')
            if row['profile'] != profile or row['query_id'] != '%s-%03d-%d' % identity:
                raise ValueError('PROFILE_QUERY_ID_CHANGED')
            if row['gate']['decision'] not in ('ACCEPT', 'WARN', 'REJECT'):
                raise ValueError('INVALID_GATE_DECISION')
            if not row['native_inverse_clip'] or row['extra_correction']:
                raise ValueError('NATIVE_DECODE_POSTPROCESSING_CHANGED')
            noise = row['noise']
            if noise['shape'] != [1, 40, 132] or not isinstance(noise['sha256'], str) or len(noise['sha256']) != 64:
                raise ValueError('INITIAL_NATIVE_NOISE_IDENTITY_REQUIRED')
            for metric in METRICS:
                values = row['errors']['pointwise'][metric]
                if len(values) != 40 or any(v is not None and (not np.isfinite(v) or v < 0) for v in values):
                    raise ValueError('INVALID_POINTWISE_ERROR_RECORD')
        indices[profile] = mapping
    for identity in expected:
        if not indices['B0'][identity]['noise'] == indices['S2048'][identity]['noise'] == indices['S4096'][identity]['noise']:
            raise ValueError('ACTUAL_INITIAL_NOISE_NOT_PAIRED')
    return indices


def aggregate(probe, records):
    indices = validate_records(probe, records)
    summary = {profile:summarize_records(records[profile]) for profile in PROFILES}
    stages, pairs, events = {}, {}, {}
    for split in ('TRAIN', 'DEV'):
        stages[split], pairs[split], events[split] = {}, {}, {}
        labels = sorted({label for row in probe['probes'] if row['split'] == split for label in row['stages']})
        for profile in PROFILES:
            rows = [row for row in records[profile] if row['split'] == split]
            stages[split][profile] = {}
            for label in labels:
                own = [row for row in rows if label in row['stages']]
                stages[split][profile][label] = dict(queries=len(own),
                    decisions=dict(Counter(row['gate']['decision'] for row in own)),
                    reasons=dict(Counter(issue['code'] for row in own for issue in row['gate']['issues'])),
                    episode_macro=summarize_records(own)[split]['episode_macro'])
            events[split][profile] = {event:dict(
                planned_queries=len(rows), missing=sum(row['errors'][event]['missing'] for row in rows),
                spurious=sum(row['errors'][event]['spurious'] for row in rows),
                expected_events=sum(bool(row['errors'][event]['expert_indices']) for row in rows),
                both_present=sum(row['errors'][event]['first_timing_error_s'] is not None for row in rows),
                timing_error_s_for_both_present=scalar_summary([row['errors'][event]['first_timing_error_s']
                    for row in rows if row['errors'][event]['first_timing_error_s'] is not None]))
                for event in ('close_events', 'open_events')}
        for candidate in ('S2048', 'S4096'):
            per_episode = {}
            identifiers = sorted({identity[0] for identity, row in indices[candidate].items() if row['split'] == split})
            for identifier in identifiers:
                identities = [identity for identity, row in indices[candidate].items()
                              if row['split'] == split and identity[0] == identifier]
                views = {}
                for view, length in (('H40', 40), ('prefix8', 8)):
                    views[view] = {}
                    for metric in METRICS:
                        differences = []
                        for identity in identities:
                            left = indices['B0'][identity]['errors']['pointwise'][metric][:length]
                            right = indices[candidate][identity]['errors']['pointwise'][metric][:length]
                            differences += [b-a if a is not None and b is not None else None for a,b in zip(left,right)]
                        views[view][metric] = scalar_summary(differences)
                per_episode[identifier] = views
            macro = {view:{metric:scalar_summary([episode[view][metric]['mean']
                        for episode in per_episode.values()]) for metric in METRICS}
                     for view in ('H40', 'prefix8')}
            pairs[split][candidate+'_minus_B0'] = dict(per_episode=per_episode, episode_macro_of_mean_differences=macro,
                interpretation='Negative mean error difference favors candidate; all Gate decisions included')
    return dict(schema='PI0292_PAIRED_OFFLINE_ATTRIBUTION_V1',
        status='PAIRED_OFFLINE_RECORDS_ANALYZED_NOT_TASK_SUCCESS', profiles=summary,
        stage_attribution=stages, events=events, paired_error_differences=pairs,
        paired_actual_noise=True, queries_per_profile=probe['per_profile_queries'],
        planned_denominator_per_profile=probe['per_profile_queries'],
        rejected_rows_excluded=False, incomplete_errors_remain_null=True,
        B1='FIXED_S4096', S2048='DIAGNOSTIC_ONLY', checkpoint_selection='NONE',
        selected='NONE', deployment='NONE')


def verify_arrays(path, row):
    if path.is_symlink() or sha(path) != row['array_sha256']:
        raise ValueError('PROFILE_ARRAY_BYTES_CHANGED')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'normalized_action', 'decoded_action', 'expert', 'actual_state', 'noise_bits'}:
            raise ValueError('OFFLINE_ARRAY_SCHEMA_CHANGED')
        normalized, decoded = arrays['normalized_action'], arrays['decoded_action']
        expert, state, noise = arrays['expert'], arrays['actual_state'], arrays['noise_bits']
        if normalized.shape != (1,40,132) or decoded.shape != (40,10) or expert.shape != (40,10) or state.shape != (10,):
            raise ValueError('OFFLINE_ARRAY_SHAPES_CHANGED')
        if not np.isfinite(expert).all() or not np.isfinite(state).all():
            raise ValueError('ACTUAL_STATE_OR_EXPERT_INVALID')
        if noise.dtype != np.uint8 or hashlib.sha256(noise.tobytes()).hexdigest() != row['noise']['sha256']:
            raise ValueError('SAVED_NATIVE_NOISE_CHANGED')
        widths = {'torch.bfloat16':2, 'torch.float16':2, 'torch.float32':4, 'torch.float64':8}
        if row['noise']['dtype'] not in widths or noise.nbytes != 40*132*widths[row['noise']['dtype']]:
            raise ValueError('SAVED_NOISE_DTYPE_OR_SHAPE_CHANGED')
        if np.count_nonzero(np.abs(normalized[...,:10]) > 1) != row['normalized_outside_unit_components']:
            raise ValueError('NORMALIZED_OUT_OF_RANGE_COUNT_CHANGED')
        if error_metrics(decoded, expert, state) != row['errors']:
            raise ValueError('ERROR_RECORD_DOES_NOT_MATCH_SAVED_ARRAYS')
        if 'decode_error' in row:
            expected_gate = dict(decision='REJECT',issues=[dict(code='NATIVE_DECODE_FAILURE',severity='REJECT')])
            if row['gate'] != expected_gate or not np.isnan(decoded).all():
                raise ValueError('DECODE_FAILURE_EVIDENCE_CHANGED')
        elif gate_action(decoded, state, row['anchor']/20, row['query_id']) != row['gate']:
            raise ValueError('GATE_RECORD_DOES_NOT_MATCH_SAVED_ARRAYS')
        return dict(expert=hashlib.sha256(expert.tobytes()).hexdigest(),
                    actual_state=hashlib.sha256(state.tobytes()).hexdigest())


def run(root, frozen_code):
    root, frozen_code = Path(root), Path(frozen_code)
    probe_path = root/'evidence/OPEN_LOOP_PROBE_SEAL.json'
    probe = read(probe_path)
    sys.path.insert(0, str(frozen_code/'src'))
    for name, digest in probe['gate_code_hashes'].items():
        if sha(frozen_code/name) != digest:
            raise ValueError('FROZEN_GATE_SOURCE_CHANGED')
        module = importlib.import_module(name.removeprefix('src/').removesuffix('.py').replace('/','.'))
        if sha(module.__file__) != digest:
            raise ValueError('IMPORTED_GATE_SOURCE_DIFFERS_FROM_SEAL')
    for name, digest in probe['evaluation_script_hashes'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError('SEALED_GENERATION_SCRIPT_CHANGED')
    records, receipts, anchors = {}, {}, {}
    for profile in PROFILES:
        directory = root/'offline'/profile
        state, summary = read(directory/'STATE.json'), read(directory/'SUMMARY.json')
        if (state['status'] != 'OFFLINE_PROFILE_EVALUATED_NOT_TASK_SUCCESS'
                or summary['status'] != state['status'] or summary['profile'] != profile
                or summary['probe_sha256'] != sha(probe_path)
                or state['queries'] != summary['queries'] or state['queries'] != probe['per_profile_queries']
                or state.get('pending')):
            raise ValueError('ACTUAL_COMPLETE_PROFILE_REQUIRED')
        records[profile] = state['records']
        anchors[profile] = {}
        receipts[profile] = dict(state_sha256=sha(directory/'STATE.json'), summary_sha256=sha(directory/'SUMMARY.json'),
                                profile_weights=summary['profile_weights'])
        if not summary['profile_weights']:
            raise ValueError('PROFILE_WEIGHT_IDENTITY_MISSING')
        for row in records[profile]:
            filename = row['query_id']
            if filename != '%s-%03d-%d' % key(row) or Path(filename).name != filename:
                raise ValueError('QUERY_PATH_ESCAPE')
            if read(directory/(filename+'.json')) != row:
                raise ValueError('PROFILE_RECORD_BYTES_CHANGED')
            anchors[profile][key(row)] = verify_arrays(directory/(filename+'.npz'), row)
        if summary['summary'] != summarize_records(records[profile]):
            raise ValueError('PROFILE_SUMMARY_CHANGED')
    result = aggregate(probe, records)
    if not anchors['B0'] == anchors['S2048'] == anchors['S4096']:
        raise ValueError('ACTUAL_STATE_OR_EXPERT_NOT_PAIRED')
    result.update(probe_sha256=sha(probe_path), actual_profile_receipts=receipts,
                  saved_arrays_recomputed=True, model_calls=0, physics_integrations=0)
    output = root/'evidence/PAIRED_OFFLINE_ATTRIBUTION.json'
    if output.exists():
        raise FileExistsError('PRESERVE_FIRST_PAIRED_COMPARISON')
    write(output,result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--frozen-code', required=True)
    arguments=parser.parse_args()
    print(run(arguments.root, arguments.frozen_code)['status'])
