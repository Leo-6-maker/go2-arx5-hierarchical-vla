"""Recompute original H40 and command-seam gates on the fixed D4-B/W/T predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.dense_g7_v6 import DT, GRIP_LIMIT, label_margin_report, previous_command_context
from go2arx5_vla.sla_adapter.multisource_d4 import C20G7Dataset
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import CHANNELS, analyze
from scripts.sla.close_goal_g1_v1 import width_metrics
from scripts.sla.export_mfree_c20_v1 import _reference


ARMS = ('D4-B', 'D5-W', 'D5-T')
FIXED_BANK = '467c9939039f8e5d0d0f9fb2c1a907a02cffc02a997aa07754f1cdb517926afc'


def summarize(rows):
    annotated = [row for row in rows if row['width']['expert_crossing_in_window'] is not None]
    crossed = [row for row in annotated if row['width']['predicted_first_crossing'] is not None]
    on_time = [row for row in crossed if row['width']['crossing_error_s'] <= .1]
    failed = [row for row in rows if not row['H40_pass']]
    first = Counter(row['first_temporal_exceedance']['channel'] for row in failed)
    short = Counter(row['original_short_circuit_component']['channel'] for row in failed)
    return dict(predictions=len(rows), annotated_crossings=len(annotated), crossings=len(crossed),
        crossings_error_at_most_100ms=len(on_time),
        crossings_with_H40_and_seam=sum(row['H40_pass'] and row['seam_pass'] for row in crossed),
        crossings_with_H40_seam_and_100ms=sum(row['H40_pass'] and row['seam_pass'] for row in on_time),
        H40_pass=len(rows)-len(failed), H40_and_seam_pass=sum(row['H40_pass'] and row['seam_pass'] for row in rows),
        only_seam_rejected=sum(row['H40_pass'] and not row['seam_pass'] for row in rows),
        also_seam_rejected=sum(not row['H40_pass'] and not row['seam_pass'] for row in rows),
        mean_width_error_m=float(np.mean([row['width']['width_MAE_m'] for row in rows])),
        first_temporal_by_channel={channel:first[channel] for channel, _ in CHANNELS},
        compiler_first_by_channel={channel:short[channel] for channel, _ in CHANNELS})


def run(arms, expected_hashes):
    from openpi_client.image_tools import resize_with_pad

    if set(arms) != set(ARMS) or set(expected_hashes) != set(ARMS):
        raise ValueError('D5_AUDIT_REQUIRES_THE_EXACT_THREE_ENDPOINTS')
    began = time.monotonic()
    output, common_population = {}, None
    for arm in ARMS:
        root = arms[arm]
        result_path = root/'RESULT.private.json'
        result = read_pin(dict(path=str(result_path), sha256=expected_hashes[arm]))
        spec = json.loads((root/'EXECUTION_SPEC.json').read_bytes())
        if (result['arm'] != arm or result['new_updates'] != 500 or result['new_visits'] != 16000
                or result['exit_code'] or spec['fixed_TRAIN48']['sha256'] != FIXED_BANK):
            raise ValueError('D5_AUDIT_EXPECTS_ORIGINAL_COMPLETE_TRAIN500')
        bank = read_pin(spec['fixed_TRAIN48'])['windows']
        contexts = read_pin(spec['portable_evaluation'])
        dataset = C20G7Dataset(spec['dataset'], time_audit=spec['time_codec_audit'],
            portable_inputs=spec['portable_inputs'], resize_with_pad=resize_with_pad)
        records = root/'TRAIN48_PREDICTIONS.private.json'
        ledger = json.loads(records.read_bytes())
        expected = [(row['index'], noise) for row in bank for noise in (0, 1)]
        if len(set(expected)) != 96 or [(row['index'], row['noise']) for row in ledger] != expected:
            raise ValueError('D5_AUDIT_FIXED_48_WINDOW_BY_TWO_NOISE_ORDER_REQUIRED')
        rows, population = [], []
        for row in ledger:
            index = row['index']
            episode, anchor = dataset.rows[index]
            context = contexts['windows'][str(index)]
            if (row['episode'], row['anchor']) != (episode['id'], anchor) or (
                    context['episode'], context['anchor']) != (episode['id'], anchor):
                raise ValueError('D5_AUDIT_WINDOW_IDENTITY_CHANGED')
            path = Path(row['output']['path']).resolve(strict=True)
            if not path.is_relative_to(root.resolve()) or sha(path) != row['output']['sha256']:
                raise ValueError('D5_AUDIT_SAVED_ARRAY_BYTES_CHANGED')
            with np.load(path, allow_pickle=False) as artifact:
                raw = artifact['physical_H40']
                for name, key in (('physical_H40', 'physical_sha256'), ('normalized', 'normalized_sha256')):
                    if hashlib.sha256(artifact[name].tobytes()).hexdigest() != row[key]:
                        raise ValueError('D5_AUDIT_ARRAY_DIGEST_CHANGED')
            state, expert, _, _ = dataset.raw(index)
            previous = dataset.reference(index)
            envelope = analyze(raw, state, np.asarray(context['policy_from_world']), _reference(context))
            width = width_metrics(raw, expert)
            if previous != row['previous_reference'] or envelope != row['envelope'] or width != row['width']:
                raise ValueError('D5_AUDIT_ORIGINAL_DIAGNOSTIC_RECOMPUTE_CHANGED')
            seam = label_margin_report(raw, previous)
            speeds = abs(np.diff(np.r_[previous_command_context(previous), .002+.040*raw[:, 9]]))/DT
            bad = np.flatnonzero(speeds > GRIP_LIMIT+1e-10)
            population.append((index, row['noise'], episode['id'], anchor))
            rows.append(dict(index=index, noise=row['noise'], phase=row['phase'], output_sha256=row['output']['sha256'],
                H40_pass=envelope['original_full_H40_pass'], seam_pass=not seam['original_execution_speed_exceeded'],
                first_temporal_exceedance=envelope['first_temporal_exceedance'],
                original_short_circuit_component=envelope['original_short_circuit_component'],
                channel_peaks={key:path['channels'] for key,path in envelope['paths'].items()},
                seam=dict(seam, first_exceedance_segment=int(bad[0]) if len(bad) else None), width=width))
        counts = summarize(rows)
        if any(counts[key] != result['TRAIN96'][original] for key, original in (
                ('predictions', 'predictions'), ('H40_and_seam_pass', 'original_H40_and_gripper_speed_pass'))):
            raise ValueError('D5_AUDIT_SAVED_SUMMARY_COUNTS_CHANGED')
        if common_population is not None and population != common_population:
            raise ValueError('D5_AUDIT_DIAGNOSTIC_POPULATION_NOT_MATCHED')
        common_population = population
        output[arm] = dict(result_sha256=expected_hashes[arm], prediction_ledger_sha256=sha(records),
                           source_sha=result['source_sha'], counts=counts, rows=rows)
    return dict(fixed_bank_sha256=FIXED_BANK, arms=output, model_generations=0,
                training_updates=0, native_requests=0, native_integrations=0, wall_s=time.monotonic()-began)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs=3, type=Path, metavar='ARM_ROOT')
    parser.add_argument('hashes', nargs=3, metavar='RESULT_SHA256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    actual = run(dict(zip(ARMS, args.paths, strict=True)), dict(zip(ARMS, args.hashes, strict=True)))
    with args.output.open('x') as stream:
        json.dump(actual, stream, indent=2, allow_nan=False)
        stream.write('\n')
