"""Replay the two archived endpoint RGB16 banks on the Core deployment provider."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import traceback

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.multisource_d4 import ARMS, C20G7Dataset, Provider
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write


def registered_providers(spec):
    mode = spec.get('stage', 'D4')
    if mode == 'D4':
        arms, provider = ARMS, Provider
    elif mode == 'D5':
        from go2arx5_vla.sla_adapter.grasp_supervision_d5 import ARMS as arms, Provider as provider
    elif mode == 'J':
        from go2arx5_vla.sla_adapter.joint_source_d4 import ARMS as arms, Provider as provider
    else:
        raise ValueError('UNKNOWN_CORE_MATCHED_RGB_PANEL')
    if set(arms) != set(spec['arms']):
        raise ValueError('EXACT_TWO_MATCHED_CORE_RGB_ENDPOINTS_REQUIRED')
    return mode, arms, provider


def original_decision(state, old, new, context, previous):
    from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    from scripts.sla.export_mfree_c20_v1 import _reference

    decisions = []
    for h40 in (old, new):
        envelope = analyze(h40, state, np.asarray(context['policy_from_world']), _reference(context))
        first = envelope['first_temporal_exceedance']
        reason = tuple(first[k] for k in ('k','path','channel')) if first is not None else None
        speed = label_margin_report(h40, previous)['original_execution_speed_exceeded']
        decisions.append((envelope['original_full_H40_pass'], reason, speed))
    return decisions[0] == decisions[1]


def replay(spec, progress):
    import jax
    from openpi_client.image_tools import resize_with_pad

    if (os.getuid() != spec['uid'] or os.uname().nodename != spec['host']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != spec['GPU_uuid']
            or os.environ.get('XLA_FLAGS') != spec['XLA_FLAGS']):
        raise PermissionError('ORIGINAL_CORE_DEPLOYMENT_RUNTIME_REQUIRED')
    dataset = C20G7Dataset(spec['dataset'], time_audit=spec['time_codec_audit'],
                           resize_with_pad=resize_with_pad, max_cache_bytes=128*1024**2)
    root = Path(spec['root'])
    mode, arms, provider_class = registered_providers(spec)
    contexts = read_pin(spec['portable_evaluation']) if mode != 'D4' else None
    references = Path(spec.get('reference_root',str(root/'reference')))
    if contexts is not None and contexts['C20_manifest_sha256'] != dataset.manifest_sha:
        raise ValueError('EXACT_FROZEN_C20_RGB_EVALUATION_CONTEXT_REQUIRED')
    for arm in arms:
        endpoint = spec['arms'][arm]
        cases = read_pin(endpoint['cases'])
        if len(cases) != 16 or len({(row['index'], row['noise']) for row in cases}) != 16:
            raise ValueError('FIXED_UNIQUE_ENDPOINT_RGB16_REQUIRED')
        provider = provider_class(endpoint['package']['path'], manifest_sha256=endpoint['package']['sha256'],
                            parent_params=spec['parent_params'], parent_manifest_path=spec['parent_manifest_path'])
        for restored in cases:
            if time.time() >= spec['deadline_epoch']-30 or (root/'STOP.json').exists():
                raise TimeoutError('FINITE_CORE_RGB_REPLAY_BEFORE_NEXT_GENERATION')
            index, noise = restored['index'], restored['noise']
            state, _, images, prompt = dataset.raw(index)
            episode, anchor = dataset.rows[index]
            if ((episode['id'], anchor, dataset.reference(index))
                    != (restored['episode'], restored['anchor'], restored['previous_reference'])):
                raise ValueError('ARCHIVED_CURRENT_RGB_AND_ACTUAL_PREVIOUS_COMMAND_REQUIRED')
            frame = dataset.frames[episode['id']]['T_policy_from_mount']
            archive = references/arm/f'{index:04d}-{noise}.npz'
            if sha(archive) != restored['output']['sha256']:
                raise ValueError('ARCHIVED_RGB_PREDICTION_CHANGED')
            progress['attempts'] += 1
            write(root/'PROGRESS.private.json', progress)
            result = provider.predict(state, images, prompt, frame, noise_id=noise,
                                      previous_reference=restored['previous_reference'])
            with np.load(archive, allow_pickle=False) as old:
                normalized = np.asarray(result['normalized'])
                physical = np.asarray(result['physical_H40'])
                if ((normalized.shape, physical.shape) != (old['normalized'].shape, old['physical_H40'].shape)
                        or not np.isfinite(normalized).all() or not np.isfinite(physical).all()):
                    raise ValueError('FINITE_COMPLETE_RGB_H40_REQUIRED')
                merged = dict(arm=arm, index=index, noise=noise, episode=episode['id'], anchor=anchor,
                              normalized_exact=bool(np.array_equal(old['normalized'], normalized)),
                              physical_exact=bool(np.array_equal(old['physical_H40'], physical)),
                              max_normalized_abs=float(np.max(np.abs(old['normalized']-normalized))),
                              max_physical_abs=float(np.max(np.abs(old['physical_H40']-physical))),
                              normalized_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
                              physical_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
                              grip_speed=result['G7_gripper_speed'])
                if contexts is not None:
                    context = contexts['windows'][str(index)]
                    if (context['episode'], context['anchor']) != (episode['id'], anchor):
                        raise ValueError('EVALUATION_COMPILER_CONTEXT_DOES_NOT_MATCH_ORIGINAL_RGB')
                    merged['same_original_compiler_and_seam_decision'] = original_decision(
                        state, old['physical_H40'], physical, context, restored['previous_reference'])
            with (root/'predictions'/f'{arm}-{index:04d}-{noise}.npz').open('xb') as handle:
                np.savez(handle, normalized=normalized, physical_H40=physical)
            progress['rows'].append(merged)
            progress['completed'] += 1
            progress['GPU_memory'] = jax.devices()[0].memory_stats()
            progress['epoch'] = time.time()
            write(root/'PROGRESS.private.json', progress)
        if contexts is not None:
            rows = [row for row in progress['rows'] if row['arm'] == arm]
            write(root/f'{arm}-CORE_RGB16.private.json', dict(arm=arm, source_sha=spec['source_sha'],
                package_sha256=endpoint['package']['sha256'], actual_generations=len(rows),
                original_H40_seam_decisions_match=all(row['same_original_compiler_and_seam_decision'] for row in rows),
                normalized_bit_exact=sum(row['normalized_exact'] for row in rows),
                physical_bit_exact=sum(row['physical_exact'] for row in rows),
                max_physical_abs=max(row['max_physical_abs'] for row in rows),native_requests=0,native_integrations=0,
                rows=rows))
        del provider
        jax.clear_caches()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    spec = read_pin(dict(path=args.spec, sha256=args.sha256))
    root = Path(spec['root'])
    progress = dict(source_sha=spec['source_sha'], spec_sha=args.sha256,
                    planned=32, attempts=0, completed=0, rows=[], updates=0, visits=0,
                    native_requests=0, native_integrations=0, epoch=time.time())
    start = time.monotonic()
    lock = None
    try:
        lock = OwnerLock(spec['gpu_owner_lock'])
        replay(spec, progress)
        exact = sum(row['normalized_exact'] and row['physical_exact'] for row in progress['rows'])
        normalized_given = sum(row['normalized_exact'] for row in progress['rows'])
        same_guards = sum(row.get('same_original_compiler_and_seam_decision', False) for row in progress['rows'])
        passed = exact == 32 if spec.get('stage', 'D4') == 'D4' else normalized_given == same_guards == 32
        result = dict(progress, both_bit_exact=exact, normalized_bit_exact=normalized_given,
                      original_H40_and_seam_decisions_identical=same_guards, passed=passed,
                      status='CORE_RGB32_MEASURED', exit_code=int(not passed))
    except Exception:
        result = dict(progress, passed=False, status='CORE_RGB32_STOPPED',
                      exit_code=1, first_failure=traceback.format_exc())
    finally:
        if lock is not None:
            lock.close()
    write(root/'RESULT.private.json', dict(result, wall_s=time.monotonic()-start))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
