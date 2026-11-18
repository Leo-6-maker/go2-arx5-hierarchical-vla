"""Finite D4 matched A/B and independent ordinary-RGB readbacks."""
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np

from scripts.pi0292.durable import read, write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from go2arx5_vla.sla_adapter.multisource_d4 import ARMS, UPDATES, VISITS
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin


def stopped(spec):
    root = Path(spec['registered_root'])
    return (time.time() >= spec['deadline_epoch']-300
            or (root/'STOP.json').exists()
            or (Path(spec['campaign_root'])/'STOP.json').exists())


def train(spec):
    from go2arx5_vla.sla_adapter.multisource_d4 import TransferEngine, export
    from go2arx5_vla.sla_adapter.multitask_initialization_v6 import match_initial_checkpoints
    from scripts.sla.train_close_time_v1 import evaluate
    from scripts.sla.train_pd4_matched_v71 import verify_restored

    root = Path(spec['registered_root'])
    fixed_bank = Path(spec['previous_campaign'])/'diagnostic/WINDOW48_PLAN.private.json'
    if str(fixed_bank) != spec['fixed_TRAIN48']['path'] or len(read_pin(spec['fixed_TRAIN48'])['windows']) != 48:
        raise ValueError('D4_EXACT_ORIGINAL_TRAIN48_DIAGNOSTIC_REQUIRED')
    begin, cpu = time.monotonic(), time.process_time()
    engine = TransferEngine(spec, root/'work')
    try:
        initial = engine.checkpoint('step0000')
        start = read_pin(dict(path=str(Path(initial['path'])/'COMPLETE.json'), sha256=initial['sha256']))
        if len(start['arrays']) != 87:
            raise ValueError('D4_FIXED_STEP_ZERO_MODEL_ADAM_RNG_COUNT')
        noise, tau = engine.draw_bank(engine.rng, np.arange(32,dtype=np.uint32))
        initial_record = dict(checkpoint=initial, match=None,
            initial_noise_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest(),
            initial_flow_time_sha256=hashlib.sha256(np.asarray(tau).tobytes()).hexdigest(),
            first32=engine.visit_map[:32].tolist(),
            visit_map_sha256=engine.bindings['visit_map_sha256'])
        if spec['arm'] == ARMS[1]:
            control = Path(spec['campaign_root'])/ARMS[0]
            if not (control/'RESTORE.private.json').is_file() or not read(control/'RESTORE.private.json')['passed']:
                raise ValueError('D4_COMPLETE_FIXED_CONTROL_RESTORE_REQUIRED')
            left = read(control/'INITIAL.private.json')
            compare_keys = ('initial_noise_sha256','initial_flow_time_sha256','first32','visit_map_sha256')
            if any(left[key] != initial_record[key] for key in compare_keys):
                raise ValueError('D4_MATCHED_ABSOLUTE_CURSOR_RANDOM_BANK_REQUIRED')
            initial_record['match'] = match_initial_checkpoints(left['checkpoint'], initial,
                engine.transfer_rows, arms=ARMS)
        write(root/'INITIAL.private.json', initial_record)

        if not stopped(spec):
            write(root/'GRADIENT_ACTUAL.private.json', engine.qualify_gradient())
        while engine.step < UPDATES and not stopped(spec):
            try:
                row = engine.update()
            except TimeoutError:
                break
            if engine.step == 1 or engine.step % 10 == 0:
                write(root/'PROGRESS.private.json', dict(phase='REAL_D4_C20_G7_TRAINING', epoch=time.time(),
                    arm=spec['arm'], updates=engine.step, target_visits=engine.cursor, source_visits=0,
                    actual=engine.counts, last=row))
            if engine.step in (250, UPDATES):
                last = engine.checkpoint(f'step{engine.step:04d}')

        effects = dict(arm=spec['arm'], source_sha=spec['source_sha'], initial_checkpoint=initial,
            new_updates=engine.step, new_visits=engine.cursor, actual=engine.counts,
            new_source_visits=0, new_native_requests=0, new_integrations=0)
        if engine.step != UPDATES:
            saved = engine.checkpoint(f'LIMIT-step{engine.step:04d}')
            return dict(exit_code=75, status='BOUNDED_PREFIX_SAVED_NO_REPLAY', checkpoint=saved,
                        wall_s=time.monotonic()-begin, CPU_s=time.process_time()-cpu, **effects)
        if engine.cursor != VISITS or engine.counts['training_full_generations'] != UPDATES*4:
            raise ValueError('D4_REAL_FIXED500_VISITS_AND_FULL_H40_AUXILIARIES_REQUIRED')
        engine.resume(last)
        restored = verify_restored(engine, last)
        if stopped(spec):
            return dict(exit_code=75, status='FIXED_ENDPOINT_DIAGNOSTIC_UNSTARTED',
                        last_checkpoint=last, full_restore=restored, **effects)
        metrics = evaluate(engine, spec)
        seam, full = 0, 0
        for row in read(root/'TRAIN48_PREDICTIONS.private.json'):
            path = row['output']['path']
            if sha(path) != row['output']['sha256']:
                raise ValueError('D4_DIAGNOSTIC_OUTPUT_HASH_CHANGED')
            with np.load(path, allow_pickle=False) as data:
                rejected = label_margin_report(data['physical_H40'], row['previous_reference'])['original_execution_speed_exceeded']
            seam += rejected
            full += not rejected and row['envelope']['original_full_H40_pass']
        metrics.update(original_H40_and_gripper_speed_pass=full, gripper_speed_reject=seam)
        return dict(exit_code=0, status='D4_FIXED500_SEALED_NATIVE_UNSTARTED', last_checkpoint=last,
            full_restore=restored, package=export(root/'RGB_PACKAGE', last, engine=engine), TRAIN96=metrics,
            RGB16_list=dict(path=str(root/'ENDPOINT_RGB16.private.json'), sha256=sha(root/'ENDPOINT_RGB16.private.json')),
            frozen_fingerprint=engine.verify_frozen(), wall_s=time.monotonic()-begin,
            CPU_s=time.process_time()-cpu, **effects)
    except BaseException:
        return dict(exit_code=1, status='D4_TERMINAL_AFTER_ENGINE_INITIALIZATION',
            failure=traceback.format_exc(), arm=spec['arm'], new_updates=engine.step,
            new_visits=engine.cursor, new_source_visits=0, new_native_requests=0,
            new_integrations=0, actual=engine.counts, wall_s=time.monotonic()-begin,
            CPU_s=time.process_time()-cpu)
    finally:
        engine.close()


def restore_rgb(spec):
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.multisource_d4 import C20G7Dataset, Provider

    root = Path(spec['registered_root'])
    result = read(root/'RESULT.private.json')
    if result['exit_code'] or (result['new_updates'], result['new_visits']) != (UPDATES, VISITS):
        raise ValueError('D4_FIXED500_TRAINED_RESULT_REQUIRED')
    cases = read_pin(result['RGB16_list'])
    if len(cases) != 16 or len({(case['index'],case['noise']) for case in cases}) != 16:
        raise ValueError('D4_FIXED_UNIQUE_RGB16_REQUIRED')
    package = result['package']
    dataset = C20G7Dataset(spec['dataset'], time_audit=spec['time_codec_audit'],
        resize_with_pad=resize_with_pad, max_cache_bytes=spec['frame_cache_bytes'])
    provider = Provider(package['path'], manifest_sha256=package['sha256'], parent_params=spec['parent_params'])
    observations = []
    for case in cases:
        if stopped(spec):
            break
        index = case['index']
        state, _, images, prompt = dataset.raw(index)
        frame = dataset.frames[dataset.identity(index)['episode_id']]['T_policy_from_mount']
        if case['previous_reference'] != dataset.reference(index):
            raise ValueError('D4_EXACT_PREVIOUS_ACTUAL_C20_COMMAND_REQUIRED_FOR_RESTORE')
        path = case['output']['path']
        if sha(path) != case['output']['sha256']:
            raise ValueError('D4_RGB16_REFERENCE_OUTPUT_CHANGED')
        output = provider.predict(state, images, prompt, frame, noise_id=case['noise'],
            previous_reference=case['previous_reference'])
        with np.load(path, allow_pickle=False) as original:
            observations.append(dict(index=index, noise=case['noise'],
                normalized_exact=bool(np.array_equal(original['normalized'], output['normalized'])),
                EEF10_exact=bool(np.array_equal(original['physical_H40'], output['physical_H40']))))
        write(root/'RESTORE_PROGRESS.private.json', dict(epoch=time.time(), actual_generations=len(observations),
            rows=observations))
    passed = len(observations) == 16 and all(row['normalized_exact'] and row['EEF10_exact'] for row in observations)
    return dict(exit_code=int(not passed), passed=passed, actual_generations=len(observations),
                observations=observations, package=package, new_native_requests=0)


def run(spec, *, restore):
    from go2arx5_vla.sla_adapter.site_v2 import validate

    root = Path(spec['registered_root'])
    output = root/('RESTORE.private.json' if restore else 'RESULT.private.json')
    try:
        site = validate(spec)
        actual = restore_rgb(spec) if restore else train(spec)
        actual['site'] = site
    except BaseException:
        actual = dict(exit_code=1, status='D4_TECHNICAL_TERMINAL_NO_AUTO_REPLAY',
            failure=traceback.format_exc(), new_native_requests=0,
            training_log=str(root/'work/updates.private.jsonl'))
    write(output, actual)
    return actual['exit_code']
