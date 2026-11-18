"""Fixed V6 G7 training endpoints and separate ordinary-RGB readback."""
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from scripts.pi0292.durable import read, write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from go2arx5_vla.sla_adapter.multitask_initialization_v6 import match_initial_checkpoints
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin

STOP = False


def flag_stop(*_):
    global STOP
    STOP = True


def endpoint(spec):
    from go2arx5_vla.sla_adapter.multitask_training_v6 import MultitaskEngine
    from go2arx5_vla.sla_adapter.provider_multitask_v6 import export
    from scripts.sla.train_close_time_v1 import evaluate
    from scripts.sla.train_pd4_matched_v71 import verify_restored

    root = Path(spec['registered_root'])
    began, cpu = time.monotonic(), time.process_time()
    bank = Path(spec['previous_campaign'])/'diagnostic/WINDOW48_PLAN.private.json'
    if spec['C20_TRAIN48_bank'] != dict(path=str(bank),sha256=sha(bank)) or len(read(bank)['windows']) != 48:
        raise ValueError('PRESERVE_FIXED_C20_TRAIN96_DIAGNOSTIC_POPULATION')
    engine = MultitaskEngine(spec, root/'work')
    try:
        initial = engine.checkpoint('step0000')
        start = read_pin(dict(path=str(Path(initial['path'])/'COMPLETE.json'),sha256=initial['sha256']))
        if len(start['arrays']) != 87:
            raise ValueError('EXACT_MODEL_ADAM_AND_RNG_STEP_ZERO_ARRAY_COUNT_REQUIRED')
        initial_match = None
        if spec['arm'] == 'M-SRC':
            initial_match = match_initial_checkpoints(spec['matched_M_BASE_initial'], initial, engine.transfer_rows)
        write(root/'INITIAL.private.json', dict(checkpoint=initial,match=initial_match,arm=spec['arm'],
            visit_bank=engine.visit_bank_binding,source_initialization=engine.transfer_receipt))
        gradient = engine.qualify_gradient()
        write(root/'GRADIENT_ACTUAL.private.json',gradient)
        while engine.step < 1000:
            if STOP or (root/'STOP.json').exists() or time.time() >= spec['deadline_epoch']-600:
                break
            try:
                row = engine.update()
            except TimeoutError:
                break
            if engine.step == 1 or engine.step % 10 == 0:
                committed = engine.visit_map[:engine.cursor]
                legacy = int((committed < len(engine.dataset.legacy)).sum())
                write(root/'PROGRESS.private.json', dict(epoch=time.time(),phase='REAL_M_G7_TRAINING',arm=spec['arm'],
                    step=engine.step,cursor=engine.cursor,actual=engine.counts,last=row,
                    committed_legacy_visits=legacy,committed_teacher_visits=engine.cursor-legacy,new_native_requests=0))
            if engine.step in (250,500,1000):
                last = engine.checkpoint(f'step{engine.step:04d}')
        counts = engine.counts
        effects = dict(arm=spec['arm'],source_sha=spec['source_sha'],new_updates=engine.step,new_visits=engine.cursor,
                       actual=counts,initial_checkpoint=initial,new_native_requests=0,new_native_ticks=0,
                       wall_s=time.monotonic()-began,CPU_s=time.process_time()-cpu)
        if engine.step != 1000:
            prefix = engine.checkpoint(f'LIMIT-step{engine.step:04d}')
            return dict(exit_code=75,status='SAVED_LIMIT_PREFIX_NO_NEW_OPTIMIZATION',checkpoint=prefix,**effects)
        if engine.cursor != 32000 or counts['training_full_generations'] != 4000:
            raise ValueError('FIXED_1000_32000_AND_4000_ACTUAL_FULL_TRAJECTORIES_REQUIRED')
        engine.resume(last)
        restored = verify_restored(engine,last)
        diagnostic = evaluate(engine,spec)
        records = read(root/'TRAIN48_PREDICTIONS.private.json')
        logits_speed = []
        for row in records:
            with np.load(row['output']['path'],allow_pickle=False) as data:
                speed = label_margin_report(data['physical_H40'], row['previous_reference'])
            logits_speed.append(dict(index=row['index'], noise=row['noise'],phase=row['phase'],
                original_H40_pass=row['envelope']['original_full_H40_pass'], **speed))
        write(root/'C20_G7_SPEED.private.json',logits_speed)
        diagnostic.update(original_H40_and_G7_seam_pass=sum(row['original_H40_pass'] and
            not row['original_execution_speed_exceeded'] for row in logits_speed),
            seam_rejections=sum(row['original_execution_speed_exceeded'] for row in logits_speed))
        package = export(root/'RGB_PACKAGE',last,engine=engine)
        effects.update(wall_s=time.monotonic()-began,CPU_s=time.process_time()-cpu)
        return dict(exit_code=0,status='M_FIXED1000_SEALED_RGB_RESTORE_AND_NATIVE_PENDING',
            last_checkpoint=last,full_restore=restored,package=package,TRAIN96=diagnostic,
            independent_RGB16=dict(path=str(root/'ENDPOINT_RGB16.private.json'),sha256=sha(root/'ENDPOINT_RGB16.private.json')),
            legacy_C20_visits=6400,teacher_visits=25600,visit_bank=engine.visit_bank_binding,
            frozen_fingerprint_unchanged=engine.verify_frozen(),residency=engine.residency(),**effects)
    finally:
        engine.close()


def provider_restore(spec):
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
    from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_MANIFEST
    from go2arx5_vla.sla_adapter.provider_multitask_v6 import Provider

    result = read_pin(spec['trained_result'])
    if result['exit_code'] or result['new_updates'] != 1000 or result['new_visits'] != 32000 or result['arm'] != spec['arm']:
        raise ValueError('ONLY_THE_FIXED_M_ENDPOINT_MAY_BE_RELOADED')
    dataset = MobileDataset(spec['legacy_C20_root'], C20_MANIFEST, resize_with_pad=resize_with_pad,
                            max_cache_bytes=spec['frame_cache_bytes'])
    package = result['package']
    cases = read_pin(result['independent_RGB16'])
    if len(cases) != 16 or len({(row['index'],row['noise']) for row in cases}) != 16:
        raise ValueError('FIXED_INDEPENDENT_RGB16_PANEL_REQUIRED')
    provider = Provider(package['path'],manifest_sha256=package['sha256'],parent_params=spec['parent_params'])
    observed = []
    for row in cases:
        if STOP or time.time() >= spec['deadline_epoch']-60 or (Path(spec['registered_root'])/'STOP.json').exists():
            break
        index = row['index']
        state, _, images, text = dataset.raw(index)
        frame = dataset.frames[dataset.identity(index)['episode_id']]['T_policy_from_mount']
        pin = row['output']
        if sha(pin['path']) != pin['sha256']:
            raise ValueError('SEALED_ENDPOINT_RGB16_BYTES_CHANGED')
        output = provider.predict(state,images,text,frame,noise_id=row['noise'],previous_reference=row['previous_reference'])
        with np.load(pin['path'],allow_pickle=False) as target:
            observed.append(dict(index=index,noise=row['noise'],normalized_exact=bool(np.array_equal(target['normalized'],output['normalized'])),
                EEF10_exact=bool(np.array_equal(target['physical_H40'],output['physical_H40'])),
                maximum_normalized_difference=float(np.max(abs(target['normalized']-output['normalized'])))))
        write(Path(spec['registered_root'])/'PROGRESS.private.json',dict(phase='M_ORDINARY_RGB16_RESTORE',
            generated=len(observed),observed=observed,new_optimizer_updates=0,new_native_requests=0))
    passed = len(observed)==16 and all(row['normalized_exact'] and row['EEF10_exact'] for row in observed)
    return dict(exit_code=0 if passed else 1,status='INDEPENDENT_M_G7_RGB16',passed=passed,actual_generations=len(observed),
        comparisons=observed,package=package,new_optimizer_updates=0,new_native_requests=0)


def main(spec, *, restore=False):
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    from scripts.sla.close_goal_g1_v1 import wait_resource_owner

    root = Path(spec['registered_root'])
    lock = None
    signal.signal(signal.SIGTERM,flag_stop)
    signal.signal(signal.SIGINT,flag_stop)
    try:
        lock = wait_resource_owner(spec)
        site = validate(spec)
        actual = provider_restore(spec) if restore else endpoint(spec)
        actual['site'] = site
    except BaseException:
        actual = dict(exit_code=1,status='PRESERVED_M_G7_TECHNICAL_FAILURE',failure=traceback.format_exc(),
            latest_progress=read(root/'PROGRESS.private.json') if (root/'PROGRESS.private.json').exists() else None,
            committed_update_log=str(root/'work/updates.private.jsonl'),new_native_requests=0)
    finally:
        if lock is not None:
            lock.close()
    write(root/'RESULT.private.json',actual)
    close(spec,actual['exit_code'])
    return actual['exit_code']
