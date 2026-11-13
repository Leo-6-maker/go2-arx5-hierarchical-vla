"""Finite CUDA seam qualification and separately named two-episode pilot."""
import argparse
import json
from pathlib import Path
import signal
import time

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin

ENDPOINTS = {'DENSE_CONTINUOUS4': 4, 'DENSE_SPLIT2': 2,
             'DENSE_RESUME4': 4, 'DENSE_PICK_HOLD_3K': 3000}
ACTIVE_ENGINE = None
STOP_REQUESTED = False


def admission(spec):
    if spec['operation'] not in ENDPOINTS or spec['closeout_reserve_s'] < 120:
        raise ValueError('FINITE_DENSE_PILOT_OPERATION_AND_CLOSEOUT_REQUIRED')
    legacy = read_pin(spec['upstream_qualification_spec'])
    from scripts.sla.train_device_v3 import candidate_admission
    candidate_admission(legacy)  # Reuse actual closed CUDA/native proofs.
    if any(spec[k] != legacy[k] for k in ('parent_manifest', 'parent_params', 'tokenizer',
                                         'initial_function_rule', 'upstream_sha')):
        raise ValueError('SAME_QUALIFIED_SLA_PARENT_MODEL_AND_NUMERIC_MODE_REQUIRED')
    resumed = spec['operation'] == 'DENSE_RESUME4'
    if resumed != bool(spec.get('resume')):
        raise ValueError('ONLY_NAMED_STEP2_QUALIFICATION_RESUMES')
    if spec['operation'] == 'DENSE_PICK_HOLD_3K':
        proof = read_pin(spec['dense_cuda_qualification'])
        if (proof.get('schema') != 'SLA_DENSE_PILOT_CUDA_RESTORE_RGB_V4_1'
                or not proof.get('passes') or proof['noncandidate_updates'] != 8
                or proof['optimization_visits'] != 256
                or not proof['full_state4_vs2plus2_bit_exact']
                or not proof['fresh_RGB16_bit_exact']
                or proof['manifest_sha256'] != spec['dense_manifest_sha256']
                or proof['source_sha'] != spec['source_sha']):
            raise ValueError('ACTUAL_NEW_DATA_CUDA_RESTORE_AND_FRESH_RGB_REQUIRED')
        for row in proof['closed_phases']:
            result = read_pin(row['result'])
            if result['exit_code'] != 0 or result.get('physical_requests') != 0:
                raise ValueError('ACTUAL_TERMINAL_DENSE_PHASE_PROOFS_REQUIRED')


def run(spec):
    admission(spec)
    from go2arx5_vla.sla_adapter.site_v2 import validate
    site = validate(spec)
    from go2arx5_vla.sla_adapter.runtime_v4 import DensePilotEngine
    from scripts.sla.train_v1 import probes
    import jax, flax, optax
    root = Path(spec['registered_root'])
    (root/'SITE_ACTUAL.private.json').write_text(json.dumps(site, indent=2))
    local = dict(spec, deadline_epoch=spec['deadline_epoch']-spec['closeout_reserve_s'],
                 gradient_capture_steps=[])
    global ACTIVE_ENGINE
    engine = ACTIVE_ENGINE = DensePilotEngine.__new__(DensePilotEngine)
    DensePilotEngine.__init__(engine, local, root/'work')
    engine.bindings['training_site'] = dict(GPU_uuid=site['GPU_uuid'], GPU_name=site['GPU_name'],
                                          jax=jax.__version__, flax=flax.__version__, optax=optax.__version__)
    if spec.get('resume'):
        engine.resume(spec['resume'])
        if (engine.step, engine.cursor) != (2, 64):
            raise ValueError('ORIGINAL_STEP2_COMPLETE_STATE_REQUIRED')
    stop = ENDPOINTS[spec['operation']]
    candidate = spec['operation'] == 'DENSE_PICK_HOLD_3K'
    started = time.time()
    result = dict(schema='SLA_DENSE_PILOT_FINITE_CUDA_ACTUAL_V4_1', operation=spec['operation'],
                  source_sha=spec['source_sha'], noncandidate=not candidate,
                  initial_step=engine.step, requested_updates=stop-engine.step,
                  manifest_sha256=spec['dense_manifest_sha256'], D100_new=False, PICK_PLACE_admitted=False)
    if engine.step == 0:
        result['initial_checkpoint'] = engine.checkpoint('step0000')
        if candidate:
            result['initial_RGB'] = probes(engine, 'step0-RGB')
    try:
        while engine.step < stop:
            if STOP_REQUESTED or time.time() >= local['deadline_epoch']:
                raise TimeoutError('FINITE_DENSE_PILOT_CLOSEOUT')
            row = engine.update()
            print(json.dumps(dict(event='ACTUAL_DENSE_PILOT_UPDATE', step=engine.step, cursor=engine.cursor,
                                  FM32=row['mean_FM32_loss'], used_lr=row['used_lr'], wall_s=row['wall_s'])), flush=True)
            if engine.step in ((500, 1500, 3000) if candidate else (2, 4)):
                result['last_checkpoint'] = engine.checkpoint(f'step{engine.step:04d}')
            if candidate and engine.step in (500, 1500):
                result[f'RGB_step{engine.step}'] = probes(engine, f'step{engine.step}-RGB')
    except TimeoutError:
        result['last_checkpoint'] = engine.checkpoint(f'LIMIT-step{engine.step:04d}')
        result.update(status='LIMIT_SAVED_COMMITTED_STATE_NOT_COMPLETE', exit_code=75)
    else:
        result.update(status='COMPLETE_NAMED_DENSE_PILOT_CUDA_OPERATION', exit_code=0)
        if engine.step in (4, 3000):
            result['endpoint_RGB'] = probes(engine, f'step{engine.step}-RGB')
    result.update(step=engine.step, cursor=engine.cursor, actual=engine.counts,
                  frozen_fingerprint_unchanged=engine.verify_frozen(), residency=engine.residency(),
                  wall_s=time.time()-started, load_wall_s=engine.load_wall_s, site=site,
                  physical_requests=0, native_task_success='NOT_ESTABLISHED_BY_TRAINING_OR_RGB')
    engine.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    root = Path(spec['registered_root'])
    if (root/'RESULT.private.json').exists() or (root/'FAILED.private.json').exists():
        raise FileExistsError('UNKNOWN_OR_FINISHED_DENSE_PILOT_NEVER_REPLAY')
    def stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        if ACTIVE_ENGINE is not None and hasattr(ACTIVE_ENGINE, 'spec'):
            ACTIVE_ENGINE.spec['deadline_epoch'] = time.time()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        result = run(spec)
    except Exception as error:
        result = dict(status='FAILED_NAMED_PILOT_CUDA_OPERATION_PRESERVED', error=repr(error),
                      actual=getattr(ACTIVE_ENGINE, 'counts', {}), step=getattr(ACTIVE_ENGINE, 'step', 0),
                      cursor=getattr(ACTIVE_ENGINE, 'cursor', 0), physical_requests=0)
        with (root/'FAILED.private.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        raise
    with (root/'RESULT.private.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
