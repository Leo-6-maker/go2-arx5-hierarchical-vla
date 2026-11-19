"""Fixed J0/J1 endpoint, source DEV and independent RGB restoration."""
from pathlib import Path
import time
import traceback

import numpy as np

from go2arx5_vla.sla_adapter import joint_source_d4
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import read, write
from scripts.sla import train_multisource_d4


def diagnose_source(engine, root):
    from go2arx5_vla.sla_adapter.source_evaluation_v6 import evaluate

    summary = evaluate(engine.source, root, 'SOURCE_DEV', endpoint=engine.spec['arm'],
                       stop=lambda: train_multisource_d4.stopped(engine.spec))
    evaluation = root/(engine.spec['arm']+'-SOURCE_DEV96.private.json')
    receipt = read(evaluation)
    if (len(receipt['rows']) != 96 or engine.source_counts['ordinary_RGB_generations'] != 96
            or engine.source.fingerprint_private() != engine.source.private_fingerprint):
        raise ValueError('FIXED_SOURCE_DEV96_AND_FROZEN_SOURCE_PRIVATE_BYTES_REQUIRED')
    return dict(SOURCE_DEV96=summary, source_actual=engine.source_counts, source_private_bit_exact=True,
                source_evaluation=dict(path=str(evaluation), sha256=sha(evaluation)))


def restore_source(spec):
    from scripts.sla.train_pd4_matched_v71 import verify_restored

    root = Path(spec['registered_root'])
    result = read(root/'RESULT.private.json')
    if result['exit_code'] or (result['new_updates'], result['new_visits']) != (500, 16000):
        raise ValueError('J_FIXED500_BEFORE_INDEPENDENT_SOURCE_RESTORE')
    rows = read_pin(result['source_evaluation'])['rows']
    bank = rows[:16]
    if len(rows) != 96 or len({(row['index'], row['noise_id']) for row in bank}) != 16:
        raise ValueError('EXACT_SIXTEEN_SEALED_SOURCE_DEV_INPUTS_REQUIRED')
    engine = joint_source_d4.Engine(spec, root/'source-restore-work')
    observations = []
    try:
        engine.resume(result['last_checkpoint'])
        restored = verify_restored(engine, result['last_checkpoint'], expected_arrays=None)
        for row in bank:
            if train_multisource_d4.stopped(spec):
                break
            reference = row['array']
            if sha(reference['path']) != reference['sha256'] or engine.source.dev.identity(row['index']) != row['input_identity']:
                raise ValueError('J_SOURCE_RESTORE_REFERENCE_OR_INPUT_CHANGED')
            normalized, reference_units = engine.source.generate(row['index'], row['noise_id'],
                                                                  dataset=engine.source.dev)
            with np.load(reference['path'], allow_pickle=False) as saved:
                observations.append(dict(index=row['index'], noise_id=row['noise_id'],
                    normalized_exact=bool(np.array_equal(normalized, saved['normalized'])),
                    reference_exact=bool(np.array_equal(reference_units, saved['reference']))))
            write(root/'SOURCE_RESTORE_PROGRESS.private.json', dict(epoch=time.time(), rows=observations,
                 actual_generations=engine.source_counts['ordinary_RGB_generations']))
        passed = (len(observations) == 16
                  and all(r['normalized_exact'] and r['reference_exact'] for r in observations)
                  and engine.source.fingerprint_private() == engine.source.private_fingerprint)
        return dict(exit_code=int(not passed), passed=passed, full_restore=restored, observations=observations,
                    actual_generations=engine.source_counts['ordinary_RGB_generations'])
    finally:
        engine.close()


def run(spec, *, mode):
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from scripts.sla.train_grasp_supervision_d5 import historical_operator_environment
    from scripts.wm02.sla_followup_population_v1 import validate_frozen

    root = Path(spec['registered_root'])
    output = {'train':'RESULT', 'restore':'RESTORE', 'restore_source':'SOURCE_RESTORE'}[mode]
    terminal = root/(output+'.private.json')
    if terminal.exists():
        raise FileExistsError('PRESERVE_EXISTING_J_TERMINAL')
    try:
        site = validate(spec)
        operator_proof = historical_operator_environment(spec, site)
        validate_frozen(read_pin(spec['sealed_followup_population']))
        if mode == 'train':
            actual = train_multisource_d4.train(spec, engine_type=joint_source_d4.Engine,
                export_package=joint_source_d4.export, endpoint_diagnostics=diagnose_source)
        elif mode == 'restore':
            actual = train_multisource_d4.restore_rgb(spec, provider_type=joint_source_d4.Provider)
        else:
            actual = restore_source(spec)
        actual.update(site=site, operator_environment=operator_proof)
    except Exception:
        actual = dict(exit_code=1, status='J_TECHNICAL_TERMINAL_NO_AUTO_REPLAY', failure=traceback.format_exc())
    write(terminal, dict(actual, new_native_requests=0, new_integrations=0))
    return actual['exit_code']
