"""Fixed J0/J1 endpoint, source DEV and independent RGB restoration."""
from pathlib import Path
import shutil
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


def recover_J0(spec):
    """Finish the original J0 endpoint from step 500 and its existing target predictions."""
    from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
    from scripts.sla.train_close_time_v1 import summarize_target_panel
    from scripts.sla.train_pd4_matched_v71 import verify_restored

    carry = spec['recover_J0']
    failed = read_pin(carry['failed_result'])
    original = Path(carry['failed_result']['path']).parent
    root = Path(spec['registered_root'])
    if (spec['arm'] != 'J0' or root == original or failed['exit_code'] != 1
            or failed['failure'].splitlines()[-1] != "ModuleNotFoundError: No module named 'h5py'"
            or (failed['new_updates'], failed['new_visits'], failed['new_source_visits']) != (500,16000,0)
            or failed['new_integrations'] or failed['new_native_requests']
            or failed['actual']['training_full_generations'] != 2000
            or failed['source_actual']['ordinary_RGB_generation_attempts'] != 0):
        raise ValueError('EXACT_COLD_J0_SOURCE_DECODER_FAILURE_REQUIRED')
    endpoint = carry['checkpoint']
    if Path(endpoint['path']) != original/'work/step0500':
        raise ValueError('RECOVERY_ONLY_FROM_ORIGINAL_J0_STEP500')
    initial_record = read_pin(carry['initial'])
    records = read_pin(carry['TRAIN96'])
    old_rgb = read_pin(carry['RGB16'])
    bank = read_pin(spec['fixed_TRAIN48'])['windows']
    identities = [(r['index'],r['phase'],noise) for r in bank for noise in (0,1)]
    if (len(identities) != 96 or identities != [(r['index'],r['phase'],r['noise']) for r in records]
            or old_rgb != [records[2*j+n] for j in (0,4,8,12,16,20,28,36) for n in (0,1)]):
        raise ValueError('J0_RECOVERY_REQUIRES_ALL_ORIGINAL_TARGET_PREDICTIONS')
    rejected, full = 0, 0
    for row in records:
        name = f"{row['index']:04d}-{row['noise']}.npz"
        previous = original/'data'/name
        saved = root/'data'/name
        if Path(row['output']['path']) != previous or sha(previous) != row['output']['sha256']:
            raise ValueError('J0_RECOVERY_ARCHIVED_TARGET_PREDICTION_CHANGED')
        shutil.copyfile(previous,saved)
        if sha(saved) != row['output']['sha256']:
            raise ValueError('J0_TARGET_OUTPUT_COPY_MISMATCH')
        row['output']['path'] = str(saved)
        with np.load(saved,allow_pickle=False) as payload:
            bad = label_margin_report(payload['physical_H40'],row['previous_reference'])['original_execution_speed_exceeded']
        rejected += bad
        full += not bad and row['envelope']['original_full_H40_pass']
    train96 = summarize_target_panel(records)
    train96.update(original_H40_and_gripper_speed_pass=full,gripper_speed_reject=rejected)
    write(root/'TRAIN48_PREDICTIONS.private.json',records)
    path = root/'ENDPOINT_RGB16.private.json'
    write(path,[records[2*j+n] for j in (0,4,8,12,16,20,28,36) for n in (0,1)])
    write(root/'INITIAL.private.json',initial_record)
    engine = joint_source_d4.Engine(spec, root/'recovery-work')
    try:
        engine.resume(endpoint)
        restored = verify_restored(engine,endpoint,expected_arrays=87)
        diagnostic = diagnose_source(engine, root)
        result = dict(exit_code=0,status='J0_ORIGINAL_FIXED500_ENDPOINT_EVALUATION_COMPLETED',arm='J0',
            source_sha=spec['source_sha'],initial_checkpoint=initial_record['checkpoint'],
            last_checkpoint=endpoint,full_restore=restored,TRAIN96=train96,
            RGB16_list=dict(path=str(path),sha256=sha(path)),
            package=joint_source_d4.export(root/'RGB_PACKAGE',endpoint,engine=engine),
            frozen_fingerprint=engine.verify_frozen(),new_updates=500,new_visits=16000,new_source_visits=0,
            actual=failed['actual'],recovery=dict(optimizer_updates=0,optimization_visits=0,
                carried_target_generations=96,new_target_generations=0,
                new_source_generations=engine.source_counts['ordinary_RGB_generations'],
                original_failure=carry['failed_result']),**diagnostic)
        if engine.step != 500 or engine.cursor != 16000 or engine.counts['optimizer_updates']:
            raise ValueError('J0_RECOVERY_MUTATED_FIXED_TRAINING_STATE')
        return result
    finally:
        engine.close()


def run(spec, *, mode):
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from scripts.sla.train_grasp_supervision_d5 import historical_operator_environment
    from scripts.wm02.sla_followup_population_v1 import validate_frozen

    root = Path(spec['registered_root'])
    output = {'train':'RESULT', 'recover_J0':'RESULT', 'restore':'RESTORE', 'restore_source':'SOURCE_RESTORE'}[mode]
    terminal = root/(output+'.private.json')
    if terminal.exists():
        raise FileExistsError('PRESERVE_EXISTING_J_TERMINAL')
    try:
        # HDF decoding is required by the source gradient and evaluation.
        import h5py

        site = validate(spec)
        operator_proof = historical_operator_environment(spec, site)
        validate_frozen(read_pin(spec['sealed_followup_population']))
        if mode == 'train':
            actual = train_multisource_d4.train(spec, engine_type=joint_source_d4.Engine,
                export_package=joint_source_d4.export, endpoint_diagnostics=diagnose_source)
        elif mode == 'recover_J0':
            actual = recover_J0(spec)
        elif mode == 'restore':
            actual = train_multisource_d4.restore_rgb(spec, provider_type=joint_source_d4.Provider)
        else:
            actual = restore_source(spec)
        actual.update(site=site, operator_environment=operator_proof,source_HDF5_decoder_version=h5py.__version__)
    except Exception:
        actual = dict(exit_code=1, status='J_TECHNICAL_TERMINAL_NO_AUTO_REPLAY', failure=traceback.format_exc())
    write(terminal, dict(actual, new_native_requests=0, new_integrations=0))
    return actual['exit_code']
