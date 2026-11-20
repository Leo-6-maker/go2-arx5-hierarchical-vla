"""Qualify the shipped SCALE20K Go2 package on sixteen sealed Core RGB inputs."""
import argparse
from pathlib import Path
import time

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import RESTORE_ORDINALS
from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha, data_sha
from go2arx5_vla.sla_adapter.scale20k_package import Provider, read_package
from go2arx5_vla.world_model import scale20k_native as scale
from scripts.pi0292.durable import OwnerLock, write


def guard_signature(envelope, seam):
    first = envelope['first_temporal_exceedance']
    reason = tuple(first[k] for k in ('k','path','channel')) if first is not None else None
    return envelope['original_full_H40_pass'],reason,seam['original_execution_speed_exceeded']


def replay_assets(spec, evaluation, panel, training):
    """Use identical archived bytes when a sealed endpoint moved between sites."""
    portable = spec.get('portable_custody')
    if portable is None:
        if evaluation['panel'] != spec['panel']:
            raise ValueError('SCALE20K_REPLAY_PANEL_CHANGED')
        return training['data']['target'], {i: evaluation['rows']['target'][i]['output']
                                           for i in RESTORE_ORDINALS}
    custody = read_pin(portable)
    root = Path(portable['path']).resolve(strict=True).parent
    expected = {'SOURCE_EVAL.private.json':spec['evaluation'], 'PANEL.private.json':spec['panel'],
                'TRAIN_SOURCE_SPEC.private.json':spec['training_spec']}
    members = custody['sealed_members']
    if (custody['status'] != 'TGT5K_CORE_CUSTODY_VERIFIED' or spec['arm'] != 'SCALE-TGT5K'
            or custody['package_sha256'] != spec['package']['sha256']
            or custody['checkpoint_sha256'] != evaluation['checkpoint']['sha256']
            or Path(spec['package']['path']).resolve(strict=True) != root/'package'
            or custody['archive_ordinals'] != sorted(RESTORE_ORDINALS)
            or evaluation['panel']['sha256'] != spec['panel']['sha256']
            or any(members[key] != pin['sha256'] or Path(pin['path']).resolve(strict=True) != root/key
                   for key,pin in expected.items())):
        raise ValueError('SCALE20K_REPLAY_REQUIRES_SEALED_PORTABLE_ENDPOINT_AND_FIXED_PANEL')
    local = read_pin(spec['core_training_spec'])
    if data_sha(local['data']) != data_sha(training['data']) or panel['data'] != data_sha(local['data']):
        raise ValueError('SCALE20K_PORTABLE_CORE_READER_DATA_IDENTITY_CHANGED')
    archived = {}
    for i in RESTORE_ORDINALS:
        name = f'archive/{i:03}.npz'
        if members[name] != evaluation['rows']['target'][i]['output']['sha256']:
            raise ValueError('SCALE20K_PORTABLE_ARCHIVE_MEMBER_CHANGED')
        archived[i] = dict(path=str(root/name),sha256=members[name])
    return local['data']['target'],archived


def run(spec):
    import jax
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.scale20k_target import ScaleTargetDataset
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze

    root = Path(spec['registered_root']).resolve(strict=True)
    result = root/'RESULT.private.json'
    arm, package = spec['arm'], spec['package']
    receipt = dict(schema=scale.RGB_SCHEMA,arm=arm,source_sha=spec['source_sha'],package_sha256=package['sha256'],
                   evaluation_sha256=spec['evaluation']['sha256'],rows=[],generation_attempts=0,
                   new_model_generations=0,new_native_requests=0,new_native_ticks=0,optimizer_updates=0)
    if (result.exists() or (root/'PROGRESS.private.json').exists() or spec['plan'] != scale.PLAN
            or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or spec['native_request_cap'] or spec['updates_cap'] or spec['model_generation_cap'] != len(RESTORE_ORDINALS)):
        raise ValueError('SCALE20K_CORE_RGB16_REQUIRES_A_FRESH_FINITE_NONINTEGRATING_SCOPE')
    began = time.monotonic()
    lock = None
    try:
        lock = OwnerLock(spec['neural_owner_lock'])
        site = validate(spec)
        receipt['site'] = site
        if site['host'] != 'corelab-4080s':
            raise PermissionError('SCALE20K_PACKAGE_REPLAY_MUST_USE_THE_ORIGINAL_CORE_SITE')
        jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
        manifest = read_package(package['path'],package['sha256'])
        evaluation, panel, training = map(read_pin,(spec['evaluation'],spec['panel'],spec['training_spec']))
        target, archives = replay_assets(spec,evaluation,panel,training)
        if ((manifest['arm'],manifest['step']) != scale.ENDPOINTS[arm]
                or (evaluation['arm'],evaluation['step']) != scale.ENDPOINTS[arm]
                or evaluation['status'] != 'ENDPOINT_PREDICTIONS_SEALED' or evaluation['exit_code']
                or spec['evaluation']['sha256'] != manifest['offline_evaluation_sha256']
                or spec['panel']['sha256'] != manifest['panel_sha256']
                or panel['data'] != data_sha(training['data']) or panel['old_TEST_access'] is not False
                or target['manifest']['sha256'] != manifest['target_data_manifest_sha256']
                or target['admission']['sha256'] != manifest['target_admission_sha256']
                or Path(spec['dataset']).resolve(strict=True) != Path(target['root']).resolve(strict=True)
                or spec['parent_manifest']['sha256'] != manifest['parent_manifest']['sha256']):
            raise ValueError('SCALE20K_REPLAY_REQUIRES_EXACT_SEALED_GO2_DATA_AND_ENDPOINT')
        data = ScaleTargetDataset(target,resize_with_pad=resize_with_pad,cache_bytes=128*1024**2)
        model = Provider(package['path'],manifest_sha256=package['sha256'],parent_params=spec['parent_params'],
                         parent_manifest_path=spec['parent_manifest']['path'])
        for ordinal in RESTORE_ORDINALS:
            if time.time() >= spec['deadline_epoch']-30 or (root/'STOP.json').exists() or Path(spec['STOP_path']).exists():
                raise TimeoutError('SCALE20K_CORE_RGB16_DEADLINE_OR_STOP')
            archived = evaluation['rows']['target'][ordinal]
            fixed = panel['rows']['target'][ordinal]
            index, noise = fixed['index'], fixed['noise_id']
            context = panel['target_contexts'][str(index)]
            output = archives[ordinal]
            if (any(archived[k] != v for k,v in fixed.items()) or fixed['identity'] != data.identity(index)
                    or context['previous_command'] != data.reference(index)
                    or sha(output['path']) != output['sha256']):
                raise ValueError('SCALE20K_CORE_RGB16_INPUT_OR_ARCHIVED_OUTPUT_CHANGED')
            state, _, images, prompt = data.raw(index)
            frame = data.codecs[fixed['identity']['episode_id']].T_policy_from_mount
            receipt['generation_attempts'] += 1
            write(root/'PROGRESS.private.json',dict(receipt,pending_ordinal=ordinal))
            modeled = model.predict(state,images,prompt,frame,noise_id=noise,previous_reference=data.reference(index))
            receipt['new_model_generations'] += 1
            action = modeled['physical_H40']
            envelope = analyze(action,state,np.asarray(context['policy_from_world']),np.asarray(context['reference']))
            recorded = archived['metrics']
            matched = (guard_signature(envelope,modeled['G7_gripper_speed']) ==
                       guard_signature(recorded['envelope'],recorded['gripper_seam']))
            actual = dict(ordinal=ordinal,index=index,noise_id=noise,normalized_sha256=bytes_sha(modeled['normalized'][0]),
                          decoded_sha256=bytes_sha(action),same_original_H40_and_seam_decision=matched)
            receipt['rows'].append(actual)
            if not matched or any(actual[k] != archived[k] for k in ('normalized_sha256','decoded_sha256')):
                raise ValueError('SCALE20K_CORE_REPLAY_STRICT_OUTPUT_OR_GUARD_MISMATCH')
        receipt.update(exit_code=0,passed=True)
    except Exception as error:
        receipt.update(exit_code=1,passed=False,first_failure=f'{type(error).__name__}: {error}')
    finally:
        if lock is not None:
            lock.close()
    write(result,dict(receipt,wall_s=time.monotonic()-began))
    return receipt['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
