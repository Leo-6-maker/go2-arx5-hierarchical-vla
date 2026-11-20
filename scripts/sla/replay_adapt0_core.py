"""Qualify one sealed ADAPT endpoint on the original sixteen Core RGB inputs."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_contract, adapt5k_package
from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_evaluation import RESTORE, bytes_sha
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla import adapt5k_endpoint_transport as transport
from scripts.sla.freeze_adapt5k_panels import bindings
from scripts.sla.replay_scale20k_core import guard_signature


def pin(path):
    return dict(path=str(path),sha256=sha(path))


def trained_inputs(spec, manifest, evaluation):
    """Resolve VANDA output pins against the immutable local receipt."""
    routed = read_pin(spec['core_custody'])
    package = spec['package']
    bundle = Path(spec['core_custody']['path']).resolve(strict=True).parent
    if (routed['schema'] != transport.RECEIPT_SCHEMA or routed['goal'] != transport.GOAL
            or routed['phase'] != manifest['phase'] or routed['phase'] != spec['phase']
            or routed['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or routed['package_sha256'] != package['sha256'] or Path(package['path']).resolve() != bundle/'package'
            or spec['evaluation'] != pin(bundle/'archive/FORMAL.private.json')
            or spec['panel'] != pin(bundle/'archive/PANEL.private.json')
            or spec['parent_manifest']['sha256'] != manifest['parent_manifest']['sha256']
            or evaluation['phase'] != routed['phase'] or evaluation['model_generations'] != 376
            or evaluation['status'] != 'ADAPT_OFFLINE_RESULTS_SEALED' or evaluation['role'] != 'formal'
            or evaluation['old_TEST_access'] or evaluation['exit_code'] or evaluation['new_updates']
            or evaluation['optimization_visits'] or evaluation['native_requests']):
        raise ValueError('ADAPT_TRAINED_REPLAY_REQUIRES_LOCAL_CUSTODY_AND_EXACT_FORMAL_EVAL')
    custody = transport.read_custody((bundle/transport.CUSTODY).read_bytes(),
                                     routed['export_custody_sha256'],routed['phase'])
    if (routed['archived'] != transport.validate_package(bundle,custody) or len(routed['archived']) != len(RESTORE)
            or sha(bundle/'archive/RESTORE_PROOF.private.json') != manifest['restore_sha256']):
        raise ValueError('ADAPT_TRAINED_REPLAY_ARCHIVE_OR_RESTORE_CHANGED')
    for ordinal in RESTORE:
        archived = routed['archived'][str(ordinal)]
        expected = evaluation['rows']['target'][ordinal]['output']['sha256']
        if archived['sha256'] != expected or sha(bundle/archived['path']) != expected:
            raise ValueError(f'ADAPT_TRAINED_REPLAY_ARCHIVED_OUTPUT_CHANGED:{ordinal}')
    return evaluation['checkpoint']['sha256']


def capture_fingerprints(provider, jax, rows):
    """Observe the real device inputs without altering the pinned Provider."""
    infer = provider.infer

    def digest(value):
        array = np.asarray(value)
        return hashlib.sha256(str((array.shape,str(array.dtype))).encode()+array.tobytes()).hexdigest()

    def capture(active, frozen, observation, noise):
        leaves,_ = jax.tree_util.tree_flatten_with_path(observation)
        rows.update(input_sha256={jax.tree_util.keystr(k):digest(v) for k,v in leaves},noise_sha256=digest(noise))
        return infer(active,frozen,observation,noise)

    provider.infer = capture


def run(spec):
    import jax
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    from scripts.sla.scale20k_core_prefix import resources

    root = Path(spec['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    unit = 'wm02-'+root.name+'.service'
    deadline, gpu = spec['deadline_epoch'],spec['GPU_uuid']
    stop = root/'STOP.json'
    pid = subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True,timeout=10).strip()
    if (spec['plan'] != adapt5k_contract.PLAN or spec['source_sha'] != (source/'SOURCE_SHA.private').read_text().strip()
            or spec['generation_cap'] != len(RESTORE) or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or root.stat().st_uid != os.getuid() or os.uname().nodename != 'corelab-4080s'
            or str(os.getpid()) != pid or os.environ['CUDA_VISIBLE_DEVICES'] != gpu
            or (root/'RESULT.private.json').exists() or (root/'PROGRESS.private.json').exists()
            or stop.exists() or not 600 < deadline-time.time() <= 3600):
        raise PermissionError('ADAPT0_CORE_REPLAY_REQUIRES_OWNED_FINITE_NONPHYSICAL_SERVICE')
    phase = spec.get('phase',0)
    result = dict(source_sha=spec['source_sha'],package=spec['package']['sha256'],phase=phase,
                  generation_attempts=0,model_generations=0,optimizer_updates=0,optimization_visits=0,
                  native_requests=0,native_ticks=0,rows=[],exit_code=1)
    compat = spec.get('deployment_compat_v2')
    lock = OwnerLock(spec['neural_owner_lock'])
    lease = root/'control/LEASE.json'
    ownership = dict(owner='EXISTING_LOCAL_PC',deadline_epoch=deadline,main_pid=os.getpid(),unit=unit)
    began = time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:write(stop,dict(reason='OPERATOR_STOP')))
    try:
        capacity = resources(gpu)
        fraction = spec['JAX_memory_fraction']
        reserved = math.ceil(capacity['total_bytes']*fraction)
        headroom = 2*1024**3
        result['resource_check'] = dict(capacity,phase=phase,reserved_bytes=reserved,headroom_bytes=headroom,
            GPU_capacity_pass=capacity['free_bytes']>reserved+headroom,
            host_capacity_pass=capacity['host_bytes']>30*1024**3)
        if not all(result['resource_check'][k] for k in ('GPU_capacity_pass','host_capacity_pass')):
            raise PermissionError('ADAPT0_CORE_REPLAY_ACTUAL_GPU_OR_HOST_HEADROOM_UNAVAILABLE')
        paths = {k:str(Path(spec[k]).resolve(strict=True)) for k in ('parent_params','dataset','compilation_cache_dir')}
        site = root/'SITE.private.json'
        write(site,dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=os.uname().nodename,uid=os.getuid(),
            GPU_uuid=gpu,GPU_name=capacity['name'],allocation_deadline_epoch=deadline,
            allocator_limit_bytes=reserved,GPU_headroom_bytes=headroom,host_headroom_bytes=30*1024**3,paths=paths,
            allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=spec['neural_owner_lock'],unit=unit)))
        lease.parent.mkdir()
        write(lease,dict(ownership,status='ACTIVE'))
        result['site'] = validate(dict(spec,site_binding=pin(site),deadline_epoch=deadline-60,
            max_walltime_s=3600,max_gpu_hours=1,save_and_exit_on_limit=True,model_service_timeout_s=1800,RPC_timeout_s=1800))
        jax.config.update('jax_compilation_cache_dir',paths['compilation_cache_dir'])
        package = spec['package']
        manifest = adapt5k_package.Provider.read_manifest(Path(package['path']),package['sha256'])
        data_spec,evaluation,panel = map(read_pin,(spec['eval_input'],spec['evaluation'],spec['panel']))
        if phase in adapt5k_package.TRAINED_PHASES and manifest['schema'] == adapt5k_package.TRAINED_SCHEMA and compat is None:
            checkpoint = trained_inputs(spec,manifest,evaluation)
            result.update(schema=transport.RGB_SCHEMA,checkpoint_sha256=checkpoint,core_custody=spec['core_custody'],
                evaluation=spec['evaluation'],panel=spec['panel'],hardware_allowed=False,physics=False,
                environment=dict(GPU_uuid=gpu,backend=jax.default_backend(),jax=jax.__version__,
                                 precision='highest',xla=os.environ.get('XLA_FLAGS')))
        elif phase == 0 and manifest['schema'] == adapt5k_package.SCHEMA:
            checkpoint = manifest['ADAPT0_sha256']
        else:
            raise ValueError('ADAPT_CORE_REPLAY_PHASE_OR_GENERATION_CONTRACT_CHANGED')
        target = data_spec['target_D1']
        if (spec['evaluation']['sha256'] != manifest['evaluation_sha256'] or evaluation['checkpoint']['sha256'] != checkpoint
                or evaluation['panel']['sha256'] != spec['panel']['sha256'] or manifest['panel_sha256'] != spec['panel']['sha256']
                or panel['old_TEST_access'] or panel['bindings'] != bindings(data_spec)
                or data_spec['D1_freeze_sha256'] != manifest['D1_freeze_sha256']
                or target['statistics']['sha256'] != manifest['state_statistics_sha256']
                or tuple(panel['restore_ordinals']) != RESTORE):
            raise ValueError('ADAPT0_CORE_REPLAY_REQUIRES_EXACT_FIXED_D1_PANEL_AND_VANDA_OUTPUT_EVIDENCE')
        data = AdaptTargetDataset(target['legacy'],target['captures'],resize_with_pad=resize_with_pad)
        data.bind_state(target['statistics'])
        if compat is not None:
            import jaxlib
            from scripts.sla.adapt0_deploy_compat_v2 import LIMITS, ROLES, SCHEMA

            declaration = read_pin(compat)
            strict = read_pin(declaration['strict_failure'])
            role = spec['compat_role']
            if (declaration['schema'] != SCHEMA or declaration['limits'] != LIMITS
                    or declaration['generation_cap'] != 32 or declaration['spent_generated_attempts']+32 > 336
                    or role not in ROLES or declaration['panel_sha256'] != spec['panel']['sha256']
                    or declaration['checkpoint_sha256'] != manifest['ADAPT0_sha256']
                    or declaration['ordinary_A40_formal_sha256'] != spec['evaluation']['sha256']
                    or strict['status'] != 'CORE_ADAPT0_RGB16_OUTPUT_GATE_FAILED' or strict['model_generations'] != 16):
                raise ValueError('ADAPT0_COMPAT_V2_REQUIRES_EXACT_PREDECLARED_CONTRACT_AND_OLD_FAILURE')
            if role == 'restore':
                prior = read_pin(spec['ordinary'])
                if (prior['status'] != 'CORE_ADAPT0_COMPAT_V2_ARRAYS_SEALED' or prior['compat_role'] != 'ordinary'
                        or prior['model_generations'] != 16 or prior['package'] != result['package']
                        or prior['deployment_compat_v2'] != compat):
                    raise ValueError('ADAPT0_COMPAT_V2_RESTORE_REQUIRES_COLD_SEALED_ORDINARY')
            result.update(deployment_compat_v2=compat,compat_role=role,
                environment=dict(source=spec['source_sha'],platform=jax.default_backend(),jax=jax.__version__,
                    jaxlib=jaxlib.__version__,numpy=np.__version__,GPU_uuid=gpu,xla=os.environ.get('XLA_FLAGS'),
                    dtype_recipe=manifest['partition_sha256']),
                process_identity=dict(pid=os.getpid(),start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
            (root/'outputs').mkdir()
        result['stage'] = 'LOADING_REAL_TARGET_PACKAGE'
        write(root/'PROGRESS.private.json',result)
        parent_override = spec['parent_manifest']['path'] if phase else None
        provider = adapt5k_package.Provider(package['path'],manifest_sha256=package['sha256'],
            parent_params=paths['parent_params'],parent_manifest_path=parent_override)
        fingerprints = {}
        if compat is not None:
            capture_fingerprints(provider,jax,fingerprints)
        for ordinal in RESTORE:
            if stop.exists() or time.time()+60 >= deadline:
                raise TimeoutError('ADAPT0_CORE_REPLAY_STOP_OR_EXPIRED')
            expected = evaluation['rows']['target'][ordinal]
            fixed = panel['formal']['target'][ordinal]
            index = fixed['index']
            ctx = panel['target_contexts'][str(index)]
            previous = data.reference(index)
            if (any(expected[k] != value for k,value in fixed.items()) or data.identity(index) != fixed['identity']
                    or ctx['previous_command'] != previous):
                raise ValueError('ADAPT0_CORE_REPLAY_ACTUAL_RGB_WINDOW_OR_PREVIOUS_COMMAND_CHANGED')
            state,_,images,prompt = data.raw(index)
            frame = data.codecs[fixed['identity']['episode_id']].T_policy_from_mount
            result['stage'] = 'RUNNING_FIXED_RGB16'
            result['generation_attempts'] += 1
            write(root/'PROGRESS.private.json',dict(result,pending_ordinal=ordinal))
            fingerprints.clear()
            out = provider.predict(state,images,prompt,frame,noise_id=fixed['noise_id'],previous_reference=previous)
            result['model_generations'] += 1
            action = out['physical_H40']
            envelope = analyze(action,state,np.asarray(ctx['policy_from_world']),np.asarray(ctx['reference']))
            signatures = [guard_signature(envelope,out['G7_gripper_speed']),
                          guard_signature(expected['metrics']['envelope'],expected['metrics']['gripper_seam'])]
            row = dict(ordinal=ordinal,index=index,normalized_exact=bytes_sha(out['normalized'][0])==expected['normalized_sha256'],
                physical_exact=bytes_sha(action)==expected['decoded_sha256'],guard_exact=signatures[0]==signatures[1])
            if phase:
                row.update(noise_id=fixed['noise_id'],identity=fixed['identity'],
                           normalized_sha256=bytes_sha(out['normalized'][0]),decoded_sha256=bytes_sha(action))
            if compat is not None:
                file = root/'outputs'/f'{ordinal:03d}.npz'
                with file.open('xb') as stream:
                    np.savez(stream,normalized=out['normalized'][0],decoded=action)
                    stream.flush()
                    os.fsync(stream.fileno())
                row.update(identity=fixed['identity'],noise_id=fixed['noise_id'],previous_actual_command=previous,
                           output=pin(file),normalized_sha256=bytes_sha(out['normalized'][0]),decoded_sha256=bytes_sha(action),
                           guard_signature=signatures[0],original_envelope=envelope,gripper_seam=out['G7_gripper_speed'],
                           **fingerprints)
            result['rows'].append(row)
        if compat is None:
            passed = all(r['normalized_exact'] and r['physical_exact'] and r['guard_exact'] for r in result['rows'])
            prefix = 'CORE_ADAPT_TRAINED_RGB16' if phase else 'CORE_ADAPT0_RGB16'
            result.update(status=prefix+('_EXACT' if passed else '_OUTPUT_GATE_FAILED'),
                          exit_code=0 if passed else 1,passed=passed)
        else:
            result.update(status='CORE_ADAPT0_COMPAT_V2_ARRAYS_SEALED',exit_code=0,native_admission=False)
    except Exception as error:
        result.update(status='CORE_ADAPT0_RGB16_TECHNICAL_TERMINAL',first_failure=f'{type(error).__name__}: {error}')
    finally:
        write(root/'RESULT.private.json',dict(result,wall_s=time.monotonic()-began))
        if lease.exists():
            write(lease,dict(ownership,status='CLOSED'))
        lock.close()
    return result['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('export','replay'))
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    card = read_pin(dict(path=args.spec,sha256=args.sha256))
    if args.operation == 'export':
        print(adapt5k_package.export(Path(card['output']),card))
    else:
        raise SystemExit(run(card))
