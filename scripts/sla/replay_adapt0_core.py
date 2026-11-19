"""Qualify the sealed ADAPT0 package on exactly sixteen saved Core RGB inputs."""
import argparse
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
from scripts.sla.freeze_adapt5k_panels import bindings
from scripts.sla.replay_scale20k_core import guard_signature


def pin(path):
    return dict(path=str(path),sha256=sha(path))


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
    result = dict(source_sha=spec['source_sha'],package=spec['package']['sha256'],phase=0,
                  generation_attempts=0,model_generations=0,optimizer_updates=0,optimization_visits=0,
                  native_requests=0,native_ticks=0,rows=[],exit_code=1)
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
        if capacity['free_bytes'] <= reserved+headroom or capacity['host_bytes'] <= 30*1024**3:
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
        target = data_spec['target_D1']
        if (spec['evaluation']['sha256'] != manifest['evaluation_sha256'] or evaluation['checkpoint']['sha256'] != manifest['ADAPT0_sha256']
                or evaluation['panel']['sha256'] != spec['panel']['sha256'] or manifest['panel_sha256'] != spec['panel']['sha256']
                or panel['old_TEST_access'] or panel['bindings'] != bindings(data_spec)
                or data_spec['D1_freeze_sha256'] != manifest['D1_freeze_sha256']
                or target['statistics']['sha256'] != manifest['state_statistics_sha256']
                or tuple(panel['restore_ordinals']) != RESTORE):
            raise ValueError('ADAPT0_CORE_REPLAY_REQUIRES_EXACT_FIXED_D1_PANEL_AND_VANDA_OUTPUT_EVIDENCE')
        data = AdaptTargetDataset(target['legacy'],target['captures'],resize_with_pad=resize_with_pad)
        data.bind_state(target['statistics'])
        result['stage'] = 'LOADING_REAL_TARGET_PACKAGE'
        write(root/'PROGRESS.private.json',result)
        provider = adapt5k_package.Provider(package['path'],manifest_sha256=package['sha256'],parent_params=paths['parent_params'])
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
            out = provider.predict(state,images,prompt,frame,noise_id=fixed['noise_id'],previous_reference=previous)
            result['model_generations'] += 1
            action = out['physical_H40']
            envelope = analyze(action,state,np.asarray(ctx['policy_from_world']),np.asarray(ctx['reference']))
            signatures = [guard_signature(envelope,out['G7_gripper_speed']),
                          guard_signature(expected['metrics']['envelope'],expected['metrics']['gripper_seam'])]
            row = dict(ordinal=ordinal,index=index,normalized_exact=bytes_sha(out['normalized'][0])==expected['normalized_sha256'],
                physical_exact=bytes_sha(action)==expected['decoded_sha256'],guard_exact=signatures[0]==signatures[1])
            result['rows'].append(row)
        passed = all(r['normalized_exact'] and r['physical_exact'] and r['guard_exact'] for r in result['rows'])
        result.update(status='CORE_ADAPT0_RGB16_EXACT' if passed else 'CORE_ADAPT0_RGB16_OUTPUT_GATE_FAILED',
                      exit_code=0 if passed else 1,passed=passed)
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
