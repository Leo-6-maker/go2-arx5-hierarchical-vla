"""One frozen SLA parent, two qualified PD4 adapters, finite serial requests."""
import copy
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.storage import digest


def read(path):return json.loads(Path(path).read_bytes())


def providers(spec):
    from go2arx5_vla.sla_adapter.provider_mfree_snapshot_v1 import Provider
    result={}
    if len(spec['policies'])!=1:raise ValueError('ONE_DECLARED_SNAPSHOT_MODEL')
    for name,policy in spec['policies'].items():
        package=policy['package'];result[name]=Provider(package['path'],manifest_sha256=package['sha256'],parent_params=spec['checkpoint'],parent_manifest_path=spec['checkpoint_manifest_path'])
    return result


def run(spec_path):
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    spec=read(spec_path);root=Path(spec['registered_root']);lease=registered(spec,spec_path)
    lock=OwnerLock(spec['neural_owner_lock']);journal=Journal(root/'control/journal',root.name)
    loads=0;attempts=0;done=[];failure=None;code=1;start=time.monotonic()
    try:
        import jax
        from go2arx5_vla.world_model.stable_task_scene_v1 import pinned
        rule=pinned(spec['SLA_runtime']['initial_function_rule'])
        if (os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['GPU_uuid']or os.environ.get('XLA_FLAGS')!=rule['XLA_FLAGS']
                or os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')!=str(spec['SLA_runtime']['JAX_memory_fraction'])):
            raise PermissionError('ORIGINAL_QUALIFIED_SLA_NUMERIC_ENVIRONMENT')
        jax.config.update('jax_compilation_cache_dir',spec['SLA_runtime']['compilation_cache_dir'])
        if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise PermissionError('ACTUAL_SINGLE_GPU')
        write(root/'MODEL_LOAD_INTENT.private.json',dict(parent_loads=1,adapters=len(spec['policies']),optimizer_updates=0))
        policies=providers(spec);loads=1
        write(root/'READY.private.json',dict(pid=os.getpid(),parent_loads=1,adapter_packages=len(policies),
            GPU_uuid=spec['GPU_uuid'],optimizer_updates=0,training_visits=0,reloads_per_episode=0))
        index=0
        while index<spec['max_queries']and time.time()<spec['deadline_epoch']:
            if(root/'STOP.json').exists()or(Path(spec['mobile_group'])/'RESULT.private.json').exists():break
            request=root/'requests'/f'{index:04d}'/'REQUEST.private.json'
            if not request.exists():time.sleep(.2);continue
            q=read(request);folder=request.parent
            if q['sequence']!=index or q['method']not in policies:raise ValueError('RESIDENT_REQUEST_IDENTITY')
            native=Path(q['native_spec']);n=read(native)
            if (digest(native)!=q['native_spec_sha256']or n['resident_server_spec']['sha256']!=digest(spec_path)
                    or n['mobile_condition']['method']!=q['method']or not Path(q['input']).is_relative_to(native.parent)):
                raise PermissionError('REGISTERED_CURRENT_NATIVE_REQUEST_ONLY')
            if digest(q['input'])!=q['input_sha256']:raise PermissionError('EXACT_CURRENT_RGB_STATE_INPUT')
            def generate():
                nonlocal attempts
                attempts+=1;wall=time.monotonic()
                with np.load(q['input'],allow_pickle=False)as data:
                    value=policies[q['method']].predict(data['state_eef10'],[data['body_rgb'],data['wrist_rgb']],
                        q['language'],data['T_policy_from_mount'],noise_id=q['seed']%2**32-2026111200)
                from scripts.pi0294.e1.artifacts import atomic_npz
                action=folder/'ACTION.npz';atomic_npz(action,physical_H40=value['physical_H40'],normalized=value['normalized'])
                return dict(method=q['method'],sequence=index,request_sha256=digest(request),action=str(action),
                    action_sha256=digest(action),generation_wall_s=time.monotonic()-wall,optimizer_updates=0,
                    model_steps=10,physical_integrations=0,task_frame=q['task_frame'])
            result=journal.once('QUERY-'+str(index),dict(request_sha256=digest(request)),generate)
            write(folder/'RESULT.private.json',result);done.append(result);index+=1
            write(root/'PROGRESS.private.json',dict(parent_loads=loads,attempted_generations=attempts,completed=len(done),epoch=time.time()))
        code=0
    except BaseException:failure=traceback.format_exc()
    finally:
        write(root/'RESULT.private.json',dict(exit_code=code,failure=failure,parent_loads=loads,
            generation_invocations=attempts,completed=len(done),records=done,optimizer_updates=0,
            training_visits=0,physical_requests=0,wall_s=time.monotonic()-start))
        write(root/'WORKER_EXIT.private.json',dict(exit_code=code))
        write(root/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
        lock.close()
    return code


if __name__=='__main__':
    import sys
    raise SystemExit(run(sys.argv[sys.argv.index('--server')+1]))
