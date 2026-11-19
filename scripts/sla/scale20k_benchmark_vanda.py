"""Bind the existing scheduler's actual spare A40 to one preregistered engineering candidate."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from go2arx5_vla.sla_adapter.adapt5k_contract import PLAN
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.benchmark_scale20k import ARM, SCOPE
from scripts.sla.scale20k_portable_endpoint import local_custody
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.wm02.multisource_d4_vanda import GIB, actual_gpu, pbs_job, pin

WALL = 5400


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    receipt = root/'RESULT.private.json'
    if (root.stat().st_uid != os.getuid() or receipt.exists() or config['plan'] != PLAN
            or config['scope'] != SCOPE or (source/'SOURCE_SHA.private').read_text().strip() != config['source_sha']
            or any(sha(source/name) != value for name,value in config['source_members'].items())):
        raise PermissionError('MIX_BENCHMARK_REQUIRES_FRESH_PINNED_ENGINEERING_RUN')
    lock = OwnerLock(root/'PBS_OWNER.lock')
    result = dict(scope=SCOPE,engineering_only=True,source_sha=config['source_sha'],
                  actual_optimizer_updates=0,actual_aux_generations=0,native_requests=0,scientific_updates=0)
    try:
        contract = read_pin(config['engineering_contract'])
        variant = contract['variants'][config['variant']]
        assets = read_pin(config['site_assets'])
        match = read_pin(config['CPU_match'])
        if (match['core'] != match['vanda'] or match['R48_manifest_sha256'] != assets['data']['robodojo']['manifest']['sha256']
                or match['LIBERO_manifest_sha256'] != assets['data']['libero']['manifest']['sha256']
                or match['T23_admission_sha256'] != assets['data']['target']['admission']['sha256']
                or len(match['core']['R48']) != 12 or len(match['core']['LIBERO']) != 6
                or len(match['core']['T23']) != 5 or variant['GPU'] != 'NVIDIA A40'):
            raise ValueError('MIX_BENCHMARK_REQUIRES_THE_MATCHED_TRAIN_INPUTS_ON_BOTH_SITES')
        parent_pin = config['parent_custody']
        custody = read_pin(parent_pin)
        if (local_custody(Path(parent_pin['path']).parent,custody['transfer']) != custody
                or custody['arm'] != ARM or custody['step'] != 500
                or custody['checkpoint']['sha256'] != contract['parent_checkpoint_sha256']):
            raise ValueError('MIX_BENCHMARK_REQUIRES_THE_ORIGINAL_COLD_MIX500_STATE')
        result['PBS_JOBID'], window = read_pbs_with_auth_retry(pbs_job,config)
        deadline = window['deadline_epoch']
        if not 1800 < deadline-time.time() <= WALL or window['PBS_wall_s'] > WALL:
            raise PermissionError('MIX_BENCHMARK_REQUIRES_SUFFICIENT_ACTUAL_SINGLE_A40_WINDOW')
        uuid,name,total,free = actual_gpu()
        fraction = .80
        cache = Path(os.environ['TMPDIR']).resolve(strict=True)/'sla-mix-benchmark-compile'
        cache.mkdir()
        paths = {k:str(Path(assets[k]).resolve(strict=True)) for k in ('parent_params','dataset')}
        paths['compilation_cache_dir'] = str(cache)
        observed = root/'PBS_ALLOCATION_OBSERVED.private.json'
        write(observed,dict(window,job_id=result['PBS_JOBID'],uid=os.getuid(),host=os.uname().nodename,
            project=config['project'],state='R',allocated_GPU_uuids=[uuid],GPU_free_bytes=free,PID=os.getpid(),
            observed_epoch=time.time()))
        site = root/'SITE.private.json'
        write(site,dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=os.uname().nodename,uid=os.getuid(),
            GPU_uuid=uuid,GPU_name=name,allocation_deadline_epoch=window['allocation_deadline_epoch'],
            allocator_limit_bytes=math.ceil(total*fraction),GPU_headroom_bytes=2*GIB,host_headroom_bytes=32*GIB,paths=paths,
            allocation=dict(kind='PBS',job_id=result['PBS_JOBID'],project=config['project'],actual_allocation=pin(observed))))
        role = root/'candidate'
        role.mkdir()
        spec = dict(scope=SCOPE,plan=PLAN,variant=config['variant'],microbatch=variant['microbatch'],
                    registered_root=str(role),source_sha=config['source_sha'],site_assets=config['site_assets'],
                    parent_checkpoint=custody['checkpoint'],parent_training_spec=custody['training_spec'],
                    engineering_contract=config['engineering_contract'],site_binding=pin(site),GPU_uuid=uuid,
                    JAX_memory_fraction=fraction,deadline_epoch=deadline-60,max_walltime_s=WALL,max_gpu_hours=WALL/3600,
                    save_and_exit_on_limit=True,model_service_timeout_s=1800,RPC_timeout_s=1800,
                    STOP_path=str(root/'STOP.json'),physics=False,hardware_allowed=False,
                    neural_owner_lock=str(root/'GPU_WORKER.lock'),**paths)
        spec_path = root/'CANDIDATE.private.json'
        write(spec_path,spec)
        result['worker_spec'] = pin(spec_path)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,JAX_PLATFORMS='cuda',XLA_FLAGS=config['XLA_FLAGS'],
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (root/'worker.log').open('x') as log:
            result.update(actual_optimizer_updates=None,actual_aux_generations=None)
            child = subprocess.run([sys.executable,'-B','-m','scripts.sla.benchmark_scale20k',str(spec_path),sha(spec_path)],
                                   cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=deadline-time.time())
        footer = role/'RESULT.private.json'
        if not footer.is_file():
            result.update(status='ENGINEERING_EFFECT_REQUIRES_RECONCILIATION',exit_code=child.returncode or 1)
            return result['exit_code']
        actual = read_pin(pin(footer))
        result.update({k:actual[k] for k in ('status','actual_optimizer_updates','actual_aux_generations')})
        result.update(candidate=pin(footer),exit_code=child.returncode or actual['exit_code'])
        return result['exit_code']
    except Exception as error:
        result.update(status='ENGINEERING_TECHNICAL_TERMINAL',exit_code=1,
                      first_failure=f'{type(error).__name__}: {error}',trace=traceback.format_exc())
        raise
    finally:
        write(receipt,result)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.config,sha256=args.sha256))))
