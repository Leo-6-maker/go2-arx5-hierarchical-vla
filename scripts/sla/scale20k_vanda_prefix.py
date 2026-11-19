"""Run the first fixed SCALE target segment inside its own VANDA PBS allocation."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, PLAN, RECIPE
from go2arx5_vla.sla_adapter.scale20k_model import parent_asset
from scripts.pi0292.durable import OwnerLock, write
from scripts.wm02.multisource_d4_vanda import GIB, actual_gpu, pbs_job, pin


ARM = 'SCALE-TGT5K'
WALL = 10800


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    if (root.stat().st_uid != os.getuid() or (root/'RESULT.private.json').exists()
            or (source/'SOURCE_SHA.private').read_text().strip() != config['source_sha']
            or config['plan_sha'] != PLAN or config['arm'] != ARM):
        raise PermissionError('SCALE_TARGET_FRESH_PBS_EXECUTION_IDENTITY_REQUIRED')
    lock = OwnerLock(root/'PBS_OWNER.lock')
    stages = {}
    status = 'PBS_PREFLIGHT_FAILED_NO_WORKER_START'
    current = None
    failure = None
    stop = root/'STOP.json'
    cancelled = False

    def cancel(_signal, _frame):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, cancel)
    try:
        job_id, window = pbs_job(config)
        deadline = window['deadline_epoch']
        if cancelled or stop.exists() or window['PBS_wall_s'] > WALL or not 900 < deadline-time.time() <= WALL:
            status = 'PBS_FINITE_START_WINDOW_ELAPSED_NO_WORKER_START'
            return 75
        custody = read_pin(config['data_custody'])
        data_proof = read_pin(config['data_cpu_proof'])
        if (data_proof['custody_sha256'] != config['data_custody']['sha256']
                or data_proof['visit_bank_sha256'] != custody['visit_bank_sha256']
                or data_proof['lr_sha256'] != custody['learning_rates_sha256']
                or data_proof['windows_by_domain'] != dict(target=1768, robodojo=50979, libero=209146)
                or data_proof['status'] != 'VANDA_MATCHED_TARGET_CPU_DATA_VERIFIED'):
            raise ValueError('SCALE_TARGET_REQUIRES_MATCHED_CROSS_SITE_DATA_PROOF')
        parent_asset(custody['parent_manifest'], custody['parent_params'])

        uuid, name, capacity, free = actual_gpu()
        fraction = .65
        allocator = math.ceil(capacity*fraction)
        if free <= allocator+2*GIB or shutil.disk_usage(root).free <= 24*GIB:
            raise PermissionError('SCALE_TARGET_ACTUAL_GPU_AND_STORAGE_HEADROOM_REQUIRED')
        cache = Path(os.environ['TMPDIR']).resolve(strict=True)/'sla-scale20k-target-compile'
        cache.mkdir()
        dataset = custody['data']['target']['root']
        paths = dict(parent_params=str(Path(custody['parent_params']).resolve(strict=True)),
                     dataset=str(Path(dataset).resolve(strict=True)), compilation_cache_dir=str(cache))
        allocation = root/'PBS_ALLOCATION_OBSERVED.private.json'
        host = os.uname().nodename
        write(allocation,dict(job_id=job_id,host=host,uid=os.getuid(),project=config['project'],state='R',
            allocated_GPU_uuids=[uuid], GPU_free_bytes=free, PID=os.getpid(), observed_epoch=time.time(), **window))
        site = root/'SITE_BINDING.private.json'
        write(site,dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=host,uid=os.getuid(),GPU_uuid=uuid,GPU_name=name,
            allocation_deadline_epoch=window['allocation_deadline_epoch'],allocator_limit_bytes=allocator,
            GPU_headroom_bytes=2*GIB,host_headroom_bytes=32*GIB,paths=paths,allocation=dict(kind='PBS',job_id=job_id,
                project=config['project'],actual_allocation=pin(allocation))))
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,JAX_PLATFORMS='cuda',
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   XLA_FLAGS=config['XLA_FLAGS'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        common = dict(arm=ARM,plan_sha=PLAN,recipe=RECIPE,source_sha=config['source_sha'],
            global_stop_step=200,microbatch=dict.fromkeys(DOMAINS,1),deadline_epoch=deadline,
            max_walltime_s=WALL,max_gpu_hours=3,save_and_exit_on_limit=True,save_reserve_s=300,
            model_service_timeout_s=1800,RPC_timeout_s=1800,hardware_allowed=False,physics=False,
            new_native_requests_cap=0,STOP_path=str(stop),neural_owner_lock=str(root/'GPU_WORKER.lock'),
            GPU_uuid=uuid,JAX_memory_fraction=fraction,site_binding=pin(site),**paths,
            **{key:custody[key] for key in ('data','parent_manifest','optimizer','tokenizer',
                                           'visit_bank_sha256','learning_rates_sha256')})
        prior = None
        for mode in ('qualify','train'):
            if cancelled or stop.exists() or time.time()+600 >= deadline:
                status = 'STOPPED_AT_COMPLETED_STAGE'
                return 75
            status = 'UNRESOLVED_GPU_EFFECT_NO_AUTOMATIC_REPLAY'
            stage = root/mode
            stage.mkdir()
            card = stage/'SPEC.private.json'
            write(card,dict(common,registered_root=str(stage),resume_from_closed_segment=prior))
            spec = pin(card)
            current = mode
            with (stage/'worker.log').open('x') as log:
                child = subprocess.Popen([sys.executable,'-B','-m','scripts.sla.train_scale20k',mode,
                    spec['path'],spec['sha256']],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(root/'PROGRESS.private.json',dict(stage=mode,child_PID=child.pid,job_id=job_id,
                                                       spec_sha256=spec['sha256'],epoch=time.time()))
                while child.poll() is None:
                    if cancelled and not stop.exists():
                        write(stop,dict(reason='PBS_OPERATOR_STOP',epoch=time.time()))
                    time.sleep(2)
            result = stage/'RESULT.private.json'
            if not result.is_file():
                status = 'UNRESOLVED_GPU_EFFECT_NO_AUTOMATIC_REPLAY'
                return 1
            prior = pin(result)
            stages[mode] = prior
            observed = read_pin(prior)
            status = observed['status']
            failure = observed.get('failure')
            if child.returncode or observed['exit_code']:
                return child.returncode or observed['exit_code']
        return 0
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        write(root/'RESULT.private.json',dict(status=status,stages=stages,last_stage=current,
            first_failure=failure,source_sha=config['source_sha'],PBS_JOBID=os.environ.get('PBS_JOBID'),
            new_native_requests=0,new_native_ticks=0))
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.config,sha256=args.sha256))))
