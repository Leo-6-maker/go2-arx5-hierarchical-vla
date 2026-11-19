"""Evaluate and independently restore a sealed SCALE-TGT milestone on an allocated A40."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import CHECKPOINTS, DOMAINS, PLAN
from go2arx5_vla.sla_adapter.scale20k_evaluation import PER_DOMAIN, RESTORE
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.evaluate_scale20k import evidence_spec, relocated_training
from scripts.sla.scale20k_vanda_prefix import ARM, completed_prefix
from scripts.wm02.multisource_d4_vanda import GIB, actual_gpu, pbs_job, pin


WALL = 5400


def read_pbs_with_auth_retry(query, *args):
    """Retry transient server authentication errors before any device work."""
    for attempt in range(4):
        try:
            return query(*args)
        except subprocess.CalledProcessError as error:
            if (error.returncode != 255 or not error.cmd or error.cmd[0] != '/opt/pbs/bin/qstat'
                    or 'errno=15019' not in (error.stderr or '')):
                raise
            print(f'PBS qstat authentication failure on attempt {attempt+1}/4: {error.stderr.strip()}',
                  file=sys.stderr, flush=True)
            if attempt == 3:
                raise
            time.sleep(4*(attempt+1))


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    if (root.stat().st_uid != os.getuid() or (root/'RESULT.private.json').exists()
            or (source/'SOURCE_SHA.private').read_text().strip() != config['source_sha']
            or config['plan'] != PLAN or config['arm'] != ARM or config['step'] not in CHECKPOINTS[ARM]):
        raise PermissionError('SCALE_VANDA_OFFLINE_REQUIRES_FRESH_OWNED_MILESTONE_SCOPE')
    lock = OwnerLock(root/'PBS_OWNER.lock')
    stop = root/'STOP.json'
    stages = {}
    result = dict(source_sha=config['source_sha'],PBS_JOBID=os.environ.get('PBS_JOBID'),
                  panel=config['panel'],stages=stages,offline_generations=0,
                  native_requests=0,native_ticks=0,updates=0)
    try:
        cpu = read_pin(config['CPU_match'])
        match = cpu['core']
        if (match != cpu['vanda'] or match['panel_sha256'] != config['panel']['sha256']
                or set(match['fingerprints']) != set(DOMAINS)
                or any(len(rows) != PER_DOMAIN for rows in match['fingerprints'].values())
                or any(match[key] != 0 for key in ('GPU_generations','optimizer_updates','native_requests'))):
            raise ValueError('SCALE_VANDA_OFFLINE_REQUIRES_MATCHED_FIXED_PANEL_CPU_INPUTS')
        job, allocation = read_pbs_with_auth_retry(pbs_job,config)
        deadline = allocation['deadline_epoch']
        if not 600 < deadline-time.time() <= WALL or allocation['PBS_wall_s'] > WALL:
            raise PermissionError('SCALE_VANDA_OFFLINE_REQUIRES_FINITE_EXISTING_PBS_WINDOW')
        checkpoint_result, _ = read_pbs_with_auth_retry(completed_prefix,config,root)
        terminal = read_pin(checkpoint_result)
        if terminal['global_step'] != config['step'] or terminal['exit_code'] != 0:
            raise ValueError('SCALE_VANDA_OFFLINE_REQUIRES_ORIGINAL_COMPLETE_MILESTONE')
        checkpoint = terminal['checkpoint']
        training = config['continuation']['training_spec']
        assets = read_pin(config['site_assets'])
        relocated = relocated_training(read_pin(training), assets)
        result.update(checkpoint=checkpoint,training_spec=training,site_assets=config['site_assets'])
        uuid,name,total,free = actual_gpu()
        fraction = .65
        reserve = math.ceil(total*fraction)
        if free <= reserve+2*GIB:
            raise PermissionError('SCALE_VANDA_OFFLINE_REQUIRES_ACTUAL_GPU_HEADROOM')
        cache = Path(os.environ['TMPDIR']).resolve(strict=True)/'sla-scale-offline-compile'
        cache.mkdir()
        paths = {key:str(Path(relocated[key]).resolve(strict=True)) for key in ('parent_params','dataset')}
        paths['compilation_cache_dir'] = str(cache)
        actual = root/'PBS_ALLOCATION_OBSERVED.private.json'
        write(actual,dict(allocation,job_id=job,uid=os.getuid(),host=os.uname().nodename,project=config['project'],
            state='R',allocated_GPU_uuids=[uuid],GPU_free_bytes=free,PID=os.getpid(),observed_epoch=time.time()))
        site = root/'SITE.private.json'
        write(site,dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=os.uname().nodename,uid=os.getuid(),GPU_uuid=uuid,
            GPU_name=name,allocation_deadline_epoch=allocation['allocation_deadline_epoch'],
            allocator_limit_bytes=reserve,GPU_headroom_bytes=2*GIB,host_headroom_bytes=32*GIB,paths=paths,
            allocation=dict(kind='PBS',job_id=job,project=config['project'],actual_allocation=pin(actual))))
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,JAX_PLATFORMS='cuda',
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   XLA_FLAGS=config['XLA_FLAGS'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        common = dict(plan=PLAN,arm=ARM,step=config['step'],source_sha=config['source_sha'],
            checkpoint=checkpoint,training_spec=training,site_assets=config['site_assets'],panel=config['panel'],
            site_binding=pin(site),physics=False,hardware_allowed=False,updates_cap=0,native_request_cap=0,
            GPU_uuid=uuid,JAX_memory_fraction=fraction,neural_owner_lock=str(root/'GPU_WORKER.lock'),
            deadline_epoch=deadline-60,max_walltime_s=WALL,max_gpu_hours=WALL/3600,save_and_exit_on_limit=True,
            model_service_timeout_s=1800,RPC_timeout_s=1800,STOP_path=str(stop),**paths)
        sealed_generations = 0
        for mode,count in (('evaluate',PER_DOMAIN),('restore',len(RESTORE))):
            if stop.exists() or time.time()+300 >= deadline:
                result.update(status='STOPPED_AT_COMPLETED_OFFLINE_STAGE',exit_code=75)
                return 75
            role = root/mode
            spec = dict(common,registered_root=str(role),model_generation_cap=len(DOMAINS)*count)
            if mode == 'restore':
                spec['evaluation_result'] = stages['evaluate']
            evidence_spec(spec,mode)
            role.mkdir()
            card = root/(mode+'.SPEC.private.json')
            write(card,spec)
            write(root/'PROGRESS.private.json',dict(role=mode,spec_sha256=pin(card)['sha256'],epoch=time.time(),job_id=job))
            with (root/(mode+'.log')).open('x') as log:
                result['offline_generations'] = None
                child = subprocess.run([sys.executable,'-B','-m','scripts.sla.evaluate_scale20k',
                    mode,str(card),pin(card)['sha256']],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                    timeout=max(1,deadline-time.time()))
            stage = role/'RESULT.private.json'
            if not stage.is_file():
                result.update(status='OFFLINE_EFFECT_UNRESOLVED',exit_code=child.returncode or 1)
                return result['exit_code']
            stages[mode] = pin(stage)
            readback = read_pin(stages[mode])
            sealed_generations += readback['new_model_generations']
            result['offline_generations'] = sealed_generations
            code = child.returncode or readback['exit_code']
            result.update(status=readback['status'],exit_code=code,first_failure=readback.get('failure'))
            if code:
                return code
        return 0
    except Exception as error:
        result.update(status='SCALE_VANDA_OFFLINE_FAILED',exit_code=1,first_failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(root/'RESULT.private.json',result)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.config,sha256=args.sha256))))
