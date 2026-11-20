"""Run one authorized SCALE20K Core segment or its first qualification."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import CHECKPOINTS, DOMAINS, PLAN, RECIPE, bank_sha256, cursors, learning_rates
from scripts.pi0292.durable import OwnerLock, read, write

GIB = 1024**3


def resources(uuid):
    row = subprocess.check_output(['nvidia-smi', '--id='+uuid, '--query-gpu=uuid,name,memory.total,memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=8)
    actual,name,total,available = [field.strip() for field in row.split(',')]
    if actual != uuid:
        raise ValueError('WRONG_CORE_GPU_BOUND_TO_PREFIX')
    host = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))
    return dict(name=name,total_bytes=int(total)*1024**2,free_bytes=int(available)*1024**2,host_bytes=host)


def closed_predecessor(config, root):
    if 'crash_reconciliation' in config:
        from scripts.sla.scale20k_core_crash import resolve

        return resolve(config,root)
    prior = config.get('previous_train_result')
    if prior is None:
        return None, None, 0
    result_file = Path(prior['path']).resolve(strict=True)
    old_root = result_file.parent.parent
    parent = read(old_root/'RESULT.private.json')
    lease = read(old_root/'control/LEASE.json')
    old_unit = lease['unit']
    if (result_file != old_root/'train/RESULT.private.json' or old_root == root
            or old_root.stat().st_uid != os.getuid() or lease['status'] != 'CLOSED'
            or old_unit != 'wm02-'+old_root.name+'.service' or parent['stages'].get('train') != prior):
        raise ValueError('SCALE20K_PRIOR_RESULT_REQUIRES_A_SEALED_CLOSED_SEGMENT')
    status = subprocess.check_output(['systemctl','--user','show',old_unit,'--value','-p','MainPID'],
                                     text=True,timeout=10).strip()
    if status != '0':
        raise PermissionError('SCALE20K_PRIOR_SERVICE_STILL_OWNED_OR_UNKNOWN')
    result = read_pin(prior)
    step = result['global_step']
    if (result['scope'] != RECIPE or result['source_sha'] != config['source_sha'] or result['arm'] != config['arm']
            or result['mode'] != 'train' or (result['status'],result['exit_code']) not in (
                ('FIXED_SEGMENT_SAVED',0),('BOUNDED_SAVED_PREFIX',75))
            or step <= 0 or result['global_cursors'] != cursors(config['arm'],step)
            or not result['checkpoint'] or result['new_native_requests'] or result['new_native_ticks']):
        raise ValueError('SCALE20K_PRIOR_TRAIN_EFFECT_NOT_SAFE_TO_RESUME')
    checkpoint = Path(result['checkpoint']['path']).resolve(strict=True)
    if (checkpoint.parent != result_file.parent
            or sha(checkpoint/'COMPLETE.json') != result['checkpoint']['sha256']):
        raise ValueError('SCALE20K_PREDECESSOR_CHECKPOINT_MUST_BE_OWNED_AND_HASH_VERIFIED')
    worker_source = Path(parent.get('training_source_root',old_root/'source')).resolve(strict=True)
    if not (worker_source/'scripts/sla/train_scale20k.py').is_file():
        raise ValueError('SCALE20K_ORIGINAL_TRAINING_WORKER_UNAVAILABLE')
    return prior, worker_source, step


def training_launch_context(config, launcher_source, worker_source, prior, env):
    launcher_source, worker_source = launcher_source.resolve(strict=True), worker_source.resolve(strict=True)
    if prior:
        members = config['training_source_files']
        if not {'scripts/sla/train_scale20k.py', 'src/go2arx5_vla/sla_adapter/scale20k_training.py'}.issubset(members):
            raise ValueError('SCALE20K_CONTINUATION_REQUIRES_ORIGINAL_TRAINER_SOURCE_MEMBERS')
        for member, expected in members.items():
            path = (worker_source/member).resolve(strict=True)
            if not path.is_relative_to(worker_source) or sha(path) != expected:
                raise ValueError('SCALE20K_CONTINUATION_ORIGINAL_WORKER_SOURCE_CHANGED:'+member)
    # A different cwd does not override PYTHONPATH for the src-layout package.
    extra = [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p and not any(
        Path(p).resolve().is_relative_to(root) for root in (launcher_source,worker_source))]
    search = os.pathsep.join([str(worker_source), str(worker_source/'src'), *extra])
    cache = Path(config['root'])/'cache'
    if prior:
        previous_stage = Path(prior['path']).parent
        previous = read(previous_stage/'SPEC.private.json')
        previous_scope = previous_stage.parent
        latest = read(previous_scope/'PROGRESS.private.json')
        previous_cache = Path(previous['compilation_cache_dir'])
        if (latest['spec_sha256'] == sha(previous_stage/'SPEC.private.json')
                and previous['source_sha'] == config['source_sha'] and previous['GPU_uuid'] == config['GPU_uuid']
                and previous_cache.is_dir() and previous_cache.stat().st_uid == os.getuid()):
            cache = previous_cache
    return dict(env, PYTHONPATH=search), cache.resolve(strict=True)


def run(config):
    root = Path(config['root']).resolve(strict=True)
    deadline = config['deadline_epoch']
    lock_path = config['neural_owner_lock']
    uuid = config['GPU_uuid']
    unit = config['unit']+'.service'
    actual = subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],
                                     text=True,timeout=10).strip()
    linger = subprocess.check_output(['loginctl','show-user',str(os.getuid()),'-p','Linger','--value'],
                                     text=True,timeout=10).strip()
    if (config['plan'] != PLAN or config['arm'] not in CHECKPOINTS or os.uname().nodename != config['host']
            or os.getuid() != config['uid'] or actual != str(os.getpid()) or linger != 'yes'
            or os.environ['CUDA_VISIBLE_DEVICES'] != uuid or not time.time() < deadline < time.time()+14400
            or root.stat().st_uid != os.getuid()):
        raise PermissionError('SCALE20K_EXACT_FINITE_DETACHED_CORE_UNIT_REQUIRED')
    prior, worker_source, initial_step = closed_predecessor(config,root)
    launcher_source = Path(__file__).resolve().parents[2]
    worker_source = worker_source or launcher_source
    worker_env,cache = training_launch_context(config,launcher_source,worker_source,prior,os.environ)
    limit = config.get('global_stop_step',200)
    end = CHECKPOINTS[config['arm']][-1] if initial_step else 200
    if type(limit) is not int or not initial_step < limit <= end:
        raise ValueError('SCALE20K_FINITE_SEGMENT_REQUIRES_FORWARD_STEP_LIMIT')
    stop = root/'STOP.json'
    progress,terminal = root/'PROGRESS.private.json',root/'RESULT.private.json'
    lease = root/'control'/'LEASE.json'
    (root/'control').mkdir()
    write(lease,dict(owner='EXISTING_LOCAL_PC',status='ACTIVE',deadline_epoch=deadline,main_pid=os.getpid(),unit=unit))
    allocation = dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=lock_path,unit=unit)
    cancelled = False

    def halt(*_):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,halt)
    status = 'CAPACITY_OR_OWNER_WAIT_ENDED_NO_GPU_EFFECT'
    stages = {}
    try:
        host_headroom_gib = 27
        if 'crash_reconciliation' in config:
            host_headroom_gib = 40
        while not cancelled and not stop.exists() and time.time()+1800 < deadline:
            capacity = resources(uuid)
            share = math.ceil(config['JAX_memory_fraction']*capacity['total_bytes'])
            can_start = (capacity['free_bytes'] > share+2*GIB and capacity['host_bytes'] > host_headroom_gib*GIB
                         and shutil.disk_usage(root).free > 24*GIB)
            owner = False
            if can_start:
                try:
                    lock = OwnerLock(lock_path)
                except BlockingIOError:
                    owner = True
                else:
                    lock.close()
                    break
            write(progress,dict(phase='WAIT_GPU_CAPACITY_AND_EXISTING_OWNER',epoch=time.time(),
                                owner_busy=owner,GPU=capacity))
            time.sleep(10)
        else:
            return 75
        paths = dict(parent_params=str(Path(config['parent_params']).resolve(strict=True)),
            dataset=str(Path(config['data']['target']['root']).resolve(strict=True)),
            compilation_cache_dir=str(cache))
        site = dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=config['host'],uid=config['uid'],
            GPU_uuid=uuid,GPU_name=capacity['name'],GPU_headroom_bytes=2*GIB,allocator_limit_bytes=share,
            allocation_deadline_epoch=deadline,host_headroom_bytes=host_headroom_gib*GIB,paths=paths,allocation=allocation)
        sitepath = root/'SITE.private.json'
        write(sitepath,site)
        common = dict(plan_sha=PLAN,recipe=RECIPE,source_sha=config['source_sha'],arm=config['arm'],
            hardware_allowed=False,physics=False,new_native_requests_cap=0,save_reserve_s=300,STOP_path=str(stop),
            deadline_epoch=deadline-60,max_walltime_s=14400,max_gpu_hours=4,save_and_exit_on_limit=True,
            model_service_timeout_s=1800,RPC_timeout_s=1800,site_binding=dict(path=str(sitepath),sha256=sha(sitepath)),
            GPU_uuid=uuid,neural_owner_lock=lock_path,global_stop_step=limit,
            microbatch=dict.fromkeys(DOMAINS,1),learning_rates_sha256=bank_sha256(learning_rates()),
            **paths,**{key:config[key] for key in ('JAX_memory_fraction','data','parent_manifest','optimizer',
                                                   'visit_bank_sha256','tokenizer')})
        stages_to_run = ('train',) if prior else ('qualify','train')
        for mode in stages_to_run:
            if cancelled or stop.exists() or time.time()+600 >= deadline:
                status = 'STOPPED_AT_COMPLETED_STAGE'
                return 75
            stage = root/mode
            stage.mkdir()
            path = stage/'SPEC.private.json'
            write(path,dict(common,registered_root=str(stage),resume_from_closed_segment=prior))
            write(progress,dict(phase=mode.upper(),epoch=time.time(),stage=str(stage),spec_sha256=sha(path)))
            command = [sys.executable,'-B','-m','scripts.sla.train_scale20k',mode,str(path),sha(path)]
            with (stage/'worker.log').open('x') as log:
                child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=worker_source,env=worker_env)
                while child.poll() is None:
                    if cancelled and not stop.exists():
                        write(stop,dict(reason='OPERATOR_STOP'))
                    time.sleep(2)
            result_path = stage/'RESULT.private.json'
            if not result_path.is_file():
                status = 'UNRESOLVED_GPU_EFFECT_NO_AUTOMATIC_REPLAY'
                return 1
            prior = dict(path=str(result_path),sha256=sha(result_path))
            observed = read_pin(prior)
            stages[mode] = prior
            status = observed['status']
            if child.returncode or observed['exit_code']:
                return child.returncode or 1
        return 0
    finally:
        write(terminal,dict(status=status,stages=stages,training_source_root=str(worker_source),
                            new_native_requests=0,new_native_ticks=0))
        write(lease,dict(owner='EXISTING_LOCAL_PC',status='CLOSED',deadline_epoch=deadline,main_pid=os.getpid(),unit=unit))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.config,sha256=args.sha256))))
