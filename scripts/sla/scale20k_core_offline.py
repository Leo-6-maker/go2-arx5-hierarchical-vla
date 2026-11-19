"""Evaluate one sealed SCALE endpoint in a bounded Core user service."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import CHECKPOINTS, DOMAINS, PLAN
from go2arx5_vla.sla_adapter.scale20k_evaluation import PER_DOMAIN, RESTORE
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.evaluate_scale20k import evidence_spec
from scripts.sla.scale20k_core_prefix import GIB, closed_predecessor, resources


WALL = 5400


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def offline_endpoint(config, root):
    prior, _, step = closed_predecessor(dict(config, source_sha=config['training_source_sha']), root)
    if prior is None or step != config['step'] or step not in CHECKPOINTS[config['arm']]:
        raise ValueError('SCALE20K_FORMAL_OFFLINE_REQUIRES_SEALED_FORMAL_TRAINING_STEP')
    result = read_pin(prior)
    training_path = Path(prior['path']).parent/'SPEC.private.json'
    training = pin(training_path)
    if training != config['training_spec']:
        raise ValueError('SCALE20K_ORIGINAL_TERMINAL_TRAINING_SPEC_REQUIRED')
    return result['checkpoint'], training


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    unit = 'wm02-'+root.name+'.service'
    gpu = config['GPU_uuid']
    deadline = config['deadline_epoch']
    pid = subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],
                                  text=True, timeout=10).strip()
    if (pid != str(os.getpid()) or root.stat().st_uid != os.getuid() or config['uid'] != os.getuid()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or os.uname().nodename != config['host']
            or (root/'RESULT.private.json').exists() or not 600 < deadline-time.time() <= WALL
            or (source/'SOURCE_SHA.private').read_text().strip() != config['source_sha']
            or config['plan'] != PLAN):
        raise PermissionError('SCALE20K_OFFLINE_REQUIRES_FRESH_OWNED_FINITE_SITE_SERVICE')
    checkpoint, training = offline_endpoint(config, root)
    training_card = read_pin(training)
    panel = read_pin(config['panel'])
    if any(len(panel['rows'][d]) != PER_DOMAIN for d in DOMAINS):
        raise ValueError('SCALE20K_OFFLINE_REQUIRES_FROZEN_COMPLETE_THREE_DOMAIN_ROWS')
    fraction = config['JAX_memory_fraction']
    capacity = resources(gpu)
    reserve = math.ceil(capacity['total_bytes']*fraction)
    if (not 0 < fraction < 1 or capacity['free_bytes'] <= reserve+2*GIB
            or capacity['host_bytes'] <= 27*GIB):
        raise PermissionError('SCALE20K_OFFLINE_ACTUAL_GPU_OR_RAM_HEADROOM_INSUFFICIENT')
    owner_lock = root/'SUPERVISOR.lock'
    worker_lock = config['neural_owner_lock']
    with_probe = OwnerLock(worker_lock)
    with_probe.close()
    lock = OwnerLock(owner_lock)
    stopped = False
    stop = root/'STOP.json'
    control = root/'control'
    control.mkdir()
    lease = control/'LEASE.json'
    ownership = dict(owner='EXISTING_LOCAL_PC', deadline_epoch=deadline, main_pid=os.getpid(), unit=unit)
    write(lease, dict(ownership, status='ACTIVE'))

    def cancel(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    stages = {}
    status = 'FAILED_BEFORE_MODEL_WORKER'
    failure = None
    try:
        paths = {key:str(Path(training_card[key]).resolve(strict=True)) for key in ('parent_params','dataset')}
        paths['compilation_cache_dir'] = str((root/'cache').resolve(strict=True))
        site = root/'SITE.private.json'
        write(site, dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=config['host'],uid=config['uid'],
            GPU_uuid=gpu,GPU_name=capacity['name'],allocation_deadline_epoch=deadline,
            allocator_limit_bytes=reserve,GPU_headroom_bytes=2*GIB,host_headroom_bytes=27*GIB,paths=paths,
            allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=worker_lock,unit=unit)))
        common = dict(plan=PLAN,arm=config['arm'],step=config['step'],source_sha=config['source_sha'],
            checkpoint=checkpoint,training_spec=training,panel=config['panel'],site_binding=pin(site),
            physics=False,hardware_allowed=False,updates_cap=0,native_request_cap=0,GPU_uuid=gpu,
            JAX_memory_fraction=fraction,neural_owner_lock=worker_lock,deadline_epoch=deadline-60,
            max_walltime_s=WALL,max_gpu_hours=WALL/3600,save_and_exit_on_limit=True,
            model_service_timeout_s=1800,RPC_timeout_s=1800,STOP_path=str(stop),**paths)
        for mode, per_domain in (('evaluate',PER_DOMAIN),('restore',len(RESTORE))):
            if stopped or stop.exists() or time.time()+300 >= deadline:
                status = 'STOPPED_AT_COMPLETED_OFFLINE_STAGE'
                return 75
            role = root/mode
            role.mkdir()
            spec = dict(common,registered_root=str(role),model_generation_cap=len(DOMAINS)*per_domain)
            if mode=='restore':
                spec['evaluation_result'] = stages['evaluate']
            evidence_spec(spec, mode)
            card = root/(mode+'.SPEC.private.json')
            write(card, spec)
            status = 'OFFLINE_EFFECT_UNRESOLVED'
            with (root/(mode+'.log')).open('x') as log:
                child = subprocess.Popen([sys.executable,'-B','-m','scripts.sla.evaluate_scale20k',
                    mode,str(card),sha(card)],cwd=source,stdout=log,stderr=subprocess.STDOUT)
                write(root/'PROGRESS.private.json', dict(role=mode,child_PID=child.pid,spec_sha256=sha(card),epoch=time.time()))
                while child.poll() is None:
                    if stopped and not stop.exists():
                        write(stop, dict(reason='OPERATOR_STOP', epoch=time.time()))
                    time.sleep(2)
            result_path = role/'RESULT.private.json'
            if not result_path.is_file():
                return child.returncode or 1
            result = read_pin(pin(result_path))
            stages[mode] = pin(result_path)
            status = result['status']
            failure = result.get('failure')
            if child.returncode or result['exit_code']:
                return child.returncode or result['exit_code']
        return 0
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write(root/'RESULT.private.json', dict(status=status,checkpoint=checkpoint,panel=config['panel'],
            stages=stages,first_failure=failure,source_sha=config['source_sha'],native_requests=0,updates=0))
        write(lease, dict(ownership, status='CLOSED'))
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.config, sha256=args.sha256))))
