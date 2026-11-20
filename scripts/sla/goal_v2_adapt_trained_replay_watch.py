"""Wait for an exact trained ADAPT bundle and release one original Core RGB16."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from go2arx5_vla.sla_adapter import adapt5k_contract, adapt5k_package
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla import adapt5k_endpoint_transport as transport
from scripts.sla.freeze_adapt5k_panels import bindings
from scripts.sla.goal_v2_adapt0_dev_tail import expected_cold
from scripts.sla.replay_adapt0_core import trained_inputs
from scripts.sla.scale20k_core_prefix import resources


PANEL = '3136404c06f890a9dd830b2cf5f40e487ef436ede16e805e14ffc0cb06b5eefd'
GIB = 1024**3
REPLAY_WINDOW_S = 1500


def replay_inputs(phase, custody, previous, gpu):
    bundle = Path(custody['path']).resolve(strict=True).parent
    local = read_pin(custody)
    fields = dict(phase=phase, core_custody=custody, eval_input=previous['eval_input'],
        evaluation=transport.pin(bundle/'archive/FORMAL.private.json'),
        panel=transport.pin(bundle/'archive/PANEL.private.json'),
        package=dict(path=str(bundle/'package'), sha256=local['package_sha256']),
        parent_manifest=gpu['parent_manifest'], parent_params=gpu['parent_params'], dataset=previous['dataset'])
    manifest = adapt5k_package.Provider.read_manifest(bundle/'package', local['package_sha256'])
    panel, evaluation, data = map(read_pin, (fields['panel'], fields['evaluation'], fields['eval_input']))
    if (manifest['schema'] != adapt5k_package.TRAINED_SCHEMA or fields['panel']['sha256'] != PANEL
            or panel['bindings'] != bindings(data) or data['D1_freeze_sha256'] != manifest['D1_freeze_sha256']
            or data['target_D1']['statistics']['sha256'] != manifest['state_statistics_sha256']
            or previous['parent_params'] != gpu['parent_params'] or previous['GPU_uuid'] != gpu['GPU_uuid']
            or previous['neural_owner_lock'] != gpu['neural_owner_lock']):
        raise ValueError('ADAPT_TRAINED_WATCH_REQUIRES_IDENTICAL_CORE_INPUT_CONTRACT')
    read_pin(gpu['parent_manifest'])
    checkpoint = trained_inputs(fields, manifest, evaluation)
    return fields, checkpoint


def resource_snapshot(gpu, root, owner):
    capacity = resources(gpu['GPU_uuid'])
    reserve = math.ceil(capacity['total_bytes']*.8)
    pressure = {row.split()[0]: float(row.split()[1].split('=')[1])
                for row in Path('/proc/pressure/memory').read_text().splitlines()}
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'], text=True, timeout=12).splitlines()
    try:
        token = OwnerLock(gpu['neural_owner_lock'])
        token.close()
        owner_clear = True
    except BlockingIOError:
        owner_clear = False
    checks = dict(allocator_possible=reserve+2*GIB < capacity['total_bytes'],
        gpu_headroom=capacity['free_bytes'] > reserve+2*GIB,
        host_headroom=capacity['host_bytes'] > 36*GIB, pressure=max(pressure.values()) <= .1,
        no_device_owner=not any(gpu['GPU_uuid'] in row for row in processes),
        owner_lock=owner_clear, disk=shutil.disk_usage(root).free > 24*GIB)
    group = Path('/sys/fs/cgroup')/owner['ControlGroup'].lstrip('/')
    cgroup = {name: (group/name).read_text().strip() for name in ('memory.current','memory.peak','memory.max')
              if (group/name).is_file()}
    return dict(capacity, GPU_uuid=gpu['GPU_uuid'], used_gpu_processes=processes, allocator_fraction=.8,
        allocator_bytes=reserve, headroom_bytes=2*GIB, host_minimum_bytes=36*GIB,
        pressure_avg10=pressure, max_pressure_avg10=.1, checks=checks, cgroup=cgroup, epoch=time.time())


def dispatch(root, target, source, fields, gpu, config, deadline, stopped):
    if stopped() or target.exists():
        raise PermissionError('ADAPT_TRAINED_WATCH_TARGET_ALREADY_USED_OR_STOPPED')
    name = 'wm02-'+target.name+'.service'
    if unit_state(name)['pid']:
        raise PermissionError('ADAPT_TRAINED_WATCH_EXISTING_GPU_OWNER')
    target.mkdir(mode=0o700)
    cache = target/'cache'
    cache.mkdir()
    card = target/'REPLAY.private.json'
    write(card, dict(fields, root=str(target), source_sha=config['source_sha'], plan=adapt5k_contract.PLAN,
        deadline_epoch=deadline, physics=False, hardware_allowed=False, GPU_uuid=gpu['GPU_uuid'],
        neural_owner_lock=gpu['neural_owner_lock'], JAX_memory_fraction=.8,
        generation_cap=16, compilation_cache_dir=str(cache)))
    fixed = transport.pin(card)
    environment = dict(gpu['native_environment'])
    environment.update(PYTHONPATH=':'.join((str(source),str(source/'src'),environment['PYTHONPATH'])),
        CUDA_VISIBLE_DEVICES=gpu['GPU_uuid'], JAX_PLATFORMS='cuda', XLA_FLAGS=config['xla'],
        XLA_PYTHON_CLIENT_MEM_FRACTION='.8', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    args = ['systemd-run','--user','--unit='+name,'--property=Restart=no','--property=UMask=0077',
            '--property=KillMode=control-group','--property=TimeoutStopSec=90',
            '--property=StandardInput=null','--property=MemoryMax=25G','--property=CPUAffinity=16 17',
            '--property=RuntimeMaxSec='+str(math.ceil(deadline-time.time())+30),
            '--property=WorkingDirectory='+str(source),'--property=StandardOutput=append:'+str(target/'replay.log'),
            '--property=StandardError=append:'+str(target/'replay.log')]
    args.extend('--setenv='+key+'='+value for key,value in environment.items())
    args.extend([gpu['python'],'-B','-m','scripts.sla.replay_adapt0_core','replay',str(card),fixed['sha256']])

    def launch():
        if stopped():
            raise InterruptedError('ADAPT_TRAINED_WATCH_STOP_BEFORE_DISPATCH')
        subprocess.run(args,check=True,capture_output=True,timeout=20)
        state = unit_state(name)
        if state['pid'] not in state['cgroup_pids']:
            raise RuntimeError('ADAPT_TRAINED_WATCH_UNKNOWN_DISPATCH_EFFECT')
        return dict(owner=state, spec=fixed, deadline_epoch=deadline)

    return Journal(root/'registration',root.name).once('ONE_TRAINED_ENDPOINT_RGB16',fixed,launch)


def run(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    owner = unit_state(config['unit'])
    campaign, tail, previous, gpu = map(read_pin, (config['campaign'],config['tail'],config['input_reference'],config['GPU']))
    window = min(config['site_deadline_epoch'], campaign['deadline_epoch'])
    target, bundle = Path(config['target']), Path(config['bundle'])
    phase = config['phase']
    if (os.getuid()!=2009 or os.uname().nodename!='corelab-4080s' or root.stat().st_uid!=os.getuid()
            or os.getpid()!=owner['pid'] or owner['pid'] not in owner['cgroup_pids']
            or config['source_sha']!=(source/'SOURCE_SHA.private').read_text().strip()
            or campaign['plan_sha']!=transport.GOAL or campaign['deadline_epoch']!=campaign['t0_epoch']+14*86400
            or tail['campaign']!=config['campaign'] or tail['site_deadline_epoch']!=config['site_deadline_epoch']
            or previous['panel']['sha256']!=PANEL or phase not in transport.PHASES
            or target.parent!=root.parent or bundle.parent!=root.parent or target.exists()
            or previous['generation_cap']!=16 or gpu['uid']!=os.getuid()):
        raise PermissionError('ADAPT_TRAINED_WATCH_REQUIRES_FIXED_EXISTING_SITE')
    stop_paths = (root/'STOP.json', target/'STOP.json', Path(campaign['stop']))
    stopped = lambda: any(p.exists() for p in stop_paths)
    result = dict(status='WAITING_EXACT_VANDA_CUSTODY', phase=phase, source_sha=config['source_sha'],
        GPU_worker_started=False, generation_attempts=0, model_generations=0, native_requests=0,
        native_ticks=0, optimizer_updates=0, optimization_visits=0)
    lock = OwnerLock(root/'OWNER.lock')
    latest = window-960

    def stop_or_time():
        if stopped() or time.time()>=latest:
            raise TimeoutError('ADAPT_TRAINED_WATCH_STOP_OR_EXISTING_SITE_WINDOW')

    try:
        write(root/'READY.private.json',dict(result, owner=owner, latest_launch_epoch=latest))
        custody = bundle/'CORE_CUSTODY.private.json'
        while not custody.is_file():
            stop_or_time()
            time.sleep(min(30,max(.1,latest-time.time())))
        fields, checkpoint = replay_inputs(phase, transport.pin(custody), previous, gpu)
        result.update(status='WAITING_ORIGINAL_ADAPT0_TAIL', custody=fields['core_custody'], checkpoint=checkpoint)
        write(root/'PROGRESS.private.json',result)
        expected_cold(tail['unit'],config['tail_owner'],deadline=latest,stop=stop_or_time)
        prior = read(Path(tail['root'])/'RESULT.private.json')
        if prior['status'] not in ('ADAPT0_ORIGINAL_DEV24_SEALED','ADAPT0_TAIL_STOP_OR_SITE_BOUNDARY') or 'active' in prior:
            raise PermissionError('ADAPT_TRAINED_WATCH_ORIGINAL_ADAPT0_REQUIRES_RECONCILIATION')
        stop_or_time()
        capacity = resource_snapshot(gpu, root, owner)
        write(root/'RESOURCE.private.json',capacity)
        failed = [key for key,value in capacity['checks'].items() if not value]
        if failed:
            raise PermissionError('ADAPT_TRAINED_WATCH_RESOURCE_REJECT:'+','.join(failed))
        worker_deadline = min(window-120, time.time()+REPLAY_WINDOW_S)
        service = dispatch(root,target,source,fields,gpu,config,worker_deadline,stopped)
        result.update(status='EXACT_TRAINED_RGB16_WORKER_DISPATCHED', GPU_worker_started=True, dispatch=service)
        write(root/'PROGRESS.private.json',result)

        def signal_stop():
            if stopped() and not (target/'STOP.json').exists():
                write(target/'STOP.json',dict(reason='GOAL_V2_TRAINED_RGB16_STOP',epoch=time.time()))

        expected_cold(service['owner']['unit'],service['owner'],deadline=window,stop=signal_stop)
        receipt = transport.pin(target/'RESULT.private.json')
        actual = read_pin(receipt)
        result.update(replay=receipt,generation_attempts=actual['generation_attempts'],
                      model_generations=actual['model_generations'],worker_outcome=actual['status'])
        if (actual['package']!=fields['package']['sha256'] or actual['phase']!=phase
                or actual['generation_attempts']>16 or actual['model_generations']>actual['generation_attempts']
                or any(actual[k] for k in ('optimizer_updates','optimization_visits','native_requests','native_ticks'))
                or read(target/'control/LEASE.json')['status']!='CLOSED'):
            raise PermissionError('ADAPT_TRAINED_WATCH_TERMINAL_EFFECT_REQUIRES_RECONCILIATION')
        result['status'] = actual['status']
        return actual['exit_code']
    except TimeoutError as error:
        result.update(first_failure=str(error),last_stage=result['status'],status='STOP_OR_EXISTING_SITE_WINDOW')
        return 0
    except Exception as error:
        result.update(first_failure=f'{type(error).__name__}: {error}',last_stage=result['status'],
                      status='ADAPT_TRAINED_RGB16_REQUIRES_RECONCILIATION')
        return 1
    finally:
        write(root/'RESULT.private.json',result)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
