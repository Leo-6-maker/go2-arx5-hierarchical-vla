"""Register a bounded ADAPT student block on the existing Corelab world."""
import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from go2arx5_vla.world_model import adapt5k_native as adapt
from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from .multitask_panel_suffix import budgets, verified_prefix
from .multitask_stage_cards_v6 import cold_native_owner, pin, read, register_student_panel
from .multitask_student_campaign_v6 import endpoint_records, preflight
from .multitask_teacher_campaign_v6 import interface_records

GIB = 1024**3


def resources(auth, native, gpu, check, root):
    require = ['nvidia-smi','--id='+gpu['GPU_uuid'],'--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits']
    uuid,name,total,free = (s.strip() for s in subprocess.check_output(require,text=True,timeout=15).strip().split(','))
    total,free = int(total)*1024**2,int(free)*1024**2
    reserve = math.ceil(.8*total)
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    psi = {s.split()[0]:float(s.split()[1].split('=')[1]) for s in Path('/proc/pressure/memory').read_text().splitlines()}
    linger = subprocess.check_output(['loginctl','show-user',str(os.getuid()),'-p','Linger','--value'],
                                     text=True,timeout=10).strip() == 'yes'
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'],text=True,timeout=10).splitlines()
    try:
        with closing(OwnerLock(native['mobile_world_lock'])), closing(OwnerLock(gpu['neural_owner_lock'])):
            owners_clear = True
    except BlockingIOError:
        owners_clear = False
    host_required = max(36*GIB,gpu['minimum_host_available_bytes'])
    checks = dict(linger=linger,uuid=uuid==gpu['GPU_uuid']==check['GPU_uuid'],
        gpu_free=free>reserve+2*GIB,host=available>host_required,psi=max(psi.values())<=.1,
        gpu_idle=not any(uuid in line for line in processes),owners_clear=owners_clear,
        disk=shutil.disk_usage(root.parent).free>100*GIB)
    site = dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=native['host'],uid=native['uid'],
        GPU_uuid=uuid,GPU_name=name,GPU_headroom_bytes=2*GIB,host_headroom_bytes=host_required,
        allocator_limit_bytes=reserve,allocation_deadline_epoch=auth['allocation_deadline_epoch'],
        allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=gpu['neural_owner_lock']))
    observed = dict(GPU_uuid=uuid,GPU_total_bytes=total,GPU_free_bytes=free,allocator_limit_bytes=reserve,
                    headroom_bytes=2*GIB,host_available_bytes=available,host_required_bytes=host_required,
                    psi_avg10=psi,checks=checks,native_requests=0)
    return site,observed


def build(config):
    root = Path(config['root'])
    source = Path(config['source_root']).resolve(strict=True)
    native = read_pin(config['native_reference'])
    gpu = read_pin(config['gpu_reference'])
    auth = read_pin(config['authority'])
    campaign = read_pin(config['campaign'])
    interface = cold_native_owner(config['interface_predecessor'])
    now = time.time()
    if (root.exists() or root.parent.stat().st_uid != os.getuid() or os.getuid() != native['uid'] or gpu['uid'] != native['uid']
            or native['host'] != gpu['host'] or os.uname().nodename != native['host']
            or native['owner'] != 'EXISTING_LOCAL_PC' or gpu['owner'] != native['owner']
            or campaign['plan_sha'] != adapt.GOAL or Path(campaign['stop']).exists()
            or auth['plan_commit'] != adapt.PLAN or auth['goal_plan_sha'] != adapt.GOAL
            or auth['source'] != adapt.AUTHORITY_SOURCE or auth['source_sha'] != config['source_sha']
            or auth['compiled_population_sha256'] != adapt.POPULATION_SHA or auth['hardware_allowed'] is not False
            or not now+1980 < auth['deadline_epoch'] <= min(now+43200,campaign['deadline_epoch'],
                                                           auth['allocation_deadline_epoch']-120)
            or not set(native['cpus']) <= os.sched_getaffinity(0)
            or any(interface[k] != native[k] for k in ('mobile_world_lock','ROS_DOMAIN_ID','registry'))):
        raise PermissionError('ADAPT_NATIVE_REQUIRES_CURRENT_FINITE_PROJECT_WORLD_AND_SITE_IDENTITY')
    interface_records(interface,native['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    if Path(root).name != config['unit'].removeprefix('wm02-'):
        raise ValueError('ADAPT_SYSTEMD_UNIT_MUST_NAME_ITS_OWN_CAPSULE')
    arm = config['arm']
    manifest = adapt.package_manifest(arm,config['package'])
    qualified = read_pin(config['deployment'])
    replay = read_pin(qualified['proof']['restore'])
    code_files = {str(p.relative_to(source)):pin(p)['sha256'] for directory in ('src','scripts')
                  for p in (source/directory).rglob('*.py')}
    if any(code_files.get(path) != expected for path,expected in config['source_members'].items()):
        raise ValueError('ADAPT_NATIVE_STAGED_RUNTIME_BYTES_CHANGED')
    seal = hashlib.sha256(json.dumps(code_files,sort_keys=True).encode()).hexdigest()
    env = dict(native['native_environment'],PYTHONPATH=':'.join((str(source),str(source/'src'),
                                                                   native['native_environment']['PYTHONPATH'])),
               ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES='')
    spec = dict(native,scope=adapt.SCOPE,plan_commit=adapt.PLAN,goal_plan_sha=adapt.GOAL,
        evaluation_stage=auth['stage'],condition_stop=auth['condition_stop'],source_sha=config['source_sha'],
        source_root=str(source),source_files=code_files,source_seal=seal,
        registered_root=str(root),campaign_root=str(root),native_environment=env,physics=True,
        deadline_epoch=auth['deadline_epoch'],campaign_deadline_epoch=campaign['deadline_epoch'],
        allocation_deadline_epoch=auth['allocation_deadline_epoch'],new_optimizer_updates_budget=0,
        new_training_visits_budget=0,new_physical_requests_budget=auth['native_request_cap'],
        new_native_tick_budget=auth['tick_cap'],new_model_generations_budget=auth['generation_cap'],
        automatic_retry=False,frontend_dependency=False,sim_only_project_authorization=config['authority'],
        neural_owner_lock=gpu['neural_owner_lock'],adapt_GPU_uuid=gpu['GPU_uuid'],
        interface_predecessor=config['interface_predecessor'],compiled_population=config['population'],
        packages={arm:config['package']},endpoints={arm:config['evaluation']},
        ordinary_RGB_restores={arm:config['deployment']},adapt_D1_sha256=manifest['D1_freeze_sha256'],
        adapt_state_statistics_sha256=manifest['state_statistics_sha256'],
        previous_panel_segments=config.get('previous_panel_segments',[]))
    conditions = adapt.conditions(read_pin(spec['compiled_population']),spec['evaluation_stage'],adapt.arms_for(spec))
    carried = verified_prefix(spec,conditions)
    block = adapt.next_block(spec,conditions,len(carried))
    if budgets(block) != tuple(auth[k] for k in ('native_request_cap','tick_cap','generation_cap')):
        raise PermissionError('ADAPT_NATIVE_AUTHORITY_MUST_MATCH_FIXED_NEXT_BLOCK')
    endpoint_records(spec)
    site, observed = resources(auth,native,gpu,replay['environment'],root)
    gpu_env = dict(gpu['native_environment'],XLA_FLAGS=replay['environment']['xla'],
                   XLA_PYTHON_CLIENT_MEM_FRACTION='0.8',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   CUDA_VISIBLE_DEVICES=gpu['GPU_uuid'],JAX_PLATFORMS='cuda')
    gpu = dict(gpu,JAX_memory_fraction=.8,allocator_limit_bytes=site['allocator_limit_bytes'],
               minimum_host_available_bytes=site['host_headroom_bytes'],native_environment=gpu_env)
    return spec,gpu,site,observed,[row['id'] for row,_ in block]


def launch(config, *, start=False):
    spec,gpu,site,observed,conditions = build(config)
    if not start:
        return dict(status='READ_ONLY_PREFLIGHT',conditions=conditions,resource=observed,started=False)
    root = Path(spec['registered_root'])
    run = root.parent/(root.name+'-launch')
    run.mkdir(mode=0o700)
    write(run/'RESOURCE.private.json',observed)
    if not all(observed['checks'].values()):
        raise PermissionError('ADAPT_NATIVE_SITE_RESOURCE_OR_LOCK_UNAVAILABLE')
    write(run/'SITE.private.json',site)
    write(run/'GPU_TEMPLATE.private.json',dict(gpu,site_binding=pin(run/'SITE.private.json')))
    spec['gpu_template'] = pin(run/'GPU_TEMPLATE.private.json')
    spec_pin = register_student_panel(spec)
    registered = read_pin(spec_pin)
    try:
        preflight(registered)
        worker = read_pin(registered['resident_server_spec'])
        replay = read_pin(read_pin(config['deployment'])['proof']['restore'])
        cache_source = Path(replay['site']['paths']['compilation_cache_dir']).resolve(strict=True)
        cache = Path(worker['compilation_cache_dir']).resolve(strict=True)
        for member in sorted(cache_source.iterdir()):
            if member.is_file() and not member.is_symlink():
                shutil.copy2(member,cache/member.name)
    except Exception as error:
        for file in [*registered['cases'],registered['resident_server_spec']['path'],spec_pin['path']]:
            close(read(file),1,not_started=True,reason='ADAPT_NATIVE_PREFLIGHT:'+type(error).__name__)
        raise
    write(run/'PRE_DISPATCH.private.json',dict(spec=spec_pin,site=observed,native_requests=0))
    receipt = launch_service(registered,config['unit'],'--multitask-student-panel',bootstrap='v6_bootstrap.py')
    actual = unit_state(config['unit'])
    if not actual['pid'] or actual['pid'] not in actual['cgroup_pids']:
        raise RuntimeError('ADAPT_NATIVE_UNKNOWN_START_REQUIRES_ONSITE_RECONCILIATION')
    result = dict(receipt,spec=spec_pin,actual=actual,conditions=conditions,started=True,
                  deadline_epoch=spec['deadline_epoch'],stop=str(root/'STOP.json'))
    write(run/'DISPATCH.private.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    parser.add_argument('--launch',action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(launch(read_pin(dict(path=args.config,sha256=args.sha256)),start=args.launch),allow_nan=False))
