"""Launch a finite SCALE native block using the original resident worker and scorer."""
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

from go2arx5_vla.world_model import multisource_d4_native as d4, scale20k_native as scale
from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.scale20k_core_prefix import resources
from .multitask_panel_suffix import budgets, verified_prefix
from .multitask_stage_cards_v6 import cold_native_owner, pin, read, register_student_panel
from .multitask_student_campaign_v6 import preflight
from .multitask_teacher_campaign_v6 import interface_records


GOAL = '1b2a25b7c1720051a2627d30831fb18130462374'
GIB = 1024**3


def resource_binding(native, gpu, replay, root, allocation_deadline):
    uuid = gpu['GPU_uuid']
    if uuid != replay['site']['GPU_uuid']:
        raise ValueError('SCALE20K_NATIVE_GPU_MUST_MATCH_THE_STRICT_CORE_REPLAY')
    observed = resources(uuid)
    fraction = gpu['JAX_memory_fraction']
    reserve = math.ceil(fraction*observed['total_bytes'])
    host_min = max(36*GIB,gpu['minimum_host_available_bytes'])
    raw = Path('/proc/pressure/memory').read_text().splitlines()
    pressure = {line.split()[0]:float(line.split()[1].split('=')[1]) for line in raw}
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader'],text=True,timeout=10).splitlines()
    try:
        with closing(OwnerLock(native['mobile_world_lock'])), closing(OwnerLock(gpu['neural_owner_lock'])):
            owners_clear = True
    except BlockingIOError:
        owners_clear = False
    linger = subprocess.check_output(['loginctl','show-user',str(os.getuid()),'-p','Linger','--value'],
                                     text=True,timeout=10).strip()=='yes'
    disk = shutil.disk_usage(root.parent).free
    checks = dict(linger=linger,gpu=observed['free_bytes']>reserve+2*GIB,host=observed['host_bytes']>host_min,
                  psi=max(pressure.values())<=.1,gpu_idle=not any(uuid in p for p in processes),
                  owners_clear=owners_clear,disk=disk>100*GIB)
    site = dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=native['host'],uid=native['uid'],
        GPU_uuid=uuid,GPU_name=observed['name'],GPU_headroom_bytes=2*GIB,host_headroom_bytes=host_min,
        allocator_limit_bytes=reserve,allocation_deadline_epoch=allocation_deadline,
        allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=gpu['neural_owner_lock']))
    return site,dict(observed,checks=checks,psi_avg10=pressure,disk_free_bytes=disk,
                     required_gpu_bytes=reserve+2*GIB,required_host_bytes=host_min,native_requests=0)


def build(config):
    root = Path(config['root'])
    source = Path(config['source_root']).resolve(strict=True)
    native,gpu,authority,campaign = map(read_pin,(config['native_reference'],config['gpu_reference'],
                                                  config['authority'],config['campaign']))
    now = time.time()
    if (root.exists() or root.parent.stat().st_uid!=os.getuid() or os.getuid()!=native['uid']
            or native['uid']!=gpu['uid'] or os.uname().nodename!=native['host'] or native['host']!=gpu['host']
            or native['owner']!='EXISTING_LOCAL_PC' or gpu['owner']!=native['owner']
            or campaign['plan_sha']!=GOAL or Path(campaign['stop']).exists()
            or authority['plan_commit']!=scale.PLAN or authority['goal_plan_sha']!=GOAL
            or authority['source']!=scale.AUTHORITY_SOURCE or authority['source_sha']!=config['source_sha']
            or authority['compiled_population_sha256']!=scale.POPULATION_SHA or authority['hardware_allowed'] is not False
            or authority['allocation_deadline_epoch']>config['site_deadline_epoch']
            or not now+1980<authority['deadline_epoch']<=min(now+43200,campaign['deadline_epoch'],
                                                           authority['allocation_deadline_epoch']-120)
            or root.name!=config['unit'].removeprefix('wm02-') or not set(native['cpus'])<=os.sched_getaffinity(0)):
        raise PermissionError('SCALE20K_REQUIRES_EXACT_FINITE_PROJECT_AND_CORE_WORLD_IDENTITY')
    interface = cold_native_owner(config['interface_predecessor'])
    if any(interface[k]!=native[k] for k in ('mobile_world_lock','ROS_DOMAIN_ID','registry')):
        raise ValueError('SCALE20K_WORLD_DIFFERS_FROM_ORIGINAL_COLD_INTERFACES')
    interface_records(interface,native['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    arm = config['arm']
    manifest,_ = scale.endpoint(arm,config['package'],config['evaluation'],config['replay'])
    replay = read_pin(config['replay'])
    replay_spec = read_pin(config['replay_worker_spec'])
    dataset = read_pin(config['dataset'])
    if (manifest['target_data_manifest_sha256']!=config['dataset']['sha256']
            or manifest['statistics_sha256']!=dataset['statistics']['sha256']
            or manifest['parent_manifest']['sha256']!=gpu['parent_manifest']['sha256']
            or replay['site']['GPU_uuid']!=gpu['GPU_uuid'] or replay_spec['GPU_uuid']!=gpu['GPU_uuid']
            or replay_spec['package']!=config['package'] or replay_spec['evaluation']!=config['evaluation']
            or replay_spec['physics'] is not False or replay_spec['JAX_memory_fraction']!=.8):
        raise ValueError('SCALE20K_PACKAGE_WORLD_DATA_AND_GPU_REPLAY_DO_NOT_MATCH')
    code_files = {str(p.relative_to(source)):pin(p)['sha256'] for directory in ('src','scripts')
                  for p in (source/directory).rglob('*.py')}
    if code_files!=read_pin(config['source_members']):
        raise ValueError('SCALE20K_STAGED_RUNTIME_BYTES_CHANGED')
    environment = dict(native['native_environment'],PYTHONPATH=':'.join((str(source),str(source/'src'),
                                                                         native['native_environment']['PYTHONPATH'])),
        ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES='')
    spec = dict(native,scope=scale.SCOPE,plan_commit=scale.PLAN,goal_plan_sha=GOAL,
        evaluation_stage=authority['stage'],condition_stop=authority['condition_stop'],source_sha=config['source_sha'],
        source_root=str(source),source_files=code_files,source_seal=hashlib.sha256(json.dumps(code_files,sort_keys=True).encode()).hexdigest(),
        registered_root=str(root),campaign_root=str(root),native_environment=environment,physics=True,
        deadline_epoch=authority['deadline_epoch'],campaign_deadline_epoch=campaign['deadline_epoch'],
        allocation_deadline_epoch=authority['allocation_deadline_epoch'],new_optimizer_updates_budget=0,
        new_training_visits_budget=0,new_physical_requests_budget=authority['native_request_cap'],
        new_native_tick_budget=authority['tick_cap'],new_model_generations_budget=authority['generation_cap'],
        automatic_retry=False,frontend_dependency=False,sim_only_project_authorization=config['authority'],
        neural_owner_lock=gpu['neural_owner_lock'],interface_predecessor=config['interface_predecessor'],
        compiled_population=config['population'],multitask_dataset=config['dataset'],
        packages={arm:config['package']},endpoints={arm:config['evaluation']},ordinary_RGB_restores={arm:config['replay']},
        previous_panel_segments=config.get('previous_panel_segments',[]))
    ordered = scale.conditions(read_pin(config['population']),spec['evaluation_stage'],scale.arms_for(spec))
    block = scale.next_block(spec,ordered,len(verified_prefix(spec,ordered)))
    if budgets(block)!=tuple(authority[k] for k in ('native_request_cap','tick_cap','generation_cap')):
        raise ValueError('SCALE20K_AUTHORITY_DOES_NOT_MATCH_THE_NEXT_ORIGINAL_BLOCK')
    fraction = replay_spec['JAX_memory_fraction']
    gpu_env = dict(gpu['native_environment'],XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),
                   XLA_PYTHON_CLIENT_PREALLOCATE='false',CUDA_VISIBLE_DEVICES=gpu['GPU_uuid'],JAX_PLATFORMS='cuda')
    gpu = dict(gpu,JAX_memory_fraction=fraction,native_environment=gpu_env)
    site,measured = resource_binding(native,gpu,replay,root,authority['allocation_deadline_epoch'])
    template = dict(gpu,allocator_limit_bytes=site['allocator_limit_bytes'],
                    minimum_host_available_bytes=site['host_headroom_bytes'])
    return spec,template,site,measured,[f"{row['id']}-{name}" for row,name in block]


def launch(config, *, start=False):
    spec,gpu,site,measured,conditions = build(config)
    if not start:
        return dict(status='READ_ONLY_PREFLIGHT',conditions=conditions,resources=measured,started=False)
    root = Path(spec['registered_root'])
    run = root.parent/(root.name+'-launch')
    run.mkdir(mode=0o700)
    write(run/'RESOURCE.private.json',measured)
    if not all(measured['checks'].values()):
        raise PermissionError('SCALE20K_NATIVE_RESOURCE_REJECT:'+','.join(k for k,v in measured['checks'].items() if not v))
    write(run/'SITE.private.json',site)
    write(run/'GPU_TEMPLATE.private.json',dict(gpu,site_binding=pin(run/'SITE.private.json')))
    spec['gpu_template'] = pin(run/'GPU_TEMPLATE.private.json')
    spec_pin = register_student_panel(spec)
    registered = read_pin(spec_pin)
    try:
        preflight(registered)
        worker = read_pin(registered['resident_server_spec'])
        replay = read_pin(config['replay'])
        original = Path(replay['site']['paths']['compilation_cache_dir']).resolve(strict=True)
        cache = Path(worker['compilation_cache_dir']).resolve(strict=True)
        for member in original.iterdir():
            if member.is_file() and not member.is_symlink():
                shutil.copy2(member,cache/member.name)
        if Path(read_pin(config['campaign'])['stop']).exists() or (root/'STOP.json').exists():
            raise PermissionError('SCALE20K_STOPPED_AFTER_REGISTRATION')
    except Exception as error:
        for file in [*registered['cases'],registered['resident_server_spec']['path'],spec_pin['path']]:
            close(read(file),1,not_started=True,reason='SCALE20K_NATIVE_PREFLIGHT:'+type(error).__name__)
        raise
    write(run/'PRE_DISPATCH.private.json',dict(spec=spec_pin,resource=measured,native_requests=0))
    receipt = launch_service(registered,config['unit'],'--multitask-student-panel',bootstrap='v6_bootstrap.py')
    owner = unit_state(config['unit'])
    if not owner['pid'] or owner['pid'] not in owner['cgroup_pids']:
        raise RuntimeError('SCALE20K_UNKNOWN_START_REQUIRES_ONSITE_RECONCILIATION')
    actual = dict(receipt,spec=spec_pin,actual=owner,conditions=conditions,started=True,
                  deadline_epoch=spec['deadline_epoch'],stop=str(root/'STOP.json'))
    write(run/'DISPATCH.private.json',actual)
    return actual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    parser.add_argument('--launch',action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(launch(read_pin(dict(path=args.config,sha256=args.sha256)),start=args.launch),allow_nan=False))
