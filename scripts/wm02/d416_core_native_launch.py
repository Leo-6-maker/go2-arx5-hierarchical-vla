"""Register and start one admitted D416 panel on the existing single Core world."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.world_model import multisource_d4_native as d4, sla_followup_native_v1 as plan
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, unit_state
from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from scripts.pi0292.durable import OwnerLock, read, write
from .d416_core_replay_launch import output
from .multitask_stage_cards_v6 import cold_native_owner as cold_owner, pin, register_student_panel
from .multitask_student_campaign_v6 import endpoint_records, preflight
from .multitask_teacher_campaign_v6 import interface_records


GIB = 1024**3


def build(config):
    root = Path(config['root'])
    source = Path(config['source_root']).resolve(strict=True)
    stage = config['stage']
    native = read_pin(config['native_reference'])
    gpu = read_pin(config['GPU_reference'])
    auth = read_pin(config['authority'])
    modes = {plan.GRASP:'D5',plan.CONFIRM:'D5',plan.JOINT:'J'}
    caps = plan.CAPS[stage]
    deadline = auth['deadline_epoch']
    now = time.time()
    cpus = native['cpus']
    if (root.exists() or root.parent.stat().st_uid != os.getuid()
            or native['uid'] != gpu['uid'] or gpu['uid'] != os.getuid()
            or native['host'] != gpu['host'] or gpu['host'] != os.uname().nodename
            or native['owner'] != gpu['owner'] or native['owner'] != 'EXISTING_LOCAL_PC'
            or not set(cpus).issubset(os.sched_getaffinity(0))
            or auth['source'] != plan.AUTHORITY_SOURCE or auth['plan_commit'] != plan.PLAN
            or auth['source_sha'] != config['source_sha'] or auth['stage'] != stage
            or auth['hardware_allowed'] is not False
            or tuple(auth[k] for k in ('native_request_cap','tick_cap','generation_cap')) != caps
            or not now + 1980 < deadline <= min(now+43200,auth['allocation_deadline_epoch']-120)):
        raise PermissionError('D416_REQUIRES_FRESH_OWN_FINITE_SINGLE_WORLD_SITE_BINDING')
    iface = cold_owner(config['E0_interface'])
    interface_records(iface,native['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    for name in ('mobile_world_lock','ROS_DOMAIN_ID','registry'):
        if iface[name] != native[name]:
            raise ValueError('D416_MUST_KEEP_THE_WORLD_AND_INTERFACE_BINDING')
    for p in (native['mobile_world_lock'],gpu['neural_owner_lock']):
        if not Path(p).is_file():
            raise ValueError('D416_ORIGINAL_SINGLE_WORLD_AND_GPU_LOCKS_REQUIRED')
        lock = OwnerLock(p)
        lock.close()
    gpu_site = read_pin(gpu['site_binding'])
    host_bytes = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                      if line.startswith('MemAvailable:'))
    free_bytes = int(output('nvidia-smi','--id='+gpu['GPU_uuid'],'--query-gpu=memory.free',
                            '--format=csv,noheader,nounits'))*1024**2
    if (free_bytes <= gpu_site['allocator_limit_bytes']+gpu_site['GPU_headroom_bytes']
            or host_bytes <= max(gpu_site['host_headroom_bytes'],gpu['minimum_host_available_bytes'])
            or shutil.disk_usage(root.parent).free <= 100*GIB
            or output('loginctl','show-user',str(os.getuid()),'-p','Linger') != 'Linger=yes'):
        raise PermissionError('D416_ACTUAL_CORE_SITE_HEADROOM_AND_LINGER_REQUIRED')
    custody = read_pin(config['custody'])
    replay = read_pin(config['Core_RGB_result'])
    if (custody['mode'] != modes[stage] or not replay['passed'] or replay['completed'] != 32
            or replay['native_requests'] or replay['native_integrations']):
        raise ValueError('D416_BOTH_LIVE_CORE_RGB16_REMARKS_REQUIRED')
    successor = {arm:row for arm,row in custody['arms'].items()}
    if stage == plan.CONFIRM:
        decision = read_pin(config['screen_decision'])
        chosen = decision['selected_arm']
        if chosen not in plan.CANDIDATES:
            raise ValueError('D416_NO_CONFIRMED_CANDIDATE_TO_REGISTER')
        successor = {chosen:successor[chosen]}
    if stage != plan.JOINT:
        successor[plan.CONTROL] = config['D4_B_control']
    core_restores = config['Core_RGB_restores']
    packages = {arm:item['package'] for arm,item in successor.items()}
    results = {arm:item['result'] for arm,item in successor.items()}
    restores = {arm:item['restore'] for arm,item in successor.items()}
    source_restores = {arm:item['source_restore'] for arm,item in successor.items() if item.get('source_restore')}
    code_files = {str(p.relative_to(source)):sha(p) for root_name in ('src','scripts')
                  for p in sorted((source/root_name).rglob('*.py')) if p.is_file()}
    for name,pinned in config['source_members'].items():
        if code_files.get(name) != pinned:
            raise ValueError('D416_STAGED_RUNTIME_BYTES_CHANGED:'+name)
    seal = hashlib.sha256(json.dumps(code_files,sort_keys=True).encode()).hexdigest()
    env = dict(native['native_environment'],PYTHONPATH=':'.join((
        str(source),str(source/'src'),native['native_environment']['PYTHONPATH'])),
        ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES='')
    spec = dict(native,scope=plan.SCOPE,evaluation_stage=stage,plan_commit=plan.PLAN,goal_plan_sha=plan.PLAN,
        source_sha=config['source_sha'],source_root=str(source),source_files=code_files,source_seal=seal,
        registered_root=str(root),campaign_root=str(root),native_environment=env,physics=True,
        deadline_epoch=deadline,campaign_deadline_epoch=deadline,allocation_deadline_epoch=auth['allocation_deadline_epoch'],
        new_optimizer_updates_budget=0,new_training_visits_budget=0,new_physical_requests_budget=caps[0],
        new_model_generations_budget=caps[2],automatic_retry=False,frontend_dependency=False,
        sim_only_project_authorization=config['authority'],neural_owner_lock=gpu['neural_owner_lock'],
        interface_predecessor=config['E0_interface'],compiled_population=config['population'],
        packages=packages,endpoints=results,ordinary_RGB_restores=restores,core_RGB_restores=core_restores,
        source_RGB_restores=source_restores)
    if stage == plan.CONFIRM:
        spec['screen_decision'] = config['screen_decision']
    if config.get('previous_panel_segments'):
        spec['previous_panel_segments'] = config['previous_panel_segments']
    plan.arms_for(spec)
    endpoint_records(spec)
    observed = dict(GPU_free_bytes=free_bytes,host_available_bytes=host_bytes,cpus=cpus,linger=True)
    return spec,gpu,gpu_site,observed


def launch(config):
    spec,gpu,site,observed = build(config)
    root = Path(spec['registered_root'])
    launch = root.parent/(root.name+'-launch')
    launch.mkdir(mode=0o700)
    binding = launch/'SITE.private.json'
    write(binding,dict(site,allocation_deadline_epoch=spec['allocation_deadline_epoch'],
          allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),
                          neural_owner_lock=spec['neural_owner_lock'])))
    template = launch/'GPU_TEMPLATE.private.json'
    write(template,dict(gpu,site_binding=pin(binding),allocation_deadline_epoch=spec['allocation_deadline_epoch']))
    spec['gpu_template'] = pin(template)
    spec_pin = register_student_panel(spec)
    registered = read_pin(spec_pin)
    try:
        preflight(registered)
    except Exception as error:
        for file in [*registered['cases'],registered['resident_server_spec']['path'],spec_pin['path']]:
            close(read(file),1,not_started=True,reason='D416_NATIVE_PREFLIGHT:'+type(error).__name__)
        raise
    unit = 'wm02-'+root.name
    write(launch/'PRE_DISPATCH.private.json',dict(spec=spec_pin,site=observed,native_requests=0))
    receipt = launch_service(registered,unit,'--multitask-student-panel',bootstrap='v6_bootstrap.py')
    actual = unit_state(unit)
    if not actual['pid'] or actual['pid'] not in actual['cgroup_pids']:
        raise RuntimeError('D416_NATIVE_UNKNOWN_START_REQUIRES_ONSITE_RECONCILIATION')
    return dict(receipt,spec=spec_pin,actual=actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(launch(read_pin(dict(path=args.config,sha256=args.sha256))),allow_nan=False))


if __name__ == '__main__':
    main()
