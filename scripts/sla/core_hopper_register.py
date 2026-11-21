"""Register the finite expert chain once on the existing isolated Core user service."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service, register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs
from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import OwnerLock, write
from scripts.wm02.multitask_stage_cards_v6 import case_card, cold_native_owner, panel_cards, pin, require_unstarted_cases
from scripts.wm02.multitask_teacher_campaign_v6 import interface_records
from scripts.sla.core_hopper_expert import AUTHORITY, OWNER, PLAN, POP, SCOPE, order


def prepare(config):
    root,source=Path(config['root']),Path(config['source_root'])
    if (os.getuid()!=2009 or os.uname().nodename!='corelab-4080s' or root.exists()
            or source!=Path(__file__).resolve().parents[2] or root.parent.stat().st_uid!=os.getuid()):
        raise PermissionError('Only an unopened Core role under the verified owner')
    mount=read_pin(config['mount'])
    if (mount['plan']!=PLAN or mount['T0_epoch']+14*86400!=mount['deadline_epoch']
            or time.time()+1980>=mount['deadline_epoch']):
        raise ValueError('Campaign deadline or registration mismatch')
    interface=cold_native_owner(config['interface'])
    interface_records(interface,interface['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    if subprocess.check_output(['loginctl','show-user','2009','-p','Linger'],text=True).strip()!='Linger=yes':
        raise PermissionError('User service persistence unverified')
    lock=OwnerLock(interface['mobile_world_lock']); lock.close()
    if config['population']['sha256']!=POP:
        raise ValueError('Geometry seal mismatch')
    pop=read_pin(config['population']); rows=order(pop)
    closure=read_pin(config['source_members'])
    if any(digest(source/name)!=expected for name,expected in closure.items()):
        raise ValueError('Source closure changed')
    require_unstarted_cases(interface['registry'],{row['id'] for row in rows})
    dirs(root)
    (root/'cases').mkdir(); (root/'snapshots').mkdir()
    source_sha=config['source_sha']
    end=mount['deadline_epoch']
    authority=root/'AUTHORITY.private.json'
    write(authority,dict(source=AUTHORITY,plan_commit=PLAN,source_sha=source_sha,frozen_producer=source_sha,
        compiled_population_sha256=POP,hardware_allowed=False,native_request_cap=len(rows),
        native_tick_cap=sum(row['native_max_ticks'] for row in rows),T0_epoch=mount['T0_epoch'],
        deadline_epoch=end,owner=OWNER,sim_only=True))
    templates={s['mobile_condition']['family']:s for s in (read_pin(pin(Path(p))) for p in interface['cases'])}
    planned=[(dict(r,noise_seed=r['scene_seed']),None) for r in rows]
    panel=panel_cards(pop,planned,templates,role='EXPERT')
    pf=root/'REGISTERED_PANEL.private.json';write(pf,panel)
    env=dict(interface['native_environment'])
    env.update(PYTHONPATH=':'.join((str(source),str(source/'src'),env['PYTHONPATH'])),
        TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),ROS_LOG_DIR=str(root/'logs/ros'),
        CUDA_VISIBLE_DEVICES='',ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
    spec=dict(interface,source_sha=source_sha,source_root=str(source),source_files=closure,
              source_seal=hashlib.sha256(json.dumps(closure,sort_keys=True).encode()).hexdigest(),
              owner=OWNER,scope=SCOPE,registry_scope=SCOPE,goal_plan_sha=PLAN,plan_commit=PLAN,
              plan_name='SLA-CORE-HOPPER-PARALLEL-V1',registered_root=str(root),campaign_root=str(root),
              mobile_group=str(root),collection_root=str(root/'admission'),unattended_control=str(root/'control'),
              native_environment=env,role=SCOPE,compiled_population=config['population'],
              registered_panel=pin(pf),interface_predecessor=config['interface'],sim_only_project_authorization=pin(authority),
              C20_statistics_path=config['statistics'],physics=True,hardware_connections_allowed=False,
              resident_server_spec=None,new_optimizer_updates_budget=0,new_training_visits_budget=0,
              new_model_generations_budget=0,new_physical_requests_budget=len(rows),allocation_deadline_epoch=end,
              campaign_deadline_epoch=end,deadline_epoch=end,automatic_retry=False,frontend_dependency=False,
              cases=[str(root/'cases'/r['id']/'EXECUTION_SPEC.json') for r in rows])
    spec.update({k:str(root/k) for k in ('logs','cache','tmp','data')})
    for item,row in zip(panel,rows,strict=True):
        _,case=case_card(spec,templates[row['family']],item,spec['registered_panel'])
        case.update(registry_scope=SCOPE,collection_split=row['split'],compiled_population=config['population'],
                    expert_training_eligible=row['split']=='TRAIN')
        validate(case)
    write(root/'PREFLIGHT.private.json',dict(validated_cards=len(rows),native_requests=0,source_sha=source_sha,
                                             population_sha256=POP,plan=PLAN))
    register(spec,[(root,spec)])
    launch=launch_service(spec,'wm02-'+root.name,'--core-hopper-collection',bootstrap='v6_bootstrap.py')
    print(json.dumps(launch))


if __name__=='__main__':
    os.umask(0o077)
    prepare(read_pin(dict(path=sys.argv[1],sha256=sys.argv[2])))
