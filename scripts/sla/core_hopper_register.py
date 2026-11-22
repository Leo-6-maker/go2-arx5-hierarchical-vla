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
    multienv=config.get('multienv')
    if multienv:
        from scripts.sla import core_multienv_scope as me
        plan, scope, authority_source = me.PLAN, me.SCOPE, me.AUTHORITY
    else:
        plan, scope, authority_source = PLAN, SCOPE, AUTHORITY
    mount=read_pin(config['mount'])
    if (mount['plan']!=plan or (multienv and mount['deadline_epoch']>mount['T0_epoch']+7*86400)
            or (not multienv and mount['T0_epoch']+14*86400!=mount['deadline_epoch'])
            or time.time()+1980>=mount['deadline_epoch']):
        raise ValueError('Campaign deadline or registration mismatch')
    interface=cold_native_owner(config['interface'])
    interface_records(interface,interface['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    if subprocess.check_output(['loginctl','show-user','2009','-p','Linger'],text=True).strip()!='Linger=yes':
        raise PermissionError('User service persistence unverified')
    repair=config.get('carry_repair')
    if multienv:
        previous = cold_native_owner(multienv['handoff_owner'])
        if (previous['owner'] != OWNER or previous['scope'] != SCOPE or previous['plan_commit'] != PLAN
                or previous['mobile_world_lock'] != interface['mobile_world_lock']
                or previous['host'] != interface['host']
                or not (Path(previous['registered_root'])/'RESULT.private.json').is_file()):
            raise ValueError('Expected the closed Core owner and the original shared world lock')
    elif not repair:
        lock=OwnerLock(interface['mobile_world_lock']); lock.close()
    if not multienv and config['population']['sha256']!=POP:
        raise ValueError('Geometry seal mismatch')
    pop=read_pin(config['population'])
    if multienv:
        if pop['wave']!='A' or repair:
            raise ValueError('Expected the original Teacher on the fixed Wave A geometry')
        rows=me.order(pop)
    elif repair:
        from scripts.sla.core_hopper_expert import repair_rows
        rows=repair_rows(pop,repair)
    else:
        rows=order(pop)
    closure=read_pin(config['source_members'])
    if any(digest(source/name)!=expected for name,expected in closure.items()):
        raise ValueError('Source closure changed')
    if repair and 'completed_pilot' in repair:
        from scripts.sla.core_hopper_repair_series import identity
        _,pilot=identity(repair['completed_pilot'])
        base=pilot['source_files']
        allowed={'scripts/sla/core_hopper_collection.py','scripts/sla/core_hopper_expert.py','scripts/sla/core_hopper_register.py'}
        if (set(closure)!=set(base)|{'scripts/sla/core_hopper_repair_series.py'}
                or any(closure[k]!=v for k,v in base.items() if k not in allowed)):
            raise ValueError('Continued collection must preserve the successful pilot and its complete native stack')
    require_unstarted_cases(interface['registry'],{row['id'] for row in rows})
    dirs(root)
    (root/'cases').mkdir(); (root/'snapshots').mkdir()
    source_sha=config['source_sha']
    end=mount['deadline_epoch']
    authority=root/'AUTHORITY.private.json'
    extra = {} if not multienv else dict(wave='A',goal_start=mount['T0_epoch'],
        compiled_population=config['population'],case_ids=[r['id'] for r in rows])
    write(authority,dict(source=authority_source,plan_commit=plan,source_sha=source_sha,frozen_producer=source_sha,
        compiled_population_sha256=config['population']['sha256'],hardware_allowed=False,native_request_cap=len(rows),
        native_tick_cap=sum(row['native_max_ticks'] for row in rows),T0_epoch=mount['T0_epoch'],
        deadline_epoch=end,owner=OWNER,sim_only=True,**extra,**({'carry_repair':repair} if repair else {})))
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
              owner=OWNER,scope=scope,registry_scope=scope,goal_plan_sha=plan,plan_commit=plan,
              plan_name='SLA-CORE-MULTIENV-DATA-V1' if multienv else 'SLA-CORE-HOPPER-PARALLEL-V1',registered_root=str(root),campaign_root=str(root),
              mobile_group=str(root),collection_root=str(root/'admission'),unattended_control=str(root/'control'),
              native_environment=env,role=scope,compiled_population=config['population'],
              registered_panel=pin(pf),interface_predecessor=config['interface'],sim_only_project_authorization=pin(authority),
              admission_python=config['admission_python'],C20_statistics_path=config['statistics'],physics=True,hardware_connections_allowed=False,
              resident_server_spec=None,new_optimizer_updates_budget=0,new_training_visits_budget=0,
              new_model_generations_budget=0,new_physical_requests_budget=len(rows),allocation_deadline_epoch=None,
              campaign_deadline_epoch=end,deadline_epoch=end,automatic_retry=False,frontend_dependency=False,
              cases=[str(root/'cases'/r['id']/'EXECUTION_SPEC.json') for r in rows])
    subprocess.run([spec['admission_python'],'-B','-c','import scripts.sla.core_hopper_admission'],
                   cwd=source,env=dict(os.environ,**env),check=True)
    subprocess.run([spec['python'],'-B','-c','import scripts.sla.core_hopper_collection'],
                   cwd=source,env=dict(os.environ,**env),check=True)
    spec.update({k:str(root/k) for k in ('logs','cache','tmp','data')})
    if repair:
        spec['carry_repair']=repair
    if multienv:
        spec['multienv']=multienv
    for item,row in zip(panel,rows,strict=True):
        _,case=case_card(spec,templates[row['family']],item,spec['registered_panel'])
        case.update(registry_scope=scope,collection_split=row['split'],compiled_population=config['population'],
                    allocation_deadline_epoch=min(end,time.time()+2160),deadline_epoch=min(end,time.time()+1980),
                    expert_training_eligible=row['split']=='TRAIN')
        if repair:
            case['carry_repair']=repair
        validate(case)
    write(root/'PREFLIGHT.private.json',dict(validated_cards=len(rows),native_requests=0,source_sha=source_sha,
                                             population_sha256=config['population']['sha256'],plan=plan))
    register(spec,[(root,spec)])
    launch=launch_service(spec,'wm02-'+root.name,'--core-hopper-collection',bootstrap='v6_bootstrap.py')
    print(json.dumps(launch))


if __name__=='__main__':
    os.umask(0o077)
    prepare(read_pin(dict(path=sys.argv[1],sha256=sys.argv[2])))
