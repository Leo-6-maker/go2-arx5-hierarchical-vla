"""Prepare and execute one contiguous student block on the existing MuJoCo world."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from go2arx5_vla.sla_adapter.multitask_contract_v6 import NATIVE_METHOD
from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
from scripts.pi0292.durable import read, write
from scripts.sla.core_hopper_expert import PLAN, POP
from scripts.sla.core_canonical_proof import VERIFIED
from scripts.sla.core_hopper_native_scope import SCOPE, AUTHORITY, accepted
from scripts.wm02.multitask_stage_cards_v6 import case_card, cold_native_owner, panel_cards, pin, require_unstarted_cases
from scripts.wm02.multitask_student_campaign_v6 import student_row
from scripts.wm02.multitask_teacher_campaign_v6 import interface_records
from scripts.wm02.sim_only_campaign_v51 import fatal_control_violation, run as run_campaign


def locations(folder):
    return {k:str(folder/k) for k in ('tmp','logs','cache','data')}


def prepare(cfg, proof_pin, arm, split, start, root, capacity):
    source = Path(cfg['source_root'])
    original = read_pin(cfg['expert'])
    template = read_pin(cfg['gpu_template'])
    receiver = read_pin(cfg['receiver'])
    numeric = read_pin(cfg['numeric_service'])
    qualified = read_pin(proof_pin)
    rows = qualified['panels'][split][start:start+6]
    ids = [r['id'] for r in rows]
    if (not rows or len(rows)>6 or start not in range(0,24,6) or any(r['request_cap']!=1 for r in rows)
            or numeric['config_sha']!=cfg['receiver']['sha256'] or original['compiled_population']['sha256']!=POP
            or qualified['compiled_population']!=original['compiled_population'] or {r['arm'] for r in rows}!={arm}
            or receiver['gpu']!=template['GPU_uuid'] or receiver['gpu_lock']!=template['neural_owner_lock']
            or cfg['deadline']!=original['campaign_deadline_epoch'] or time.time()+2100>=cfg['deadline']):
        raise PermissionError('Invalid endpoint, world, GPU or remaining allocation')
    interface = cold_native_owner(original['interface_predecessor'])
    interface_records(interface,original['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    require_unstarted_cases(original['registry'],set(ids))
    actual = read_pin(cfg['source_members'])
    if (source != Path(__file__).resolve().parents[2] or original['owner']!=cfg['expert_owner']
            or hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()!=cfg['source_seal']):
        raise ValueError('Native code or registered expert binding changed')
    now=time.time()
    allocation=min(cfg.get('dispatch_deadline',cfg['deadline']),cfg['deadline'],now+len(rows)*1980+1200)
    end=allocation-120
    if end-now < 1980:
        raise TimeoutError('Insufficient time for one original condition')
    layers = original['native_environment']['PYTHONPATH']
    native_env=dict(original['native_environment'],PYTHONPATH=f'{source}:{source}/src:{layers}',CUDA_VISIBLE_DEVICES='',
        TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),ROS_LOG_DIR=str(root/'logs/ros'))
    root.parent.mkdir(parents=True,exist_ok=True)
    dirs(root)
    (root/'cases').mkdir()
    ticks=sum(r['tick_cap'] for r in rows)
    queries=sum(r['query_cap'] for r in rows)
    proof=qualified['gate']
    authority=dict(source=AUTHORITY,owner='EXISTING_LOCAL_PC',plan_commit=PLAN,source_sha=cfg['source'],
        compiled_population_sha256=POP,student_proof=proof_pin,hardware_allowed=False,sim_only=True,
        T0_epoch=cfg['T0'],deadline_epoch=end,case_ids=ids,native_request_cap=len(ids),native_tick_cap=ticks,
        generation_cap=queries,panel_start=start,scope=SCOPE,arm=arm,split=split)
    af=root/'AUTHORITY.private.json'
    write(af,authority)
    payload = Path(cfg['receiver']['path']).parent/'endpoints'/arm/'payload'
    canonical = proof['status']==VERIFIED
    package = proof['package'] if canonical else dict(path=str(payload),sha256=proof['manifest'])
    parent=dict(interface,**locations(root))
    parent.update(owner=authority['owner'],source_sha=cfg['source'],source_root=str(source),source_files=actual,source_seal=cfg['source_seal'],
        scope=SCOPE,registry_scope=SCOPE,plan_commit=PLAN,goal_plan_sha=PLAN,registered_root=str(root),campaign_root=str(root),
        mobile_group=str(root),unattended_control=str(root/'control'),role='SLA_CORE_HOPPER_STUDENT_BLOCK',arm=arm,
        evaluation_split=split,compiled_population=original['compiled_population'],interface_predecessor=original['interface_predecessor'],
        registered_panel=None,student_proof=proof_pin,receiver=cfg['receiver'],sim_only_project_authorization=pin(af),native_environment=native_env,
        packages={arm:package},ordinary_RGB_restores={arm:proof['core_result']},new_optimizer_updates_budget=0,new_training_visits_budget=0,
        new_physical_requests_budget=len(ids),new_native_tick_budget=ticks,new_model_generations_budget=queries,physics=True,
        campaign_deadline_epoch=cfg['deadline'],allocation_deadline_epoch=allocation,deadline_epoch=end,
        neural_owner_lock=receiver['gpu_lock'],hardware_connections_allowed=False)
    accepted(parent)
    population=read_pin(original['compiled_population'])
    original_rows={r['condition_id']:r for r in population['conditions'][split]}
    t={c['mobile_condition']['family']:c for c in (read(p) for p in interface['cases'])}
    plan=[(dict(original_rows[r['condition']],noise_seed=r['scene_seed']),arm) for r in rows]
    panel=panel_cards(population,plan,t,role='STUDENT')
    if [r['id'] for r in panel]!=ids:
        raise ValueError('Student panel order changed')
    write(root/'PANEL.private.json',panel)
    parent['registered_panel']=pin(root/'PANEL.private.json')
    parent['cases']=[str(root/'cases'/eid/'EXECUTION_SPEC.json') for eid in ids]
    worker_dir=root/'worker'
    dirs(worker_dir)
    ipc=worker_dir/'ipc'
    ipc.mkdir()
    env=dict(numeric['environment'],CUDA_VISIBLE_DEVICES=receiver['gpu'],JAX_PLATFORMS='cuda',
             XLA_PYTHON_CLIENT_MEM_FRACTION=str(receiver['memory_fraction']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             TMPDIR=str(worker_dir/'tmp'),XDG_CACHE_HOME=str(worker_dir/'cache'),
             PYTHONPATH=numeric['environment']['PYTHONPATH']+f':{source}:{source}/src')
    if canonical:
        # The native supervisor carries the frozen WBC loader paths. The
        # resident numerical process must inherit only its qualified loader.
        env.update(SLA_CORE_CANONICAL_WORKER='1',
                   LD_LIBRARY_PATH=numeric['environment'].get('LD_LIBRARY_PATH',''),
                   LD_PRELOAD=numeric['environment'].get('LD_PRELOAD',''))
    gpu = dict(parent,**locations(worker_dir))
    for k in ('parent_params','parent_manifest','tokenizer','initial_function_rule','upstream_sha','dataset',
              'RPC_timeout_s','save_and_exit_on_limit','frame_cache_bytes'):
        gpu[k]=template[k]
    if (gpu['parent_manifest']!=receiver['model']['parent_manifest'] or gpu['parent_params']!=receiver['model']['parent_params']
            or gpu['tokenizer']!=receiver['tokenizer']):
        raise ValueError('GPU template and qualified model foundation or tokenizer differ')
    reserve=math.ceil(receiver['memory_fraction']*capacity['total_bytes'])
    gpu.update(registered_root=str(worker_dir),role='SLA_CORE_HOPPER_STUDENT_INFERENCE',physics=False,
        new_physical_requests_budget=0,new_native_tick_budget=0,selected_combo=NATIVE_METHOD,archive_all_branches=False,
        model_service_timeout_s=30,new_complete_generations_budget=queries,unified_requests_budget=queries,
        live_root=str(worker_dir),ipc_root=str(ipc),GPU_uuid=receiver['gpu'],max_gpu_hours=4,max_walltime_s=14400,
        python=numeric['python'],parent_host_cast_v7=True,native_environment=env,gpu_environment=env,
        allocator_limit_bytes=reserve,minimum_host_available_bytes=template['minimum_host_available_bytes'],
        JAX_memory_fraction=receiver['memory_fraction'],compilation_cache_dir=str(worker_dir/'cache'))
    gpu['request_bindings_by_arm']={arm:dict(arm=arm,policy_package_sha256=package['sha256'],
        statistics_sha256=receiver['action']['sha256'],codec=NATIVE_METHOD,source_sha=cfg['source'],
        plan_commit=PLAN,compiled_population_sha256=POP)}
    site=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=original['host'],uid=original['uid'],
        GPU_uuid=receiver['gpu'],GPU_name=capacity['name'],GPU_headroom_bytes=2*1024**3,
        host_headroom_bytes=template['minimum_host_available_bytes'],allocator_limit_bytes=reserve,
        allocation_deadline_epoch=allocation,allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),
        neural_owner_lock=receiver['gpu_lock']),paths={k:str(Path(gpu[k]).resolve(strict=True)) for k in ('parent_params','dataset','compilation_cache_dir')})
    write(worker_dir/'SITE.json',site)
    gpu['site_binding']=pin(worker_dir/'SITE.json')
    cache=Path(cfg['receiver']['path']).parent/'jax-cache'
    if not canonical:
        for entry in cache.iterdir():
            if entry.is_file() and not entry.is_symlink():
                shutil.copy2(entry,worker_dir/'cache'/entry.name)
    write(worker_dir/'EXECUTION_SPEC.json',gpu)
    parent['resident_server_spec']=pin(worker_dir/'EXECUTION_SPEC.json')
    children=[]
    for item in panel:
        folder,case=case_card(parent,t[item['condition']['family']],item,parent['registered_panel'])
        case.update(student_proof=proof_pin,receiver=cfg['receiver'],compiled_population=parent['compiled_population'],
                    evaluation_split=split,registry_scope=SCOPE)
        dirs(folder)
        validate(case)
        children.append((folder,case))
    register(parent,[(root,parent),(worker_dir,gpu),*children])
    return parent


def run_block(path, spec):
    def preflight(parent):
        accepted(parent)
        interface=cold_native_owner(parent['interface_predecessor'])
        prior=interface_records(interface,parent['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
        for card in parent['cases']:
            validate(read(card))
        return prior
    return run_campaign(path,preflight=preflight,bootstrap='v6_bootstrap.py',expected_count=len(spec['cases'])+2,
        result_schema='SLA_CORE_HOPPER_NATIVE_STUDENT_RESULT_V1',pilot_evaluate=lambda _:False,
        load_carried=lambda rows:rows,compact=student_row,case_lead_s=1980)


def audit_block(root):
    """Every paid native request and IPC generation must have a terminal receipt."""
    r=read(root/'RESULT.private.json')
    cards=read(root/'EXECUTION_SPEC.json')['cases']
    students=[s for s in r['records'] if s['role']=='STUDENT']
    if (r['exit_code'] or r['unresolved_case_launches'] or len(students)!=len(cards)
            or r['new_verified_native_requests']!=len(students)):
        raise RuntimeError('Reconcile technical or partial block before dispatching a suffix')
    generations=ticks=0
    for card,row in zip(cards,students,strict=True):
        child=Path(card).parent
        result=read_pin(dict(path=str(child/'RESULT.private.json'),sha256=row['result_sha256']))
        if (row['id']!=child.name or result['physical_requests']!=1 or row['technical_failure'] or fatal_control_violation(row)
                or (row['hard_STOP'] and row['integrations_after_STOP']!=0)
                or result['actual_native_integrations']>read(card)['max_ticks'] or result['optimizer_updates']):
            raise RuntimeError('Student control, case identity or denominator differs')
        ticks+=result['actual_native_integrations']
        generations+=result['model_generations']
    worker=r['model_service']
    if not worker or worker['exit_code'] or not(worker['attempted_generations']==worker['completed_generations']==generations):
        raise RuntimeError('Unresolved student generation counts')
    return dict(block=pin(root/'RESULT.private.json'),requests=len(students),ticks=ticks,generations=generations,
                success=sum(s['task_and_mobile_success'] for s in students),records=[s['result_sha256'] for s in students])
