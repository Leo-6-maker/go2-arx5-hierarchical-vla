"""Finite server-side E1 -> RGB/offline -> E2 successor, using existing owners.

No retry, scientific selection, new training, confirmation, DEV or TEST is
scheduled here. A failed child is terminal and its unstarted suffix stays so.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,close,unit_state
from go2arx5_vla.sla_adapter.mobile_successor_v52 import PLAN,PARENT


def read(path):return json.loads(Path(path).read_bytes())
def pin(path):return dict(path=str(path),sha256=digest(path))


def common(spec,root,role):
    value={k:spec[k]for k in('source_sha','source_root','source_files','trusted_source_parent','source_seal','registry','owner','host','uid','world_control','plan_commit','goal_plan_sha','allocation_deadline_epoch','campaign_deadline_epoch','deadline_epoch','python')}
    value.update(registered_root=str(root),campaign_root=str(root),role=role,scope='SLA_MOBILE_F2_V52',registry_scope='SLA_MOBILE_F2_V52',cpus=[12,13],ROS_DOMAIN_ID=198,automatic_retry=False,frontend_dependency=False,WBC_changes=False,safety_changes=False,real_hardware=False,RL=False,optimizer_updates=0,new_optimizer_updates_budget=0,new_training_visits_budget=0)
    return value


def site_binding(spec,root,campaign):
    if digest(spec['site_binding']['path'])!=spec['site_binding']['sha256']:raise ValueError('SOURCE_SITE_BINDING_CHANGED')
    value=read(spec['site_binding']['path']);value['paths']={k:str(Path(spec[k]).resolve(strict=True))for k in('parent_params','dataset','compilation_cache_dir')}
    value['allocation']=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(campaign),neural_owner_lock=spec['neural_owner_lock'])
    value['permission']='CURRENT_USER_EXPLICIT_PR404_V52_BOUNDED_OFFLINE_AND_E2'
    write(root/'SITE_BINDING.private.json',value);return pin(root/'SITE_BINDING.private.json')


def wait_endpoints(spec):
    if any(digest(p['path'])!=p['sha256']for p in spec['E1_specs'].values()):raise ValueError('SEALED_E1_SPEC_CHANGED')
    root=Path(spec['registered_root']);ends={a:Path(p['path']).parent for a,p in spec['E1_specs'].items()}
    while time.time()<spec['deadline_epoch']-1800 and not(root/'STOP.json').exists():
        complete={}
        for arm,folder in ends.items():
            if not(folder/'RESULT.private.json').exists():continue
            result=read(folder/'RESULT.private.json')
            if result.get('exit_code')!=0:raise RuntimeError('E1_CHILD_TERMINAL_NO_AUTOMATIC_REPAIR_'+arm)
            state=unit_state('wm02-'+folder.name)
            if state['pid']or state['cgroup_pids']or read(folder/'control/LEASE.json')['status']!='CLOSED':continue
            if (result['status']!='FIXED500_COMPLETE'or result['new_updates']!=500 or result['new_visits']!=16000
                    or not result['full_restore']['bit_exact']or result['full_restore']['arrays']!=87
                    or result['parent_package']['sha256']!=PARENT or not result['frozen_fingerprint_unchanged']):
                raise ValueError('TWO_REAL_MATCHED500_RESTORED_ENDPOINTS_REQUIRED')
            complete[arm]=result
        write(root/'PROGRESS.private.json',dict(phase='WAIT_ACTUAL_E1_ENDPOINTS',epoch=time.time(),complete_arms=sorted(complete),new_generations=0,new_native_requests=0))
        if len(complete)==2:
            initials={a:read(p/'INITIAL_MATCH.private.json')for a,p in ends.items()}
            for key in('parent_package','all_state_array_hashes','visit_map_sha256','LR_sha256','noise_sha256','tau_sha256','state_dropout','first_indices','fresh_Adam'):
                if initials['B1'][key]!=initials['B2'][key]:raise ValueError('ACTUAL_MATCHED_INITIAL_STATE_MISMATCH_'+key)
            write(root/'E1_MATCHED_ENDPOINTS.private.json',dict(all87_initial_arrays_equal=True,matched_keys=list(initials['B1']),training_results={a:pin(p/'RESULT.private.json')for a,p in ends.items()}))
            return complete
        time.sleep(30)
    raise TimeoutError('FINITE_FOLLOWUP_NO_ENDPOINT_REPLAY')


def run_child(spec,child,operation,journal,message):
    root=Path(spec['registered_root']);folder=Path(child['registered_root']);path=folder/'EXECUTION_SPEC.json'
    def execute():
        with(folder/'logs/followup-child.log').open('xb')as log:
            process=subprocess.Popen([child['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/mobile_v52_bootstrap.py'),operation,str(path)],env=dict(os.environ,**child['native_environment']),stdout=log,stderr=subprocess.STDOUT)
        write(root/'CURRENT_CHILD.private.json',dict(pid=process.pid,operation=operation,spec=pin(path),epoch=time.time()))
        while process.poll()is None:
            if(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-180:
                write(folder/'STOP.json',dict(reason='FINITE_PARENT_DEADLINE_OR_OPERATOR_STOP'))
            time.sleep(5)
        result=folder/'RESULT.private.json'
        if not result.exists():raise RuntimeError('CHILD_TERMINAL_WITHOUT_RESULT_NO_REPLAY')
        return dict(result=pin(result),exit_code=process.returncode)
    actual=journal.once(message,dict(spec=pin(path),operation=operation),execute)
    if actual['exit_code']or read(actual['result']['path'])['exit_code']:
        raise RuntimeError('PRESERVED_CHILD_FAILURE_NO_AUTOMATIC_RETRY_'+message)
    return actual


def prepare_offline(spec,endpoints):
    root=Path(spec['registered_root'])/'offline';dirs(root)
    e1=read(spec['E1_specs']['B1']['path']);s=dict(e1);s.update(common(spec,root,'SLA_MOBILE_V52_E1_OFFLINE600'))
    training={a:pin(Path(p['path']).parent/'RESULT.private.json')for a,p in spec['E1_specs'].items()}
    training['B0']=spec['B0_training_result'];packages={a:r['package']for a,r in endpoints.items()};packages['B0']=read(training['B0']['path'])['package']
    s.update(physics=False,new_physical_requests_budget=0,new_complete_generations_budget=600,new_model_generations_budget=600,packages=packages,training_results=training,CN38_bank=spec['CN38_bank'],B0_R1_records=spec['B0_R1_records'],ROLL96=spec['ROLL96'],evaluation_split=spec['evaluation_split'],compilation_cache_dir=str(root/'cache'))
    env=dict(e1['native_environment']);env.update(TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),ROS_LOG_DIR=str(root/'logs/ros'));s['native_environment']=env
    s['site_binding']=site_binding(s,root,Path(spec['registered_root']));register(spec,[(root,s)])
    return s


def prepare_screen(spec,offline,offline_result):
    if digest(spec['V51_terminal_spec']['path'])!=spec['V51_terminal_spec']['sha256']or digest(spec['E2_screen']['path'])!=spec['E2_screen']['sha256']:
        raise ValueError('ORIGINAL_TERMINAL_OR_PREDECLARED_PANEL_CHANGED')
    root=Path(spec['registered_root'])/'E2';dirs(root);(root/'cases').mkdir();old=read(spec['V51_terminal_spec']['path'])
    all_cases=[read(x['spec']['path'])for x in old.get('prior_case_results',[])]+[read(x)for x in old['cases']]
    templates={s['mobile_condition']['family']:s for s in all_cases if s['sim_only_role']=='STUDENT'}
    prior=[dict(spec=pin(Path(s['registered_root'])/'EXECUTION_SPEC.json'),result=pin(Path(s['registered_root'])/'RESULT.private.json'))for s in all_cases if s['sim_only_role']=='INTERFACE']
    old_server=read(old['resident_server_spec']['path']);server_root=root/'worker';dirs(server_root);(server_root/'ipc').mkdir()
    server=dict(offline);server.update(common(spec,server_root,'SLA_MOBILE_V52_E2_SINGLE_MODEL_POOL'))
    server.update(campaign_root=str(root),physics=False,new_physical_requests_budget=0,new_complete_generations_budget=2880,new_model_generations_budget=2880,unified_requests_budget=2880,ipc_root=str(server_root/'ipc'),live_root=str(server_root),selected_combo='SHIFT7_SINGLE',archive_all_branches=False,SHIFT7_package=offline['packages']['B0'])
    server['request_bindings_by_arm']={a:dict(old_server['request_bindings'],source_sha=spec['source_sha'],policy_package_sha256=p['sha256'],arm=a,plan_commit=PLAN)for a,p in offline['packages'].items()}
    server['gpu_environment']=dict(offline['native_environment']);server['native_environment']=dict(offline['native_environment']);server['site_binding']=site_binding(server,server_root,root)
    authority=root/'PROJECT_AUTHORIZATION.private.json';write(authority,dict(schema='USER_PROJECT_AUTHORIZED_ISOLATED_MUJOCO_V52',source='CURRENT_USER_EXPLICIT_PR404_V52_INSTRUCTION',source_sha=spec['source_sha'],plan_commit=PLAN,hardware_allowed=False,deadline_epoch=spec['deadline_epoch'],new_native_requests_max=24,full_H40_P10_TTL_STOP_unchanged=True))
    # Register the immutable worker first so its actual spec SHA binds every case.
    register(spec,[(server_root,server)]);server_pin=pin(server_root/'EXECUTION_SPEC.json')
    panel=read(spec['E2_screen']['path']);entries=[];paths=[]
    for row in panel:
        folder=root/'cases'/row['id'];dirs(folder);s=dict(templates[row['family']]);previous_model=s['model_path'];s.update(common(spec,folder,'SLA_MOBILE_V52_E2_'+row['arm']))
        for key in('old_condition_result','old_V3_seed','previous_v2_result','same_old_scene_new_expert_condition'):s.pop(key,None)
        s.update(mobile_group=str(root),campaign_root=str(root),mobile_experiment_version='MFREE_SHIFT7_SIM_ONLY_V52',sim_only_role='STUDENT',sim_only_project_authorization=pin(authority),native_backend='MUJOCO_ONLY',hardware_connections_allowed=False,physics=True,max_ticks=35703,physics_dt_s=.002,model_service_timeout_s=30,new_physical_requests_budget=1,cpus=[10,11],arm=row['arm'],packages=offline['packages'],SHIFT7_package=offline['packages'][row['arm']],ordinary_RGB_restore=offline_result['ordinary_RGB_restore'][row['arm']],resident_server_spec=server_pin,registered_panel=spec['E2_screen'],panel_entry_id=row['id'],model_path=row['scene']['path'],language=row['language'],expert_training_eligible=False,unattended_control=str(root/'control'))
        s['frozen_files']=dict(s['frozen_files']);s['frozen_files'].pop(previous_model,None);s['frozen_files'][s['model_path']]=row['scene']['sha256']
        s['mobile_condition']=dict(s['mobile_condition'],family=row['family'],layout=row['layout'],reset=row['reset'],seed=row['noise_seed'],collection_episode=row['id'])
        s['native_environment']=dict(s['native_environment']);s['native_environment'].update(ROS_DOMAIN_ID='198',ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES='',TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),ROS_LOG_DIR=str(folder/'logs/ros'))
        for name in('tmp','cache','logs','data'):s[name]=str(folder/name)
        entries.append((folder,s));paths.append(str(folder/'EXECUTION_SPEC.json'))
    parent=dict(old);parent.update(common(spec,root,'SLA_MOBILE_V52_E2_24_CONTROLLER'));parent.update(physics=True,new_physical_requests_budget=24,cases=paths,prior_case_results=prior,resident_server_spec=server_pin,registered_panel=spec['E2_screen'],native_environment=dict(spec['native_environment']),cpus=[10,11,12,13],mobile_world_lock=entries[0][1]['mobile_world_lock'])
    register(spec,[(root,parent),*entries])
    from scripts.wm02.mobile_campaign_v52 import preflight
    from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
    preflight(parent)
    for folder,case in entries:TargetSequentialAdapter(case,folder/'native-01')
    write(root/'ZERO_INTEGRATION_PREFLIGHT.private.json',dict(cases=24,actual_adapter_constructors=24,process_starts=0,new_native_requests=0,new_integrations=0))
    return parent


def main(spec):
    root=Path(spec['registered_root']);j=Journal(root/'control/journal',root.name);result={}
    try:
        if (spec['scope']!='SLA_MOBILE_F2_V52'or spec['plan_commit']!=PLAN
                or spec['new_physical_requests_budget']!=24 or spec['new_complete_generations_budget']!=600
                or spec['new_optimizer_updates_budget']):raise PermissionError('EXACT_E1_FOLLOWUP_AUTHORIZED_BOUND')
        endpoints=wait_endpoints(spec)
        receipt=j.once('REGISTER-OFFLINE',dict(endpoints={a:r['package']for a,r in endpoints.items()}),lambda:dict(spec=pin(Path(prepare_offline(spec,endpoints)['registered_root'])/'EXECUTION_SPEC.json')))
        offline=read(receipt['spec']['path']);actual=run_child(spec,offline,'--offline',j,'RUN-OFFLINE');qualification=read(actual['result']['path'])
        receipt=j.once('REGISTER-E2',dict(offline=actual['result']),lambda:dict(spec=pin(Path(prepare_screen(spec,offline,qualification)['registered_root'])/'EXECUTION_SPEC.json')))
        panel=read(receipt['spec']['path']);actual=run_child(spec,panel,'--campaign',j,'RUN-E2')
        result=dict(exit_code=0,status='E1_OFFLINE_AND_E2_TERMINAL_REQUIRES_ACTUAL_SELECTION',offline=pin(Path(offline['registered_root'])/'RESULT.private.json'),E2=actual['result'],new_updates=0,automatic_confirmation_DEV_TEST_E3=False)
    except BaseException:result=dict(exit_code=1,status='PRESERVED_NO_REPLAY',failure=traceback.format_exc(),automatic_retry=False)
    write(root/'RESULT.private.json',result);close(spec,result['exit_code']);return result['exit_code']
