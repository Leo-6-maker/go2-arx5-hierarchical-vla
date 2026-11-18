"""Finite target endpoint -> fixed offline -> new24 -> conditional confirm8."""
import copy,json,os,subprocess,time,traceback
from pathlib import Path
from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,close,unit_state
from go2arx5_vla.sla_adapter.target_contract_v6 import PLAN,B1


def read(path):return json.loads(Path(path).read_bytes())
def pin(path):return dict(path=str(path),sha256=digest(path))
def pinned(value):
    if digest(value['path'])!=value['sha256']:raise ValueError('PINNED_V6_TARGET_PREDECESSOR_CHANGED')
    return read(value['path'])


def common(spec,root,role):
    value={k:spec[k]for k in('source_sha','source_root','source_files','trusted_source_parent','source_seal','registry','owner','host','uid','world_control','plan_commit','goal_plan_sha','allocation_deadline_epoch','campaign_deadline_epoch','deadline_epoch','python')}
    value.update(registered_root=str(root),campaign_root=str(root),role=role,scope='SLA_V6_TARGET_NATIVE',registry_scope='SLA_V6_TARGET_NATIVE',
        cpus=[12,13],ROS_DOMAIN_ID=198,automatic_retry=False,frontend_dependency=False,WBC_changes=False,safety_changes=False,real_hardware=False,RL=False,
        optimizer_updates=0,new_optimizer_updates_budget=0,new_training_visits_budget=0)
    return value


def site_binding(spec,root,campaign):
    value=pinned(spec['site_binding']);value['paths']={k:str(Path(spec[k]).resolve(strict=True))for k in('parent_params','dataset','compilation_cache_dir')}
    value['allocation']=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(campaign),neural_owner_lock=spec['neural_owner_lock']);value['permission']='CURRENT_USER_EXPLICIT_FIXED_SLA_V6_TARGET_BOUNDED'
    write(root/'SITE_BINDING.private.json',value);return pin(root/'SITE_BINDING.private.json')


def wait_dependencies(spec):
    root=Path(spec['registered_root']);pair=pinned(spec['target_pair']);source=pinned(spec['queued_source_campaign'])
    while time.time()<spec['deadline_epoch']-3600 and not(root/'STOP.json').exists():
        ready=True
        for s in(pair,source):
            folder=Path(s['registered_root']);lease=read(folder/'control/LEASE.json');state=unit_state('wm02-'+folder.name)
            ready&=lease['status']=='CLOSED'and not state['pid']and not state['cgroup_pids']and(folder/'RESULT.private.json').exists()
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='WAIT_EXISTING_T_AND_S_CAPACITY_COLD_BOUNDARY',new_updates=0,new_generations=0,new_native_requests=0))
        if ready:
            terminal=read(Path(pair['registered_root'])/'RESULT.private.json')
            if terminal['exit_code']:raise RuntimeError('T_PAIR_TECHNICAL_TERMINAL_NO_RETRAIN_OR_SCREEN')
            endpoints={}
            for arm in('T-FM','T-H40'):
                path=Path(pair['registered_root'])/arm/'RESULT.private.json';result=read(path)
                if result['exit_code']or result['new_updates']!=500 or result['new_visits']!=16000 or not result['full_restore']['bit_exact']:
                    raise ValueError('TWO_REAL_MATCHED_FIXED500_ENDPOINTS_REQUIRED')
                endpoints[arm]=pin(path)
            return endpoints
        time.sleep(30)
    raise TimeoutError('FINITE_TARGET_FOLLOWUP_QUEUE_LIMIT_NO_PREEMPTION')


def run_child(spec,child,route,journal,key):
    root=Path(spec['registered_root']);folder=Path(child['registered_root']);path=folder/'EXECUTION_SPEC.json'
    def execute():
        with(folder/'logs/followup-child.log').open('xb')as log:
            process=subprocess.Popen([child['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/v6_bootstrap.py'),route,str(path)],
                env=dict(os.environ,**child['native_environment']),stdout=log,stderr=subprocess.STDOUT)
        write(root/'CURRENT_CHILD.private.json',dict(pid=process.pid,route=route,spec=pin(path),epoch=time.time()))
        while process.poll()is None:
            if time.time()>=spec['deadline_epoch']-300 or(root/'STOP.json').exists():write(folder/'STOP.json',dict(reason='FINITE_TARGET_PARENT_LIMIT_OR_STOP'))
            time.sleep(5)
        result=folder/'RESULT.private.json'
        if not result.exists():raise RuntimeError('TARGET_CHILD_TERMINAL_WITHOUT_RESULT_RECONCILE_NO_REPLAY')
        return dict(result=pin(result),process_exit=process.returncode)
    actual=journal.once(key,dict(spec=pin(path),route=route),execute)
    if actual['process_exit']or pinned(actual['result'])['exit_code']:raise RuntimeError('PRESERVED_TARGET_CHILD_TECHNICAL_FAILURE_'+key)
    return actual


def prepare_offline(spec,endpoints):
    root=Path(spec['registered_root'])/'offline';dirs(root);template=pinned(spec['target_pair']);training=dict(endpoints);training['T-B1']=spec['B1_training_result']
    packages={arm:pinned(value)['package']for arm,value in training.items()}
    if packages['T-B1']['sha256']!=B1:raise ValueError('FROZEN_REAL_B1_ENDPOINT_REQUIRED')
    e1=read(Path(template['registered_root'])/'T-FM/EXECUTION_SPEC.json');s=dict(e1);s.update(common(spec,root,'SLA_V6_TARGET_OFFLINE615'))
    s.update(physics=False,new_physical_requests_budget=0,new_complete_generations_budget=615,new_model_generations_budget=615,packages=packages,training_results=training,
        CN38_bank=spec['CN38_bank'],ROLL96=spec['ROLL96'],E2_REJECT5=spec['E2_REJECT5'],B1_saved_offline=spec['B1_saved_offline'],historical_reject3_ids=spec['historical_reject3_ids'],compilation_cache_dir=str(root/'cache'))
    s['native_environment']=dict(e1['native_environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'));s['site_binding']=site_binding(s,root,Path(spec['registered_root']));register(spec,[(root,s)])
    return s


def prepare_native(spec,offline,qualification,stage,arm=None,screen_decision=None):
    population=pinned(spec['T_population']);items=[r for r in population['conditions']if r['phase']==stage];count=24 if stage=='SCREEN'else 8
    if len(items)!=8:raise ValueError('EXACT_NEW_EIGHT_MATCHED_CONDITIONS_REQUIRED')
    root=Path(spec['registered_root'])/stage;dirs(root);(root/'cases').mkdir();panel=[]
    for item in items:
        template=pinned(dict(path=item['source_case_spec'],sha256=item['source_case_spec_sha256']))
        if template['frozen_files'][template['model_path']]!=item['source_scene_sha256']:raise ValueError('SEALED_TRAIN_SUPPORT_SCENE_CHANGED')
        for selected in item['arm_order']if stage=='SCREEN'else(arm,):
            panel.append(dict(id=stage+'-'+item['family']+'-'+item['layout']+'-'+selected,arm=selected,family=item['family'],layout=item['layout'],reset=item['reset'],noise_seed=item['noise_seed'],
                language=template['language'],scene=dict(path=template['model_path'],sha256=item['source_scene_sha256']),template_spec=dict(path=item['source_case_spec'],sha256=item['source_case_spec_sha256'])))
    write(root/'REGISTERED_PANEL.private.json',panel);panel_pin=pin(root/'REGISTERED_PANEL.private.json')
    authority=root/'PROJECT_AUTHORIZATION.private.json';write(authority,dict(schema='USER_PROJECT_AUTHORIZED_ISOLATED_MUJOCO_V6_T',source='CURRENT_USER_EXPLICIT_FIXED_SLA_V6_PLAN',
        source_sha=spec['source_sha'],plan_commit=PLAN,hardware_allowed=False,deadline_epoch=spec['deadline_epoch'],new_native_requests_max=count,full_H40_P10_TTL_STOP_unchanged=True))
    old=pinned(spec['old_E2_spec']);old_server=pinned(old['resident_server_spec']);worker_root=root/'worker';dirs(worker_root);(worker_root/'ipc').mkdir()
    server=dict(offline);server.update(common(spec,worker_root,'SLA_V6_TARGET_'+stage+'_POOL'))
    server.update(campaign_root=str(root),physics=False,new_physical_requests_budget=0,new_complete_generations_budget=count*120,new_model_generations_budget=count*120,
        unified_requests_budget=count*120,ipc_root=str(worker_root/'ipc'),live_root=str(worker_root),selected_combo='SHIFT7_SINGLE',archive_all_branches=False,SHIFT7_package=offline['packages']['T-B1'])
    server['request_bindings_by_arm']={a:dict(old_server['request_bindings_by_arm']['B1'],source_sha=spec['source_sha'],policy_package_sha256=p['sha256'],arm=a,plan_commit=PLAN)for a,p in offline['packages'].items()}
    server['gpu_environment']=dict(offline['native_environment']);server['native_environment']=dict(offline['native_environment']);server['site_binding']=site_binding(server,worker_root,root)
    register(spec,[(worker_root,server)]);server_pin=pin(worker_root/'EXECUTION_SPEC.json');entries=[]
    for row in panel:
        folder=root/'cases'/row['id'];dirs(folder);case=copy.deepcopy(pinned(row['template_spec']));case.update(common(spec,folder,'SLA_V6_TARGET_'+stage+'_'+row['arm']))
        # Native interpreter, NumPy, MuJoCo, ROS, physics and guard parameters
        # remain those of the reviewed template, never the learner environment.
        case['python']=pinned(row['template_spec'])['python']
        case.update(evaluation_stage=stage,mobile_group=str(root),campaign_root=str(root),mobile_experiment_version='MFREE_B1_FULL_H40_SIM_ONLY_V6',sim_only_role='STUDENT',
            sim_only_project_authorization=pin(authority),native_backend='MUJOCO_ONLY',hardware_connections_allowed=False,physics=True,max_ticks=35703,physics_dt_s=.002,
            model_service_timeout_s=30,new_physical_requests_budget=1,cpus=[10,11],arm=row['arm'],packages=offline['packages'],SHIFT7_package=offline['packages'][row['arm']],
            ordinary_RGB_restore=qualification['ordinary_RGB_restore'][row['arm']],resident_server_spec=server_pin,registered_panel=panel_pin,panel_entry_id=row['id'],
            expert_training_eligible=False,unattended_control=str(root/'control'))
        case['mobile_condition']=dict(case['mobile_condition'],family=row['family'],layout=row['layout'],reset=row['reset'],seed=row['noise_seed'],collection_episode=row['id'])
        case['native_environment']=dict(case['native_environment'],ROS_DOMAIN_ID='198',ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES='',
            TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),ROS_LOG_DIR=str(folder/'logs/ros'))
        for name in('tmp','cache','logs','data'):case[name]=str(folder/name)
        entries.append((folder,case))
    parent=dict(old);parent.update(common(spec,root,'SLA_V6_TARGET_'+stage+'_CONTROLLER'));parent.update(evaluation_stage=stage,physics=True,new_physical_requests_budget=count,
        cases=[str(p/'EXECUTION_SPEC.json')for p,_ in entries],resident_server_spec=server_pin,registered_panel=panel_pin,native_environment=dict(spec['native_environment']),
        cpus=[10,11,12,13],mobile_world_lock=entries[0][1]['mobile_world_lock'],prior_case_results=old['prior_case_results'])
    if stage=='CONFIRM':parent.update(frozen_arm=arm,screen_decision=screen_decision)
    register(spec,[(root,parent),*entries])
    from scripts.wm02.target_campaign_v6 import preflight
    from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
    preflight(parent)
    for folder,case in entries:TargetSequentialAdapter(case,folder/'native-01')
    write(root/'ZERO_INTEGRATION_PREFLIGHT.private.json',dict(cases=count,actual_adapter_constructors=count,process_starts=0,new_native_requests=0,new_integrations=0))
    return parent


def main(spec):
    root=Path(spec['registered_root']);journal=Journal(root/'control/journal',root.name)
    try:
        if(spec['plan_commit']!=PLAN or spec['scope']!='SLA_V6_TARGET_NATIVE' or spec['new_physical_requests_budget']!=32
           or spec['new_optimizer_updates_budget']or spec['new_complete_generations_budget']!=615):raise PermissionError('EXACT_V6_FINITE_TARGET_FOLLOWUP_BOUND')
        endpoints=wait_dependencies(spec)
        actual=journal.once('REGISTER-OFFLINE',dict(endpoints=endpoints),lambda:dict(spec=pin(Path(prepare_offline(spec,endpoints)['registered_root'])/'EXECUTION_SPEC.json')))
        offline=pinned(actual['spec']);receipt=run_child(spec,offline,'--target-offline',journal,'RUN-OFFLINE');qualification=pinned(receipt['result'])
        from scripts.wm02.target_offline_report_v6 import run as report,decision
        audit=report(offline);write(root/'OFFLINE377_ACTUAL.private.json',audit)
        actual=journal.once('REGISTER-SCREEN',dict(offline=receipt['result']),lambda:dict(spec=pin(Path(prepare_native(spec,offline,qualification,'SCREEN')['registered_root'])/'EXECUTION_SPEC.json')))
        screen=pinned(actual['spec']);receipt=run_child(spec,screen,'--target-campaign',journal,'RUN-SCREEN');screen_result=pinned(receipt['result']);selected=decision(screen_result['records'],audit)
        write(root/'SCREEN_DECISION_ACTUAL.private.json',selected);confirm=None
        if selected['qualified']:
            selected_pin=pin(root/'SCREEN_DECISION_ACTUAL.private.json');arm=selected['selected_arm']
            actual=journal.once('REGISTER-CONFIRM',dict(screen=receipt['result'],selection=selected_pin),lambda:dict(spec=pin(Path(prepare_native(spec,offline,qualification,'CONFIRM',arm,selected_pin)['registered_root'])/'EXECUTION_SPEC.json')))
            confirmation=pinned(actual['spec']);receipt=run_child(spec,confirmation,'--target-campaign',journal,'RUN-CONFIRM');native=pinned(receipt['result'])
            from go2arx5_vla.sla_adapter.mobile_progression_v52 import confirmation_decision
            confirm=confirmation_decision([r for r in native['records']if r['role']=='STUDENT'],pinned(confirmation['registered_panel']),arm)
            write(root/'CONFIRM_DECISION_ACTUAL.private.json',dict(confirm,schema='SLA_V6_TARGET_CONFIRMATION_ACTUAL',large_DEV_TEST_authorized=False))
        result=dict(exit_code=0,status='TARGET_FIXED_SCREEN_AND_CONDITIONAL_CONFIRMATION_TERMINAL',selection=selected,confirmation=confirm,
            source_sha=spec['source_sha'],new_updates=0,new_native_requests=24+(8 if confirm is not None else 0),new_offline_generations=615,
            large_DEV_TEST_auto_started=False,scientific_retry=False)
    except BaseException:result=dict(exit_code=1,status='PRESERVED_TARGET_FOLLOWUP_TECHNICAL_FAILURE_NO_REPLAY',failure=traceback.format_exc(),new_updates=0,actual_native_evidence=str(root/'SCREEN/RESULT.private.json'))
    write(root/'RESULT.private.json',result);close(spec,result['exit_code']);return result['exit_code']
