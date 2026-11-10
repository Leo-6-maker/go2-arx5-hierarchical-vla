"""One serial DEV panel group using existing native/neural workers and Journal.

The caller has already registered the group, fresh P1/Fusion bindings and a
sealed source. This module registers pairs, starts each worker once, waits for
both actual exits, and retains valid failures. It never submits PBS or trains.
"""
import copy,json,os,signal,subprocess,time
from pathlib import Path
from contextlib import closing
from . import task_learning_grasp_v1 as task,task_learning_student_v1 as student
from .storage import digest
from .stable_task_scene_v1 import pinned

SCHEMA='TASK_LEARNING_STUDENT_DEV_GROUP_INPUT_V1'
CONTINUATION_SCHEMA='TASK_LEARNING_STUDENT_DEV_GROUP_INPUT_V2'


def selection(inp,panel):
    if inp.get('schema')=='CORE_C500_EXPLICIT_CONIC_GROUP_INPUT_V3':
        from .task_learning_conic_native_v3 import selection as conic_selection
        return conic_selection(inp,panel)
    if inp.get('schema')=='TASK_LEARNING_CORE_LOCAL_GROUP_INPUT_V1':
        from .task_learning_local_policy_v1 import selection as local_selection
        return local_selection(inp,panel)
    rows=student.validate(panel);span=inp.get('assignment_slice')
    version=inp.get('schema')
    task.need(version in(SCHEMA,CONTINUATION_SCHEMA) and inp.get('source_sha')==panel['source_sha']
        and isinstance(span,list)and len(span)==2 and all(type(v)is int for v in span)
        and 0<=span[0]<span[1]<=80,
        'FIXED_CONTIGUOUS_MATCHED_DEV_PAIRS_WITHOUT_RETRY')
    if version==SCHEMA:
        task.need(all(v%2==0 for v in span),'ORIGINAL_COMPLETE_PAIR_SLICE')
    elif span[0]or inp.get('population_predecessor')is not None:
        # A finite scope can end between the two methods of a pair. Preserve
        # that actual condition and continue only the still-unexecuted suffix.
        prior=pinned(inp['population_predecessor'])
        task.need(prior.get('schema')=='TASK_LEARNING_FULL_NATIVE_DEV_POPULATION_ACTUAL_V2'
            and prior.get('panel')==dict(path=inp['panel'],sha256=inp['panel_sha256'])
            and prior.get('source_sha')==inp['source_sha']
            and prior.get('cold_continuation_safe')is True
            and [v['native_run']for v in prior['outcomes']]==[r['native_run']for r in rows[:span[0]]],
            'ACTUAL_CLOSED_PREFIX_BEFORE_UNEXECUTED_DEV_SUFFIX_NOT_RETRY')
        for group in prior['groups']:
            lease=pinned(group['lease']);result=pinned(group['result'])
            task.need(lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==result.get('exit_code')
                and result.get('unresolved_pair')is None,'ALL_PRIOR_DEV_GROUPS_ACTUALLY_COLD')
        task.need(all(prior['summary']['methods'][m]['hard_safety_stop_conditions']==
            prior['summary']['methods'][m]['technical_failure_conditions']==0 for m in('B0','B1')),
            'SAFETY_OR_TECHNICAL_FAILURE_NEEDS_SEPARATE_REPAIR_NOT_SCOPE_RETRY')
    else:
        task.need(span[0]==0,'FIRST_DEV_GROUP_STARTS_AT_ORIGINAL_CONDITION0')
    return rows[span[0]:span[1]]


def specs(inp,panel,row,nbase,gbase,nroot,groot,deadline):
    """Build a pair before registration; original runtime admission still applies."""
    from . import task_learning_live_pair_v1 as pair
    source=Path(inp['source_root']);descriptor=inp['scenes'][row['layout']];proof=pinned(descriptor)
    task.need(proof['layout']==row['layout']and proof['parent']==row['parent']
        and proof['compiled']['scene_sha256']==row['model_sha256'],'FIXED_DEV_SCENE_ROW_JOIN')
    local=panel.get('task_learning_role')==task.CORE_LOCAL_ROLE
    conic=panel.get('task_learning_role')==task.CONIC_ROLE
    role=task.CONIC_ROLE if conic else task.CORE_LOCAL_ROLE if local else task.STUDENT_DEV_ROLE
    common=dict(source_root=str(source),source_sha=inp['source_sha'],source_files=inp['source_files'],
        deadline_epoch=deadline,wall_limit_s=1200,pilot1_binding=inp['pilot1_binding'],
        pilot1_binding_sha256=inp['pilot1_binding_sha256'],task_learning_grasp=task.protocol(role),
        target_gr00t_policy=panel['policies'][row['method']],target_gr00t_query_seeds=row['query_seeds'],
        target_gr00t_pair_id=row['native_run'],target_gr00t_B1_global_step=panel['B1_global_step'])
    if local or conic:common['core_local_protocol']=role
    native=copy.deepcopy(nbase)
    for key in('task_learning_teacher_mode','task_learning_control','night1_binding','next1_binding','next2_binding',
        'fusion_binding','fusion_binding_sha256','target_gr00t_collection','paused_query_pair',
        'paused_query_gpu_spec','stable_task_candidate','stable_task_evaluation','program_reference_selection'):
        native.pop(key,None)
    native.update(common,registered_root=str(nroot),role=nroot.name.upper(),lease=str(nroot/'control/LEASE.json'),
        registry_snapshot=str(nroot/'control/REGISTRY_SNAPSHOT.private.json'),task_learning_role=role,
        experiment_role=('CORE_C500_RAW_VERSUS_EXPLICIT_CONIC_TRAIN_DIAGNOSTIC_V3'if conic else 'CORE_LOCAL_B0_SFT2000_DIAGNOSTIC_NOT_DEV_TEST'if local else student.EXPERIMENT),task_condition=role,
        task_learning_scene_reference=descriptor,scene_id=row['parent'],data_parent=row['parent'],data_role='TRAIN'if local or conic else'DEV',
        model_path=str(Path(descriptor['path']).parent/'SCENE.private.mjb'),stable_task_panel=inp['panel'],
        stable_task_panel_sha256=inp['panel_sha256'],scene_allowlist=inp['scene_allowlist'],
        scene_allowlist_sha256=inp['scene_allowlist_sha256'],cpus=[0,1,2,3],ROS_DOMAIN_ID=220,
        target_gr00t_gpu_root=str(groot))
    gpu=copy.deepcopy(gbase)
    for key in('night1_binding','next1_binding','next2_binding','phase','max_ticks','stage_identity_sha256',
        'episode_manifest','episode_manifest_sha256','target_capture_result','target_capture_result_sha256',
        'target_capture_step','actor_numpy','actor_numpy_sha256','task','task_sha256','ARX','ARX_sha256',
        'teacher_metadata','teacher_metadata_sha256','recorded_observation','recorded_observation_sha256',
        'paused_query_pair','paused_query_pair_sha256','producer_native_spec_sha256'):
        gpu.pop(key,None)
    gpu.update(common,registered_root=str(groot),role=groot.name.upper(),lease=str(groot/'control/LEASE.json'),
        registry_snapshot=str(groot/'control/REGISTRY_SNAPSHOT.private.json'),profile=pair.PROFILE,kind=pair.KIND,
        scope='WM02_FUSION_REFERENCE_SIM_ONLY',execution_site='CORE_FUSION_NEURAL',physics=False,device='cuda',
        max_queries=15,cpus=[8,9,10,11],baseline_changed=False,GR00T_updates=0,WM_updates=0,Actor_updates=0,
        fusion_binding=inp['fusion_binding'],fusion_binding_sha256=inp['fusion_binding_sha256'],
        memory_execution=inp['memory_execution'],memory_execution_reference=inp['memory_execution_reference'])
    if conic:
        gpu['conic_solver_dependency']=panel['registration']['conic_solver_dependency']
    for s,root in((native,nroot),(gpu,groot)):
        for name in('cache','logs','data','tmp'):s[name]=str(root/name)
    native['native_environment']=dict(native['native_environment'],
        PYTHONPATH=str(source/'src')+':'+str(source)+':'+native['native_environment'].get('PYTHONPATH',''))
    return native,gpu


def environment(s,*,gpu):
    source=Path(s['source_root']);root=Path(s['registered_root'])
    env=dict(os.environ)if gpu else dict(os.environ,**s['native_environment'])
    for key in('LD_PRELOAD','MUJOCO_EGL_DEVICE_ID'):env.pop(key,None)
    env.update(PYTHONPATH=str(source/'src')+':'+str(source)+':'+env.get('PYTHONPATH',''),
        CUDA_VISIBLE_DEVICES=s['GPU_uuid']if gpu else'',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4'if gpu else'1',MKL_NUM_THREADS='4'if gpu else'1',OPENBLAS_NUM_THREADS='1',
        TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),XDG_CONFIG_HOME=str(root/'cache/config'))
    if gpu and s.get('core_local_protocol')==task.CONIC_ROLE:
        dependency=s['conic_solver_dependency'];dep=Path(dependency['path'])
        for name,pin in dependency['files'].items():
            task.need(digest(dep/name)==pin,'EXACT_PRIVATE_CONIC_SOLVER_MEMBER')
        env['PYTHONPATH']=str(dep)+':'+env['PYTHONPATH']
    if gpu:
        env.update(LD_LIBRARY_PATH='',HF_HOME=str(root/'cache/hf'),HF_MODULES_CACHE=str(root/'cache/hf-modules'),
            TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    else:env['ROS_LOG_DIR']=str(root/'logs')
    return env


def resources(spec):
    """Read the shared device before registering a new pair; change nothing."""
    from .pilot1_scope import GPU_UUIDS
    task.need(spec['GPU_uuid']in GPU_UUIDS and set(range(4))|set(range(8,12))<=os.sched_getaffinity(0),
        'ONE_KNOWN_GPU_AND_ORIGINAL_SEPARATE_CPU_AFFINITIES')
    q=subprocess.run(['nvidia-smi','--id='+spec['GPU_uuid'],
        '--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=15,check=True)
    uuid,name,free=[v.strip()for v in q.stdout.split(',')]
    available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'))
    task.need(uuid==spec['GPU_uuid']and name=='NVIDIA GeForce RTX 4080 SUPER'
        and spec['checkpoint_parameter_bytes']<=spec['allocator_limit_bytes']<int(free)*1024**2-1024**3
        and available>32*1024**3,'FRESH_SHARED_GPU_AND_HOST_HEADROOM_BEFORE_DEV_PAIR')
    return dict(GPU_uuid=uuid,free_GPU_bytes=int(free)*1024**2,host_available_bytes=available,epoch=time.time())


def execute_pair(registered,journal,stage,outcome_count):
    """Both exits are required; never redraw or restart a lost serving process."""
    from scripts.pi0292.durable import write
    values={name:json.loads(Path(registered[name]['path']).read_bytes())for name in('native','gpu')}
    children={};handles=[]
    for name in('native','gpu'):
        task.need(digest(registered[name]['path'])==registered[name]['sha256'],'SEALED_REGISTERED_PAIR')
        task.need(not journal.path(Path(values[name]['registered_root']).name.upper()+'-RUN').exists(),
            'RECONCILE_EXISTING_OR_UNKNOWN_PAIR_DISPATCH')
    try:
        for name in('gpu','native'):
            if name=='native':task.need(children['gpu'].poll()is None,'GPU_EXIT_BEFORE_NATIVE_START')
            s=values[name];root=Path(s['registered_root']);gpu=name=='gpu'
            task.need(not(root/'SUPERVISOR.private.json').exists(),'NEVER_REPLAY_A_STARTED_PAIR_WORKER')
            command=[s['python'],'-B','-m','scripts.wm02.pilot1_neural_worker'if gpu else'scripts.wm02.pilot1_native_worker',
                '--spec',registered[name]['path']]
            def start(s=s,root=root,gpu=gpu,command=command,name=name):
                out=(root/'logs/worker-out.log').open('xb');err=(root/'logs/worker-err.log').open('xb');handles.extend((out,err))
                child=subprocess.Popen(command,cwd=s['source_root'],env=environment(s,gpu=gpu),stdout=out,stderr=err)
                children[name]=child
                return dict(pid=child.pid,epoch=time.time(),command=command)
            journal.once(root.name.upper()+'-RUN',registered[name],start)
        deadline=min(v['deadline_epoch']for v in values.values())
        while any(c.poll()is None for c in children.values()):
            write(stage/'PROGRESS.private.json',dict(current_run=Path(values['native']['registered_root']).name,
                completed=outcome_count,children={n:dict(pid=c.pid,exit_code=c.poll())for n,c in children.items()},
                updated_epoch=time.time()))
            gpu=children['gpu'];native=children['native']
            if gpu.poll()not in(None,0)or time.time()>=deadline or(stage/'STOP.json').exists():
                write(Path(values['native']['registered_root'])/'STOP.json',dict(reason='PAIRED_GPU_FAILURE_OR_FINITE_DEV_ENDPOINT'))
            if time.time()>deadline+90:raise TimeoutError('FINITE_PAIRED_DEV_CLOSEOUT')
            time.sleep(.5)
    finally:
        for name,child in children.items():
            if child.poll()is None:
                write(Path(values['native']['registered_root'])/'STOP.json',dict(reason='OWNED_DEV_COORDINATOR_EXIT'))
                try:child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
        for h in handles:h.close()
    return {n:c.returncode for n,c in children.items()}


def run(input_path):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .pilot1_scope import validate as validate_scope
    from . import task_learning_live_pair_v1 as pair
    from .stable_task_scene_v1 import episode
    from .task_learning_grasp_native_v1 import episode_config
    inp=json.loads(input_path.read_bytes());stage=input_path.parent;lp=stage/'control/LEASE.json'
    lease=json.loads(lp.read_bytes());panel=pinned(dict(path=inp['panel'],sha256=inp['panel_sha256']))
    rows=selection(inp,panel);nbase=pinned(inp['native_base']);gbase=pinned(inp['gpu_base'])
    task.need(os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
        and lease.get('status')=='ACTIVE'and lease.get('run_uuid')==stage.name
        and lease.get('input_sha256')==digest(input_path)and not(stage/'SUPERVISOR.private.json').exists(),
        'FRESH_ACTUAL_REGISTERED_STUDENT_DEV_GROUP')
    campaign=Path(nbase['campaign_root']);source=Path(inp['source_root'])
    for base in(nbase,gbase):
        root=Path(base['registered_root']);old=json.loads((root/'control/LEASE.json').read_bytes())
        worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
        task.need(old.get('status')=='CLOSED'and old.get('worker_exit_code')==worker.get('exit_code')==0,
            'ACTUAL_CLOSED0_IMMUTABLE_NATIVE_AND_NEURAL_BASES')
    task.need(json.loads((Path(nbase['registered_root'])/'native-01/TASK_LEARNING_SCORE.private.json').read_bytes()).get('task_success')is True,
        'ACTUAL_FULL_TEACHER_CONSTRUCTION_BASE')
    for name,pin in inp['source_files'].items():task.need(digest(source/name)==pin,'SEALED_DEV_SOURCE_'+name)
    validation=dict(nbase,registered_root=str(stage),source_sha=inp['source_sha'],deadline_epoch=inp['deadline_epoch'],
        pilot1_binding=inp['pilot1_binding'],pilot1_binding_sha256=inp['pilot1_binding_sha256'])
    validate_scope(validation)
    local=panel.get('task_learning_role')==task.CORE_LOCAL_ROLE
    conic=panel.get('task_learning_role')==task.CONIC_ROLE
    validate_policy=pair.policy_contract;summarize=student.summarize
    methods=('B0','B1')
    if conic:
        from .task_learning_conic_native_v3 import policy_contract as validate_policy,summarize
        from .task_learning_conic_policy_v3 import METHODS as methods
    if local:
        from .task_learning_local_policy_v1 import policy_contract as validate_policy,summarize
    for method in methods:
        validate_policy(panel['policies'][method],gbase['checkpoint_files'],expected_global_step=panel['B1_global_step'])
    journal=Journal(campaign/'control/journal',campaign.name);outcomes=[];code=1;failure=None;start=time.monotonic()
    current_row=None;current_pair=None;unresolved=None
    write(stage/'SUPERVISOR.private.json',dict(pid=os.getpid(),source_sha=inp['source_sha'],started_epoch=time.time()))
    def interrupted(sig,frame):raise KeyboardInterrupt('OWNED_STUDENT_DEV_SIGNAL_'+str(sig))
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}
    def register(row):
        nr=stage/'runs'/row['native_run'];gr=stage/'runs'/(row['native_run']+'-gpu');key=nr.name.upper()
        with closing(OwnerLock(campaign/'OWNER.lock')),closing(OwnerLock(nbase['physical_owner_lock'])),closing(OwnerLock(gbase['neural_owner_lock'])):
            task.need(not nr.exists()and not gr.exists()and not journal.path(key+'-REGISTER').exists(),'NO_EXISTING_OR_UNKNOWN_DEV_PAIR')
            actual_resources=resources(gbase)
            def effect():
                for root in(nr,gr):
                    root.mkdir()
                    for name in('control','logs','cache','tmp','data'):(root/name).mkdir()
                n,g=specs(inp,panel,row,nbase,gbase,nr,gr,min(time.time()+1200,inp['deadline_epoch']))
                ep=nr/'EPISODE_MANIFEST.private.json';write(ep,episode(n));n.update(episode_manifest=str(ep),
                    episode_manifest_sha256=digest(ep),stage_identity_sha256=digest(ep))
                ns=nr/'EXECUTION_SPEC.json';write(ns,n)
                g['producer_native_spec']=dict(path=str(ns),sha256=digest(ns))
                alias=gr/'model_inputs/nvidia/Cosmos-Reason2-2B';alias.parent.mkdir(parents=True)
                alias.symlink_to(g['neural_bindings']['cosmos'],target_is_directory=True)
                gs=gr/'EXECUTION_SPEC.json';write(gs,g)
                episode_config(n);pair.validate_spec(g,verify_model=False)
                reg,pin=snapshot(n['registry'])
                for s in(n,g):
                    root=Path(s['registered_root']);task.need(not any(e.get('path')==str(root)for e in reg['entries']),'FRESH_DEV_REGISTRY')
                    entry=dict(role=s['role'],path=str(root),run_uuid=root.name,scope=s['scope'],
                        source_sha=inp['source_sha'],owner=s['owner'],executor_node=s['host'],pilot1_binding=s['pilot1_binding'],local_lease=s['lease'])
                    if s is g:entry['fusion_binding']=s['fusion_binding']
                    reg['entries'].append(entry)
                once(Path(n['registry']),reg,key+'-PAIR-REGISTRY',expected_before_sha256=pin)
                for s in(n,g):
                    write(s['registry_snapshot'],reg);write(s['lease'],dict(status='ACTIVE',run_uuid=Path(s['registered_root']).name,
                        spec_sha256=digest(Path(s['registered_root'])/'EXECUTION_SPEC.json'),owner=s['owner'],scope=s['scope'],deadline_epoch=s['deadline_epoch']))
                pair.preflight(n)
                write(nr/'PAIR_RESOURCE_READBACK.private.json',actual_resources)
                return dict(native=dict(path=str(ns),sha256=digest(ns)),gpu=dict(path=str(gs),sha256=digest(gs)))
            return journal.once(key+'-REGISTER',dict(row=row,source_sha=inp['source_sha']),effect)
    try:
        for row in rows:
            task.need(time.time()+1200<inp['deadline_epoch']and not(stage/'STOP.json').exists(),'NO_LATE_NEW_DEV_CONDITION')
            current_row=row;current_pair=None;registered=register(row);current_pair=registered
            execute_pair(registered,journal,stage,len(outcomes))
            result=student.classify_actual(row,stage/'runs'/row['native_run'],inp['source_sha'],core_local=local,conic=conic)
            outcomes.append(result);write(stage/'OUTCOMES.private.json',outcomes)
            write(stage/'SUMMARY.private.json',summarize(panel,outcomes))
            task.need(not result['hard_safety_stop'],'ORIGINAL_HARD_SAFETY_DEV_FAMILY_PAUSED')
            task.need(result['technical_valid'],'ENGINEERING_FAILURE_PRESERVED_NO_BLIND_NEXT_CASE')
        code=0
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
        if current_pair and not any(v['native_run']==current_row['native_run']for v in outcomes):
            try:
                outcomes.append(student.classify_actual(current_row,stage/'runs'/current_row['native_run'],inp['source_sha'],core_local=local,conic=conic))
                write(stage/'OUTCOMES.private.json',outcomes)
            except BaseException as audit_error:
                unresolved=dict(pair=current_pair,observation_error=type(audit_error).__name__+': '+str(audit_error))
        elif current_row and current_pair is None:
            unresolved=dict(native_root=str(stage/'runs'/current_row['native_run']),stage='REGISTRATION_NOT_COMPLETED_NO_REPLAY')
    finally:
        # Revoke only permissions that provably never reached a dispatch intent.
        # No worker result or zero exit is invented for an unstarted partner.
        if current_pair:
            for descriptor in current_pair.values():
                root=Path(descriptor['path']).parent
                if not journal.path(root.name.upper()+'-RUN').exists()and not(root/'SUPERVISOR.private.json').exists():
                    permission=json.loads((root/'control/LEASE.json').read_bytes())
                    task.need(permission.get('status')=='ACTIVE'and permission.get('spec_sha256')==descriptor['sha256'],
                        'OWN_UNDISPATCHED_PARTNER_PERMISSION')
                    permission.update(status='CLOSED',worker_exit_code=None,closed_epoch=time.time(),
                        terminal_reason='NEVER_DISPATCHED_PERMISSION_REVOKED')
                    write(root/'control/LEASE.json',permission)
        summary=summarize(panel,outcomes);write(stage/'SUMMARY.private.json',summary)
        write(stage/'RESULT.private.json',dict(status=('CORE_C500_CONIC_NATIVE_GROUP_COMPLETE'if code==0 else'CORE_C500_CONIC_NATIVE_GROUP_INCOMPLETE_PRESERVED')if conic else('TASK_LEARNING_CORE_LOCAL_GROUP_COMPLETE'if code==0 else'TASK_LEARNING_CORE_LOCAL_GROUP_INCOMPLETE_PRESERVED')if local else('TASK_LEARNING_STUDENT_DEV_GROUP_COMPLETE'if code==0 else'TASK_LEARNING_STUDENT_DEV_GROUP_INCOMPLETE_PRESERVED'),
            exit_code=code,failure=failure,summary=summary,assignment_slice=inp['assignment_slice'],
            selected_requests=len(rows),completed_in_group=len(outcomes),source_sha=inp['source_sha'],panel_sha256=inp['panel_sha256'],
            wall_s=time.monotonic()-start,CPU_s=sum(v['CPU_s']for v in outcomes),
            GPU_worker_wall_s=sum(v.get('GPU_worker_wall_s')or 0 for v in outcomes),unresolved_pair=unresolved,
            GR00T_updates=0,qualified_training_demonstrations=0))
        current=json.loads(lp.read_bytes());task.need(current.get('status')=='ACTIVE'and current.get('input_sha256')==digest(input_path),'OWN_DEV_GROUP_LEASE')
        current.update(status='CLOSED',worker_exit_code=code,closed_epoch=time.time());write(lp,current)
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return code
