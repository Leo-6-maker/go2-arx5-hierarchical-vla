"""Finite two-slot native batch under one existing physical owner, no queue."""
from pathlib import Path
import fcntl,json,os,subprocess,time
from contextlib import contextmanager
from .storage import digest,write_json


def recorded_split(batch,child):
    # Commissioning has no learning episode. Do not infer TRAIN from raw atoms.
    if batch.get('schema')=='WM02_PILOT2_TWO_COMMISSION_BATCH_V1':return 'QUALIFICATION'
    return json.loads(Path(child['episode_manifest']).read_text())['split']


def validate_duration_commission(batch,children,proof,value):
    """HOLD-only full-duration successor after actual single and short dual runs."""
    prior=json.loads((proof.parent.parent/'EXECUTION_SPEC.json').read_text())
    for child in children:
        episode=json.loads(Path(child['episode_manifest']).read_text())
        single=json.loads(Path(child['previous_stage']).read_text())
        if (episode.get('purpose')!='ACTION_COVERAGE'or episode.get('forced_action')!=0
                or single.get('actual_target_integrations')!=4103
                or prior.get('native_binary')!=child['native_binary']
                or prior.get('observer_library')!=child['observer_library']):
            raise PermissionError('DURATION_COMMISSION_REQUIRES_SINGLE_QUALIFIED_HOLD_AND_SAME_EXECUTOR')
    for row in value['children']:
        if Path(row['run']).name!=row['run']:raise PermissionError('PRIOR_COMMISSION_CHILD_IDENTITY_INVALID')
        native=json.loads((Path(prior['native_runs_root'])/row['run']/'native-01/RESULT.json').read_text())
        if (native.get('status')!='NATIVE_STAGE_PASSED'or not native.get('shutdown_clean')
                or native.get('physics_stops')or native.get('actual_target_integrations',0)<3103
                or native.get('native_binary_sha256')!=digest(children[0]['native_binary'])):
            raise PermissionError('ACTUAL_SAME_EXECUTOR_SHORT_DUAL_NATIVE_RESULTS_REQUIRED')


def validate_layout(batch, children):
    if len(children)!=2 or len(batch['slots'])!=2:raise PermissionError('EXACT_TWO_SLOT_PILOT_REQUIRED')
    feedback=batch.get('schema')=='WM02_PILOT1_TWO_FREE_FEEDBACK_BATCH_V1'
    coverage=batch.get('schema')=='WM02_PILOT2_TWO_COVERAGE_BATCH_V1'
    supported=batch.get('schema')=='WM02_PILOT2_TWO_SUPPORTED_POLICY_BATCH_V1'
    commissioning=batch.get('schema')=='WM02_PILOT2_TWO_COMMISSION_BATCH_V1'
    free=feedback or coverage or batch.get('schema')=='WM02_PILOT1_TWO_FREE_EVALUATION_BATCH_V1'
    pilot= supported or free or batch.get('schema')in('WM02_PILOT1_TWO_CONTACT_BATCH_V1','WM02_PILOT1_TWO_CONTACT_POLICY_BATCH_V1')
    runs=Path(batch['native_runs_root']).resolve() if pilot or commissioning else Path(batch['campaign_root'])/'runs'
    domains=[];cpus=set();roots=set()
    for slot,child in zip(batch['slots'],children):
        root=Path(child['registered_root']).resolve()
        if (root.parent!=runs or str(root)in roots
                or child.get('parallel_batch_spec')!=str(Path(batch['registered_root'])/'EXECUTION_SPEC.json')
                or child['campaign_root']!=batch['campaign_root']
                or child['physical_owner_lock']!=batch['physical_owner_lock']
                or child['source_sha']!=batch['source_sha']
                or child['ROS_DOMAIN_ID']not in (220,221)
                or not set(child['cpus'])or cpus.intersection(child['cpus'])
                or child['phase']not in (('collect-target-pilot2-supported',)if supported else('collect-target-pilot1',)if free else('collect-target-pilot1-contact',)if pilot else('cold-1','g2-standing-hold-short-v2','collect-target-parallel-01','collect-target-parallel-02','collect-target-night1','collect-target-next1','collect-target-next2'))
                or child['max_ticks']not in ((6703,)if supported else(4103,)if free else(5703,)if pilot else(50,2528,3103,3703,4103,4903,6503))
                or digest(slot['spec'])!=slot['spec_sha256']):
            raise PermissionError('ISOLATED_PINNED_FINITE_TARGET_SLOTS_REQUIRED')
        domains.append(child['ROS_DOMAIN_ID']);cpus.update(child['cpus']);roots.add(str(root))
    if len(set(domains))!=2 or not cpus<=set(batch['cpus']):raise PermissionError('DISTINCT_DOMAINS_AND_AVAILABLE_CPUS_REQUIRED')
    if commissioning:
        from .pilot2_coverage import PLAN
        if not batch.get('pilot1_binding')or batch.get('pilot2_plan_sha')!=PLAN:
            raise PermissionError('FRESH_PILOT2_COMMISSION_SCOPE_REQUIRED')
        for child in children:
            if any(child.get(k)!=batch.get(k) for k in ('pilot1_binding','pilot1_binding_sha256','pilot2_plan_sha','native_binary','observer_library','candidate_source_sha')):
                raise PermissionError('SAME_CANDIDATE_AND_SCOPE_FOR_NATIVE_COMMISSION_REQUIRED')
    if pilot:
        if supported:
            from .pilot2_supported import validate
        elif free:
            from .pilot1_episode import episode_limit as validate
        else:
            from .pilot1_contact import validate
        for child in children:
            if (child.get('pilot1_binding')!=batch.get('pilot1_binding')
                    or child.get('pilot1_binding_sha256')!=batch.get('pilot1_binding_sha256')):
                raise PermissionError('SAME_ACTUAL_PILOT1_WINDOW_REQUIRED')
            previous=json.loads(Path(child['previous_stage']).read_text())
            if validate(child,previous)!=(6703 if supported else 4103 if free else 5703):raise PermissionError('QUALIFIED_SINGLE_PILOT1_SCENE_REQUIRED')
            if supported and (child.get('supported_policy_parallel_schema')!=batch['schema']
                    or child.get('pilot2_plan_sha')!=batch.get('pilot2_plan_sha')):
                raise PermissionError('SAME_QUALIFIED_SUPPORTED_POLICY_BATCH_AND_PLAN_REQUIRED')
            if free:
                episode=json.loads(Path(child['episode_manifest']).read_text())
                if episode.get('purpose')!=('ACTION_COVERAGE'if coverage else'TRAIN_FEEDBACK'if feedback else'EVALUATION')or feedback and episode.get('split')!='TRAIN':
                    raise PermissionError('TWO_SLOT_FREE_BATCH_REQUIRES_PREQUALIFIED_EVALUATION_SCENES_OR_SEPARATE_TRAIN_FEEDBACK')
                if coverage and child.get('pilot2_plan_sha')!=batch.get('pilot2_plan_sha'):
                    raise PermissionError('SAME_PREDECLARED_PILOT2_PLAN_REQUIRED')
        proof=Path(batch['parallel_qualification']);value=json.loads(proof.read_text())
        duration_commission=coverage and batch.get('purpose')=='COVERAGE_DUAL_DURATION_COMMISSION'
        if (digest(proof)!=batch['parallel_qualification_sha256']or value['status']!='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'
                or len(value['children'])!=2 or not all(c['ticks']>=(3103 if duration_commission else 4103)and c['shutdown_clean']and c['exit_code']==0 for c in value['children'])):
            raise PermissionError('ACTUAL_PRIOR_TWO_NATIVE_SLOT_QUALIFICATION_REQUIRED')
        if duration_commission:validate_duration_commission(batch,children,proof,value)
        if supported:
            episodes=[json.loads(Path(c['episode_manifest']).read_text())for c in children]
            if batch.get('purpose')=='SUPPORTED_POLICY_DUAL_COMMISSION':
                from .pilot2_supported import GOALS
                if (any(e['method']!='GOAL_FSM' or e['split']!='TRAIN'for e in episodes)
                        or {e['task_goal']for e in episodes}!=set(GOALS)
                        or len({e['parent']for e in episodes})!=1):
                    raise PermissionError('DUAL_COMMISSION_ONLY_BOTH_SINGLE_QUALIFIED_TRAIN_FSM_GOALS')
            elif batch.get('purpose')!='SUPPORTED_POLICY_PRODUCTION' or any(c['ticks']!=6703 for c in value['children']):
                raise PermissionError('ACTUAL_FULL_DURATION_TWO_SUPPORTED_POLICY_QUALIFICATION_REQUIRED')
        return
    night1=all(c['phase']=='collect-target-night1'for c in children)
    next2=all(c['phase']=='collect-target-next2'for c in children)
    if any(c['phase']=='collect-target-next2'for c in children)and not next2:raise PermissionError('NO_MIXED_NEXT2_LEGACY_BATCH')
    next1=all(c['phase']=='collect-target-next1'for c in children)or next2
    if any(c['phase']=='collect-target-next1'for c in children)and not next1:raise PermissionError('NO_MIXED_NEXT1_LEGACY_BATCH')
    if next1:
        from .next1_episode import episode_config
        if next2:
            from .next2_episode import episode_config
        episodes=[episode_config(c)for c in children]
        if len({e['task_atoms']for e in episodes})!=1 or any(c.get('next2_binding'if next2 else'next1_binding')!=batch.get('next2_binding'if next2 else'next1_binding')for c in children):raise PermissionError('SAME_NEXT1_WINDOW_AND_LENGTH_REQUIRED')
        if all(e['purpose']=='STEADY_DUAL'for e in episodes):
            proofs=batch['steady_single_qualifications']
            if len(proofs)!=2 or len({p['path']for p in proofs})!=2:raise PermissionError('TWO_INDEPENDENT_SINGLE_PREFIX_RUNS_REQUIRED')
            for proof in proofs:
                value=json.loads(Path(proof['path']).read_text())
                if digest(proof['path'])!=proof['sha256']or value.get('qualified')is not True or value.get('screen_atom')!=48 or not value.get('independent_initialization'):raise PermissionError('ACTUAL_SAFE_STABLE_SINGLE_PREFIX_REQUIRED')
        else:
            proof=Path(batch['parallel_qualification']);value=json.loads(proof.read_text())
            if digest(proof)!=batch['parallel_qualification_sha256']or value['status']!='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'or len(value['children'])!=2 or not all(c['ticks']==4103 and c['shutdown_clean']and c['exit_code']==0 for c in value['children']):raise PermissionError('ACTUAL_NEXT1_TWO_SLOT_LONG_PREFIX_QUALIFICATION_REQUIRED')
    if any(c['phase']=='collect-target-night1'for c in children)and not night1:raise PermissionError('NO_MIXED_NIGHT1_LEGACY_BATCH')
    if night1:
        from .target_manifest import episode_config
        episodes=[episode_config(c)for c in children]
        if len({e['task_atoms']for e in episodes})!=1:raise PermissionError('EXACT_FINITE_SAME_LENGTH_BATCH_REQUIRED')
        if any(e['purpose']in('PRODUCTION','DECISION_PRODUCTION')for e in episodes):
            proof=Path(batch['duration_qualification']);value=json.loads(proof.read_text())
            if (digest(proof)!=batch['duration_qualification_sha256']or value['status']!='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'
                    or len(value['children'])!=2 or not all(c['ticks']==2503+25*episodes[0]['task_atoms']and c['shutdown_clean']and c['exit_code']==0 for c in value['children'])):raise PermissionError('ACTUAL_CONCURRENT_DURATION_QUALIFICATION_REQUIRED')
    if any(c['phase'].startswith('collect-target-')for c in children)and not night1 and not next1:
        if [c['phase']for c in children]!=['collect-target-parallel-01','collect-target-parallel-02']:raise PermissionError('EXACT_FINITE_PARALLEL_COLLECTION_PAIR_REQUIRED')
        proof=Path(batch['parallel_qualification']);value=json.loads(proof.read_text())
        if (digest(proof)!=batch['parallel_qualification_sha256']or value['status']!='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'
                or len(value['children'])!=2 or not all(c['ticks']==2528 and c['shutdown_clean']and c['exit_code']==0 for c in value['children'])):raise PermissionError('ACTUAL_CONCURRENT_HOLD_QUALIFICATION_REQUIRED')


def process_start(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def child_environment(child,fds):
    env=dict(os.environ,**child['native_environment'])
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=child['source_root']+'/src:'+child['source_root']+':'+env.get('PYTHONPATH',''),WM02_AUTO1_OWNER_FDS=json.dumps(fds),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for key,field in [('TMPDIR','tmp'),('ROS_LOG_DIR','logs'),('XDG_CACHE_HOME','cache'),('XDG_CONFIG_HOME','cache')]:
        if field in child:env[key]=child[field]
    return env


def wait_children(spec,jobs):
    """Finite native batch supervision; a failed peer closes this batch."""
    from scripts.pi0292.durable import write
    campaign=Path(spec['campaign_root'])
    while any(p.poll()is None for _,p in jobs):
        failed_child=any(p.poll()not in(None,0)for _,p in jobs)
        if failed_child or time.time()>=spec['deadline_epoch']or(campaign/'STOP.json').exists()or(Path(spec['registered_root'])/'STOP.json').exists():
            for child,p in jobs:
                if p.poll()is None:write(Path(child['registered_root'])/'STOP.json',dict(reason='PEER_FAILED'if failed_child else'BATCH_FINITE_STOP'))
        time.sleep(.05)


def verify_inherited_owner(fd,path):
    """An inherited locked open-file description, not a permission boolean."""
    expected=os.stat(path);actual=os.fstat(fd)
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise PermissionError('OWNER_FD_PATH_MISMATCH')
    # An independent open must contend. The inherited description must succeed.
    with Path(path).open('a')as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise PermissionError('CANONICAL_OWNER_LOCK_NOT_HELD')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)


@contextmanager
def inherited_batch_owner(spec_path,spec):
    batch_path=Path(spec['parallel_batch_spec']);batch=json.loads(batch_path.read_text())
    root=Path(batch['registered_root']);lease=json.loads(Path(batch['lease']).read_text());started=json.loads((root/'batch-01/INHERITED_OWNER.private.json').read_text())
    from .auto1_native import validate_scope
    campaign=Path(spec['campaign_root']);validate_scope(batch,json.loads((campaign/'control/LEASE.json').read_text()),json.loads(Path(spec['registry']).read_text()),host=os.uname().nodename,uid=os.getuid(),now=time.time())
    if (lease['status']!='ACTIVE'or lease['spec_sha256']!=digest(batch_path)
            or started['batch_spec_sha256']!=digest(batch_path)
            or started['pid']!=os.getppid()or started['process_start']!=process_start(os.getppid())
            or Path(f'/proc/{os.getppid()}').stat().st_uid!=os.getuid()):raise PermissionError('ACTUAL_LIVE_REGISTERED_BATCH_PARENT_REQUIRED')
    children=[json.loads(Path(s['spec']).read_text())for s in batch['slots']];validate_layout(batch,children)
    if not any(s['spec']==str(spec_path)and s['spec_sha256']==digest(spec_path)for s in batch['slots']):raise PermissionError('CHILD_NOT_PINNED_IN_BATCH')
    binding=json.loads(Path(spec['campaign_binding']).read_text())
    if binding.get('max_target_instances')!=2 or binding.get('owner_model')!='ONE_PARENT_INHERITED_CANONICAL_LOCKS':raise PermissionError('REGISTERED_TWO_INSTANCE_SCOPE_REQUIRED')
    fds=json.loads(os.environ.get('WM02_AUTO1_OWNER_FDS','[]'))
    if len(fds)!=2 or fds!=started['fds']:raise PermissionError('TRUSTED_INHERITED_OWNER_FDS_REQUIRED')
    for fd,path in zip(fds,(campaign/'OWNER.lock',spec['physical_owner_lock'])):verify_inherited_owner(fd,path)
    try:yield
    finally:
        for fd in fds:os.close(fd)


def run_batch(spec_path,output,run_id):
    from .auto1_native import validate_scope
    from .runtime import admitted_run
    from scripts.pi0292.durable import OwnerLock,write
    spec=json.loads(Path(spec_path).read_text());campaign=Path(spec['campaign_root'])
    validate_scope(spec,json.loads((campaign/'control/LEASE.json').read_text()),json.loads(Path(spec['registry']).read_text()),host=os.uname().nodename,uid=os.getuid(),now=time.time())
    if spec.get('pilot1_binding'):
        from .pilot1_scope import validate
        validate(spec,json.loads(Path(spec['registry']).read_text()))
        runs=Path(spec['native_runs_root']).resolve(strict=True)
        proof=Path(spec['output_storage_proof']);value=json.loads(proof.read_text())
        if (digest(proof)!=spec['output_storage_proof_sha256']or value.get('status')!='PASSED_NODE_LOCAL_HDD_IO'
                or value.get('same_root_device')is not False or runs.stat().st_dev!=value['device']
                or runs.stat().st_dev==Path('/').stat().st_dev):
            raise PermissionError('ACTUAL_QUALIFIED_NON_ROOT_LOCAL_DATA_DISK_REQUIRED')
    if spec.get('next2_binding'):
        from .next2_scope import validate
        validate(spec,json.loads(Path(spec['registry']).read_text()))
    if spec.get('next1_binding'):
        from .next1_scope import validate
        validate(spec,json.loads(Path(spec['registry']).read_text()))
    children=[json.loads(Path(s['spec']).read_text())for s in spec['slots']];validate_layout(spec,children)
    with admitted_run(spec_path,run_id,'wm02-auto1-target-batch',output,physics=True):
        campaign_owner=OwnerLock(campaign/'OWNER.lock')
        try:
            physical_owner=OwnerLock(spec['physical_owner_lock'])
            try:
                fds=[campaign_owner.handle.fileno(),physical_owner.handle.fileno()]
                write_json(Path(output)/'INHERITED_OWNER.private.json',dict(pid=os.getpid(),process_start=process_start(os.getpid()),fds=fds,batch_spec_sha256=digest(spec_path)))
                jobs=[];handles=[];start=time.monotonic()
                try:
                    for slot,child in zip(spec['slots'],children):
                        env=child_environment(child,fds)
                        argv=[child['python'],'-B','-m','scripts.wm02.commission_target','--spec',slot['spec'],'--output',str(Path(child['registered_root'])/'native-01'),'--run-id','native-01']
                        h=(Path(output)/(Path(child['registered_root']).name+'.log')).open('x');handles.append(h)
                        jobs.append((child,subprocess.Popen(argv,cwd=child['source_root'],env=env,stdout=h,stderr=subprocess.STDOUT,pass_fds=fds)))
                    wait_children(spec,jobs)
                    rows=[]
                    for child,p in jobs:
                        result=Path(child['registered_root'])/'native-01/RESULT.json';value=json.loads(result.read_text())if result.exists()else{}
                        lease_path=Path(child['lease']);lease=json.loads(lease_path.read_text());lease.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=p.returncode);write(lease_path,lease)
                        diag=Path(child['registered_root'])/'native-01/DYNAMIC_DIAGNOSTIC.json';atoms=json.loads(diag.read_text())['complete_task_atoms']if diag.exists()else 0
                        rows.append(dict(run=Path(child['registered_root']).name,exit_code=p.returncode,status=value.get('status'),ticks=value.get('actual_target_integrations',0),shutdown_clean=value.get('shutdown_clean',False),ROS_DOMAIN_ID=child['ROS_DOMAIN_ID'],cpus=child['cpus'],task_atoms=atoms,collection=child['phase'].startswith('collect-target-')))
                        if spec.get('pilot1_binding'):rows[-1]['split']=recorded_split(spec,child)
                        if child['phase']in('collect-target-next1','collect-target-next2'):
                            episode=json.loads(Path(child['episode_manifest']).read_text())
                            rows[-1]['sampled_WM_atoms']=24 if episode['purpose']in('ACTION_PILOT','ACTION_DEV','HOLD_REPEAT','PROSPECTIVE')and value.get('status')=='NATIVE_STAGE_PASSED'else 0
                    result=dict(status='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'if all(r['exit_code']==0 and r['status']=='NATIVE_STAGE_PASSED'and r['shutdown_clean']for r in rows)else'TWO_ENVIRONMENT_NATIVE_PILOT_FAILED_PRESERVED',children=rows,wall_s=time.monotonic()-start,physical_owners=1,WM_updates=0,Actor_updates=0,learning_atoms=sum(r['task_atoms']for r in rows if r['collection']and r['status']=='NATIVE_STAGE_PASSED'))
                    if spec.get('pilot1_binding'):
                        result['recorded_task_atoms']=sum(r['task_atoms']for r in rows)
                        result['learning_atoms']=sum(r['task_atoms']for r in rows if r['split']=='TRAIN')if result['status']=='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'else 0
                        result['learning_atoms_definition']='ELIGIBLE_TRAIN_ONLY_NOT_DEV_TEST_OR_OPTIMIZER_UPDATES'
                        if spec.get('schema')=='WM02_PILOT2_TWO_COVERAGE_BATCH_V1':
                            passed=result['status']=='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'
                            result['learning_atoms']=24*sum(r['split']=='TRAIN'for r in rows)if passed else 0
                            result['validation_replay_atoms']=24*sum(r['split']=='DEV'for r in rows)if passed else 0
                            result['learning_atoms_definition']='ELIGIBLE_LATE_DECISION_TRAIN_REPLAY_ATOMS_24_PER_EPISODE'
                            if spec.get('purpose')=='COVERAGE_DUAL_DURATION_COMMISSION':
                                result.update(qualification_only=True,learning_atoms=0,validation_replay_atoms=0,
                                    learning_atoms_definition='FULL_DURATION_HOLD_COMMISSION_NOT_TRAIN_DATA')
                    if spec.get('next1_binding')or spec.get('next2_binding'):
                        result['recorded_task_atoms']=sum(r['task_atoms']for r in rows)
                        result['learning_atoms']=sum(r.get('sampled_WM_atoms',0)for r in rows)if result['status']=='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'else 0
                    write_json(Path(output)/'RESULT.json',result);return result
                finally:
                    for child,p in jobs:
                        if p.poll()is None:
                            write(Path(child['registered_root'])/'STOP.json',dict(reason='BATCH_EXCEPTION_CLOSEOUT'))
                            try:p.wait(timeout=45)
                            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
                    for h in handles:h.close()
                    for child in children:
                        owned=next((p for c,p in jobs if c is child),None)
                        if owned is None or owned.poll()is not None:
                            path=Path(child['lease']);lease=json.loads(path.read_text())
                            if lease['status']=='ACTIVE':
                                lease.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=None if owned is None else owned.returncode,started=owned is not None);write(path,lease)
            finally:physical_owner.close()
        finally:campaign_owner.close()
