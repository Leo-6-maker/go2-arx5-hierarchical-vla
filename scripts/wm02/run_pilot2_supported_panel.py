"""Finite supported-policy panel using the existing registry, Journal and workers.

The parent scope and frozen input are registered separately at a real cold
boundary. This entry grants no site permission and launches no GPU physics.
"""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from contextlib import closing


def run(input_path):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.pilot2_supported import validate
    from go2arx5_vla.world_model.pilot2_contact_evaluation import summarize
    from go2arx5_vla.world_model.target_parallel import validate_layout
    inp=json.loads(input_path.read_text());stage=input_path.parent;lease_path=stage/'control/LEASE.json'
    lease=json.loads(lease_path.read_text())
    if lease['status']!='ACTIVE'or lease['run_uuid']!=stage.name or lease['input_sha256']!=digest(input_path):
        raise PermissionError('ACTUAL_FRESH_SUPPORTED_PANEL_SCOPE_REQUIRED')
    plan=json.loads(Path(inp['panel']).read_text());base=json.loads(Path(inp['base_spec']).read_text())
    if digest(inp['panel'])!=inp['panel_sha256']or plan['schema']!='WM02_PILOT2_REPEATED_SUPPORTED_POLICY_PANEL_V1':
        raise ValueError('FROZEN_SUPPORTED_POLICY_PANEL_REQUIRED')
    summarize(plan,[])  # Check scientific condition uniqueness before any grant.
    source=Path(inp['source_root']);runs=Path(inp['output_parent']);campaign=Path(base['campaign_root'])
    assets=json.loads(Path(base['scene_allowlist']).read_text());variants={v['name']:v for v in inp['variants']}
    if any(variants[v['name']]['method']!=v['method']for v in plan['variants']):raise ValueError('PREREGISTERED_POLICY_VARIANTS_CHANGED')
    j=Journal(campaign/'control/journal',campaign.name);proofs={};results=[];observations=[];start=time.monotonic();cpu=time.process_time()
    def case(row):return(row['parent'],row['task_goal'],row['variant'])
    def spec_for(root,row,batch=None):
        scene=next(s for s in assets['scenes']if s['scene_id']==row['scene_id']);variant=variants[row['variant']]
        root.mkdir();(root/'control').mkdir()
        for n in ['data','cache','logs','tmp']:(root/n).mkdir()
        e=json.loads((source/'experiments/wm02/pilot2_supported_collection_v1.json').read_text())
        e.update(run=str(root),schema='WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1',purpose='SUPPORTED_POLICY_EVALUATION',
            command_program='SELECT_KEEP_RELEASE_AFTER_SUPPORTED_HOLD',close_start_atom=56,policy_decision_atom=112,
            scene_id=row['scene_id'],parent=row['parent'],split=row['split'],method=row['method'],task_goal=row['task_goal'],repeat_index=row['repeat_index'])
        write(root/'EPISODE_MANIFEST.private.json',e);q=inp['scene_qualifications'][row['scene_id']];s=dict(base)
        for key in ['parallel_batch_spec','single_policy_qualification','supported_policy_parallel_schema',
                    'supported_policy_numpy','supported_policy_numpy_sha256','supported_policy_training_result','supported_policy_training_result_sha256']:
            s.pop(key,None)
        s.update(source_sha=inp['source_sha'],source_root=str(source),source_files=inp['source_files'],phase=e['phase'],max_ticks=6703,
            role='WM02_'+root.name.upper().replace('-','_'),registered_root=str(root),registry_snapshot=str(root/'REGISTRY_SNAPSHOT.private.json'),
            lease=str(root/'control/LEASE.json'),deadline_epoch=min(time.time()+600,inp['deadline_epoch']),episode_manifest=str(root/'EPISODE_MANIFEST.private.json'),
            episode_manifest_sha256=digest(root/'EPISODE_MANIFEST.private.json'),model_path=scene['model_path'],pilot2_plan_sha=inp['pilot2_plan_sha'],
            cpus=list(range(row['slot']*4,row['slot']*4+4)),ROS_DOMAIN_ID=220+row['slot'],
            previous_stage=q['path'],previous_stage_sha256=q['sha256'],scene_qualification=q['path'],scene_qualification_sha256=q['sha256'],
            supported_program_qualifications=inp['program_qualifications'][row['scene_id']],
            output_storage_proof=inp['storage_proof'],output_storage_proof_sha256=inp['storage_proof_sha256'])
        s['frozen_files']=dict(base['frozen_files']);s['frozen_files'].pop(base['model_path'],None);s['frozen_files'][scene['model_path']]=scene['model_sha256']
        if variant['method']=='TRAINED_GOAL_POLICY':
            for k in ['supported_policy_numpy','supported_policy_numpy_sha256','supported_policy_training_result','supported_policy_training_result_sha256']:s[k]=variant[k]
        if batch:s.update(parallel_batch_spec=str(batch/'EXECUTION_SPEC.json'),supported_policy_parallel_schema='WM02_PILOT2_TWO_SUPPORTED_POLICY_BATCH_V1',single_policy_qualification=proofs[case(row)])
        for n in ['data','cache','logs','tmp']:s[n]=str(root/n)
        validate(s,json.loads(Path(q['path']).read_text()));write(root/'EXECUTION_SPEC.json',s);return s
    def grant(specs,msg):
        reg,pin=snapshot(base['registry'])
        for s in specs:reg['entries'].append(dict(role=s['role'],path=s['registered_root'],run_uuid=Path(s['registered_root']).name,
            executor_node=s['host'],owner=s['owner'],scope='WM02_PILOT1_SIM_ONLY',source_sha=inp['source_sha'],
            pilot1_binding=s['pilot1_binding'],pilot2_plan_sha=inp['pilot2_plan_sha'],local_lease=s['lease']))
        once(Path(base['registry']),reg,msg+'-REGISTER',expected_before_sha256=pin)
        for s in specs:
            root=Path(s['registered_root']);write(root/'REGISTRY_SNAPSHOT.private.json',reg)
            write(Path(s['lease']),dict(status='ACTIVE',run_uuid=root.name,spec_sha256=digest(root/'EXECUTION_SPEC.json')))
    def worker(s,batch=False):
        root=Path(s['registered_root']);env=dict(os.environ,**base['native_environment'])
        env.update(PYTHONPATH=str(source/'src')+':'+str(source)+':'+env.get('PYTHONPATH',''),CUDA_VISIBLE_DEVICES='',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(root/'tmp'),
            ROS_LOG_DIR=str(root/'logs'),XDG_CACHE_HOME=str(root/'cache'),XDG_CONFIG_HOME=str(root/'cache'))
        argv=['taskset','-c',','.join(map(str,s['cpus'])),s['python'],'-B','-m',
            'scripts.wm02.pilot1_parallel_worker'if batch else'scripts.wm02.pilot1_native_worker','--spec',str(root/'EXECUTION_SPEC.json')]
        def execute():
            with(root/'logs/worker-out.log').open('xb')as out,(root/'logs/worker-err.log').open('xb')as err:
                q=subprocess.Popen(argv,cwd=source,env=env,stdout=out,stderr=err)
                try:
                    while q.poll()is None:
                        write(stage/'PROGRESS.private.json',dict(current_run=root.name,pid=q.pid,completed_items=len(results),evaluations=len(observations),updated_epoch=time.time()))
                        if time.time()>=s['deadline_epoch']:write(root/'STOP.json',dict(reason='FINITE_SUPPORTED_PANEL_ENDPOINT'))
                        time.sleep(.5)
                finally:
                    if q.poll()is None:
                        write(root/'STOP.json',dict(reason='SUPPORTED_PANEL_COORDINATOR_EXIT'))
                        try:q.wait(timeout=90)
                        except subprocess.TimeoutExpired:q.terminate();q.wait(timeout=45)
            return dict(argv=argv,exit_code=q.returncode)
        return j.once(root.name.upper()+'-RUN',dict(spec_sha256=digest(root/'EXECUTION_SPEC.json')),execute)
    def observe(root,row):
        native=root/'native-01';files={k:native/n for k,n in [('native','RESULT.json'),('decision','SUPPORTED_POLICY_DECISION.private.json'),('score','SUPPORTED_CLOSE_RESULT.private.json')]}
        values={k:json.loads(p.read_text())if p.exists()else{}for k,p in files.items()};n,d,s=[values[k]for k in ['native','decision','score']]
        passed=bool(n.get('status')=='NATIVE_STAGE_PASSED'and n.get('shutdown_clean')and not n.get('physics_stops')and n.get('actual_target_integrations')==6703)
        model=variants[row['variant']].get('supported_policy_numpy_sha256')
        valid=bool(passed and all(values.values())and d.get('native_step')==5303 and d.get('future_response_observed')is False
            and d.get('method')==row['method']and d.get('task_goal')==row['task_goal']and d.get('policy_sha256')==model
            and n.get('native_binary_sha256')==inp['native_binary_sha256'])
        obs=dict(row,attempted=native.exists(),native_passed=passed,evidence_valid=valid,safety_stop=bool(n.get('physics_stops')),
            decision_prefix_eligible=bool(d.get('decision_prefix_eligible')),task_success=bool(s.get('goal_success')))
        observations.append(obs);write(stage/'OBSERVATIONS.private.json',observations)
        return dict(run=root.name,row=row,observation=obs,score={k:v for k,v in s.items()if k!='samples'},
            native_ticks=n.get('actual_target_integrations'),evidence_sha256={k:digest(p)for k,p in files.items()if p.exists()})
    code=1;failure=None
    if (stage/'SUPERVISOR.private.json').exists():raise FileExistsError('RECONCILE_EXISTING_SUPPORTED_PANEL')
    write(stage/'SUPERVISOR.private.json',dict(pid=os.getpid(),input_sha256=digest(input_path),started_epoch=time.time()))
    def stop(signum,frame):raise KeyboardInterrupt('SUPPORTED_PANEL_SIGNAL_'+str(signum))
    old_handlers={s:signal.signal(s,stop)for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for index,row in enumerate(plan['single_rows']):
            root=runs/(stage.name+'-single-%02d'%index);msg=root.name.upper()
            def register():
                with closing(OwnerLock(campaign/'OWNER.lock')),closing(OwnerLock(base['physical_owner_lock'])):
                    s=spec_for(root,row);grant([s],msg);return dict(root=str(root))
            j.once(msg+'-SPEC',dict(row=row,source_sha=inp['source_sha']),register)
            s=json.loads((root/'EXECUTION_SPEC.json').read_text());w=worker(s);o=observe(root,row)
            results.append(dict(kind='SINGLE_QUALIFICATION_AND_EVALUATION',worker=w,child=o));write(stage/'RESULTS.private.json',results)
            if w['exit_code']or not o['observation']['native_passed']:raise RuntimeError('SUPPORTED_SINGLE_NATIVE_FAILURE_PRESERVED')
            if o['observation']['evidence_valid']and o['observation']['decision_prefix_eligible']:
                proofs[case(row)]=dict(path=str(root/'native-01/RESULT.json'),sha256=digest(root/'native-01/RESULT.json'),
                    decision_sha256=digest(root/'native-01/SUPPORTED_POLICY_DECISION.private.json'),episode_sha256=digest(root/'EPISODE_MANIFEST.private.json'))
        rows=plan['paired_repeat_rows']
        for index in range(0,len(rows),2):
            group=rows[index:index+2];root=runs/(stage.name+'-pair-%02d'%(index//2));msg=root.name.upper()
            if any(case(row)not in proofs for row in group):
                results.append(dict(kind='NOT_RUN_MISSING_SAFE_ELIGIBLE_SINGLE',rows=group));write(stage/'RESULTS.private.json',results);continue
            def register():
                with closing(OwnerLock(campaign/'OWNER.lock')),closing(OwnerLock(base['physical_owner_lock'])):
                    root.mkdir();(root/'control').mkdir()
                    for n in ['data','cache','logs','tmp']:(root/n).mkdir()
                    children=[spec_for(runs/(root.name+'-slot-'+str(i)),row,root)for i,row in enumerate(group)]
                    s=dict(base);s.update(schema='WM02_PILOT2_TWO_SUPPORTED_POLICY_BATCH_V1',purpose='SUPPORTED_POLICY_PRODUCTION',
                        profile='wm02-auto1-target-batch',source_sha=inp['source_sha'],source_root=str(source),source_files=inp['source_files'],
                        registered_root=str(root),registry_snapshot=str(root/'REGISTRY_SNAPSHOT.private.json'),lease=str(root/'control/LEASE.json'),
                        role='WM02_'+root.name.upper().replace('-','_'),deadline_epoch=min(time.time()+600,inp['deadline_epoch']),cpus=list(range(8)),
                        native_runs_root=str(runs),slots=[dict(spec=str(Path(s['registered_root'])/'EXECUTION_SPEC.json'),spec_sha256=digest(Path(s['registered_root'])/'EXECUTION_SPEC.json'))for s in children],
                        parallel_qualification=inp['parallel_qualification'],parallel_qualification_sha256=inp['parallel_qualification_sha256'],
                        pilot2_plan_sha=inp['pilot2_plan_sha'],output_storage_proof=inp['storage_proof'],output_storage_proof_sha256=inp['storage_proof_sha256'])
                    for n in ['data','cache','logs','tmp']:s[n]=str(root/n)
                    validate_layout(s,children);write(root/'EXECUTION_SPEC.json',s);grant(children+[s],msg);return dict(root=str(root))
            j.once(msg+'-SPEC',dict(rows=group,source_sha=inp['source_sha']),register);s=json.loads((root/'EXECUTION_SPEC.json').read_text());w=worker(s,True)
            children=[observe(runs/(root.name+'-slot-'+str(i)),row)for i,row in enumerate(group)]
            results.append(dict(kind='PAIRED_INDEPENDENT_REPEAT',worker=w,children=children));write(stage/'RESULTS.private.json',results)
            if w['exit_code']or any(not c['observation']['native_passed']for c in children):raise RuntimeError('SUPPORTED_PAIR_NATIVE_FAILURE_PRESERVED')
        code=0
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        summary=summarize(plan,observations);write(stage/'SUMMARY.private.json',summary)
        result=dict(status='SUPPORTED_POLICY_PANEL_COMPLETE'if code==0 else'SUPPORTED_POLICY_PANEL_FAILED_PRESERVED',exit_code=code,failure=failure,
            summary_sha256=digest(stage/'SUMMARY.private.json'),wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,WM_updates=0,Actor_updates=0)
        write(stage/'RESULT.private.json',result);v=json.loads(lease_path.read_text());v.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=code);write(lease_path,v)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args(argv);return run(a.input)


if __name__=='__main__':raise SystemExit(main())
