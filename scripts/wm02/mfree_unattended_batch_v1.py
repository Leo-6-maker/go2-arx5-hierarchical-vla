"""Frozen native batch with the new shared expert-family dispatch fence."""
from contextlib import closing
import os,subprocess,time,traceback
from pathlib import Path
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.storage import digest
from .mobile_boundary_runner_v1 import registered,read
def run_batch(spec_path):
    from scripts.pi0294.e1.journal import Journal
    spec=read(spec_path);group=Path(spec['registered_root']);lease=registered(spec,spec_path)
    if lease['status']!='ACTIVE'or lease['spec_sha256']!=digest(spec_path):raise PermissionError('FINITE_REGISTERED_MOBILE_BATCH')
    if (group/'SUPERVISOR.private.json').exists():raise PermissionError('MOBILE_BATCH_ALREADY_LAUNCHED_NO_REPLAY')
    write(group/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time()))
    journal=Journal(group/'control/journal',group.name)
    outcomes=[];quarantine=set();code=1;failure=None
    try:
        with closing(OwnerLock(spec['mobile_world_lock']))as world:
            for path in spec['cases']:
                if time.time()>=spec['deadline_epoch']-900:break
                case=read(path);axis=case['mobile_condition']['axis'].lstrip('+-')
                if case.get('mobile_stage')=='L1':axis=case['mobile_condition']['family']+':'+case['mobile_condition']['mode']
                if spec.get('mobile_experiment_version')=='MFREE_STUDENT_60S_V1':axis=case['mobile_condition']['family']+':'+case['mobile_condition']['method']
                if spec.get('mobile_experiment_version')in('MOBILE_CONTINUOUS_TEACHER_V2','MOBILE_GRASP_BUDGET_V3'):axis=case['mobile_condition']['mode']
                shared=read(Path(spec['unattended_control'])/'ISOLATION.private.json')
                if case['mobile_condition']['family']in shared:continue
                if axis in quarantine:continue
                if(group/'STOP.json').exists():break
                if case.get('mobile_stage')=='L1'and case['mobile_condition']['method']!='EXPERT':
                    server=Path(read(case['resident_server_spec']['path'])['registered_root']);until=time.monotonic()+900
                    while not(server/'READY.private.json').exists():
                        if(server/'RESULT.private.json').exists():raise RuntimeError('SLA_RESIDENT_FAILED_BEFORE_STUDENT')
                        if time.monotonic()>until:raise TimeoutError('FINITE_OWN_MODEL_READINESS')
                        time.sleep(.2)
                write(group/'PROGRESS.private.json',dict(current=case['mobile_condition'],terminal=len(outcomes),epoch=time.time()))
                argv=[spec['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/mfree_unattended_bootstrap_v1.py'),'--case',path,
                      '--world-lock-fd',str(world.handle.fileno())]
                log=group/'logs'/(Path(path).parent.name+'.log')
                children=[]
                def launch():
                    with log.open('xb')as f:
                        children.append(subprocess.Popen(argv,cwd=spec['source_root'],env=dict(os.environ,**spec['native_environment']),
                            stdout=f,stderr=subprocess.STDOUT,pass_fds=(world.handle.fileno(),)))
                    return dict(pid=children[0].pid)
                journal.once(Path(path).parent.name,dict(spec_sha256=digest(path)),launch)
                if not children:raise PermissionError('PRIOR_CHILD_IDENTITY_MUST_BE_RECONCILED_NO_REPLAY')
                child=children[0]
                try:result_code=child.wait(timeout=case['wall_limit_s']+180)
                except BaseException:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
                root=Path(path).parent
                if not(root/'RESULT.private.json').exists():raise RuntimeError('MOBILE_CHILD_EXIT_WITHOUT_RESULT:'+str(result_code))
                result=read(root/'RESULT.private.json');outcomes.append(dict(root=root.name,result=result,result_sha256=digest(root/'RESULT.private.json')))
                write(group/'OUTCOMES.private.json',outcomes)
                if result['unsafe']:quarantine.add(axis)
                elif result['technical_failure']:raise RuntimeError('MOBILE_TECHNICAL_FAILURE_RETAINED_NO_RETRY')
            code=0
    except BaseException:failure=traceback.format_exc()
    if spec.get('mobile_experiment_version')in('MOBILE_CONTINUOUS_TEACHER_V2','MOBILE_GRASP_BUDGET_V3','MFREE_STUDENT_60S_V1'):
        for path in spec['cases']:
            root=Path(path).parent
            if not(root/'SUPERVISOR.private.json').exists():
                pending=read(root/'control/LEASE.json')
                c=read(path)['mobile_condition'];key=c['family']+':'+c['method']if spec.get('mobile_experiment_version')=='MFREE_STUDENT_60S_V1'else c['mode']
                reason='NATIVE_SAFETY_QUARANTINE'if key in quarantine else'BATCH_STOP_OR_DEADLINE'
                write(root/'control/LEASE.json',dict(pending,status='CLOSED_UNSTARTED',physical_requests=0,native_integrations=0,reason=reason,closed_epoch=time.time()))
    value=dict(schema='MOBILE_L1_FINITE_BATCH_ACTUAL_V1'if spec.get('mobile_stage')=='L1'else 'MOBILE_M0_FINITE_BATCH_ACTUAL_V1',exit_code=code,failure=failure,
        requested=len(spec['cases']),terminal=len(outcomes),outcomes=outcomes,quarantined_axes=sorted(quarantine),
        physical_requests=sum(v['result']['physical_requests']for v in outcomes),
        native_integrations=sum(v['result']['actual_native_integrations']for v in outcomes),
        new_training_updates=0,new_model_generations=sum(v['result']['model_generations']for v in outcomes),automatic_retry=False,goal_complete=False)
    if spec.get('mobile_experiment_version')=='MOBILE_GRASP_BUDGET_V3':
        from go2arx5_vla.world_model.mobile_grasp_budget_v3 import qualification_record
        qualifications=[qualification_record(v['result'])for v in outcomes]
        value.update(qualifications=qualifications,
            qualified_complete_experts=sum(v['qualified']for v in qualifications),
            all_four_qualified=len(qualifications)==4 and all(v['qualified']for v in qualifications),
            automatic_collection_started=False,automatic_training_started=False)
    write(group/'RESULT.private.json',value);write(group/'WORKER_EXIT.private.json',dict(exit_code=code))
    write(group/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
    return code


if __name__=='__main__':
 import sys
 raise SystemExit(run_batch(sys.argv[sys.argv.index('--batch')+1]))
