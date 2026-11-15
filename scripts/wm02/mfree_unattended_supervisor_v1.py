"""Finite registered C20/C60/C100 successor; execution never needs a chat."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal,MutationOutcomeUnknown
from scripts.wm02.mobile_boundary_runner_v1 import registered
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import Ledger,read,sha
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,fields,register,unit_state,started_unit,launch_service,memory_admission,disk_admission,close
GIB=1024**3

def data_bytes(root):
    total=0
    for folder,_,files in os.walk(root):
        for name in files:
            try:total+=(Path(folder)/name).stat().st_blocks*512
            except FileNotFoundError:pass
    return total

def exact_job(spec,job):
    """No unknown or vanished owner can be converted into a new launch."""
    state=unit_state(job['unit']);p=Path(job['path']);result=p/'RESULT.private.json'
    if state['pid']:
        prior=job.get('process')
        if prior and (prior['pid']!=state['pid']or prior['proc_start']!=state['proc_start']or prior['ControlGroup']!=state['ControlGroup']):raise RuntimeError('UNKNOWN_JOB_PROCESS_CHANGED_NO_RELAUNCH')
        if sha(p/'EXECUTION_SPEC.json')!=job['spec_sha256']:raise RuntimeError('JOB_SPEC_CHANGED')
        return 'RUNNING',state,None
    if not state['cgroup_pids']and result.exists()and read(p/'control/LEASE.json')['status']=='CLOSED':return 'DONE',state,dict(path=str(result),sha256=sha(result))
    return 'UNKNOWN',state,None

def run(path):
    s=read(path);registered(s,path);root=Path(s['registered_root']);control=root/'control';ledger=Ledger(control);journal=Journal(control/'journal',root.name)
    owner=OwnerLock(control/'supervisor-owner.lock');ledger.start_epoch(os.getpid(),Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],Path('/proc/self/cgroup').read_text())
    old=Path(s['old_pipeline_root']);counter=0;failure=None;final=False;waiting_since=None;trial=None;baseline=None;four_allowed=False;finished_neural=set()
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time(),source_sha=s['source_sha']))
    try:
        for item in s['collectors']:
            # Exactly-once launch. A recorded INTENT/vanished process blocks;
            # supervisor restart only reconciles live independent child units.
            job=read(ledger.path)['jobs'].get(item['id'])
            if job is None:
                worker=Path(item['path']);dirs(worker);(worker/'runs').mkdir()
                value=dict(s,registered_root=str(worker),role=(root.name+'-'+item['id']).upper(),worker_id=item['id'],ROS_DOMAIN_ID=item['domain'],cpus=item['cpus'],mobile_world_lock=str(Path(s['world_control'])/f"ros-{item['domain']}.lock"))
                env=dict(s['native_environment']);env['ROS_DOMAIN_ID']=str(item['domain']);value['native_environment']=env
                register(s,[(worker,value)]);job=dict(path=str(worker),unit=item['unit'],spec_sha256=sha(worker/'EXECUTION_SPEC.json'),state='LAUNCH_INTENT')
                ledger.job_update(item['id'],**job)
                journal.once('LAUNCH-'+item['id'],dict(spec_sha256=job['spec_sha256']),lambda:launch_service(value,item['unit'],'--collector'))
                actual=started_unit(item['unit']);ledger.job_update(item['id'],state='RUNNING',process=actual)
            elif job['state']in ('LAUNCH_INTENT','UNKNOWN'):
                raise RuntimeError('COLLECTOR_LAUNCH_UNKNOWN_REQUIRES_HUMAN_RECONCILIATION')
            else:
                state,actual,result=exact_job(s,job)
                ledger.job_update(item['id'],state=state,process=actual,result=result)
        while True:
            now=time.time();state=read(ledger.path);stop=(root/'STOP.json').exists();deadline=now>=s['deadline_epoch']-3600
            if counter%5==0:write(control/'DATA_BYTES.private.json',dict(epoch=now,bytes=data_bytes(root)))
            ledger.seal_ready();state=read(ledger.path)
            old_unit=unit_state(s['old_pipeline_unit']);old_lease=read(old/'control/LEASE.json');old_closed=old_lease['status']=='CLOSED'and old_unit['pid']==0 and not old_unit['cgroup_pids']
            if old_unit['pid']and (old_unit['pid']!=s['old_pipeline_identity']['pid']or old_unit['proc_start']!=s['old_pipeline_identity']['proc_start']or sha(old/'EXECUTION_SPEC.json')!=s['old_pipeline_spec_sha256']):raise RuntimeError('OLD_OWNER_CHANGED_NO_TAKEOVER')
            old_progress=read(old/'candidate1000/PROGRESS.private.json')if(old/'candidate1000/PROGRESS.private.json').exists()else{}
            old_step=old_progress.get('step',0)
            if(old/'candidate1000/RESULT.private.json').exists():old_step=max(old_step,read(old/'candidate1000/RESULT.private.json').get('step',0))
            ledger.job_update('C20',state='IMPORTED_TERMINAL'if old_closed else'HEALTHY_EXISTING_UNCHANGED',root=str(old),source_sha=s['old_pipeline_source_sha'],spec_sha256=s['old_pipeline_spec_sha256'],real_updates=old_step,real_visits=32*old_step)
            neural=None;unknown=False
            for name,job in state['jobs'].items():
                if not name.startswith('neural-')or job['state']in ('DONE','BLOCKED','DEDUPLICATED'):continue
                status,actual,result=exact_job(s,job)
                ledger.job_update(name,state=status,process=actual,result=result)
                if status=='RUNNING':neural=dict(job,process=actual)
                elif status=='UNKNOWN':unknown=True
            for item in s['collectors']:
                job=read(ledger.path)['jobs'][item['id']]
                if job['state']=='RUNNING':
                    status,actual,result=exact_job(s,job);ledger.job_update(item['id'],state=status,process=actual,result=result)
                    if status=='UNKNOWN':unknown=True
            queue=[n for n in ('C60','C100')if n in state['snapshots']and 'neural-'+n not in state['jobs']]
            reserved=1 if not old_closed or neural or queue else 0
            # Conservative initial commitment = two total worlds, including the
            # legacy evaluator's future world. No borrowing that reservation.
            base_slots=max(0,2-reserved);total_limit=4 if four_allowed or trial else 2
            slots=total_limit-reserved
            good=[r for r in state['attempts']if not r.get('carried_over')and r['state']=='QUALIFIED']
            ready=[r for r in s['collectors']if read(ledger.path)['jobs'][r['id']]['state']=='RUNNING']
            # First prove the extra domain by rotating one ordinary planned
            # attempt into it, with the same total commitment.
            proved={198,197}|{r.get('world')for r in good}
            unproved=[d for d in (195,194)if d not in proved]
            if len(good)>=1 and unproved:ready=sorted(ready,key=lambda r:r['domain']!=unproved[0])
            now_good=len(good)
            if baseline is None:baseline=dict(epoch=now,good=now_good)
            if (not trial and not four_allowed and len(good)>=4 and {194,195,197,198}<=proved
                    and now-baseline['epoch']>60 and memory_admission(s,4)):
                rate=(now_good-baseline['good'])/max(1,now-baseline['epoch'])
                trial=dict(epoch=now,good=now_good,baseline_qualified_per_hour=rate*3600)
            if trial and now_good-trial['good']>=6:
                rate=(now_good-trial['good'])/max(1,now-trial['epoch'])*3600
                four_allowed=rate>=1.15*trial['baseline_qualified_per_hour'] and not state['isolated_families']
                write(root/'FOUR_WORLD_TRIAL.private.json',dict(**trial,completed_epoch=now,qualified_per_hour=rate,retained=four_allowed,threshold_gain=.15))
                trial=None
                # A completed unfavorable trial is not repeatedly re-tried.
                if not four_allowed:baseline=dict(epoch=now+10**9,good=now_good)
            resource_spec=dict(s)
            if neural:resource_spec['resource_units']=[*s['resource_units'],neural['unit']]
            resource_ok=memory_admission(resource_spec,total_limit)and disk_admission(s,6*GIB)
            queue=[n for n in ('C60','C100')if n in state['snapshots']and 'neural-'+n not in state['jobs']]
            queued=bool(queue and not neural)
            if neural:
                progress_path=Path(neural['path'])/'PROGRESS.private.json'
                queued=progress_path.exists()and read(progress_path).get('phase','').startswith('RESOURCE_QUEUE')
            if queued:
                if waiting_since is None:waiting_since=now
            else:waiting_since=None
            if waiting_since and now-waiting_since>=1800:slots=base_slots
            # Pause means drain at episode boundary: leases/healthy native child
            # are retained. Shared dispatch expires if this controller dies.
            live_attempts=[r for r in state['attempts']if r['state']in ('CLAIMED','REGISTERED','RUNNING','LAUNCHED','UNKNOWN')]
            unknown=unknown or any(r['state']=='UNKNOWN'for r in live_attempts)
            desired=[x['id']for x in ready[:max(0,slots)]]
            draining={r['worker']for r in live_attempts if r['worker']not in desired}
            allowed=desired[:max(0,slots-len(draining))]if resource_ok and not stop and not deadline and not unknown else[]
            if 'C100'in state['snapshots']:allowed=[]
            write(control/'DISPATCH.private.json',dict(epoch=now,collectors=allowed,total_world_commitment=total_limit,legacy_or_candidate_evaluation_reserved=reserved,closed=stop or deadline,reason='RESOURCE_OR_UNKNOWN_QUEUE'if not resource_ok or unknown else'FINITE_BUDGET'))
            if queue and old_closed and not neural and not unknown and not stop and not deadline:
                stage=queue[0];snapshot=state['snapshots'][stage];descriptor=read(snapshot['pin']['path'])
                key=hashlib.sha256(json.dumps(dict(episodes=sorted((r['id'],r['result']['sha256'])for r in descriptor['episodes']),recipe='FM1000_B32_SEED_FIXED_RAW_PARENT',parent=s['raw_parent_revision']),sort_keys=True).encode()).hexdigest()
                same=next((j for j in state['jobs'].values()if j.get('run_key')==key),None)
                # C20 is imported under the same semantic key at registration.
                if same:ledger.job_update('neural-'+stage,state='DEDUPLICATED',run_key=key,reused=same)
                elif resource_ok and len({r['worker']for r in live_attempts})<=max(0,total_limit-1)and disk_admission(s,12*GIB):
                    old_cost=min(18*3600,max(0,(old_lease.get('closed_epoch',now)-s['old_pipeline_started_epoch'])))
                    jobs=[j for n,j in state['jobs'].items()if n.startswith('neural-')and 'started_epoch'in j]
                    used=old_cost+sum(max(0,(read(Path(j['path'])/'control/LEASE.json').get('closed_epoch',now)-j['started_epoch']))for j in jobs)
                    remaining=36*3600-used
                    if remaining<3600:ledger.job_update('neural-'+stage,state='BLOCKED',reason='36_GPU_HOUR_FINITE_ENVELOPE')
                    else:
                        jobroot=root/'snapshots'/stage;dirs(jobroot)
                        value=dict(s,registered_root=str(jobroot),role=(root.name+'-NEURAL-'+stage).upper(),snapshot_id=stage,snapshot_descriptor=snapshot['pin'],deadline_epoch=min(s['deadline_epoch']-300,now+remaining),dataset=str(jobroot/'dataset'),cpus=[12,13],ROS_DOMAIN_ID=196,pipeline_program_sha256=sha(Path(s['source_root'])/'scripts/wm02/mfree_unattended_neural_v1.py'))
                        register(s,[(jobroot,value)]);unit=s['unit_prefix']+'-neural-'+stage.lower();job=dict(path=str(jobroot),unit=unit,state='LAUNCH_INTENT',started_epoch=now,spec_sha256=sha(jobroot/'EXECUTION_SPEC.json'),run_key=key)
                        ledger.job_update('neural-'+stage,**job)
                        journal.once('LAUNCH-NEURAL-'+stage,dict(spec_sha256=job['spec_sha256']),lambda:launch_service(value,unit,'--neural'))
                        actual=started_unit(unit);ledger.job_update('neural-'+stage,state='RUNNING',process=actual)
            active=[r for r in state['attempts']if r['state']in ('CLAIMED','REGISTERED','RUNNING','LAUNCHED','UNKNOWN')]
            hashed=[r['initial_state_hashes']for r in state['attempts']if r.get('initial_state_hashes')]
            unique_initial=len({json.dumps(v,sort_keys=True)for v in hashed})
            write(root/'HEARTBEAT.private.json',dict(initial_state_diversity=dict(measured=len(hashed),unique=unique_initial,duplicate_rate=1-unique_initial/len(hashed)if hashed else None,seed_labels_do_not_prove_distinct_states=True),schema='MFREE_UNATTENDED_72H_HEARTBEAT_V1',epoch=now,sequence=counter,pid=os.getpid(),source_sha=s['source_sha'],spec_sha256=sha(path),deadline_epoch=s['deadline_epoch'],attempts=len(state['attempts']),physical_requests=sum(r.get('physical_requests',0)for r in state['attempts']),qualified=sum(r['state']=='QUALIFIED'for r in state['attempts']),active=active,snapshots=state['snapshots'],jobs=read(ledger.path)['jobs'],old_C20_real_updates=old_step,old_C20_real_visits=32*old_step,worlds_committed=total_limit,collectors_admitted=allowed,neural_reserved=reserved,resource_ok=resource_ok,unknown_owner_blocks=unknown,M_ASSIST='QUARANTINED',old_RP8_passed=False))
            state=read(ledger.path)
            endpoints=all('neural-'+n in state['jobs']and state['jobs']['neural-'+n]['state']in ('DONE','BLOCKED','DEDUPLICATED')for n in ('C60','C100'))
            if (stop or deadline or ('C100'in state['snapshots']and endpoints))and not active and not neural:final=True;break
            if now>=s['deadline_epoch']-180:break
            counter+=1;time.sleep(60)
    except BaseException:failure=traceback.format_exc()
    finally:
        write(control/'DISPATCH.private.json',dict(epoch=time.time(),collectors=[],closed=not bool(failure and len(read(ledger.path)['supervisor_epochs'])<3),reason='CONTROLLER_TERMINAL',failure=failure))
        epochs=len(read(ledger.path)['supervisor_epochs'])
        if failure and epochs<3:
            write(root/f'CONTROLLER_EPOCH_{epochs}_FAILURE.private.json',dict(failure=failure,epoch=time.time(),restart_with_reconciliation_only=True))
        else:
            write(root/'RESULT.private.json',dict(exit_code=0 if final else 1,failure=failure,plan_complete=final,all_evidence=str(ledger.path),WBC_changes=False,safety_changes=False))
            close(s,0 if final else 1)
        owner.close()
    return 0 if final else 1

if __name__=='__main__':raise SystemExit(run(sys.argv[sys.argv.index('--supervisor')+1]))
