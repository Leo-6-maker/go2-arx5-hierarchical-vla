"""Finite capture-only successor plus two zero-update diagnostics."""
import json,os,sys,time,traceback,struct
from pathlib import Path
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal
from scripts.wm02.mobile_boundary_runner_v1 import registered
from scripts.wm02.mfree_unattended_supervisor_v1 import exact_job,data_bytes
from go2arx5_vla.world_model.mfree_recovery_v2 import RecoveryLedger,register,launch_service
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,unit_state,started_unit,disk_admission,close
from go2arx5_vla.world_model.durable_control_v2 import read_control
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import sha
GIB=1024**3

def run(path):
    s=read_control(path);registered(s,path);root=Path(s['registered_root']);control=root/'control';ledger=RecoveryLedger(control);journal=Journal(control/'journal',root.name);owner=OwnerLock(control/'supervisor-owner.lock');ledger.start_epoch(os.getpid(),Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],Path('/proc/self/cgroup').read_text());failure=None;sequence=0;terminal=False
    def dispatch(allowed,reason,*,ready=True,closed=False,validated=False):write(control/'DISPATCH.private.json',dict(schema='MFREE_RECOVERY_DISPATCH_V2',generation=s['dispatch_generation'],ready=ready,epoch=time.time(),collectors=allowed,closed=closed,state=reason,single_world_validation=not validated))
    def launch_job(name,jobpath,value,operation,cpus):
        value=dict(value,registered_root=str(jobpath),role=(root.name+'-'+name).upper(),cpus=cpus);dirs(jobpath)
        if operation=='--collector':(jobpath/'runs').mkdir()
        register(s,[(jobpath,value)]);job=dict(path=str(jobpath),unit=s['unit_prefix']+'-'+name,spec_sha256=sha(jobpath/'EXECUTION_SPEC.json'),state='LAUNCH_INTENT',started_epoch=time.time())
        ledger.job_update(name,**job);journal.once('LAUNCH-'+name,dict(spec_sha256=job['spec_sha256']),lambda:launch_service(value,job['unit'],operation));ledger.job_update(name,state='RUNNING',process=started_unit(job['unit']))
    # Publish a valid READY barrier with dispatch closed BEFORE starting workers.
    dispatch([],'WAIT_DISPATCH',ready=False)
    try:
        for item in s['collectors']:
            name=item['id'];jobs=read_control(ledger.path)['jobs']
            if name not in jobs:
                value=dict(s,worker_id=name,ROS_DOMAIN_ID=item['domain'],mobile_world_lock=str(Path(s['world_control'])/f"ros-{item['domain']}.lock"));value['native_environment']=dict(s['native_environment'],ROS_DOMAIN_ID=str(item['domain']))
                launch_job(name,Path(item['path']),value,'--collector',item['cpus'])
            else:
                state,actual,result=exact_job(s,jobs[name]);ledger.job_update(name,state=state,process=actual,result=result)
                if state=='UNKNOWN':raise RuntimeError('UNKNOWN_EXISTING_RECOVERY_WORKER_NO_REPLAY')
        jobs=read_control(ledger.path)['jobs']
        if 'query-audit'not in jobs:launch_job('query-audit',root/'query-audit',dict(s,ROS_DOMAIN_ID=196),'--audit',[6,7])
        dispatch([],'READY_WAIT_CAPACITY')
        while time.time()<s['deadline_epoch']-180:
            now=time.time();state=read_control(ledger.path);stop=(root/'STOP.json').exists();closed=now>=s['deadline_epoch']-3600
            if sequence%5==0:write(control/'DATA_BYTES.private.json',dict(epoch=now,bytes=data_bytes(root)+s['prior_new_data_bytes']))
            unknown=False
            for name in [c['id']for c in s['collectors']]+['query-audit','diagnostic']:
                job=state['jobs'].get(name)
                if not job or job['state']in ('DONE','BLOCKED'):continue
                status,actual,result=exact_job(s,job);ledger.job_update(name,state=status,process=actual,result=result)
                if status=='UNKNOWN':unknown=True
                if status=='DONE'and result and read_control(result['path']).get('exit_code',0):ledger.job_update(name,state='BLOCKED',result=result,reason='TERMINAL_DIAGNOSTIC_OR_WORKER_FAILURE_NO_BLIND_RETRY')
            state=read_control(ledger.path);attempts=state['attempts'];active=[r for r in attempts if r['state']in ('CLAIMED','REGISTERED','LAUNCHED','RUNNING','UNKNOWN')];unknown|=any(r['state']=='UNKNOWN'for r in active)
            fresh=[r for r in attempts if r.get('execution_generation')==s['dispatch_generation']or r.get('worker')in {c['id']for c in s['collectors']}];qualified=[r for r in fresh if r['state']=='QUALIFIED'];first_done=bool(qualified)
            diagnostic=state['jobs'].get('diagnostic');diag_done=diagnostic and diagnostic['state']in ('DONE','BLOCKED');reserve_model=first_done and not diag_done
            available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
            measured=[];native_current=0;model_current=0
            for item in s['collectors']:
                actual=unit_state(s['unit_prefix']+'-'+item['id']);measured.append(actual)
                if actual['pid']:
                    if not actual['MemoryCurrent'].isdigit():raise RuntimeError('UNKNOWN_NATIVE_MEMORY_MEASUREMENT')
                    native_current+=min(int(actual['MemoryCurrent']),s['native_peak_bound_bytes'])
            if diagnostic and diagnostic['state']=='RUNNING':
                actual=unit_state(diagnostic['unit']);measured.append(actual)
                if actual['pid']and actual['MemoryCurrent'].isdigit():model_current=int(actual['MemoryCurrent'])
            target=2 if first_done else 1;reserve=max(0,15*GIB-model_current)if reserve_model else 0
            required=max(2*GIB,2*GIB+target*s['native_peak_bound_bytes']+reserve-native_current)
            if available<required:target=1;required=max(2*GIB,2*GIB+s['native_peak_bound_bytes']+reserve-native_current)
            resource_ok=available>=required and disk_admission(s,6*GIB)
            eligible=[c['id']for c in s['collectors']if state['jobs'][c['id']]['state']=='RUNNING'];desired=eligible[:target];draining={r['worker']for r in active if r['worker']not in desired};allowed=desired[:max(0,target-len(draining))]if resource_ok and not unknown and not stop and not closed else[]
            reason='RESOURCE_QUEUE'if not resource_ok else'BLOCKED_UNKNOWN'if unknown else'COLLECT_C60'
            # First complete the single approved continuation. Then give the
            # finite zero-update GPU job priority at an episode seam.
            if first_done and not diagnostic:
                allowed=[];reason='DRAIN_FOR_ZERO_UPDATE_DIAGNOSTIC'
                if not active and available>=17*GIB and not stop and not unknown:
                    try:lock=OwnerLock(s['neural_owner_lock'])
                    except BlockingIOError:lock=None
                    if lock:
                        lock.close();value=dict(s,python=s['SLA_python'],ROS_DOMAIN_ID=196,deadline_epoch=min(s['deadline_epoch']-300,now+3*3600));launch_job('diagnostic',root/'diagnostic',value,'--diagnostic',[12,13])
            ledger.seal_ready();state=read_control(ledger.path)
            if 'C60'in state['snapshots']:allowed=[];reason='DATA_READY_TRAINING_PAUSED'
            if stop or closed:allowed=[];reason='OPERATOR_STOP'if stop else'FINITE_DEADLINE'
            dispatch(allowed,reason,closed=bool(stop or closed or 'C60'in state['snapshots']),validated=first_done)
            native_ticks=[]
            for r in active:
                if r.get('spec'):
                    p=Path(r['spec']['path']).parent/'native-01/state.bin'
                    if p.exists():
                        try:
                            with p.open('rb')as h:header=h.read(36)
                            magic,tick,*_=struct.unpack('<8sQdIII',header)
                            if magic==b'GRTSTATE':native_ticks.append(dict(id=r['id'],tick=tick))
                        except (FileNotFoundError,struct.error):pass
            done=[r for r in attempts if r.get('completed_epoch')];last=max(done,key=lambda r:r['completed_epoch'])if done else None
            write(root/'HEARTBEAT.private.json',dict(schema='MFREE_RECOVERY_PROGRESS_V2',epoch=now,sequence=sequence,pid=os.getpid(),state=reason,alive_workers=[x['unit']for x in measured if x['pid']],eligible_workers=allowed,active_attempts=active,last_completed_attempt=last['id']if last else None,last_native_tick=native_ticks,last_committed_update=1000,new_committed_updates=0,last_snapshot=state['snapshots'].get('C60',state['snapshots']['C20']),qualified_total=sum(r['state']=='QUALIFIED'for r in attempts),ledger_attempts=len(attempts),actual_physical_requests=sum(r.get('physical_requests',0)for r in attempts),new_qualified_since_revision=len(qualified),available_host_bytes=available,required_host_bytes=required,first_one_world_qualified=first_done,maximum_worlds=2,diagnostic_job=state['jobs'].get('diagnostic'),query_audit_job=state['jobs'].get('query-audit'),C60_C100_training='PAUSED_BY_USER_REVISION',F1_student='QUARANTINED',M_ASSIST='QUARANTINED',deadline_epoch=s['deadline_epoch'],next_action='WAIT_RESOURCE_OR_OWNER'if not allowed else'CONTINUE_PREDECLARED_CAPTURE'))
            diagnostic=state['jobs'].get('diagnostic');audit=state['jobs'].get('query-audit')
            if (stop or closed or 'C60'in state['snapshots'])and not active and (not diagnostic or diagnostic['state']in ('DONE','BLOCKED'))and audit['state']in ('DONE','BLOCKED'):terminal=True;break
            sequence+=1;time.sleep(30)
    except BaseException:failure=traceback.format_exc()
    finally:
        dispatch([],'BLOCKED_ENGINEERING'if failure else'DATA_READY_TRAINING_PAUSED',closed=True)
        write(root/'RESULT.private.json',dict(exit_code=0 if terminal else 1,failure=failure,collection_terminal=terminal,training_submitted=False,new_optimizer_updates=0,deadline_epoch=s['deadline_epoch']))
        close(s,0 if terminal else 1);owner.close()
    return 0 if terminal else 1

if __name__=='__main__':raise SystemExit(run(sys.argv[sys.argv.index('--supervisor')+1]))
