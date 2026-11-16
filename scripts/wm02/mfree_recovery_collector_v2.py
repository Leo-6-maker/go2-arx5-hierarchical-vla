"""Cold V2 expert worker: bounded control reads, registered launch phases."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal
from scripts.wm02.mobile_boundary_runner_v1 import registered
from scripts.wm02.mfree_collection_v1 import capture_check
from go2arx5_vla.world_model.mobile_grasp_budget_v3 import qualification_record
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,fields,close,disk_admission
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha
from go2arx5_vla.world_model.mfree_recovery_v2 import RecoveryLedger as Ledger,register
from go2arx5_vla.world_model.durable_control_v2 import valid_dispatch

def run(path):
    s=read(path);lease=registered(s,path);root=Path(s['registered_root']);campaign=Path(s['campaign_root']);control=campaign/'control';ledger=Ledger(control)
    if(root/'SUPERVISOR.private.json').exists():raise PermissionError('EXISTING_COLLECTOR_REQUIRES_RECONCILIATION_NO_RESTART')
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time(),proc_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
    journal=Journal(root/'control/journal',root.name);failure=None;code=0;active=None;finished=0
    try:
        while time.time()<s['deadline_epoch']-3600:
            write(root/'HEARTBEAT.private.json',dict(epoch=time.time(),pid=os.getpid(),active=active))
            if(campaign/'STOP.json').exists():break
            dispatch=valid_dispatch(control/'DISPATCH.private.json',s['dispatch_generation'])
            if dispatch.get('closed'):break
            if time.time()-dispatch['epoch']>180 or s['worker_id']not in dispatch['collectors']:
                time.sleep(10);continue
            if dispatch.get('single_world_validation')and finished>=1:time.sleep(3);continue
            if not disk_admission(s,6*1024**3):time.sleep(30);continue
            world=OwnerLock(s['mobile_world_lock'])
            try:
                row=ledger.claim(s['worker_id'])
                if row is None:time.sleep(3);continue
                active=row['id'];case=root/'runs'/active;dirs(case)
                template=read(s['family_templates'][row['family']]['path']);scene=next(x for x in read(s['scene_catalog']['path'])if x['family']==row['family']and x['layout_id']==row['layout_id'])
                if sha(scene['path'])!=scene['sha256']or not scene['static_initial_contact_pass']:raise ValueError('FROZEN_TRAIN_LAYOUT_CHANGED')
                contract=dict(template['collection_contract'],episode_id=active,reset=row['reset'],seed=row['seed'],attempt=row['ordinal'],session=campaign.name,split='TRAIN',family=row['family'],layout_id=row['layout_id'])
                spec=dict(template,**fields(s));spec.update(registered_root=str(case),role=(root.name+'-'+active).upper(),campaign_root=str(campaign),unattended_control=str(control),collection_root=str(campaign),mobile_group=str(root),
                    ROS_DOMAIN_ID=s['ROS_DOMAIN_ID'],cpus=s['cpus'],deadline_epoch=s['deadline_epoch']-120,model_path=scene['path'],mobile_world_lock=s['mobile_world_lock'],collection_contract=contract,native_environment=s['native_environment'])
                spec['mobile_condition']=dict(template['mobile_condition'],reset=row['reset'],seed=row['seed'],layout=row['layout_id'],collection_episode=active)
                spec['frozen_files']=dict(template['frozen_files']);spec['frozen_files'][scene['path']]=scene['sha256']
                spec['stage_identity_sha256']=__import__('hashlib').sha256((s['plan_sha']+'|'+active).encode()).hexdigest()
                for name in ('logs','tmp','cache','data'):spec[name]=str(case/name)
                journal.once('REGISTER-'+active,dict(spec_sha256=__import__('hashlib').sha256((json.dumps(spec,indent=2,allow_nan=False)+'\n').encode()).hexdigest()),lambda:register(s,[(case,spec)]));ledger.update(active,state='REGISTERED',spec=dict(path=str(case/'EXECUTION_SPEC.json'),sha256=sha(case/'EXECUTION_SPEC.json')),world=s['ROS_DOMAIN_ID'])
                children=[]
                def launch():
                    with(case/'logs/case.log').open('xb')as handle:
                        child=subprocess.Popen([s['python'],'-B',str(Path(s['source_root'])/'scripts/wm02/mfree_recovery_bootstrap_v2.py'),'--case',str(case/'EXECUTION_SPEC.json'),'--world-lock-fd',str(world.handle.fileno())],env=dict(os.environ,**s['native_environment']),cwd=s['source_root'],stdout=handle,stderr=subprocess.STDOUT,pass_fds=(world.handle.fileno(),));children.append(child)
                    start=Path('/proc')/str(child.pid)/'stat'
                    receipt=dict(pid=child.pid,proc_start=start.read_text().rsplit(')',1)[1].split()[19],epoch=time.time(),cgroup=Path('/proc').joinpath(str(child.pid),'cgroup').read_text());write(case/'LAUNCH_RECEIPT.private.json',receipt);return receipt
                launched=journal.once(active,dict(spec_sha256=sha(case/'EXECUTION_SPEC.json')),launch)
                if not children:raise PermissionError('UNKNOWN_EXISTING_NATIVE_CHILD_NEVER_RELAUNCH')
                ledger.update(active,state='RUNNING',process=launched,physical_requests=1)
                child=children[0]
                while child.poll()is None:
                    write(root/'HEARTBEAT.private.json',dict(epoch=time.time(),pid=os.getpid(),active=active,child_pid=child.pid))
                    if time.time()>launched['epoch']+spec['wall_limit_s']+180:
                        # This belongs to this worker's registered cgroup only.
                        child.terminate();child.wait(timeout=60);raise TimeoutError('OWN_NATIVE_WALL_LIMIT_NO_REPLAY')
                    time.sleep(10)
                result_path=case/'RESULT.private.json'
                if not result_path.exists():raise RuntimeError('NATIVE_EXIT_WITHOUT_TERMINAL_EVIDENCE_UNKNOWN')
                result=read(result_path)
                if read(case/'control/LEASE.json')['status']!='CLOSED':raise RuntimeError('NATIVE_RESULT_LEASE_NOT_CLOSED')
                qualified=qualification_record(result)['qualified'];capture=capture_check(result)if qualified else None
                state='QUALIFIED'if qualified else'UNSAFE'if result['unsafe']else'ISOLATED_ABORT'if result.get('quarantine_abort')else'TECHNICAL'if result['technical_failure']else'TASK_FAILED'
                # Real qpos/qvel/ctrl at the initial simulator packet, not seed labels.
                import struct
                from scripts.pi0294.task_state import parse_state
                with(case/'native-01/state.bin.journal').open('rb')as handle:
                    size=handle.read(4);first=parse_state(handle.read(struct.unpack('<I',size)[0]))if size else None
                initial={}
                if first is not None:
                    import numpy as np,hashlib
                    # Packet field names are preserved if present; unsupported
                    # representations stay explicitly unmeasured, never "unique".
                    for key in ('qpos','qvel','ctrl'):
                        value=first.get(key)if isinstance(first,dict)else getattr(first,key,None)
                        if value is not None:initial[key]=hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                ledger.update(active,state=state,result=dict(path=str(result_path),sha256=sha(result_path)),qualified=qualified,capture=capture,physical_requests=1,native_integrations=result['actual_native_integrations'],wall_s=result['wall_s'],unsafe=result['unsafe'],technical_failure=result['technical_failure'],initial_state_hashes=initial,initial_diversity_measured=bool(initial),completed_epoch=time.time())
                if qualified:finished+=1
                if result['unsafe']:ledger.isolate(row['family'],dict(case=str(case),reason='ORIGINAL_NATIVE_SAFETY'))
                if result['technical_failure']:
                    # New/semantic/unknown failures have no pre-approved repair;
                    # block this cell instead of turning a retry budget into a loop.
                    with ledger.transaction()as value:value['blocked_cells'][row['cell']]=dict(reason='UNATTENDED_TECHNICAL_FAILURE_NO_KNOWN_REPAIR',evidence=str(result_path))
                active=None;ledger.seal_ready()
            finally:world.close()
    except BaseException:
        failure=traceback.format_exc();code=1
        if active:ledger.update(active,state='UNKNOWN',failure=failure)
    write(root/'RESULT.private.json',dict(exit_code=code,failure=failure,unknown_attempt=active,new_optimizer_updates=0))
    close(s,code);return code

if __name__=='__main__':raise SystemExit(run(sys.argv[sys.argv.index('--collector')+1]))
