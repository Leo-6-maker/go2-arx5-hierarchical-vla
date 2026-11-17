"""V3 registered nonphysical controller: finite RGB IPC panel, no native dispatch."""
from contextlib import closing
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from scripts.pi0292.durable import write, OwnerLock
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import Receiver, read, identity, wait_receipt
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,close


def run(spec):
    root=Path(spec['registered_root']);start=time.monotonic();journal=Journal(root/'control/journal',root.name)
    pin=lambda p:dict(path=str(p),sha256=digest(p))
    def state(phase,**kw):write(root/'goal_state.json',dict(phase=phase,epoch=time.time(),source_sha=spec['source_sha'],
        new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,goal_met=False,**kw))
    bank=spec['input_bank'];assert digest(bank['path'])==bank['sha256'];records=read(bank['path'])['records']
    bindings=spec['request_bindings'];rows=[];generated=0;worker_results=[];loads=[]
    # Fixed input order before model load. Same known bank, no requalification claim.
    normal=[dict(kind='normal',entry=i//2,noise=i%2,partial=bool(i%2))for i in range(16)]
    faults=['pre_stop','pre_cancel','pre_restart','binding','mid_EEF_stop','mid_EEF_cancel',
        'mid_EEF_restart','mid_EEF_advance','mid_GRIP_stop','mid_GRIP_cancel','mid_GRIP_restart',
        'mid_GRIP_advance','pre_install_stop','pre_install_cancel','response_identity','response_hash']
    panel=normal+[dict(kind=k,entry=i%8,noise=0)for i,k in enumerate(faults)]
    write(root/'PANEL.private.json',panel)
    def wait(path,p,seconds=90):
        return wait_receipt(path,alive=lambda:p.poll() is None,
            stopped=lambda:time.time()>=spec['deadline_epoch']-60 or(root/'STOP.json').exists(),
            timeout_s=seconds)
    spans=((0,8),(8,32))
    prior_request_count=prior_generations=0
    if spec.get('technical_continuation'):
        old=Path(spec['technical_continuation']['root']);c=spec['technical_continuation']
        assert digest(old/'RESULT.private.json')==c['result_sha256']
        assert read(old/'control/LEASE.json')['status']=='CLOSED'
        from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
        assert not unit_state('wm02-'+old.name)['pid']
        assert read(old/'PANEL.private.json')==panel
        rows=read(old/'REQUEST_RECORDS.private.json')
        assert len(rows)==10 and [r['index']for r in rows]==list(range(10))
        q=read(old/'request-10/REQUEST.private.json');response=read(old/'request-10/RESPONSE.json')
        receiver_state=read(old/'request-10/receiver/live.json')
        assert receiver_state['emitted_atoms']==0 and receiver_state['installed'] is None
        assert not list((old/'request-10/receiver').glob('install-*'))
        assert digest(response['action']['path'])==response['action']['sha256']
        assert response['request_id']==q['request_id'] and response['status']=='GENERATED_NOT_INSTALLED'
        assert all(not(old/('request-%02d'%i)).exists()for i in range(11,32))
        for child in sorted(old.glob('gpu-worker-*')):
            assert read(child/'control/LEASE.json')['status']=='CLOSED'
            result=read(child/'RESULT.private.json');assert result['exit_code']==0
            generated+=result['attempted_generations'];worker_results.append(pin(child/'RESULT.private.json'))
            loads.append(read(child/'READY.private.json'))
        assert generated==22
        # Preserve the failed11th request. Its old service deadline is expired;
        # never install it later, extend TTL, or regenerate it to improve counts.
        rows.append(dict(index=10,kind='normal',request_id=q['request_id'],input_sha256=q['input']['sha256'],
            receiver=None,consumption=None,failure='PRESERVED_RECEIPT_ENOENT_NO_INSTALL_NO_REGENERATION',
            response_status=response['status'],generated=2,branch_wall_s=response['branch_wall_s'],
            end_to_end_read_IPC_compile_ACK_wall_s=None,clock_mode='SIM_PAUSED_ONLY',physical_integrations=0,
            original_result_sha256=c['result_sha256']))
        spans=((11,32),);prior_request_count=11;prior_generations=generated
        write(root/'REQUEST_RECORDS.private.json',rows)
    with closing(OwnerLock(root/'control/supervisor-owner.lock')):
        # Two cold registered processes prove service restart/cache invalidation.
        for batch,(lo,hi)in enumerate(spans):
            child=root/('gpu-worker-%d'%batch);dirs(child);bus=child/'bus';bus.mkdir()
            value=dict(spec,registered_root=str(child),campaign_root=str(root),role='MFREE_HYBRID_GPU_IPC_V3',
                ipc_root=str(bus),compilation_cache_dir=str(child/'cache'),parent_params=spec['raw_parent'],
                new_complete_generations_budget=64-generated,unified_requests_budget=hi-lo,
                max_walltime_s=math.ceil(spec['deadline_epoch']-time.time())+60,
                max_gpu_hours=(spec['deadline_epoch']-time.time())/3600+1)
            allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=spec['neural_owner_lock'])
            binding=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=spec['host'],uid=spec['uid'],
                GPU_uuid=spec['GPU_uuid'],GPU_name=spec['GPU_name'],GPU_headroom_bytes=1024**3,
                host_headroom_bytes=2*1024**3,allocator_limit_bytes=spec['allocator_limit_bytes'],
                allocation_deadline_epoch=spec['allocation_deadline_epoch'],
                paths={k:str(Path(value[k]).resolve())for k in ('parent_params','dataset','compilation_cache_dir')},allocation=allocation)
            write(child/'SITE_BINDING.private.json',binding);value['site_binding']=pin(child/'SITE_BINDING.private.json')
            register(spec,[(child,value)]);sp=child/'EXECUTION_SPEC.json'
            def execute():
                with(child/'logs/worker.log').open('xb')as log:
                    p=subprocess.Popen([value['python'],'-B',str(Path(value['source_root'])/'scripts/wm02/close_goal_bootstrap_v1.py'),
                        '--hybrid-ipc-worker-v3',str(sp)],env={**os.environ,**value['native_environment']},stdout=log,stderr=subprocess.STDOUT)
                write(child/'LAUNCH_RECEIPT.private.json',dict(pid=p.pid,proc_start=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],epoch=time.time()))
                try:
                    state('A2_LOADING_GPU',batch=batch,finished_requests=len(rows),completed_generations=generated)
                    ready=wait(child/'READY.private.json',p,600);loads.append(ready)
                    for index in range(lo,hi):
                        item=panel[index];entry=records[item['entry']];folder=root/('request-%02d'%index);folder.mkdir()
                        # Input is captured ordinary RGB/state. Truth action never enters input.
                        t=time.monotonic()
                        if digest(entry['input']['path'])!=entry['input']['sha256']:raise ValueError('CAPTURED_INPUT_CHANGED')
                        if digest(entry['truth']['path'])!=entry['truth']['sha256']:raise ValueError('CAPTURED_REFERENCE_SOURCE_CHANGED')
                        with np.load(entry['input']['path'],allow_pickle=False)as z:
                            state10=z['state_eef10'].copy();images=[z['body_rgb'].copy(),z['wrist_rgb'].copy()];frame=z['T_policy_from_mount'].copy()
                        with np.load(entry['truth']['path'],allow_pickle=False)as z:reference=z['accepted_reference_TCP'].copy()
                        f=folder/'INPUT.npz'
                        with f.open('xb')as out:np.savez(out,state_eef10=state10,body_rgb=images[0],wrist_rgb=images[1],T_policy_from_mount=frame,policy_from_world=np.asarray(entry['policy_from_world']),accepted_reference_TCP=reference)
                        receiver=Receiver(folder/'receiver',entry['episode']+'-replay-%02d'%index,
                            int(round(entry['previous_reference']['query_native_s']/.002)),entry['previous_reference'],
                            dict(kind='RECORDED_NATIVE_COMMAND',bank=bank,input_sha256=entry['input']['sha256']),bindings,operator_stop=root/'STOP.json')
                        q=receiver.request(pin(f),entry['language'],item['noise']);write(folder/'REQUEST.private.json',q)
                        kind=item['kind']
                        def mutate(kind):
                            if kind.endswith('stop'):receiver.stop()
                            elif kind.endswith('cancel'):receiver.cancel()
                            elif kind.endswith('restart'):receiver.restart()
                            elif kind.endswith('advance'):receiver.state['native_step']+=25;receiver.save()
                        if kind.startswith('pre_')and not kind.startswith('pre_install'):mutate(kind)
                        if kind=='binding':
                            q['bindings']=dict(bindings,wrong=True);q['request_id']=identity({k:v for k,v in q.items()if k!='request_id'})
                        message=dict(request=q,live_path=str(receiver.root/'live.json'),response_root=str(folder))
                        write(bus/'REQUEST.json',message)
                        # Both branches use a real IPC handshake, enabling mid-branch faults.
                        for name in ('EEF','GRIP'):
                            while not(folder/(name+'_DONE.json')).exists()and not(folder/'RESPONSE.json').exists():
                                if p.poll()is not None or time.monotonic()-t>90:raise TimeoutError('BRANCH_OR_RESPONSE_TIMEOUT')
                                time.sleep(.002)
                            if(folder/(name+'_DONE.json')).exists():
                                if kind.startswith('mid_'+name+'_'):mutate(kind)
                                write(folder/(name+'_CONTINUE.json'),dict(request_id=q['request_id']))
                        response=wait(folder/'RESPONSE.json',p);inference_end=time.monotonic()
                        if kind.startswith('pre_install'):mutate(kind)
                        if kind=='response_identity':response['request_id']='WRONG_QUERY'
                        if kind=='response_hash':response['action']['sha256']='0'*64
                        ack=None;failure=None;consumption=None
                        try:
                            if response['status']!='GENERATED_NOT_INSTALLED':raise RuntimeError(response.get('failure','NO_RESPONSE_ACTION'))
                            ack=receiver.install(q,response)
                            ack_latency=time.monotonic()-t
                            if kind!='normal':raise AssertionError('FAULT_WAS_INSTALLED')
                            # Same input/noise and unchanged numerical route must reproduce old outputs.
                            old=Path(spec['old_prediction_root'])/('%03d-%d.npz'%(entry['index'],item['noise']))
                            with np.load(old,allow_pickle=False)as a,np.load(response['action']['path'],allow_pickle=False)as b:
                                for key in ('FM_physical_H40','grip_physical_H40','physical_H40'):
                                    if not np.array_equal(a[key],b[key]):raise ValueError('UNCHANGED_ROUTE_OUTPUT_REGRESSION:'+key)
                            # Duplicate delivery is idempotent, with zero extra model generations.
                            assert receiver.install(q,response)==ack
                            consumption=receiver.consume(7 if item['partial'] else 10)
                            if item['partial']:receiver.cancel('PARTIAL7_TEST')
                            else:receiver.finish()
                        except (ValueError,RuntimeError,TimeoutError)as e:
                            if kind=='normal':raise
                            failure=str(e)
                        e2e=ack_latency if ack is not None else time.monotonic()-t
                        row=dict(index=index,kind=kind,request_id=q['request_id'],input_sha256=q['input']['sha256'],
                            receiver=ack,consumption=consumption,failure=failure,response_status=response['status'],
                            generated=response['new_generations_this_request'],branch_wall_s=response.get('branch_wall_s'),
                            camera_recorded_read_to_response_s=inference_end-t,
                            end_to_end_read_IPC_compile_ACK_wall_s=e2e,exceeds_350ms=e2e>.35,exceeds_400ms=e2e>=.4,
                            clock_mode='SIM_PAUSED_ONLY',physical_integrations=0)
                        rows.append(row);write(root/'REQUEST_RECORDS.private.json',rows)
                        state('A2_REAL_GPU_IPC',batch=batch,finished_requests=len(rows),last=row)
                    write(bus/'FINISH.json',dict(reason='FINITE_PANEL_COMPLETE'))
                    while p.poll()is None:
                        if time.time()>spec['deadline_epoch']-30:raise TimeoutError('FINITE_WORKER_SHUTDOWN')
                        time.sleep(.05)
                    result=read(child/'RESULT.private.json')
                    if p.returncode or result['exit_code']:raise RuntimeError('PRESERVED_GPU_WORKER_FAILURE')
                    return pin(child/'RESULT.private.json')
                finally:
                    if p.poll()is None:
                        write(child/'STOP.json',dict(reason='CONTROLLER_FAILURE_NO_REPLAY',epoch=time.time()))
                        # Graceful bounded shutdown; only own process.
                        try:p.wait(timeout=15)
                        except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=15)
            result_pin=journal.once('GPU-IPC-BATCH-%d'%batch,pin(sp),execute)
            result=read(result_pin['path']);generated+=result['attempted_generations'];worker_results.append(result_pin)
            assert generated<=64
        normal_rows=[r for r in rows if r['kind']=='normal' and r['receiver'] is not None];fault_rows=[r for r in rows if r['kind']!='normal']
        times=[r['end_to_end_read_IPC_compile_ACK_wall_s']for r in normal_rows]
        result=dict(schema='MFREE_HYBRID_GPU_IPC_ACTUAL_V3',exit_code=0,status='SIM_PAUSED_ONLY_PRESERVED_FAILURE_REALTIME_AND_NATIVE_UNQUALIFIED',
            unified_requests=len(rows),unified_requests_this_role=len(rows)-prior_request_count,
            prior_unified_requests=prior_request_count,new_complete_generations=generated,
            new_complete_generations_this_role=generated-prior_generations,prior_generations=prior_generations,
            normal_requests=sum(r['kind']=='normal'for r in rows),normal_install_ACK=len(normal_rows),
            preserved_technical_uninstalled=sum(r['kind']=='normal' and r['receiver'] is None for r in rows),
            full_P10=sum(bool(r['consumption']['full_P10'])for r in normal_rows),partial7=sum(r['consumption']['consumed_atoms']==7 for r in normal_rows),
            fault_cases=len(fault_rows),fault_misinstalls=sum(r['receiver']is not None for r in fault_rows),
            latency_p50_p95_p99_max_s=np.percentile(times,[50,95,99,100]).tolist(),
            over_350ms=sum(t>.35 for t in times),over_400ms=sum(t>=.4 for t in times),
            startup_loads=loads,worker_results=worker_results,new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,
            realtime_qualified=False,native_WBC_certified=False,physical_admission=False,goal_met=False,wall_s=time.monotonic()-start)
        state('A2_COMPLETE_WAIT_INDEPENDENT_READMISSION',result=result);return result


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:r=run(s)
    except BaseException:r=dict(exit_code=1,status='PRESERVED_TECHNICAL_FAILURE_NO_REPLAY',failure=traceback.format_exc(),
        actual_child_counts_authoritative=True,new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,goal_met=False)
    write(root/'RESULT.private.json',r);close(s,r['exit_code']);raise SystemExit(r['exit_code'])
