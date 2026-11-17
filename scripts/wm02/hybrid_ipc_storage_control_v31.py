"""V3.1 storage qualification, reusing the registered V3 GPU worker and compiler.

Fixed fresh scope <=32 joint requests /64 branch generations. No native API.
"""
from contextlib import closing
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from unittest.mock import patch
import numpy as np
from scripts.pi0292.durable import write,OwnerLock
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import Receiver,read,wait_receipt
from go2arx5_vla.world_model.hybrid_ipc_storage_v31 import initialize,check_size,archive_request
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs


def panel(records):
    chosen=[]
    for family in ('F1','F2'):
        episode=next(r['episode']for r in records if r['family']==family)
        for phase in ('PREGRASP','DESCEND','CLOSE','LIFT'):
            chosen.append(next(i for i,r in enumerate(records)if r['episode']==episode and r['phase']==phase))
    rows=[dict(kind='normal',entry=i,noise=n,partial=n==1,cached=False,clock_mode='SIM_PAUSED_ONLY')for i in chosen for n in(0,1)]
    faults=['service_restart','binding','mid_EEF_stop','mid_EEF_cancel','mid_EEF_restart','mid_EEF_advance',
        'mid_GRIP_stop','mid_GRIP_advance','wall_expire_after_GRIP','compile_expire','wire_stop',
        'after_ACK_stop','renew_ACK_late','strict_50ms_margin','response_identity','response_hash']
    for kind in faults:
        cached=kind in ('compile_expire','wire_stop','after_ACK_stop','renew_ACK_late','strict_50ms_margin')
        rows.append(dict(kind=kind,entry=0 if cached else chosen[0],noise=1 if cached else 0,partial=False,cached=cached,
            clock_mode='WALL_CLOCK_STRICT'if kind in ('wall_expire_after_GRIP','compile_expire','renew_ACK_late','strict_50ms_margin')else'SIM_PAUSED_ONLY'))
    return rows


def run(spec):
    if(spec['physics']or spec['new_physical_requests_budget']!=0 or spec['new_optimizer_updates_budget']!=0
            or spec['unified_requests_budget']!=32 or spec['new_complete_generations_budget']!=64):
        raise PermissionError('EXACT_USER_ADDITIONAL32_64_NONPHYSICAL_ZERO_TRAIN_SCOPE')
    root=Path(spec['registered_root']);pin=lambda p:dict(path=str(p),sha256=digest(p));begun=time.monotonic()
    bank=spec['input_bank'];assert digest(bank['path'])==bank['sha256'];records=read(bank['path'])['records'];cases=panel(records)
    write(root/'PANEL.private.json',cases);live=initialize(root);out=root/'requests';out.mkdir();rows=[];generated=0;worker_results=[]
    journal=Journal(root/'control/journal',root.name)
    def state(phase,progress_generations=None,**kw):write(root/'goal_state.json',dict(phase=phase,epoch=time.time(),source_sha=spec['source_sha'],
        completed_unified_requests=len(rows),additional_generations=generated if progress_generations is None else progress_generations,new_updates=0,new_visits=0,
        new_native_requests=0,new_integrations=0,goal_met=False,**kw))
    def wait(path,p,seconds=90):return wait_receipt(path,alive=lambda:p.poll()is None,
        stopped=lambda:(root/'STOP.json').exists()or time.time()>spec['deadline_epoch']-60,timeout_s=seconds)
    def until(at):
        while time.monotonic()<at:
            if(root/'STOP.json').exists():raise RuntimeError('OPERATOR_STOP')
            time.sleep(min(.001,max(.0001,at-time.monotonic())))
    with closing(OwnerLock(root/'control/supervisor-owner.lock')):
        for batch,(lo,hi)in enumerate(((0,17),(17,32))):
            child=root/('gpu-worker-%d'%batch);dirs(child);clive=live/child.name;clive.mkdir();bus=clive/'bus';bus.mkdir()
            value=dict(spec,registered_root=str(child),campaign_root=str(root),role='MFREE_HYBRID_GPU_IPC_V31_STORAGE',
                live_root=str(clive),ipc_root=str(bus),archive_all_branches=True,parent_params=spec['raw_parent'],
                compilation_cache_dir=str(child/'cache'),new_complete_generations_budget=min(64-generated,2*(hi-lo)),
                unified_requests_budget=hi-lo,max_walltime_s=math.ceil(spec['deadline_epoch']-time.time())+60,
                max_gpu_hours=(spec['deadline_epoch']-time.time())/3600+1)
            allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=spec['neural_owner_lock'])
            binding=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=spec['host'],uid=spec['uid'],GPU_uuid=spec['GPU_uuid'],
                GPU_name=spec['GPU_name'],GPU_headroom_bytes=1024**3,host_headroom_bytes=2*1024**3,
                allocator_limit_bytes=spec['allocator_limit_bytes'],allocation_deadline_epoch=spec['allocation_deadline_epoch'],
                paths={k:str(Path(value[k]).resolve())for k in('parent_params','dataset','compilation_cache_dir')},allocation=allocation)
            write(child/'SITE_BINDING.private.json',binding);value['site_binding']=pin(child/'SITE_BINDING.private.json')
            register(spec,[(child,value)]);sp=child/'EXECUTION_SPEC.json'
            def execute():
                with(child/'logs/worker.log').open('xb')as log:
                    p=subprocess.Popen([value['python'],'-B',str(Path(value['source_root'])/'scripts/wm02/close_goal_bootstrap_v1.py'),
                        '--hybrid-ipc-worker-v3',str(sp)],env={**os.environ,**value['native_environment']},stdout=log,stderr=subprocess.STDOUT)
                write(child/'LAUNCH_RECEIPT.private.json',dict(pid=p.pid,proc_start=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],epoch=time.time()))
                try:
                    state('V31_LOADING_SHARED_PARENT',batch=batch);wait(child/'READY.private.json',p,600)
                    for index in range(lo,hi):
                        item=cases[index];entry=records[item['entry']];kind=item['kind'];folder=clive/('request-%02d'%index);folder.mkdir()
                        # Durable reservation precedes measured capture. It bounds UNKNOWN,
                        # never claims two actual generations if a crash loses live counters.
                        write(child/'RESERVATION.private.json',dict(index=index,request_upper_branch_cost=0 if item['cached']or kind in('service_restart','binding')else 2,
                            actual_counts=read(clive/'GENERATION_COUNTS.private.json')if(clive/'GENERATION_COUNTS.private.json').exists()else{},automatic_retry=False))
                        state('V31_GPU_REQUEST_DISPATCH',inflight_index=index,progress_generations=generated+(read(clive/'GENERATION_COUNTS.private.json')['attempted']if(clive/'GENERATION_COUNTS.private.json').exists()else 0))
                        phases={};capture=time.monotonic()
                        if digest(entry['input']['path'])!=entry['input']['sha256']or digest(entry['truth']['path'])!=entry['truth']['sha256']:
                            raise ValueError('FROZEN_CAPTURE_OR_REFERENCE_CHANGED')
                        with np.load(entry['input']['path'],allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
                        with np.load(entry['truth']['path'],allow_pickle=False)as z:reference=z['accepted_reference_TCP'].copy()
                        f=folder/'INPUT.npz'
                        with f.open('xb')as handle:np.savez(handle,**data,policy_from_world=np.asarray(entry['policy_from_world']),accepted_reference_TCP=reference)
                        phases['input_read_pack_s']=time.monotonic()-capture
                        flag=folder/'OPERATOR_STOP';r=Receiver(folder/'receiver',entry['episode']+'-storage-%02d'%index,
                            entry['native_step'],entry['previous_reference'],dict(kind='RECORDED_NATIVE_COMMAND',bank=bank),
                            spec['request_bindings'],operator_stop=flag)
                        q=r.request(pin(f),entry['language'],item['noise'],clock_mode=item['clock_mode']);write(folder/'REQUEST.private.json',q)
                        def mutate(suffix):
                            if suffix.endswith('stop'):r.stop()
                            elif suffix.endswith('cancel'):r.cancel()
                            elif suffix.endswith('restart'):r.restart()
                            elif suffix.endswith('advance'):r.state['native_step']+=25;r.save()
                        message=dict(request=q,live_path=str(r.root/'live.json'),response_root=str(folder),delivery_id=1,
                            branch_barriers=kind.startswith('mid_')or kind=='wall_expire_after_GRIP',
                            restart_before_branches=kind=='service_restart',cached_protocol_replay=item['cached'])
                        if kind=='binding':message['request']=dict(q,bindings=dict(q['bindings'],wrong=True))
                        sent=time.monotonic();write(bus/'REQUEST.json',message)
                        if message['branch_barriers']:
                            for name in('EEF','GRIP'):
                                while not(folder/(name+'_DONE.json')).exists()and not(folder/'RESPONSE.json').exists():
                                    if p.poll()is not None or time.monotonic()-sent>90:raise TimeoutError('FINITE_BRANCH_WAIT')
                                    time.sleep(.0005)
                                if(folder/(name+'_DONE.json')).exists():
                                    if kind.startswith('mid_'+name+'_'):mutate(kind)
                                    if kind=='wall_expire_after_GRIP'and name=='GRIP':until(q['created_monotonic']+.401)
                                    write(folder/(name+'_CONTINUE.json'),dict(request_id=q['request_id']))
                        response=wait(folder/'RESPONSE.json',p);phases['IPC_to_response_s']=time.monotonic()-sent
                        if kind=='service_restart':
                            r.restart()
                            try:r.install(q,dict(request_id=q['request_id'],bindings=q['bindings'],action={}))
                            except RuntimeError:pass
                            else:raise AssertionError('OLD_QUERY_ACCEPTED_AFTER_SERVICE_RESTART')
                        if kind=='response_identity':response['request_id']='WRONG_QUERY'
                        if kind=='response_hash':response['action']['sha256']='0'*64
                        from go2arx5_vla.world_model import sla_dense_policy_v5 as compiler
                        from go2arx5_vla.world_model import hybrid_nonphysical_v3 as receiver_module
                        from scripts.pi0294 import base_intent
                        original_compile=compiler.compile_proposal;original_wire=base_intent.eef_document;original_write=receiver_module.write
                        def compile_timed(*args,**kw):
                            t=time.monotonic();v=original_compile(*args,**kw)
                            if kind=='compile_expire':until(q['created_monotonic']+.401)
                            phases['compiler_s']=time.monotonic()-t;return v
                        def wire_timed(*args,**kw):
                            t=time.monotonic();v=original_wire(*args,**kw)
                            if kind=='wire_stop':
                                subprocess.run([value['python'],'-B','-c','from pathlib import Path;import sys;Path(sys.argv[1]).write_text("EXTERNAL_STOP")',str(flag)],check=True,timeout=10)
                            phases['wire_s']=time.monotonic()-t;return v
                        def renewal_delayed(path,v):
                            original_write(path,v)
                            if kind=='renew_ACK_late'and path.name.startswith('renew-'):until(q['created_monotonic']+.401)
                        ack=None;error=None;consumption=None;late_guard=None;duplicate=None
                        try:
                            if response['status']!='GENERATED_NOT_INSTALLED':raise RuntimeError(response.get('failure',response['status']))
                            t=time.monotonic()
                            with patch.object(compiler,'compile_proposal',compile_timed),patch.object(base_intent,'eef_document',wire_timed):ack=r.install(q,response)
                            phases['install_total_s']=time.monotonic()-t;phases['capture_to_ACK_s']=time.monotonic()-capture
                            if kind=='normal':
                                old=Path(spec['old_prediction_root'])/('%03d-%d.npz'%(entry['index'],item['noise']))
                                with np.load(old,allow_pickle=False)as a,np.load(response['action']['path'],allow_pickle=False)as b:
                                    assert all(np.array_equal(a[k],b[k])for k in('FM_physical_H40','grip_physical_H40','physical_H40'))
                                before=read(clive/'GENERATION_COUNTS.private.json');message['delivery_id']=2;write(bus/'REQUEST.json',message)
                                while True:
                                    delivery=wait(folder/'DELIVERY_ACK.json',p)
                                    if delivery['delivery_id']==2:break
                                    time.sleep(.0005)
                                after=read(clive/'GENERATION_COUNTS.private.json');assert before['attempted']==after['attempted']
                                assert r.install(q,response)==ack;duplicate=True
                                consumption=r.consume(7 if item['partial'] else 10)
                                if item['partial']:r.cancel('PARTIAL7')
                                else:r.finish()
                            elif kind=='after_ACK_stop':
                                subprocess.run([value['python'],'-B','-c','from pathlib import Path;import sys;Path(sys.argv[1]).write_text("EXTERNAL_STOP")',str(flag)],check=True,timeout=10);r.consume(1)
                            elif kind=='strict_50ms_margin':
                                until(q['created_monotonic']+.36);r.consume(1)
                            elif kind=='renew_ACK_late':
                                r.consume(7);until(q['created_monotonic']+.35)
                                with patch.object(receiver_module,'write',renewal_delayed):r.consume(1)
                            else:raise AssertionError('FAULT_INSTALLED_UNEXPECTEDLY')
                        except (ValueError,RuntimeError,TimeoutError)as exc:
                            error=str(exc);phases['failure_elapsed_s']=time.monotonic()-capture
                            if kind=='normal':raise
                            if kind in('after_ACK_stop','strict_50ms_margin','renew_ACK_late'):late_guard=True
                        allowed_install=kind in('normal','after_ACK_stop','strict_50ms_margin','renew_ACK_late')
                        if ack is not None and not allowed_install:raise AssertionError('FAULT_MISINSTALL')
                        emitted=r.state['emitted_atoms']
                        if kind=='after_ACK_stop'and(ack is None or emitted!=0 or not r.state['stop']):raise AssertionError('STOP_AFTER_ACK_NOT_EXERCISED')
                        if kind=='strict_50ms_margin'and(ack is None or emitted!=0 or not late_guard):raise AssertionError('STRICT_WINDOW_NOT_EXERCISED')
                        if kind=='renew_ACK_late'and(ack is None or emitted!=7 or not late_guard):raise AssertionError('RENEWAL_GUARD_NOT_EXERCISED')
                        if kind=='compile_expire'and('compiler_s'not in phases or ack is not None):raise AssertionError('COMPILER_TTL_NOT_EXERCISED')
                        if kind=='wire_stop'and('wire_s'not in phases or ack is not None):raise AssertionError('WIRE_STOP_NOT_EXERCISED')
                        row=dict(index=index,kind=kind,family=entry['family'],entry=item['entry'],noise=item['noise'],cached_protocol_replay=item['cached'],
                            request_id=q['request_id'],clock_mode=item['clock_mode'],phases=phases,model_branch_s=response.get('branch_wall_s',[]),
                            new_generations=response['new_generations_this_request'],ACK=ack,duplicate_no_extra_generations=duplicate,
                            error=error,consumption=consumption,emitted_atoms=emitted,guard_exercised=late_guard,
                            native_requests=0,integrations=0)
                        # Archive after the measured ACK, never hide its cost in GPU busy time.
                        t=time.monotonic();row['archive']=archive_request(folder,out/folder.name)
                        write(child/'GENERATION_COUNTS.private.json',read(clive/'GENERATION_COUNTS.private.json'))
                        row['post_ACK_archive_s']=time.monotonic()-t;row['live_total_bytes']=check_size(live)
                        rows.append(row);write(root/'REQUEST_RECORDS.private.json',rows)
                        state('V31_GPU_IPC_QUALIFYING',progress_generations=generated+read(clive/'GENERATION_COUNTS.private.json')['attempted'],last_index=index,normal_ACK=sum(x['kind']=='normal'and x['ACK']is not None for x in rows))
                        if kind=='service_restart':break
                    if p.poll()is None:write(bus/'FINISH.json',dict(reason='FINITE_BATCH_COMPLETE'))
                    while p.poll()is None:
                        if time.time()>spec['deadline_epoch']-30:raise TimeoutError('WORKER_SHUTDOWN_DEADLINE')
                        time.sleep(.01)
                    result=read(child/'RESULT.private.json');assert p.returncode==result['exit_code']==0
                    return pin(child/'RESULT.private.json')
                finally:
                    if p.poll()is None:
                        write(child/'STOP.json',dict(reason='CONTROLLER_FAILURE_PRESERVE_NO_REPLAY'))
                        try:p.wait(timeout=15)
                        except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=15)
            result_pin=journal.once('V31_STORAGE_GPU_BATCH_%d'%batch,pin(sp),execute);wr=read(result_pin['path'])
            generated+=wr['attempted_generations'];worker_results.append(result_pin);assert generated<=64
    ordinary=[r for r in rows if r['kind']=='normal'];times=[r['phases']['capture_to_ACK_s']for r in ordinary]
    warm=[r for r in ordinary if r['index']!=0];faults=[r for r in rows if r['kind']!='normal']
    result=dict(schema='V31_STORAGE_GPU_QUALIFICATION_ACTUAL',exit_code=0,additional_unique_requests=len(rows),additional_generations=generated,
        normal_ACK=len(ordinary),bit_exact_replays=len(ordinary),by_family={f:sum(r['family']==f for r in ordinary)for f in('F1','F2')},
        complete_P10=sum(r['consumption']['full_P10']for r in ordinary),partial7=sum(r['consumption']['consumed_atoms']==7 for r in ordinary),
        fault_cases=len(faults),fault_misinstalls=sum(r['ACK']is not None for r in faults if r['kind']not in('after_ACK_stop','strict_50ms_margin','renew_ACK_late')),
        strict_after_ACK_guards=sum(bool(r['guard_exercised'])for r in faults),all_end_to_end_p50_p95_p99_max_s=np.percentile(times,[50,95,99,100]).tolist(),
        warm_end_to_end_p50_p95_p99_max_s=np.percentile([r['phases']['capture_to_ACK_s']for r in warm],[50,95,99,100]).tolist(),
        shared_parent_loads=len(worker_results),worker_results=worker_results,wall_s=time.monotonic()-begun,
        realtime_target_s=.35,realtime_target_met=all(t<=.35 for t in times),native_admission=False,goal_met=False,
        new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0)
    state('V31_NONPHYSICAL_COMPLETE_NATIVE_SAFETY_BLOCKED',result=result);return result
