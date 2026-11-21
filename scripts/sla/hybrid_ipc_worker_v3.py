"""Finite real GPU server for the nonintegrating V3 receiver. No native API."""
import json
import os
from pathlib import Path
import time
import traceback
import shutil
import numpy as np
from scripts.pi0292.durable import write
from scripts.sla.close_goal_g1_v1 import wait_resource_owner
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import read, check_live, load_input
from go2arx5_vla.world_model.storage import digest


def poll_message(path):
    """Absence at an IPC publication boundary is not a failed model request.

    Read directly instead of trusting exists(). Only ENOENT is transient;
    malformed bytes and other I/O errors remain failures. The caller's finite
    deadline, STOP and request-identity deduplication remain authoritative.
    """
    try:
        return read(path)
    except FileNotFoundError:
        return None


def run(spec):
    import jax
    from go2arx5_vla.sla_adapter.site_v2 import validate
    root=Path(spec['registered_root']);bus=Path(spec['ipc_root']);live=Path(spec.get('live_root',root));site=validate(spec)
    from go2arx5_vla.sla_adapter.compilation_cache_binding_v1 import bind_declared_cache
    from go2arx5_vla.world_model.multisource_d4_native import SCOPE as D4_SCOPE
    from go2arx5_vla.world_model.sla_followup_native_v1 import SCOPE as D416_SCOPE
    from go2arx5_vla.sla_adapter.scale20k_package import SCOPE as SCALE_SCOPE
    cache_binding=bind_declared_cache(spec,jax.config)
    start=time.monotonic();scope=spec.get('scope');g7=scope in ('SLA_V6_MULTITASK_NATIVE', D4_SCOPE, D416_SCOPE, SCALE_SCOPE, 'SLA_CORE_HOPPER_V1_STUDENT', 'SLA_CORE_CANONICAL_EVAL_V1')
    if g7:
        from go2arx5_vla.sla_adapter.multitask_contract_v6 import NATIVE_METHOD
        if spec['selected_combo']!=NATIVE_METHOD:raise ValueError('EXACT_G7_SINGLE_H40_GENERATION')
    single=g7 or spec['selected_combo']=='SHIFT7_SINGLE'
    pool=g7 or scope in ('SLA_MOBILE_F2_V52','SLA_V6_TARGET_NATIVE')
    if pool:
        if not single:raise ValueError('V52_ONE_SHIFT7_ARCHITECTURE_PER_REQUEST')
        if scope=='SLA_CORE_HOPPER_V1_STUDENT':
            from scripts.sla.core_hopper_native_model import Pool
        elif scope=='SLA_CORE_CANONICAL_EVAL_V1':
            from scripts.sla.core_canonical_ipc import Pool
        elif scope==SCALE_SCOPE:
            from go2arx5_vla.sla_adapter.scale20k_package import Pool
        elif g7:
            from go2arx5_vla.sla_adapter.multitask_pool_v6 import Pool
        elif scope=='SLA_V6_TARGET_NATIVE':
            from go2arx5_vla.sla_adapter.target_pool_v6 import Pool
        else:
            from go2arx5_vla.sla_adapter.mobile_provider_pool_v52 import Pool
        one=Pool(spec);two=None
    elif single:
        from go2arx5_vla.sla_adapter.provider_time_shift7_v1 import Provider
        package=spec['SHIFT7_package']
        one=Provider(package['path'],manifest_sha256=package['sha256'],parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
        two=None
    else:
        from scripts.sla.hybrid_cn38_infer_v1 import load_pair
        from go2arx5_vla.sla_adapter.hybrid_request_v1 import predict_request
        one,two=load_pair(spec)
    adapters=len(one.providers) if pool else 1 if single else 2
    write(root/'READY.private.json',dict(pid=os.getpid(),parent_loads=1,adapters=adapters,
        load_and_import_wall_s=time.monotonic()-start,GPU_memory=jax.devices()[0].memory_stats(),site=site,
        compilation_cache_binding=cache_binding))
    attempted=completed=0;handled={};timings=[];deliveries=set()
    def counts():
        write(live/'GENERATION_COUNTS.private.json',dict(attempted=attempted,completed=completed,
            generation_cap=spec['new_complete_generations_budget'],requests=len(handled),new_updates=0,new_visits=0,
            new_native_requests=0,new_integrations=0))
    queue=None
    if spec.get('d2_world_queue') is not None:
        from go2arx5_vla.world_model.multitask_queue_d2 import WorldQueue

        if not g7 or spec['physics']:
            raise PermissionError('D2_QUEUE_REQUIRES_NONINTEGRATING_SINGLE_G7_WORKER')
        queue=WorldQueue(bus,spec['d2_world_queue'])
    while time.time()<spec['deadline_epoch']-60 and not(root/'STOP.json').exists():
        if(bus/'FINISH.json').exists():break
        path=bus/'REQUEST.json'
        message=queue.next(handled) if queue is not None else poll_message(path)
        if message is None:time.sleep(.005);continue
        q=message['request'];qid=q['request_id'];folder=Path(message['response_root'])
        # Transport duplicates return the original response without generation.
        if qid in handled:
            # Immutable response: do not replace the same file every polling tick.
            if not (folder/'RESPONSE.json').exists():
                write(folder/'RESPONSE.json',handled[qid])
            delivery=(qid,message.get('delivery_id',0))
            if delivery not in deliveries:
                current_state=read(message['live_path'])
                try:check_live(q,current_state);status='SAME_LIVE_QUERY_CACHED_RESPONSE'
                except (RuntimeError,ValueError,TimeoutError):status='STALE_CACHE_REJECTED'
                write(folder/'DELIVERY_ACK.json',dict(request_id=qid,delivery_id=delivery[1],
                    status=status,attempted=attempted,completed=completed))
                deliveries.add(delivery)
            time.sleep(.005);continue
        if len(handled)>=spec['unified_requests_budget']:raise PermissionError('FINITE_UNIFIED_REQUEST_CAP')
        branch=[];t=time.monotonic()
        live_path=Path(message['live_path'])
        def current():
            s=read(live_path)
            if(root/'STOP.json').exists()or(Path(spec['campaign_root'])/'STOP.json').exists():s['stop']=True
            check_live(q,s)
            return dict(stop=s['stop'],cancelled=s['cancelled'],request_id=s['pending'],
                        consumed_atoms=0,query_native_s=s['native_step']*.002)
        class Counted:
            def __init__(self,provider,name):self.provider=provider;self.name=name
            def predict(self,*a,**kw):
                nonlocal attempted,completed
                current()
                if attempted>=spec['new_complete_generations_budget']:raise PermissionError('FINITE_BRANCH_GENERATION_CAP')
                attempted+=1;counts();begin=time.monotonic()
                value=self.provider.predict(*a,**kw);completed+=1
                branch.append(dict(name=self.name,output=value,wall_s=time.monotonic()-begin));counts()
                if spec.get('archive_all_branches'):
                    with(folder/(self.name+'_OUTPUT.npz')).open('xb')as f:
                        np.savez(f,normalized=value['normalized'],physical_H40=value['physical_H40'])
                # Actual interprocess cancellation barrier, not a local callback mock.
                write(folder/(self.name+'_DONE.json'),dict(request_id=qid,epoch=time.time()))
                if message.get('branch_barriers',True):
                    barrier=folder/(self.name+'_CONTINUE.json');until=time.monotonic()+10
                    while not barrier.exists():
                        if time.monotonic()>=until:raise TimeoutError('IPC_BRANCH_BARRIER_TIMEOUT')
                        time.sleep(.001)
                return value
        def respond(response):
            if queue is not None:
                response['d2_route']=message['d2_route']
            handled[qid]=response
            deliveries.add((qid,message.get('delivery_id',0)))
            counts()
            write(folder/'DELIVERY_ACK.json',dict(request_id=qid,delivery_id=message.get('delivery_id',0),
                status='ORIGINAL_RESPONSE',attempted=attempted,completed=completed))
            # RESPONSE is last: no branch/receipt mutation remains after it.
            write(folder/'RESPONSE.json',response)
        try:
            if pool:
                arms=[arm for arm,binding in spec['request_bindings_by_arm'].items()if binding==q['bindings']]
                if len(arms)!=1:raise ValueError('EXACT_PREREGISTERED_V52_ARM_BINDINGS_REQUIRED')
                one.select(arms[0])
            elif q['bindings']!=spec['request_bindings']:raise ValueError('REGISTERED_PACKAGE_CODEC_STATISTICS_REQUIRED')
            current();data=load_input(q)
            if message.get('restart_before_branches'):
                response=dict(status='SERVICE_RESTART_ABORTED_NO_ACTION',request_id=qid,bindings=q['bindings'],
                    worker_wall_s=time.monotonic()-t,new_generations_this_request=0)
                respond(response)
                break
            if message.get('cached_protocol_replay'):
                cache=spec['cached_protocol_replay']
                if digest(cache['action']['path'])!=cache['action']['sha256'] or digest(cache['request']['path'])!=cache['request']['sha256']:
                    raise ValueError('PINNED_CACHED_PROTOCOL_SOURCE_CHANGED')
                old=read(cache['request']['path']);old_data=load_input(old)
                if q['noise_id']!=old['noise_id'] or q['language']!=old['language'] or q['previous_reference']!=old['previous_reference']:
                    raise ValueError('CACHE_INPUT_NOISE_REFERENCE_IDENTITY')
                if any(not np.array_equal(data[k],old_data[k])for k in data):raise ValueError('CACHE_CAPTURED_INPUT_ARRAYS_DIFFER')
                action=folder/'ACTION.npz';shutil.copyfile(cache['action']['path'],action);current()
                response=dict(status='GENERATED_NOT_INSTALLED',request_id=qid,bindings=q['bindings'],
                    action=dict(path=str(action),sha256=digest(action)),branch_wall_s=[],
                    cached_protocol_replay=True,worker_wall_s=time.monotonic()-t,new_generations_this_request=0,
                    independent_gripper_write=False,native_execution_admitted=False)
                respond(response);continue
            if single:
                value=Counted(one,'G7' if g7 else 'SHIFT7').predict(data['state_eef10'],[data['body_rgb'],data['wrist_rgb']],
                    q['language'],data['T_policy_from_mount'],noise_id=q['noise_id'],previous_reference=q['previous_reference'])
            else:
                value=predict_request(Counted(one,'EEF'),Counted(two,'GRIP'),state=data['state_eef10'],
                    images=[data['body_rgb'],data['wrist_rgb']],language=q['language'],frame=data['T_policy_from_mount'],
                    noise_id=q['noise_id'],previous_reference=q['previous_reference'],request_id=qid,
                    deadline_monotonic=q['service_deadline_monotonic'],now=time.monotonic,live_state=current)
            current();action=folder/'ACTION.npz'
            with action.open('xb')as f:
                if single:np.savez(f,physical_H40=value['physical_H40'],normalized=value['normalized'])
                else:np.savez(f,physical_H40=value['physical_H40'],normalized=branch[0]['output']['normalized'],
                    FM_physical_H40=branch[0]['output']['physical_H40'],
                    grip_physical_H40=branch[1]['output']['physical_H40'],
                    grip_normalized=branch[1]['output']['normalized'])
            response=dict(status='GENERATED_NOT_INSTALLED',request_id=qid,bindings=q['bindings'],
                action=dict(path=str(action),sha256=digest(action)),
                branch_wall_s=[r['wall_s']for r in branch],independent_gripper_write=False,
                native_execution_admitted=False)
        except (RuntimeError,ValueError,TimeoutError):
            response=dict(status='REJECTED_NO_ACTION',request_id=qid,bindings=q['bindings'],
                failure=traceback.format_exc(),branches_completed=len(branch))
        response['worker_wall_s']=time.monotonic()-t
        response.setdefault('branch_wall_s',[r['wall_s']for r in branch])
        response['new_generations_this_request']=len(branch)
        timings.append(response['worker_wall_s']);respond(response)
    counts()
    if live!=root:
        write(root/'GENERATION_COUNTS.private.json',read(live/'GENERATION_COUNTS.private.json'))
    result=dict(exit_code=0,requests=len(handled),attempted_generations=attempted,completed_generations=completed,
        parent_loads=1,adapters=adapters,peak_GPU_memory=jax.devices()[0].memory_stats(),new_updates=0,new_visits=0,
        new_native_requests=0,new_integrations=0,wall_s=time.monotonic()-start,site=site)
    write(root/'RESULT.private.json',result);return result


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root']);lock=None
    try:lock=wait_resource_owner(s);r=run(s)
    except BaseException:
        live=Path(s.get('live_root',root))
        r=dict(exit_code=1,failure=traceback.format_exc(),status='PRESERVE_NO_REPLAY',
            actual_counts=read(live/'GENERATION_COUNTS.private.json')if(live/'GENERATION_COUNTS.private.json').exists()else{},
            new_updates=0,new_native_requests=0,new_integrations=0)
        write(root/'RESULT.private.json',r)
    finally:
        if lock:lock.close()
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,r['exit_code']);raise SystemExit(r['exit_code'])
