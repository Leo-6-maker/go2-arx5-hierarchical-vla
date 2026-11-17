"""Finite real GPU server for the nonintegrating V3 receiver. No native API."""
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.pi0292.durable import write
from scripts.sla.close_goal_g1_v1 import wait_resource_owner
from scripts.sla.hybrid_cn38_infer_v1 import load_pair
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import read, check_live, load_input
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.sla_adapter.hybrid_request_v1 import predict_request


def run(spec):
    import jax
    from go2arx5_vla.sla_adapter.site_v2 import validate
    root=Path(spec['registered_root']);bus=Path(spec['ipc_root']);site=validate(spec)
    start=time.monotonic();one,two=load_pair(spec)
    write(root/'READY.private.json',dict(pid=os.getpid(),parent_loads=1,adapters=2,
        load_and_import_wall_s=time.monotonic()-start,GPU_memory=jax.devices()[0].memory_stats(),site=site))
    attempted=completed=0;handled={};timings=[]
    def counts():
        write(root/'GENERATION_COUNTS.private.json',dict(attempted=attempted,completed=completed,
            generation_cap=64,requests=len(handled),new_updates=0,new_visits=0,
            new_native_requests=0,new_integrations=0))
    while time.time()<spec['deadline_epoch']-60 and not(root/'STOP.json').exists():
        if(bus/'FINISH.json').exists():break
        path=bus/'REQUEST.json'
        if not path.exists():time.sleep(.005);continue
        message=read(path);q=message['request'];qid=q['request_id'];folder=Path(message['response_root'])
        # Transport duplicates return the original response without generation.
        if qid in handled:
            # Immutable response: do not replace the same file every polling tick.
            if not (folder/'RESPONSE.json').exists():
                write(folder/'RESPONSE.json',handled[qid])
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
                # Actual interprocess cancellation barrier, not a local callback mock.
                write(folder/(self.name+'_DONE.json'),dict(request_id=qid,epoch=time.time()))
                barrier=folder/(self.name+'_CONTINUE.json');until=time.monotonic()+10
                while not barrier.exists():
                    if time.monotonic()>=until:raise TimeoutError('IPC_BRANCH_BARRIER_TIMEOUT')
                    time.sleep(.002)
                return value
        try:
            if q['bindings']!=spec['request_bindings']:raise ValueError('REGISTERED_PACKAGE_CODEC_STATISTICS_REQUIRED')
            current();data=load_input(q)
            value=predict_request(Counted(one,'EEF'),Counted(two,'GRIP'),state=data['state_eef10'],
                images=[data['body_rgb'],data['wrist_rgb']],language=q['language'],frame=data['T_policy_from_mount'],
                noise_id=q['noise_id'],previous_reference=q['previous_reference'],request_id=qid,
                deadline_monotonic=q['service_deadline_monotonic'],now=time.monotonic,live_state=current)
            current();action=folder/'ACTION.npz'
            with action.open('xb')as f:
                np.savez(f,physical_H40=value['physical_H40'],normalized=branch[0]['output']['normalized'],
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
        response['new_generations_this_request']=len(branch)
        handled[qid]=response;timings.append(response['worker_wall_s']);counts()
        write(folder/'RESPONSE.json',response)
    result=dict(exit_code=0,requests=len(handled),attempted_generations=attempted,completed_generations=completed,
        parent_loads=1,adapters=2,peak_GPU_memory=jax.devices()[0].memory_stats(),new_updates=0,new_visits=0,
        new_native_requests=0,new_integrations=0,wall_s=time.monotonic()-start,site=site)
    write(root/'RESULT.private.json',result);return result


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root']);lock=None
    try:lock=wait_resource_owner(s);r=run(s)
    except BaseException:
        r=dict(exit_code=1,failure=traceback.format_exc(),status='PRESERVE_NO_REPLAY',
            actual_counts=read(root/'GENERATION_COUNTS.private.json')if(root/'GENERATION_COUNTS.private.json').exists()else{},
            new_updates=0,new_native_requests=0,new_integrations=0)
        write(root/'RESULT.private.json',r)
    finally:
        if lock:lock.close()
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,r['exit_code']);raise SystemExit(r['exit_code'])
