"""One finite G1-dependent matched pair. No physics or automatic fallback pair."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts.pi0292.durable import write, OwnerLock
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,close,unit_state
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha


def run(spec):
    root=Path(spec['registered_root']);goal=Path(spec['goal_root']);journal=Journal(root/'control/journal',root.name)
    owner=OwnerLock(root/'control/supervisor-owner.lock');completed=[];started=time.time()
    pin=lambda p:dict(path=str(p),sha256=sha(p))

    def state(phase,**extra):
        write(goal/'goal_state.json',dict(schema='SLA_CLOSE_SAFE_MOBILE_GOAL_STATE_V1',phase=phase,
            epoch=time.time(),pid=os.getpid(),controller_source_sha=spec['source_sha'],
            plan_sha=spec['goal_plan_sha'],G1_source_sha=spec['G1_source_sha'],
            old_C20_updates=1000,old_C20_visits=32000,new_physical_requests=0,
            goal_met=False,completed=completed,**extra))

    def child(name,operation,coefficient,route,training=None):
        folder=root/name;dirs(folder)
        value=copy.deepcopy(spec['training_template']);value.update(spec)
        for key in ('training_template','G1_unit','G1_source_sha'):value.pop(key,None)
        value.update(registered_root=str(folder),campaign_root=str(root),role='SLA_CLOSE_'+name.upper(),
            dataset=spec['training_template']['dataset'],operation='SLA_CLOSE_PARENT1000_NEW500_MATCHED_V1',
            planned_candidate_updates=500,planned_target_visits=16000,planned_source_visits=0,
            new_optimizer_updates_budget=500 if operation=='--close-train'else 0,
            new_training_visits_budget=16000 if operation=='--close-train'else 0,
            microbatch=4,gripper_lambda=coefficient,inference_route=route,frame_cache_bytes=512*1024**2,
            compilation_cache_dir=str(folder/'cache'),max_walltime_s=max(1,math.ceil(spec['deadline_epoch']-time.time())+60),
            max_gpu_hours=max(1.,(spec['deadline_epoch']-time.time())/3600+1),save_and_exit_on_limit=True,
            parent_params=spec['raw_parent'],initial_function_rule=spec['numeric_rule'],
            model_service_timeout_s=7200,RPC_timeout_s=7200,JAX_memory_fraction=.75)
        value['model_service_timeout_s']=min(7200,value['max_walltime_s'])
        value['RPC_timeout_s']=min(7200,value['max_walltime_s'])
        if training:value['training_result']=training
        allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=spec['neural_owner_lock'])
        binding=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host='corelab-4080s',uid=2009,
            GPU_uuid=spec['GPU_uuid'],GPU_name='NVIDIA GeForce RTX 4080 SUPER',
            GPU_headroom_bytes=1024**3,host_headroom_bytes=2*1024**3,
            allocator_limit_bytes=spec['allocator_limit_bytes'],allocation_deadline_epoch=spec['allocation_deadline_epoch'],
            paths={k:str(Path(value[k]).resolve())for k in ('parent_params','dataset','compilation_cache_dir')},allocation=allocation)
        write(folder/'SITE_BINDING.private.json',binding);value['site_binding']=pin(folder/'SITE_BINDING.private.json')
        register(spec,[(folder,value)])
        sp=folder/'EXECUTION_SPEC.json';result=folder/'RESULT.private.json'

        def execute():
            with(folder/'logs/worker.log').open('xb')as handle:
                process=subprocess.Popen([value['python'],'-B',str(Path(value['source_root'])/'scripts/wm02/close_goal_bootstrap_v1.py'),
                    operation,str(sp)],env=dict(os.environ,**value['native_environment']),stdout=handle,stderr=subprocess.STDOUT)
            actual=dict(pid=process.pid,start=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                        cgroup=Path(f'/proc/{process.pid}/cgroup').read_text(),epoch=time.time())
            write(folder/'LAUNCH_RECEIPT.private.json',actual)
            while process.poll()is None:
                current=read(folder/'PROGRESS.private.json')if(folder/'PROGRESS.private.json').exists()else None
                state('OFFLINE_TRAINING'if operation=='--close-train'else'ORDINARY_RGB_RESTORE',
                      active=name,child=actual,progress=current)
                time.sleep(30)
            if not result.exists():raise RuntimeError('CHILD_EXIT_WITHOUT_RESULT_UNKNOWN_NO_RELAUNCH')
            outcome=read(result)
            if process.returncode!=0 or outcome['exit_code']!=0:raise RuntimeError('PRESERVED_CHILD_FAILURE_NO_AUTOMATIC_RETRY:'+name)
            if read(folder/'control/LEASE.json')['status']!='CLOSED':raise RuntimeError('CHILD_NOT_CLOSED')
            return pin(result)
        return journal.once('EXECUTE-'+name,pin(sp),execute)

    try:
        # Explicitly poll the known G1 handle, never treat observer timeout as exit.
        while time.time()<spec['deadline_epoch']-3600:
            process=unit_state(spec['G1_unit'])
            if process['pid']:
                write(root/'WAIT_G1.private.json',dict(epoch=time.time(),process=process,new_optimizer_updates=0,physical_requests=0))
                time.sleep(30);continue
            if not(goal/'RESULT.private.json').exists():raise RuntimeError('G1_HANDLE_MISSING_WITHOUT_TERMINAL_RESULT')
            if read(goal/'control/LEASE.json')['status']!='CLOSED':raise RuntimeError('G1_TERMINAL_LEASE_UNRESOLVED')
            decision=read(goal/'RESULT.private.json')
            if decision['exit_code']!=0:raise RuntimeError('G1_FAILED_NO_TRAINING_SUBMITTED')
            break
        else:raise TimeoutError('FINITE_G1_DEPENDENCY_QUEUE')
        write(root/'G1_DECISION.private.json',dict(result=pin(goal/'RESULT.private.json'),
            chosen_global_route=decision['chosen_global_route'],G2_needed=decision['G2_needed']))
        if not decision['G2_needed']:
            state('BLOCKED_SAFETY',new_optimizer_updates=0,new_training_visits=0,
                  reason='ZERO_UPDATE_NUMERIC_CANDIDATE_REQUIRES_INDEPENDENT_ADMISSION')
            return dict(exit_code=0,training_skipped=True,reason='G1_OFFLINE_TARGETS_MET',physical_requests=0,goal_met=False)
        route=decision['chosen_global_route']
        for name,coefficient in [('fm-control',0.),('grip-temporal',1.)]:
            if(root/'STOP.json').exists():raise TimeoutError('OPERATOR_STOP_BEFORE_NEXT_ARM')
            training=child(name,'--close-train',coefficient,route)
            actual=read(training['path'])
            if actual['new_updates']!=500 or actual['new_training_visits']!=16000:raise ValueError('REAL_MATCHED500_ENDPOINT_REQUIRED')
            replay=child(name+'-fresh-rgb','--close-rgb',coefficient,route,training)
            if not read(replay['path'])['all_bit_exact']:raise ValueError('FRESH_RGB_NOT_QUALIFIED')
            completed.append(dict(arm=name,lambda_value=coefficient,training=training,replay=replay,
                                  new_updates=500,new_training_visits=16000,metrics=actual['metrics']))
        # Common function, fresh optimizer, RNG, and sample visit map are proven
        # from complete state and actual gradient/input records, not intention.
        a,b=[read(root/name/'work/step0000/COMPLETE.json')for name in ('fm-control','grip-temporal')]
        if any(any(x[k]!=y[k]for k in ('key','shape','dtype','sha256'))for x,y in zip(a['arrays'],b['arrays']))or len(a['arrays'])!=len(b['arrays']):
            raise ValueError('MATCHED_NEW_OPTIMIZER_INITIAL_ARRAYS_DIFFER')
        ga,gb=[read(root/name/'GRADIENT_ACTUAL.private.json')for name in ('fm-control','grip-temporal')]
        if any(ga[k]!=gb[k]for k in ('noise_bank_sha256','tau_bank_sha256','visit_map_sha256')):
            raise ValueError('MATCHED_ACTUAL_INPUT_STREAM_DIFFERENCE')
        state('BLOCKED_SAFETY',new_optimizer_updates=1000,new_training_visits=32000,
              reason='OFFLINE_PAIR_COMPLETE_PHYSICAL_QUARANTINE_REMAINS')
        return dict(schema='SLA_CLOSE_MATCHED_PAIR_ACTUAL_V1',exit_code=0,completed=completed,
            common_initial_arrays_bit_exact=True,actual_input_stream_matched=True,
            new_optimizer_updates=1000,new_training_visits=32000,physical_requests=0,goal_met=False,
            optional_fallback_submitted=False,wall_s=time.time()-started,CPU_s=time.process_time())
    finally:owner.close()


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:result=run(s)
    except BaseException:
        result=dict(exit_code=1,status='BOUNDED_FAILURE_NO_REPLAY',failure=traceback.format_exc(),
                    physical_requests=0,goal_met=False,raw_update_logs_authoritative=True)
    write(root/'RESULT.private.json',result);close(s,result['exit_code']);raise SystemExit(result['exit_code'])
