"""One explicitly authorized SHIFT7 round; reuse TIME500 and FM500; no physics."""
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
        updates=1500;visits=48000
        for name in ('shift7',):
            outcome=root/name/'RESULT.private.json';progress=root/name/'PROGRESS.private.json'
            value=read(outcome)if outcome.exists()else read(progress)if progress.exists()else{}
            updates+=value.get('new_updates',0);visits+=value.get('new_training_visits',0)
        write(goal/'goal_state.json',dict(schema='SLA_CLOSE_SAFE_MOBILE_GOAL_STATE_V1',phase=phase,
            epoch=time.time(),pid=os.getpid(),controller_source_sha=spec['source_sha'],
            plan_sha=spec['goal_plan_sha'],G1_source_sha=spec['G1_source_sha'],
            old_C20_updates=1000,old_C20_visits=32000,first_pair_new_updates=1000,first_pair_new_visits=32000,prior_time_new_updates=500,prior_time_new_visits=16000,new_physical_requests=0,
            goal_met=False,completed=completed,last_reported_committed_updates=updates,
            last_reported_training_visits=visits,**extra))

    def effect(kind,**details):
        with(goal/'effects.jsonl').open('a')as handle:
            handle.write(json.dumps(dict(epoch=time.time(),kind=kind,source_sha=spec['source_sha'],**details))+'\n')
            handle.flush();os.fsync(handle.fileno())

    def child(name,operation,coefficient,route,training=None):
        folder=root/name;dirs(folder)
        value=copy.deepcopy(spec['training_template']);value.update(spec)
        for key in ('training_template','G1_unit','G1_source_sha'):value.pop(key,None)
        value.update(registered_root=str(folder),campaign_root=str(root),role='SLA_CLOSE_'+name.upper(),
            dataset=spec['training_template']['dataset'],operation='SLA_TIME_PARAMETER_LAYOUT_SHIFT7_PARENT1000_NEW500_V1',
            planned_candidate_updates=500,planned_target_visits=16000,planned_source_visits=0,
            new_optimizer_updates_budget=500 if operation=='--shift-train'else 0,
            new_training_visits_budget=16000 if operation=='--shift-train'else 0,
            new_complete_generations_budget=96 if operation=='--shift-train'else 0 if operation=='--shift-prepare'else 16,
            microbatch=4,gripper_lambda=coefficient,inference_route=route,frame_cache_bytes=512*1024**2,
            compilation_cache_dir=str(folder/'cache'),max_walltime_s=max(1,math.ceil(spec['deadline_epoch']-time.time())+60),
            max_gpu_hours=max(1.,(spec['deadline_epoch']-time.time())/3600+1),save_and_exit_on_limit=True,
            parent_params=spec['raw_parent'],initial_function_rule=spec['numeric_rule'],
            model_service_timeout_s=7200,RPC_timeout_s=7200,JAX_memory_fraction=.75)
        value['model_service_timeout_s']=min(7200,value['max_walltime_s'])
        value['RPC_timeout_s']=min(7200,value['max_walltime_s'])
        if operation!='--shift-prepare':value['shift_layout_audit']=pin(root/'qualification/RESULT.private.json')
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
            effect('TIME_CHILD_STARTED',name=name,operation=operation,spec=pin(sp))
            with(folder/'logs/worker.log').open('xb')as handle:
                process=subprocess.Popen([value['python'],'-B',str(Path(value['source_root'])/'scripts/wm02/close_goal_bootstrap_v1.py'),
                    operation,str(sp)],env={**os.environ,**value['native_environment'],**({'CUDA_VISIBLE_DEVICES':''}if operation=='--shift-prepare'else{})},stdout=handle,stderr=subprocess.STDOUT)
            actual=dict(pid=process.pid,start=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                        cgroup=Path(f'/proc/{process.pid}/cgroup').read_text(),epoch=time.time())
            write(folder/'LAUNCH_RECEIPT.private.json',actual)
            while process.poll()is None:
                current=read(folder/'PROGRESS.private.json')if(folder/'PROGRESS.private.json').exists()else None
                state('OFFLINE_TRAINING'if operation=='--shift-train'else'PREPARING_LAYOUT'if operation=='--shift-prepare'else'ORDINARY_RGB_RESTORE',
                      active=name,child=actual,progress=current)
                time.sleep(30)
            if not result.exists():raise RuntimeError('CHILD_EXIT_WITHOUT_RESULT_UNKNOWN_NO_RELAUNCH')
            outcome=read(result)
            if process.returncode!=0 or outcome['exit_code']!=0:raise RuntimeError('PRESERVED_CHILD_FAILURE_NO_AUTOMATIC_RETRY:'+name)
            if read(folder/'control/LEASE.json')['status']!='CLOSED':raise RuntimeError('CHILD_NOT_CLOSED')
            effect('TIME_CHILD_COMMITTED',name=name,result=pin(result),
                   new_updates=outcome.get('new_updates',0),new_training_visits=outcome.get('new_training_visits',0))
            return pin(result)
        return journal.once('EXECUTE-'+name,pin(sp),execute)

    try:
        old=Path(spec['old_time_root'])
        if unit_state(spec['old_time_unit'])['pid']or read(old/'control/LEASE.json')['status']!='CLOSED':
            raise RuntimeError('OLD_TIME_HANDLE_MUST_BE_TERMINAL_NO_DUPLICATE_WORK')
        if sha(old/'RESULT.private.json')!=spec['old_time_result_sha256']:
            raise ValueError('PRESERVED_TIME500_CONTROL_RESULT_CHANGED')
        previous=read(old/'RESULT.private.json')
        if previous['new_optimizer_updates_this_role']!=500 or previous['total_goal_new_optimizer_updates']!=1500:
            raise ValueError('EXACT_PRIOR_COST_REQUIRED')
        if spec['new_additional_updates_budget']!=500 or spec['parameter_slot_shift']!=7 or spec['additional_round']!=1:
            raise ValueError('ONE_EXPLICITLY_AUTHORIZED_SHIFT7_500_ROUND')
        if(root/'STOP.json').exists():raise TimeoutError('OPERATOR_STOP_BEFORE_SHIFT7_ROUND')
        qualification=child('qualification','--shift-prepare',0.,spec['inference_route'])
        if not read(qualification['path'])['passes']:raise ValueError('SHIFT7_PREPARATION_FAILED_NO_TRAINING')
        training=child('shift7','--shift-train',0.,spec['inference_route'])
        actual=read(training['path'])
        if actual['new_updates']!=500 or actual['new_training_visits']!=16000:
            raise ValueError('REAL_TIME500_ENDPOINT_REQUIRED')
        replay=child('shift7-fresh-rgb','--shift-rgb',0.,spec['inference_route'],training)
        if not read(replay['path'])['all_bit_exact']:raise ValueError('TIME_FRESH_RGB_NOT_QUALIFIED')
        control=old/'time-params';control_result=read(control/'RESULT.private.json')
        if sha(control/'RESULT.private.json')!=spec['reused_control_result_sha256']:
            raise ValueError('REUSED_MATCHED_CONTROL_CHANGED')
        a,b=[read(p/'work/step0000/COMPLETE.json')for p in(control,root/'shift7')]
        if len(a['arrays'])!=87 or len(b['arrays'])!=87 or any(
            any(x[k]!=y[k]for k in ('key','shape','dtype','sha256'))for x,y in zip(a['arrays'],b['arrays'])):
            raise ValueError('SAME_C20_MODEL_NEW_ADAM_RNG_BYTES_REQUIRED')
        common=('parent_manifest','tokenizer','target_audit','visit_map_sha256','lr_sha256','seed',
            'frozen_fingerprint','upstream_sha','optimizer','initial_function_rule','XLA_FLAGS',
            'microbatch','accumulation_steps','effective_batch','random_stream','sample_cursor',
            'training_matmul_precision','parent_C20_package','parent_step','parent_visits',
            'dense_manifest_sha256','state_token_dropout','inference_route','gripper_lambda')
        if any(a['bindings'][k]!=b['bindings'][k]for k in common):
            raise ValueError('REUSED_CONTROL_HAS_AN_UNMATCHED_SECOND_VARIABLE')
        ga,gb=[read(p/'GRADIENT_ACTUAL.private.json')for p in(control,root/'shift7')]
        if any(ga[k]!=gb[k]for k in('noise_bank_sha256','tau_bank_sha256','visit_map_sha256')):
            raise ValueError('REUSED_CONTROL_RANDOM_BANK_OR_VISITS_DIFFER')
        logs=[]
        for p in(control,root/'shift7'):
            rows=[json.loads(line)for line in(p/'work/updates.private.jsonl').read_text().splitlines()]
            if len(rows)!=500:raise ValueError('FULL500_ACTUAL_LOG_REQUIRED')
            logs.append([{k:r[k]for k in('step','cursor','used_lr','micro')}for r in rows])
        if logs[0]!=logs[1]:raise ValueError('ACTUAL_INPUT_DROPOUT_OR_LR_MISMATCH')
        match=dict(schema='SLA_TIME_SHIFT7_REUSED_TIME500_ACTUAL_MATCH_V1',initial_arrays_bit_exact=87,
            actual_matched_updates=500,actual_matched_visits=16000,first_actual_noise_time_bank_bit_exact=True,
            subsequent_random_identity='SAME_INITIAL_RNG_FIXED32_PRODUCER_ABSOLUTE_CURSOR',
            all_banks_individually_saved=False,changed_variable='CYCLIC_PARAMETER_SLOT_SHIFT_PLUS7_ONLY',
            parameter_values_and_robot_time_decode_unchanged=True,control_result=pin(control/'RESULT.private.json'),
            candidate_result=training,candidate_RGB=replay,physical_requests=0)
        write(root/'CONTROL_MATCH_ACTUAL.private.json',match)
        from scripts.sla.audit_close_intent_v1 import audit as intent_audit
        intents=intent_audit(spec['training_template']['dataset'],{'original_time500':str(control/'TRAIN48_PREDICTIONS.private.json'),'shift7_500':str(root/'shift7/TRAIN48_PREDICTIONS.private.json')})
        write(root/'INTENT_ACTUAL.private.json',intents)
        from scripts.sla.audit_time_shift7_v1 import analyze as layout_audit
        layout=layout_audit(control/'TRAIN48_PREDICTIONS.private.json',root/'shift7/TRAIN48_PREDICTIONS.private.json')
        write(root/'LAYOUT_CURVATURE_ACTUAL.private.json',layout)
        completed.append(dict(arm='shift7',training=training,replay=replay,new_updates=500,
                              new_training_visits=16000,metrics=actual['metrics']))
        events=actual['metrics']['events'];metrics=actual['metrics']
        meets=(metrics['CLOSE_pass']>=18 and metrics['nonCLOSE_pass']==76 and
            intents['conditions']['shift7_500']['summary']['ALL']['terminal_opposite_command_state']==0 and
            metrics['maximum_terminal_width_error_m']<=.001 and len(events)==20 and all(e['crossing_error_s']is not None and e['crossing_error_s']<=.100000001
                                   and e['terminal_width_error_m']<=.001 for e in events))
        state('BLOCKED_SAFETY'if meets else'BOUNDED_OFFLINE_FAILURE',new_optimizer_updates=2000,
            new_training_visits=64000,reason='USER_AUTHORIZED_SINGLE_SHIFT7_ROUND_FINISHED_NO_AUTOMATIC_SUCCESSOR',
            offline_development_target_met=meets,independent_admission=False)
        return dict(schema='SLA_ONE_TIME_SHIFT7_ROUND_ACTUAL_V1',exit_code=0,
            completed=completed,reused_control=pin(control/'RESULT.private.json'),matching=pin(root/'CONTROL_MATCH_ACTUAL.private.json'),qualification=qualification,
            new_optimizer_updates_this_role=500,new_training_visits_this_role=16000,
            total_historical_plus_new_optimizer_updates=2000,total_historical_plus_new_training_visits=64000,
            explicit_additional_round_consumed=True,further_round_authorized=False,
            layout_curvature=pin(root/'LAYOUT_CURVATURE_ACTUAL.private.json'),intent=pin(root/'INTENT_ACTUAL.private.json'),
            offline_development_target_met=meets,physical_requests=0,goal_met=False,
            independent_admission=False,wall_s=time.time()-started,CPU_s=time.process_time())
    finally:owner.close()


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:result=run(s)
    except BaseException:
        result=dict(exit_code=1,status='BOUNDED_FAILURE_NO_REPLAY',failure=traceback.format_exc(),
                    physical_requests=0,goal_met=False,raw_update_logs_authoritative=True)
    write(root/'RESULT.private.json',result);close(s,result['exit_code']);raise SystemExit(result['exit_code'])
