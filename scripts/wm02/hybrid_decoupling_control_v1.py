"""Finite conditional successor. Wait for the sealed SHIFT7 handle; never restart it."""
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from contextlib import closing
from scripts.pi0292.durable import write,OwnerLock
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs,unit_state,close
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha


def run(spec):
    root=Path(spec['registered_root']);prior=Path(spec['SHIFT_root']);journal=Journal(root/'control/journal',root.name)
    pin=lambda p:dict(path=str(p),sha256=sha(p))
    def state(phase,**detail):write(root/'goal_state.json',dict(phase=phase,epoch=time.time(),source_sha=spec['source_sha'],
        new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,goal_met=False,**detail))
    with closing(OwnerLock(root/'control/supervisor-owner.lock')):
        while time.time()<spec['deadline_epoch']-600:
            if(root/'STOP.json').exists():raise TimeoutError('OPERATOR_STOP_BEFORE_SUCCESSOR')
            process=unit_state(spec['SHIFT_unit'])
            if process['pid']:
                if process['pid']!=64423 or process.get('proc_start')!='461549814':raise RuntimeError('SHIFT_OWNER_CHANGED_NO_TAKEOVER')
                state('WAIT_SHIFT7_TERMINAL',process=process);time.sleep(30);continue
            if not(prior/'RESULT.private.json').exists():raise RuntimeError('SHIFT_HANDLE_TERMINAL_WITHOUT_RESULT_NO_REPLAY')
            break
        else:raise TimeoutError('FINITE_SHIFT_WAIT_EXPIRED_NO_REPLAY')
        if sha(prior/'EXECUTION_SPEC.json')!=spec['SHIFT_spec_sha256']or read(prior/'control/LEASE.json')['status']!='CLOSED':
            raise ValueError('EXACT_SEALED_SHIFT_EXECUTION_REQUIRED')
        result=read(prior/'RESULT.private.json')
        if result['exit_code']!=0 or result['new_optimizer_updates_this_role']!=500:
            raise RuntimeError('SHIFT_NOT_A_SCIENTIFIC500_ENDPOINT_NO_NEW_TRAINING')
        write(root/'SHIFT_TERMINAL_PIN.private.json',dict(result=pin(prior/'RESULT.private.json'),source_sha=read(prior/'EXECUTION_SPEC.json')['source_sha'],process=process))
        for name in ('shift7','shift7-fresh-rgb'):
            if read(prior/name/'control/LEASE.json')['status']!='CLOSED':raise ValueError('SHIFT_CHILD_NOT_CLOSED')
        fm=Path(spec['FM_root']);timed=Path(spec['TIME_root']);shift=prior/'shift7'
        for p,k in ((fm,'FM_result_sha256'),(timed,'TIME_result_sha256')):
            if sha(p/'RESULT.private.json')!=spec[k]:raise ValueError('PRESERVED_CONTROL_RESULT_CHANGED')
        state('SAVED96_ZERO_MODEL_COMPOSITION')
        folder=root/'saved96';folder.mkdir()
        saved=dict(registered_root=str(folder),dataset=spec['dataset'],dense_manifest_sha256=spec['dense_manifest_sha256'],
            FM_predictions=pin(fm/'TRAIN48_PREDICTIONS.private.json'),grip_predictions={
                'FM_EEF_TIME_GRIP':pin(timed/'TRAIN48_PREDICTIONS.private.json'),
                'FM_EEF_SHIFT7_GRIP':pin(shift/'TRAIN48_PREDICTIONS.private.json')})
        write(folder/'INPUT.private.json',saved)
        from scripts.sla.hybrid_saved96_v1 import run as compose96
        composite=journal.once('SAVED96-FIXED-COMBINATIONS',pin(folder/'INPUT.private.json'),lambda:compose96(saved))
        choice=composite['selected_global_candidate']
        if choice is None:
            state('NO_COMPLETE_TRAIN_CANDIDATE_NO_CN38_INFERENCE',saved96=pin(folder/'RESULT.private.json'))
            return dict(exit_code=0,status='BOUNDED_COMBINATION_FAILURE',saved96=pin(folder/'RESULT.private.json'),
                new_updates=0,new_model_generations=0,new_native_requests=0,physical_admission=False,goal_met=False)
        from scripts.sla.cn38_hybrid_bank_v1 import build
        state('SEAL_CN38_EPISODE_HELDOUT_BANK',selected_global_candidate=choice)
        bank=journal.once('SEAL-FIXED-CN38-BANK',dict(snapshot=spec['CN38_snapshot'],anchor_rule='FIVE_PHASE_LOWER_MEDIANS'),
            lambda:build(spec,root/'cn38-bank'))
        grip=timed if choice=='FM_EEF_TIME_GRIP'else shift
        child=root/'cn38-infer';dirs(child)
        value=dict(spec,registered_root=str(child),campaign_root=str(root),role='MFREE_HYBRID_CN38_OFFLINE_INFERENCE',
            selected_combo=choice,CN38_bank=bank,FM_package=read(fm/'RESULT.private.json')['package'],
            grip_package=read(grip/'RESULT.private.json')['package'],FM_RGB16=pin(fm/'ENDPOINT_RGB16.private.json'),
            grip_RGB16=pin(grip/'ENDPOINT_RGB16.private.json'),new_optimizer_updates_budget=0,new_training_visits_budget=0,
            new_complete_generations_budget=376,parent_params=spec['raw_parent'],compilation_cache_dir=str(child/'cache'),
            JAX_memory_fraction=.75,model_service_timeout_s=7200,RPC_timeout_s=7200,
            max_walltime_s=math.ceil(spec['deadline_epoch']-time.time())+60,
            max_gpu_hours=(spec['deadline_epoch']-time.time())/3600+1,save_and_exit_on_limit=True)
        allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=spec['neural_owner_lock'])
        binding=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host='corelab-4080s',uid=2009,GPU_uuid=spec['GPU_uuid'],
            GPU_name='NVIDIA GeForce RTX 4080 SUPER',GPU_headroom_bytes=1024**3,host_headroom_bytes=2*1024**3,
            allocator_limit_bytes=spec['allocator_limit_bytes'],allocation_deadline_epoch=spec['allocation_deadline_epoch'],
            paths={k:str(Path(value[k]).resolve())for k in ('parent_params','dataset','compilation_cache_dir')},allocation=allocation)
        write(child/'SITE_BINDING.private.json',binding);value['site_binding']=pin(child/'SITE_BINDING.private.json')
        register(spec,[(child,value)]);sp=child/'EXECUTION_SPEC.json'
        def execute():
            with(child/'logs/worker.log').open('xb')as handle:
                p=subprocess.Popen([value['python'],'-B',str(Path(value['source_root'])/'scripts/wm02/close_goal_bootstrap_v1.py'),
                    '--hybrid-cn38',str(sp)],env={**os.environ,**value['native_environment']},stdout=handle,stderr=subprocess.STDOUT)
            write(child/'LAUNCH_RECEIPT.private.json',dict(pid=p.pid,start=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],epoch=time.time()))
            while p.poll()is None:
                if(root/'STOP.json').exists()and not(child/'STOP.json').exists():
                    write(child/'STOP.json',dict(reason='CONTROLLER_OPERATOR_STOP',epoch=time.time()))
                state('CN38_DUAL_ADAPTER_INFERENCE',selected_global_candidate=choice,
                    progress=read(child/'PROGRESS.private.json')if(child/'PROGRESS.private.json').exists()else None)
                time.sleep(30)
            if not(child/'RESULT.private.json').exists():raise RuntimeError('CHILD_UNKNOWN_NO_REPLAY')
            r=read(child/'RESULT.private.json')
            if p.returncode!=0 or r['exit_code']!=0:raise RuntimeError('PRESERVED_INFERENCE_FAILURE_NO_REPLAY')
            if read(child/'control/LEASE.json')['status']!='CLOSED':raise RuntimeError('INFERENCE_CHILD_NOT_CLOSED')
            return pin(child/'RESULT.private.json')
        actual=journal.once('CN38-360-PLUS16-SHARED-SEAM',pin(sp),execute)
        state('OFFLINE_COMPLETE_SAFETY_NOT_ADMITTED',result=actual,selected_global_candidate=choice)
        return dict(schema='MFREE_EEF_GRIP_DECOUPLING_AND_SAFETY_V1',exit_code=0,SHIFT_terminal=pin(prior/'RESULT.private.json'),
            saved96=pin(folder/'RESULT.private.json'),CN38_bank=bank,CN38_result=actual,new_updates=0,new_visits=0,
            new_model_generations=376,new_native_requests=0,new_integrations=0,physical_admission=False,goal_met=False,
            native_STOP_TTL_P10_lifecycle_qualification=False,no_new_head_training_authorized=True)


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:r=run(s)
    except BaseException:r=dict(exit_code=1,status='BOUNDED_FAILURE_NO_REPLAY',failure=traceback.format_exc(),
        raw_child_counts_authoritative=True,new_updates=0,new_native_requests=0,new_integrations=0,goal_met=False)
    write(root/'RESULT.private.json',r);close(s,r['exit_code']);raise SystemExit(r['exit_code'])
