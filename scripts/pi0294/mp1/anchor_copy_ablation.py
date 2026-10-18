"""Compare native copies and integration restore at one reconstructed TRAIN state."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time
from types import SimpleNamespace

import numpy as np

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.e1.collision import inspect_prefix
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.s1_preflight import clone
from . import anchor_dynamic_config as config
from .anchor_dynamic import bound_inputs,compare_trace,TraceMismatch,save_arrays,require_disjoint_cpu
from .anchor_restore import load_owned_snapshot,restore_and_check
from .anchor_trace_decode import component_differences
from .failure_config import selected_sequences
from .failure_replay import replay_case
from .science import Ledger,validate_runtime


def compare_branch(engine,record,poses,opening,ledger,reference,output,label,*,mj=None):
    result,rows=inspect_prefix(engine,poses,opening,ledger,'NATIVE-COPY-ABLATION-'+label)
    path=Path(output)/(label+'.npz');save_arrays(path,observed_rows=rows)
    if (not result['live_state_unchanged'] or result['actions_modified']
            or result['decision']!='REJECT' or result['shadow_integrations']!=63
            or result['first_violation_offset_ticks']!=63):
        raise ValueError('ORIGINAL63TICK_REJECT_AND_READONLY_S1_BRANCH_REQUIRED')
    value=dict(branch=label,shadow=result,trace_sha256=sha(path),fullstate_parity=False)
    try:value.update(maximum_absolute_difference=compare_trace(rows,reference,label),fullstate_parity=True)
    except TraceMismatch as exc:value['mismatch']=exc.mismatch
    if mj is None:import mujoco as mj
    value['native_components']=component_differences(engine.model,rows,reference,mj=mj)
    write(Path(output)/(label+'.json'),value)
    return value


def ablate(engine,record,old,assets,poses,opening,ledger,reference,output,*,make_engine):
    arrays,history,_=load_owned_snapshot(old,record);results=[]
    # The native-copy branch sees the replay state before any extra forward call.
    for label in ['FULL_NATIVE_COPY','FRESH_ENGINE_INTEGRATION_RESTORE','NATIVE_COPY_INTEGRATION_RESTORE']:
        if label=='FRESH_ENGINE_INTEGRATION_RESTORE':
            ledger.charge('main_starts',1);candidate=make_engine(record['card'],assets)
        else:candidate=clone(engine)
        try:
            static=None
            if label!='FULL_NATIVE_COPY':static=restore_and_check(candidate,record,arrays,history)
            value=compare_branch(candidate,record,poses,opening,ledger,reference,output,label)
            value['static_restore']=static;results.append(value)
            write(Path(output)/'PROGRESS.json',dict(completed=len(results),rows=results))
        finally:candidate.close()
    return results


def run(root):
    root=Path(root);manifest=validate_runtime(root,'4080-simulation');stage=read(root/'MP1_STAGE_BINDING.json')
    if stage['schema']!=config.ABLATION_SCHEMA:raise PermissionError('SEPARATE_SINGLE_STATE_ABLATION_SCOPE_REQUIRED')
    window=config.validate_window(root,manifest,time.time());old,source,states,binding=bound_inputs(root)
    if len(states)!=1 or states[0]['id']!=config.FAILED_STATE:raise PermissionError('ONLY_FIXED_RECORDED_FAILURE')
    record=states[0];require_disjoint_cpu(source,window['CPU_ids']);os.sched_setaffinity(0,set(window['CPU_ids']))
    ledger=Ledger(root,'core');ledger.guard();remaining=int(stage['CPU_core_s']['core']-read(ledger.path)['Core_CPU_s'])
    if remaining<=0:raise PermissionError('FINITE_ABLATION_CPU_REMAINING_REQUIRED')
    resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining));lock=OwnerLock(root/'ANCHOR_COPY_ABLATION_OWNER.lock')
    output=root/'evidence/ANCHOR_DYNAMIC_PARITY';output.mkdir(mode=0o700);began=time.process_time();branches=[]
    def guard():
        ledger.guard()
        if time.time()>=window['deadline_epoch'] or time.process_time()-began>=remaining or (source/'SOURCE_PHYSICS_STOP.json').exists():
            raise PermissionError('ORIGINAL_FINITE_ABLATION_WINDOW_AND_SOURCE_STOP')
    bounded=SimpleNamespace(guard=guard,steps=ledger.steps,charge=ledger.charge)
    def expire(signum,frame):raise TimeoutError('FINITE_NATIVE_COPY_ABLATION_WALLTIME')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(window['deadline_epoch']-time.time())))
    try:
        reference_path=old/'evidence'/(record['original_attempt_id']+'-rejected-shadow.npz')
        with np.load(reference_path,allow_pickle=False) as z:reference=z['observed_rows'].copy()
        if len(reference)!=63:raise ValueError('EXACT_SAVED63TICK_REFERENCE')
        from scripts.pi0294.scene import make_engine
        assets=read(root/'INPUT_BINDINGS.private.json')['asset_root']
        def observe(engine,sequence,poses,opening):
            if sequence!=record['sequence']:return
            if branches:raise PermissionError('ONE_ABLATION_STATE_CALLBACK_ONLY')
            branches.extend(ablate(engine,record,old,assets,poses,opening,bounded,reference,output,make_engine=make_engine))
            original=read(source/'episodes'/record['original_attempt_id']/'S1_PREFLIGHTS.json')[sequence]
            if any(b['shadow']['predicted_native_safety']!=original['predicted_native_safety'] for b in branches):
                raise ValueError('EXACT_ORIGINAL_PREDICTED_NATIVE_SAFETY_IN_ALL_BRANCHES')
        case=dict(attempt_id=record['original_attempt_id'],sequences=selected_sequences(record['original_attempt_id']))
        history=replay_case(root,source,case,bounded,guard,prefix_observer=observe)
        bound_inputs(root);usage=read(ledger.path)
        if (len(branches)!=3 or history['original_main_steps']!=5200 or usage['main_starts']!=2
                or usage['shadow_integrations']!=5452 or usage['live_and_recovery_integrations']!=0 or usage['reservations']):
            raise ValueError('EXACT_TWO_STARTS_5200HISTORY_PLUS_FOUR63TICK_SHADOWS')
        value=dict(status='COMPLETE_SINGLE_STATE_NATIVE_COPY_ABLATION_NOT_FULL_EIGHT_GATE',
            state=record['id'],source_sha=manifest['code']['vla_commit'],branches=branches,
            history=history,reference_sha256=sha(reference_path),input_binding_sha256=sha(root/config.INPUTS),
            native_copy_vs_fresh_restore_difference_demonstrated=branches[0]['fullstate_parity'] and not branches[1]['fullstate_parity'],
            exact_missing_derived_cache_or_extra_forward_cause_proven=False,full_eight_anchor_gate_passed=False,
            original_comparison_absolute_tolerance=1e-10,tolerance_changed=False,
            actual_shadow_integrations=5452,actual_diagnostic_starts=2,new_live_integrations=0,
            new_neural_queries=0,new_optimizer_updates=0,new_teacher_labels=0,
            original_B1_dynamic01_trace02_remain_closed=True,endpoint_anchor_NN16='NOT_RUN')
        write(output/'RESULT.json',value);return value
    except BaseException as exc:
        write(output/'FAILURE.private.json',dict(status='NATIVE_COPY_ABLATION_GAP_PRESERVED_NO_RETRY',
            type=type(exc).__name__,error=str(exc),completed=len(branches),full_eight_anchor_gate_passed=False,
            ledger=read(ledger.path)));raise
    finally:signal.alarm(0);ledger.cpu(time.process_time()-began);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(dict(status=run(a.root)['status'])),flush=True)
