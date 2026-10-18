"""Qualify eight frozen TRAIN anchors through complete history and native copies."""
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
from scripts.pi0294.s1_preflight import observed_state
from . import anchor_dynamic_config as config
from .anchor_dynamic import bound_inputs,compare_trace,save_arrays,project_observed_to_highrate,require_disjoint_cpu
from .anchor_restore import load_owned_snapshot
from .failure_config import ATTEMPTS,selected_sequences
from .failure_replay import replay_case
from .science import Ledger,validate_runtime


def qualify_native(engine,record,states,old,source,poses,opening,ledger,output):
    arrays,_,_=load_owned_snapshot(old,record)
    initial=compare_trace(observed_state(engine)[None,:],arrays['observed'][None,:],record['id']+'-INITIAL')
    result,rows=inspect_prefix(engine,poses,opening,ledger,'NATIVE-HISTORY-'+record['id'])
    path=Path(output)/(record['id']+'.npz');save_arrays(path,observed_rows=rows)
    original=read(Path(source)/'episodes'/record['original_attempt_id']/'S1_PREFLIGHTS.json')[record['sequence']]
    if (not result['live_state_unchanged'] or result['actions_modified'] or result['decision']!=original['decision']
            or result['predicted_native_safety']!=original['predicted_native_safety']):
        raise ValueError('ORIGINAL_ACTION_S1_SAFETY_AND_READONLY_NATIVE_COPY_REQUIRED')
    value=dict(id=record['id'],original_attempt_id=record['original_attempt_id'],sequence=record['sequence'],
        observation_sha256=record['observation_sha256'],initial_saved_fullstate_maximum_difference=initial,
        shadow=result,trace_sha256=sha(path),restoration='COMPLETE_ORIGINAL_ACKNOWLEDGED_HISTORY_THEN_NATIVE_MJDATA_COPY')
    last=record['sequence']==ATTEMPTS[record['original_attempt_id']]['last_sequence']
    if last:
        prior=Path(old)/'evidence'/(record['original_attempt_id']+'-rejected-shadow.npz')
        with np.load(prior,allow_pickle=False) as z:reference=z['observed_rows'].copy()
        if (result['shadow_integrations']!=ATTEMPTS[record['original_attempt_id']]['offset']
                or result['first_violation_offset_ticks']!=ATTEMPTS[record['original_attempt_id']]['offset']):
            raise ValueError('EXACT_ORIGINAL_REJECTED_SHADOW_OFFSET')
        value.update(kind='ORIGINAL_REJECTED_ZERO_LIVE_PREFIX',maximum_fullstate_trace_difference=compare_trace(rows,reference,record['id']))
    else:
        prior=Path(source)/'episodes'/record['original_attempt_id']/'relay'/f'physics-{record["sequence"]:03}.npz'
        with np.load(prior,allow_pickle=False) as z:
            reference=z['rows'].copy()
            if int(z['tick_before'])!=record['tick'] or int(z['tick_after'])!=record['tick']+200:
                raise ValueError('EXACT_ORIGINAL_ACKNOWLEDGED_PREFIX_TICKS')
        following=next(s for s in states if s['original_attempt_id']==record['original_attempt_id'] and s['sequence']==record['sequence']+1)
        following_arrays,_,_=load_owned_snapshot(old,following)
        if result['decision']!='PASS' or result['shadow_integrations']!=200:
            raise ValueError('EXACT_ORIGINAL_ACKNOWLEDGED200TICK_PREFIX')
        projected=project_observed_to_highrate(engine,rows)
        value.update(kind='PREVIOUS_ACTUAL_ACKED_PREFIX',maximum_actual_highrate_difference=compare_trace(projected,reference,record['id']),
            maximum_next_saved_fullstate_difference=compare_trace(rows[-1:],following_arrays['observed'][None,:],record['id']+'-NEXT'))
    write(Path(output)/(record['id']+'.json'),value);return value


def run(root):
    root=Path(root);manifest=validate_runtime(root,'4080-simulation');stage=read(root/'MP1_STAGE_BINDING.json')
    if stage['schema']!=config.NATIVE_GATE_SCHEMA:raise PermissionError('EXACT_NEW_EIGHT_NATIVE_HISTORY_SCOPE')
    window=config.validate_window(root,manifest,time.time());old,source,states,_=bound_inputs(root)
    require_disjoint_cpu(source,window['CPU_ids']);os.sched_setaffinity(0,set(window['CPU_ids']))
    ledger=Ledger(root,'core');ledger.guard();remaining=int(stage['CPU_core_s']['core']-read(ledger.path)['Core_CPU_s'])
    if remaining<=0:raise PermissionError('FINITE_NATIVE_GATE_CPU_REMAINING')
    resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining));lock=OwnerLock(root/'ANCHOR_NATIVE_HISTORY_OWNER.lock')
    output=root/'evidence/ANCHOR_DYNAMIC_PARITY';output.mkdir(mode=0o700);began=time.process_time();rows=[];histories=[]
    def guard():
        ledger.guard()
        if time.time()>=window['deadline_epoch'] or time.process_time()-began>=remaining or (source/'SOURCE_PHYSICS_STOP.json').exists():
            raise PermissionError('ORIGINAL_NATIVE_GATE_WINDOW_OR_SOURCE_SAFETY_STOP')
    bounded=SimpleNamespace(guard=guard,steps=ledger.steps,charge=ledger.charge)
    def expire(signum,frame):raise TimeoutError('FINITE_NATIVE_HISTORY_GATE_WALLTIME')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(window['deadline_epoch']-time.time())))
    try:
        for identifier in ATTEMPTS:
            def observe(engine,sequence,poses,opening):
                record=next(s for s in states if s['original_attempt_id']==identifier and s['sequence']==sequence)
                if any(r['id']==record['id'] for r in rows):raise PermissionError('NO_DUPLICATE_NATIVE_ANCHOR_INSPECTION')
                rows.append(qualify_native(engine,record,states,old,source,poses,opening,bounded,output))
                write(output/'PROGRESS.json',dict(completed=len(rows),rows=rows))
            case=dict(attempt_id=identifier,sequences=selected_sequences(identifier))
            histories.append(replay_case(root,source,case,bounded,guard,prefix_observer=observe))
        bound_inputs(root);usage=read(ledger.path)
        if (len(rows)!=8 or len(histories)!=4 or usage['main_starts']!=4 or usage['shadow_integrations']!=29202
                or usage['live_and_recovery_integrations']!=0 or usage['reservations']):
            raise ValueError('EXACT_EIGHT_ANCHORS_FOUR_PARENTS_29202_SHADOW_TICKS')
        value=dict(status='PASS_EIGHT_ORIGINAL_HISTORY_NATIVE_COPY_FUTURE_PREFIXES',rows=rows,histories=histories,
            source_sha=manifest['code']['vla_commit'],input_binding_sha256=sha(root/config.INPUTS),
            mechanism_result_sha256=config.COPY_MECHANISM_RESULT,full_eight_anchor_gate_passed=True,
            state_reconstruction='COMPLETE_ORIGINAL_ACKNOWLEDGED_HISTORY_THEN_NATIVE_MJDATA_COPY',
            comparison_absolute_tolerance=1e-10,tolerance_changed=False,
            future_consumer_must_reconstruct_and_validate_again=True,
            actual_shadow_integrations=29202,actual_diagnostic_starts=4,new_live_integrations=0,
            new_neural_queries=0,new_optimizer_updates=0,new_teacher_labels=0,
            original_historical_process_RNG_identity_claimed=False,endpoint_anchor_NN16='NOT_RUN_SEPARATE_BOUND_INFERENCE_REQUIRED')
        write(output/'RESULT.json',value);return value
    except BaseException as exc:
        write(output/'FAILURE.private.json',dict(status='NATIVE_HISTORY_GATE_GAP_PRESERVED_NO_RETRY',type=type(exc).__name__,
            error=str(exc),mismatch=getattr(exc,'mismatch',None),completed=len(rows),full_eight_anchor_gate_passed=False,ledger=read(ledger.path)));raise
    finally:signal.alarm(0);ledger.cpu(time.process_time()-began);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(dict(status=run(a.root)['status'])),flush=True)
