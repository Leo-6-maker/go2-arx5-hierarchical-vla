"""Matched diagnostic prefixes after fresh original-history native parity."""
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
from scripts.pi0292.eval_contract import gate_action,matrix10
from scripts.pi0294.e1.collision import inspect_prefix
from scripts.pi0294.native_dataset import sha
from . import anchor_dynamic_config as config,endpoint_ni_delivery as delivery
from .anchor_dynamic import bound_inputs,require_disjoint_cpu,checked_member,save_arrays
from .anchor_native_gate import qualify_native
from .anchor_restore import load_owned_snapshot
from .failure_config import ATTEMPTS,selected_sequences
from .failure_replay import replay_case
from .science import Ledger,validate_runtime


def delivered_inputs(root,source):
    root=Path(root);source=Path(source);q=read(root/config.INPUTS)
    folder=root.parent.parent/'datasets'/delivery.NAME
    if Path(q['NI16_dataset_root']).resolve()!=folder.resolve() or folder.is_symlink():
        raise PermissionError('ONLY_FIXED_ALREADY_CONSUMED_NI16_DATASET')
    for name,checksum in q['NI16_files'].items():checked_member(folder,name,checksum)
    receipt=read(folder/'CONSUMED.private.json')
    if (receipt['status']!='PASS_CORE_NI16_ORIGINAL_INPUT_NOISE_ARRAYS_FIXED_REVISION'
            or receipt['new_queries'] or receipt['new_physics'] or receipt['new_labels']
            or receipt['repo_id']!=delivery.REPO or receipt['revision']!=q['NI16_revision']):
        raise PermissionError('ACTUAL_FIXED_CORE_NI16_CONSUMPTION_REQUIRED')
    original=folder/receipt['prefix'];result=read(original/'NI_RESULT.json')
    if sha(original/'NI_RESULT.json')!=delivery.RESULT_SHA:raise ValueError('ACTUAL_NATIVE_NI16_RESULT')
    rows=delivery.checked_rows(result,delivery.fixed_binding(source))
    required={'CONSUMED.private.json'}|{receipt['prefix']+'/'+n for n in receipt['files']}|{receipt['prefix']+'/READY.json'}
    if set(q['NI16_files'])!=required:raise PermissionError('EXACT_CONSUMED_NI16_FILES_NO_ADDITIONAL_ACTIONS')
    for row in rows:
        path=original/delivery.filename(row['model'],row['id'])
        if sha(path)!=row['arrays_sha256']:raise ValueError('ACTUAL_FIXED_ENDPOINT_ARRAY_HASH')
    gate=root.parent.parent/'children'/config.NATIVE_GATE_CHILD
    if (read(read(gate/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED'
            or sha(gate/'evidence/ANCHOR_DYNAMIC_PARITY/RESULT.json')!=delivery.ni.GATE_RESULT):
        raise PermissionError('OLD_FULL_EIGHT_GATE_REMAINS_CLOSED')
    return original,{(r['model'],r['id']):r for r in rows}


def endpoint_prefix(engine,record,old,folder,row,ledger,output):
    _,_,body=load_owned_snapshot(old,record)
    path=folder/delivery.filename(row['model'],row['id'])
    if sha(path)!=row['arrays_sha256']:raise ValueError('FIXED_ENDPOINT_ARRAY_CHANGED')
    with np.load(path,allow_pickle=False) as z:action=z['physical_H40'].copy()
    numerical=gate_action(action,np.asarray(body['payload']['state_eef10']),body['tick']/500,
        'N1-ENDPOINT-SHADOW-'+row['model']+'-'+record['id'])
    value=dict(id=record['id'],model=row['model'],task=record['task'],original_parent=record['original_attempt_id'],
        arrays_sha256=row['arrays_sha256'],original_query_uuid=row['original_query_uuid'],noise=row['noise'],
        numerical_reference_gate=numerical,new_query=False,new_live_integrations=0,new_Teacher_labels=False,
        task_success_claimed=False,scope='ONE_READONLY_PREFIX_NOT_A_POLICY_ROLLOUT')
    if numerical['decision'] not in ['ACCEPT','WARN']:
        value.update(status='ORIGINAL_REFERENCE_GATE_REJECT_NO_PHYSICS',shadow=None,actual_shadow_ticks=0)
    else:
        poses=matrix10(action)
        check,trace=inspect_prefix(engine,poses[:8],action[:8,9],ledger,'ENDPOINT-'+row['model']+'-'+record['id'])
        target=Path(output)/(row['model']+'--'+record['id']+'.npz');save_arrays(target,observed_rows=trace)
        if not check['live_state_unchanged'] or check['actions_modified']:
            raise ValueError('FROZEN_SOURCE_AND_ACTIONS_REQUIRED')
        value.update(status='ORIGINAL_S1_PREFIX_'+check['decision'],shadow=check,
            actual_shadow_ticks=check['shadow_integrations'],trace_sha256=sha(target))
    write(Path(output)/(row['model']+'--'+record['id']+'.json'),value)
    return value


def run(root):
    root=Path(root);manifest=validate_runtime(root,'4080-simulation');stage=read(root/'MP1_STAGE_BINDING.json')
    if stage['schema']!=config.ENDPOINT_SCHEMA:raise PermissionError('EXACT_NEW_NI16_SHADOW_CHILD')
    window=config.validate_window(root,manifest,time.time());old,source,states,_=bound_inputs(root)
    folder,index=delivered_inputs(root,source);require_disjoint_cpu(source,window['CPU_ids'])
    os.sched_setaffinity(0,set(window['CPU_ids']));ledger=Ledger(root,'core');ledger.guard()
    remaining=int(stage['CPU_core_s']['core']-read(ledger.path)['Core_CPU_s'])
    if remaining<=0:raise PermissionError('FINITE_MATCHED_SHADOW_CPU_BUDGET')
    resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining));lock=OwnerLock(root/'ENDPOINT_NI16_SHADOW_OWNER.lock')
    output=root/'evidence/ENDPOINT_NI16_MATCHED_SHADOWS';output.mkdir(mode=0o700)
    baseline=output/'baseline';baseline.mkdir(mode=0o700);began=time.process_time();rows=[];parity=[];histories=[]
    def guard():
        ledger.guard()
        if time.time()>=window['deadline_epoch'] or time.process_time()-began>=remaining or (source/'SOURCE_PHYSICS_STOP.json').exists():
            raise PermissionError('MATCHED_SHADOW_ORIGINAL_CLOCK_CPU_AND_SOURCE_SAFETY')
    bounded=SimpleNamespace(guard=guard,steps=ledger.steps,charge=ledger.charge)
    def expire(signum,frame):raise TimeoutError('FINITE_MATCHED_SHADOW_WALL')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(window['deadline_epoch']-time.time())))
    try:
        for identifier in ATTEMPTS:
            def observe(engine,sequence,poses,opening):
                record=next(s for s in states if s['original_attempt_id']==identifier and s['sequence']==sequence)
                if any(p['id']==record['id'] for p in parity):raise PermissionError('NO_REPEATED_ENDPOINT_ANCHOR')
                parity.append(qualify_native(engine,record,states,old,source,poses,opening,bounded,baseline))
                for model in delivery.ni.CHECKPOINTS:
                    rows.append(endpoint_prefix(engine,record,old,folder,index[model,record['id']],bounded,output))
                write(output/'PROGRESS.json',dict(completed=len(rows),parity=parity,rows=rows))
            histories.append(replay_case(root,source,dict(attempt_id=identifier,sequences=selected_sequences(identifier)),bounded,guard,prefix_observer=observe))
        bound_inputs(root);delivered_inputs(root,source);usage=read(ledger.path)
        if (len(rows)!=16 or len(parity)!=8 or len(histories)!=4 or usage['main_starts']!=4
                or usage['shadow_integrations']!=29202+sum(r['actual_shadow_ticks'] for r in rows)
                or usage['shadow_integrations']>32402 or usage['live_and_recovery_integrations'] or usage['reservations']):
            raise ValueError('EXACT_16_PREFIXES_8_REVALIDATIONS_4_HISTORIES_COUNTED')
        value=dict(status='COMPLETE_NI16_SAME_STATE_NOISE_MATCHED_SHADOWS_NOT_POLICY_EVALUATION',
            rows=rows,baseline_parity=parity,histories=histories,source_sha=manifest['code']['vla_commit'],
            NI16_result_sha256=delivery.RESULT_SHA,actual_shadow_integrations=usage['shadow_integrations'],
            revalidation_shadow_ticks=29202,actual_endpoint_shadow_ticks=sum(r['actual_shadow_ticks'] for r in rows),
            actual_diagnostic_starts=4,original_states=8,original_parents=4,new_live_integrations=0,
            new_neural_queries=0,new_optimizer_updates=0,new_Teacher_labels=0,task_success_claimed=False,
            comparison_absolute_tolerance=1e-10,original_WBC_and_S1_unchanged=True)
        write(output/'RESULT.json',value);return value
    except BaseException as exc:
        write(output/'FAILURE.private.json',dict(error=type(exc).__name__,message=str(exc),completed=len(rows),
            new_live_integrations=0,new_neural_queries=0,original_gate_not_relabelled=True));raise
    finally:signal.alarm(0);ledger.cpu(time.process_time()-began);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    value=run(args.root);print(json.dumps(dict(status=value['status'],prefixes=len(value['rows']))),flush=True)
