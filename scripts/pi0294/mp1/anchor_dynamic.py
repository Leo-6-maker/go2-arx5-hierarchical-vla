"""Prove saved full-state future-prefix parity before new endpoint inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
from types import SimpleNamespace

import numpy as np

from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.e1.collision import inspect_prefix
from scripts.pi0294.native_dataset import sha
from .anchor_restore import load_owned_snapshot,restore_and_check
from . import anchor_dynamic_config as config
from .failure_config import ATTEMPTS,selected_sequences
from .science import Ledger,validate_runtime

TOLERANCE=1e-10


class TraceMismatch(ValueError):
    def __init__(self,label,actual,expected):
        difference=np.abs(actual-expected);at=tuple(int(x) for x in np.unravel_index(np.argmax(difference),difference.shape))
        first=tuple(int(x) for x in np.argwhere(difference>TOLERANCE)[0])
        self.mismatch=dict(label=label,maximum_absolute_difference=float(difference[at]),
            maximum_at_row=at[0],maximum_at_column=at[1],first_outside_tolerance_row=first[0],
            first_outside_tolerance_column=first[1],actual_at_maximum=float(actual[at]),
            expected_at_maximum=float(expected[at]),absolute_tolerance=TOLERANCE)
        super().__init__('DYNAMIC_RESTORE_FUTURE_PREFIX_MISMATCH:'+label)


def save_arrays(path,**arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    Path(path).chmod(0o400)


def checked_member(root,relative,digest):
    relative=Path(relative);p=Path(root)/relative
    if (relative.is_absolute() or '..' in relative.parts or p.is_symlink()
            or not p.resolve().is_relative_to(Path(root).resolve()) or sha(p)!=digest):
        raise ValueError('EXACT_OWNED_DYNAMIC_INPUT_FILE_HASH')
    return p


def bound_inputs(root):
    root=Path(root);parent=root.parent.parent;q=read(root/config.INPUTS)
    old=parent/'children/sbc512-failure-history-01';source=parent/'children/source-pair-01'
    if (Path(q['B1_root']).resolve()!=old.resolve() or Path(q['source_root']).resolve()!=source.resolve()
            or old.is_symlink() or source.is_symlink() or (source/'SOURCE_PHYSICS_STOP.json').exists()):
        raise PermissionError('ONLY_REGISTERED_CLOSED_B1_AND_ORIGINAL_SOURCE_INPUTS')
    if (sha(old/'evidence/FAILURE_HISTORY_RESULT.json')!=config.B1_RESULT
            or sha(old/'evidence/CLOSED_RECEIPT.private.json')!=config.B1_CLOSED
            or sha(old/'RUN_MANIFEST.json')!=config.B1_MANIFEST
            or sha(old/'plans/SOURCE_FAILURE_INPUTS.private.json')!=config.B1_SOURCE_INPUTS
            or read(read(old/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED'):
        raise PermissionError('ORIGINAL_B1_CLOSED_RECEIPT_NEVER_REOPENED')
    for relative,digest in q['B1_files'].items():checked_member(old,relative,digest)
    original=read(old/'plans/SOURCE_FAILURE_INPUTS.private.json')
    if (original['files']!=q['original_source_files']
            or Path(original['source_root']).resolve()!=source.resolve()):
        raise PermissionError('ORIGINAL_CLOSED_B1_SOURCE_FILE_BINDINGS')
    for relative,digest in original['files'].items():checked_member(source,relative,digest)
    result=read(old/'evidence/FAILURE_HISTORY_RESULT.json')
    if result['status']!='PASS_FOUR_CURRENT_FAILURES_EXACT_HISTORY_AND_FIRST_ILLEGAL_GEOMETRY':
        raise PermissionError('QUALIFIED_ORIGINAL_FAILURE_HISTORY_ONLY')
    states=[s for c in result['cases'] for s in c['states']]
    expected={(a,s) for a in ATTEMPTS for s in selected_sequences(a)}
    if len(states)!=8 or len({s['id'] for s in states})!=8 or {(s['original_attempt_id'],s['sequence']) for s in states}!=expected:
        raise PermissionError('EXACT_FIXED_EIGHT_TRAIN_ANCHORS_NO_NEW_SELECTION')
    spec=config.specification(read(root/'MP1_STAGE_BINDING.json'))
    if spec.get('native_gate_only'):
        mechanism=parent/'children'/config.ABLATION_CHILD
        if (Path(q['native_copy_mechanism_root']).resolve()!=mechanism.resolve() or mechanism.is_symlink()
                or sha(mechanism/'evidence/ANCHOR_DYNAMIC_PARITY/RESULT.json')!=config.COPY_MECHANISM_RESULT
                or sha(mechanism/'evidence/CLOSED_RECEIPT.private.json')!=q['native_copy_mechanism_closed_sha256']):
            raise PermissionError('EXACT_CLOSED_NATIVE_COPY_MECHANISM_EVIDENCE_REQUIRED')
        for prior in [mechanism,parent/'children'/config.CHILD,parent/'children'/config.TRACE_CHILD]:
            if read(read(prior/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED':
                raise PermissionError('ALL_PREVIOUS_ANCHOR_LEASES_MUST_REMAIN_CLOSED')
    if spec['trace_only']:
        prior=parent/'children'/config.CHILD
        if (Path(q['prior_diagnostic_root']).resolve()!=prior.resolve() or prior.is_symlink()
                or sha(prior/'evidence/ANCHOR_DYNAMIC_PARITY/FAILURE.private.json')!=config.PRIOR_FAILURE
                or sha(prior/'RESOURCE_LEDGER.json')!=config.PRIOR_LEDGER
                or sha(prior/'evidence/CLOSED_RECEIPT.private.json')!=q['prior_diagnostic_closed_sha256']
                or read(read(prior/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED'):
            raise PermissionError('EXACT_OLD_FAILED_DIAGNOSTIC_REMAINS_CLOSED')
        if spec.get('ablation_only'):
            trace=parent/'children'/config.TRACE_CHILD
            if (Path(q['trace02_root']).resolve()!=trace.resolve() or trace.is_symlink()
                    or sha(trace/'evidence/ANCHOR_DYNAMIC_PARITY/FAILURE.private.json')!=config.TRACE02_FAILURE
                    or sha(trace/'evidence/CLOSED_RECEIPT.private.json')!=q['trace02_closed_sha256']
                    or read(read(trace/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED'):
                raise PermissionError('EXACT_CLOSED_TRACE02_COMPONENT_FAILURE_REQUIRED')
        states=[s for s in states if s['id']==config.FAILED_STATE]
        if len(states)!=1:raise PermissionError('EXACT_SINGLE_RECORDED_MISMATCH_INPUT')
    return old,source,states,q


def compare_trace(actual,expected,label):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=expected.shape or actual.ndim!=2 or not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('DYNAMIC_RESTORE_TRACE_SHAPE_OR_FINITE:'+label)
    maximum=float(np.max(np.abs(actual-expected)))
    if maximum>TOLERANCE:raise TraceMismatch(label,actual,expected)
    return maximum


def project_observed_to_highrate(engine,rows,*,mj=None):
    if mj is None:import mujoco as mj
    rows=np.asarray(rows);kind=mj.mjtState.mjSTATE_INTEGRATION;n=mj.mj_stateSize(engine.model,kind)
    if rows.ndim!=2 or not len(rows) or rows.shape[1]<2*n+2 or not np.isfinite(rows).all():
        raise ValueError('COMPLETE_DYNAMIC_LIVE_IK_CONTROLLER_ROWS')
    data=mj.MjData(engine.model);projected=[]
    for row in rows:
        if row[-2] not in [0,1]:raise ValueError('EXACT_DYNAMIC_NATIVE_SAFETY_FLAG')
        mj.mj_setState(engine.model,data,row[:n],kind)
        projected.append(np.r_[data.time,data.qpos,data.qvel,data.ctrl,data.xfrc_applied[engine.ids['tool']],row[-2]])
    return np.asarray(projected)


def require_disjoint_cpu(source,cpus):
    if not set(cpus)<=os.sched_getaffinity(0):raise PermissionError('PREBOUND_DYNAMIC_CPU_UNAVAILABLE')
    for folder in ['worker_processes','aux_worker_processes']:
        for p in (Path(source)/folder).glob('*.json'):
            w=read(p);stat=Path('/proc',str(w['pid']),'stat')
            if stat.exists():
                fields=stat.read_text().split()
                if fields[2]!='Z' and fields[21]==str(w['proc_start']) and set(cpus)&os.sched_getaffinity(w['pid']):
                    raise PermissionError('DYNAMIC_CPU_OVERLAPS_ACTIVE_SOURCE_CONSUMER')


def qualify(engine,record,states,old,source,ledger,*,trace_folder=None):
    arrays,history,body=load_owned_snapshot(old,record)
    static=restore_and_check(engine,record,arrays,history)
    if np.max(np.abs(np.asarray(static.pop('actual_state_eef10'))-body['payload']['state_eef10']))>1e-12:
        raise ValueError('EXACT_ORIGINAL_MODEL_INPUT_STATE_AFTER_RESTORE')
    folder=Path(old)/'evidence/TRAIN_STATES'/record['snapshot_relative']
    bundle=read(folder/'ORIGINAL_ACTION_BUNDLE.private.json');action=unseal(bundle['action'])
    trajectory=np.asarray(action['payload']['trajectory_eef10'],dtype=np.float64)
    poses,opening=matrix10(trajectory),trajectory[:,9]
    forecasts=read(Path(source)/'episodes'/record['original_attempt_id']/'S1_PREFLIGHTS.json')
    original=forecasts[record['sequence']]
    digest=hashlib.sha256(poses[:8].tobytes()+opening[:8].tobytes()).hexdigest()
    if original['action_sha256']!=digest or original['sequence']!=record['sequence']*8:
        raise ValueError('EXACT_ORIGINAL_INSTALLED_OR_REJECTED_PREFIX_ACTION')
    result,rows=inspect_prefix(engine,poses[:8],opening[:8],ledger,'DYNAMIC-'+record['id'])
    projected=project_observed_to_highrate(engine,rows)
    if trace_folder is not None:
        save_arrays(Path(trace_folder)/(record['id']+'.actual-prefix.npz'),observed_rows=rows,projected_highrate=projected)
        write(Path(trace_folder)/(record['id']+'.prefix-check.json'),dict(id=record['id'],shadow=result,
            original_S1=original,actual_arrays_sha256=sha(Path(trace_folder)/(record['id']+'.actual-prefix.npz'))))
    expected=ATTEMPTS[record['original_attempt_id']];last=record['sequence']==expected['last_sequence']
    if (not result['live_state_unchanged'] or result['actions_modified']
            or result['decision']!=original['decision'] or result['predicted_native_safety']!=original['predicted_native_safety']):
        raise ValueError('ORIGINAL_S1_DECISION_AND_UNMODIFIED_LIVE_STATE_PARITY')
    value=dict(id=record['id'],original_attempt_id=record['original_attempt_id'],task=record['task'],
        sequence=record['sequence'],source_observation_sha256=record['observation_sha256'],
        original_action_sha256=bundle['action']['sha256'],original_prefix_sha256=digest,
        static_restore=static,shadow=result,initial_anchor_is_physically_valid=True,
        historical_original_process_RNG_identity_claimed=False,new_teacher_labels=False)
    if last:
        if result['shadow_integrations']!=expected['offset'] or result['first_violation_offset_ticks']!=expected['offset']:
            raise ValueError('EXACT_ORIGINAL_REJECTED_PREFIX_OFFSET')
        prior=Path(old)/'evidence'/(record['original_attempt_id']+'-rejected-shadow.npz')
        with np.load(prior,allow_pickle=False) as z:reference=z['observed_rows'].copy()
        if trace_folder is not None:save_arrays(Path(trace_folder)/(record['id']+'.reference-prefix.npz'),observed_rows=reference)
        value.update(kind='ORIGINAL_REJECTED_ZERO_LIVE_PREFIX',
            maximum_saved_shadow_row_difference=compare_trace(rows,reference,record['id']),
            rejected_reference_sha256=sha(prior),rejected_reference_freshly_sealed_not_historical_closed_member=True,
            actual_live_rows_after_rejection_exist=False)
    else:
        if result['decision']!='PASS' or result['shadow_integrations']!=200:
            raise ValueError('EXACT_PREVIOUS_ACTUAL_ACKED_200_TICKS')
        prior=Path(source)/'episodes'/record['original_attempt_id']/'relay'/f'physics-{record["sequence"]:03}.npz'
        with np.load(prior,allow_pickle=False) as z:
            reference=z['rows'].copy()
            if int(z['tick_before'])!=record['tick'] or int(z['tick_after'])!=record['tick']+200:
                raise ValueError('ORIGINAL_ACTUAL_PREFIX_TICK_BINDING')
        following=next(s for s in states if s['original_attempt_id']==record['original_attempt_id'] and s['sequence']==record['sequence']+1)
        following_arrays,_,_=load_owned_snapshot(old,following)
        if trace_folder is not None:
            save_arrays(Path(trace_folder)/(record['id']+'.reference-prefix.npz'),highrate=reference,next_observed=following_arrays['observed'])
        value.update(kind='PREVIOUS_ACTUAL_ACKED_PREFIX',actual_physics_receipt_sha256=sha(prior),
            maximum_actual_200_tick_highrate_difference=compare_trace(projected,reference,record['id']),
            maximum_next_saved_full_state_difference=compare_trace(rows[-1:],following_arrays['observed'][None,:],record['id']+'-NEXT_FULL_STATE'))
    return value,rows,projected


def run(root):
    root=Path(root);manifest=validate_runtime(root,'4080-simulation');stage=read(root/'MP1_STAGE_BINDING.json')
    if not config.is_diagnostic(stage):raise PermissionError('EXACT_DYNAMIC_PARITY_STAGE')
    window=config.validate_window(root,manifest,time.time());spec=config.specification(stage)
    if spec.get('ablation_only') or spec.get('native_gate_only'):
        raise PermissionError('USE_SEPARATELY_BOUND_NATIVE_COPY_ENTRYPOINT')
    old,source,states,binding=bound_inputs(root)
    require_disjoint_cpu(source,window['CPU_ids']);os.sched_setaffinity(0,set(window['CPU_ids']))
    ledger=Ledger(root,'core');ledger.guard();remaining=int(spec['CPU_core_s']['core']-read(ledger.path)['Core_CPU_s'])
    if remaining<=0:raise PermissionError('DYNAMIC_CPU_BUDGET_REMAINING_REQUIRED')
    resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining));owner=OwnerLock(root/'ANCHOR_DYNAMIC_OWNER.lock')
    output=root/'evidence/ANCHOR_DYNAMIC_PARITY';output.mkdir(mode=0o700)
    began=time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN);results=[]
    def guard():
        ledger.guard()
        if (time.time()>=window['deadline_epoch'] or time.process_time()-began>=remaining
                or (source/'SOURCE_PHYSICS_STOP.json').exists()):
            raise PermissionError('DYNAMIC_FINITE_WINDOW_OR_ORIGINAL_SOURCE_SAFETY_STOP')
    bounded=SimpleNamespace(guard=guard,steps=ledger.steps)
    def expire(signum,frame):raise TimeoutError('DYNAMIC_ORIGINAL_FINITE_WALLTIME')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(window['deadline_epoch']-time.time())))
    try:
        from scripts.pi0294.scene import make_engine
        assets=read(old/'INPUT_BINDINGS.private.json')['asset_root']
        for record in states:
            guard();ledger.charge('main_starts',1);engine=make_engine(record['card'],assets)
            try:value,rows,projected=qualify(engine,record,states,old,source,bounded,trace_folder=output)
            finally:engine.close()
            path=output/(record['id']+'.npz')
            with path.open('xb') as f:
                np.savez_compressed(f,observed_rows=rows,projected_highrate=projected);f.flush();os.fsync(f.fileno())
            path.chmod(0o400);value['new_shadow_arrays_sha256']=sha(path);results.append(value)
            write(output/'PROGRESS.json',dict(completed=len(results),rows=results))
        bound_inputs(root);usage=read(ledger.path)
        if usage['main_starts']!=spec['states'] or usage['shadow_integrations']!=(63 if spec['trace_only'] else 1201) or usage['reservations']:
            raise ValueError('EXACT_DYNAMIC_EIGHT_STARTS_1201_SHADOW_TICKS_CLOSED')
        value=dict(status='SINGLE_TRACE_MATCHED_NOT_FULL_EIGHT_ANCHOR_QUALIFICATION' if spec['trace_only'] else 'PASS_EIGHT_RESTORED_ORIGINAL_FUTURE_PREFIXES',rows=results,
            full_eight_anchor_gate_passed=not spec['trace_only'],states=spec['states'],distinct_original_parents=spec['parents'],source_sha=manifest['code']['vla_commit'],
            actual_shadow_integrations=usage['shadow_integrations'],actual_diagnostic_starts=usage['main_starts'],new_live_integrations=0,
            new_neural_queries=0,new_optimizer_updates=0,new_teacher_labels=0,
            original_B1_result_sha256=config.B1_RESULT,original_B1_closed_sha256=config.B1_CLOSED,
            input_binding_sha256=sha(root/config.INPUTS),historical_unrecorded_RNG_identity_claimed=False,
            endpoint_NI_and_new_model_shadow_still_require_separate_binding=True)
        write(output/'RESULT.json',value);return value
    except BaseException as exc:
        failed_id=record['id'] if 'record' in locals() else None
        write(output/'FAILURE.private.json',dict(status='DYNAMIC_QUALIFICATION_GAP_PRESERVED_NO_RETRY',
            type=type(exc).__name__,error=str(exc),mismatch=getattr(exc,'mismatch',None),
            failed_state_id=failed_id,
            actual_failed_prefix_preserved=failed_id is not None and (output/(failed_id+'.actual-prefix.npz')).is_file(),
            reference_failed_prefix_preserved=failed_id is not None and (output/(failed_id+'.reference-prefix.npz')).is_file(),
            full_eight_anchor_gate_passed=False,
            completed=len(results),ledger=read(ledger.path)));raise
    finally:
        signal.alarm(0);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        user=after.ru_utime-child.ru_utime;system=after.ru_stime-child.ru_stime
        if user<0 or system<0:raise ValueError('MONOTONIC_DYNAMIC_CHILD_CPU_COUNTERS')
        ledger.cpu(time.process_time()-began+user+system);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(dict(status=run(a.root)['status'])),flush=True)
