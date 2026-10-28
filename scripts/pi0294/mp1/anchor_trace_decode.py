"""Decode saved MuJoCo state vectors through its API, without integration."""
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

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.native_dataset import sha
from .anchor_dynamic import project_observed_to_highrate
from .anchor_restore import load_owned_snapshot
from . import anchor_dynamic_config as config
from . import n1_limits as n1
from .scope import validate,charge_cpu

ID='N1_TRACE02_STATE_COMPONENT_DECODE_01'
FAILURE='6054695caaac588df42a36529a7793c83d63c2d375140aca991b3abd5cbd7294'


def component_differences(model,actual,reference,*,mj):
    kind=mj.mjtState.mjSTATE_INTEGRATION;n=mj.mj_stateSize(model,kind)
    actual,reference=np.asarray(actual),np.asarray(reference)
    if (actual.shape!=reference.shape or actual.ndim!=2 or not len(actual)
            or actual.shape[1]<2*n+2 or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError('EXACT_FINITE_SAVED_LIVE_IK_CONTROLLER_TRACE_ROWS')
    flags=[(name,flag) for name,flag in mj.mjtState.__members__.items()
        if int(flag)>0 and int(flag)&(int(flag)-1)==0 and int(kind)&int(flag)]
    flags.sort(key=lambda row:int(row[1]))
    if sum(mj.mj_stateSize(model,flag) for _,flag in flags)!=n:
        raise ValueError('COMPLETE_NATIVE_INTEGRATION_COMPONENT_API')
    result={}
    for group,offset in [('live',0),('IK',n)]:
        a,b=mj.MjData(model),mj.MjData(model);values={}
        for name,flag in flags:
            size=mj.mj_stateSize(model,flag);values[name]=dict(size=size,maximum_absolute_difference=0.)
        for row,other in zip(actual,reference):
            mj.mj_setState(model,a,row[offset:offset+n],kind);mj.mj_setState(model,b,other[offset:offset+n],kind)
            for name,flag in flags:
                size=values[name]['size']
                if not size:continue
                av,bv=np.empty(size),np.empty(size)
                mj.mj_getState(model,a,av,flag);mj.mj_getState(model,b,bv,flag)
                maximum=float(np.max(np.abs(av-bv)))
                values[name]['maximum_absolute_difference']=max(values[name]['maximum_absolute_difference'],maximum)
        result[group]=values
    result.update(controller_and_metadata_tail_maximum=float(np.max(np.abs(actual[:,2*n:]-reference[:,2*n:]))),
        full_observed_maximum=float(np.max(np.abs(actual-reference))),integration_vector_size=n,
        steps_and_safety_exact=np.array_equal(actual[:,-2:],reference[:,-2:]),
        scalar_tolerance=1e-10,tolerance_changed=False)
    return result


def run(root,binding_relative):
    root=Path(root);m=validate(root,'4080-cpu-data');q=read(root/binding_relative)
    card_path=root/'plans'/(ID+'_MACHINE_CARD.json');card=read(card_path)
    for path in [root/binding_relative,card_path]:
        if path.is_symlink() or path.stat().st_mode&0o222:raise PermissionError('IMMUTABLE_DECODER_BINDING_AND_CARD')
    if (binding_relative!='plans/'+ID+'.json' or m['execution'].get('trace02_decode_binding_sha256')!=sha(root/binding_relative)
            or m['execution'].get('trace02_machine_card_sha256')!=sha(card_path)
            or q['machine_card_sha256']!=sha(card_path)
            or q['authority_release']!=n1.RELEASE or q['N1_adoption_sha256']!=n1.ADOPTED_DELTA_SHA
            or sha(root/n1.FIRST_ADOPTION)!=n1.ADOPTED_DELTA_SHA or q['maximum_CPU_s']!=600
            or q['worker_CPU_s']!=420 or q['new_physics']!=0 or q['new_neural_queries']!=0):
        raise PermissionError('SEALED_CPU_ONLY_TRACE_COMPONENT_DECODER_BINDING')
    if (card['schema']!='MP1_N1_READONLY_TRACE02_COMPONENT_DECODE_V1' or card['id']!=ID
            or card['authority_release']!=n1.RELEASE or card['only_state']!=config.FAILED_STATE
            or card['CPU_core_s_max']!=600 or card['worker_CPU_s_max']!=420 or card['new_physics_integrations']!=0
            or card['new_neural_queries']!=0 or card['new_engine_rollout_starts']!=0 or card['GPU_PBS']!=0):
        raise PermissionError('EXACT_PUBLISHED_READONLY_TRACE_DECODER_CARD')
    delta=n1.validated_delta(read(root/n1.FIRST_ADOPTION),read(root/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json'))
    if not delta['autonomy_started_epoch']<=q['created_epoch']<=time.time()<q['deadline_epoch']<=min(q['created_epoch']+600,delta['new_experiment_cutoff_epoch']):
        raise PermissionError('ORIGINAL_N1_CLOCK_AND_SINGLE600S_DECODE_WINDOW')
    old=root/'children/sbc512-failure-history-01';trace=root/'children'/config.TRACE_CHILD
    if (sha(old/'evidence/FAILURE_HISTORY_RESULT.json')!=config.B1_RESULT
            or sha(old/'evidence/CLOSED_RECEIPT.private.json')!=config.B1_CLOSED
            or sha(trace/'evidence/ANCHOR_DYNAMIC_PARITY/FAILURE.private.json')!=FAILURE
            or read(read(old/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED'
            or read(read(trace/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED'):
        raise PermissionError('ONLY_EXACT_ALREADY_CLOSED_OWNED_DIAGNOSTIC_BYTES')
    for relative,digest in q['input_files'].items():
        p=root/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or sha(p)!=digest:
            raise PermissionError('EXACT_SAVED_TRACE_DECODER_INPUT_HASH')
    state=next(s for c in read(old/'evidence/FAILURE_HISTORY_RESULT.json')['cases'] for s in c['states'] if s['id']==config.FAILED_STATE)
    arrays,history,_=load_owned_snapshot(old,state)
    d=trace/'evidence/ANCHOR_DYNAMIC_PARITY';actual_path=d/(state['id']+'.actual-prefix.npz');reference_path=d/(state['id']+'.reference-prefix.npz')
    with np.load(actual_path,allow_pickle=False) as z:actual=z['observed_rows'].copy();projected=z['projected_highrate'].copy()
    with np.load(reference_path,allow_pickle=False) as z:reference=z['observed_rows'].copy()
    if len(actual)!=63 or hashlib.sha256(history['engine_python']['xml'].encode()).hexdigest()!=state['model_xml_sha256']:
        raise PermissionError('ORIGINAL63TICK_REJECTED_PREFIX_AND_MODEL_XML_IDENTITY')
    output=root/'evidence'/ID;output.mkdir(mode=0o700);owner=OwnerLock(root/(ID+'.lock'));began=time.process_time()
    os.sched_setaffinity(0,{q['CPU_id']});resource.setrlimit(resource.RLIMIT_CPU,(420,420))
    def expire(signum,frame):raise TimeoutError('FINITE_ZERO_INTEGRATION_TRACE_DECODE_WINDOW')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(q['deadline_epoch']-time.time())))
    try:
        import mujoco
        def forbidden(*args,**kwargs):raise PermissionError('NO_PHYSICAL_INTEGRATION_IN_SAVED_TRACE_DECODER')
        mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
        model=mujoco.MjModel.from_xml_string(history['engine_python']['xml'])
        value=component_differences(model,actual,reference,mj=mujoco)
        engine=SimpleNamespace(model=model,ids=history['engine_python']['ids'])
        decoded=project_observed_to_highrate(engine,reference,mj=mujoco)
        if projected.shape!=decoded.shape:raise ValueError('MATCHED_SAVED_NATIVE_HIGHRATE_PROJECTION_SHAPE')
        value.update(status='PASS_READONLY_STATE_COMPONENT_DECODE_NOT_DYNAMIC_QUALIFICATION',
            source_sha=m['code']['vla_commit'],state=state['id'],MuJoCo_version=mujoco.__version__,
            decoded_native_highrate_maximum_difference=float(np.max(np.abs(projected-decoded))),
            original_live_integration_size=len(arrays['live']),nq=model.nq,nv=model.nv,
            static_MjModel_compilations=1,static_MjData_buffers=5,new_engine_rollout_starts=0,
            new_physics_integrations=0,new_neural_queries=0,new_optimizer_updates=0,new_Teacher_labels=0,
            original_failed_trace_maximum=read(d/'FAILURE.private.json')['mismatch'],
            actual_trace_sha256=sha(actual_path),reference_trace_sha256=sha(reference_path),
            original_safety_or_parity_thresholds_changed=False,full_eight_anchor_gate_passed=False,
            unique_restoration_mechanism_proven=False,full_in_process_state_vs_roundtrip_ablation='NOT_RUN')
        write(output/'RESULT.json',value);return value
    except BaseException as exc:
        write(output/'FAILURE.private.json',dict(status='READONLY_COMPONENT_DECODE_GAP_NO_RETRY',type=type(exc).__name__,error=str(exc)));raise
    finally:signal.alarm(0);charge_cpu(root,time.process_time()-began);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--binding',required=True);a=p.parse_args()
    print(json.dumps(dict(status=run(a.root,a.binding)['status'])),flush=True)
