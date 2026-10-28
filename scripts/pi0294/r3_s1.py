"""Finite S1 known-failure qualification on original D0, no neural queries."""
import argparse,json,os,sys,time,uuid
from pathlib import Path
if not __package__:
    sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'src')]
    __package__='scripts.pi0294'
from scripts.pi0292.durable import OwnerLock,read,write
from .continuation import validate_entry,CUTOFF,RUN
from .gripper_safety import saved_inputs,create_original
from .s1_preflight import predict,bind_install,observed_state,VERSION

ENGINEERING_START=1791982997  # first S1 source creation:2026-10-14 13:03:17UTC
ENGINEERING_DEADLINE=ENGINEERING_START+7200


def known(root,output,case,ledger,guard):
    import numpy as np
    from scripts.pi0292.eval_contract import matrix10
    from .scene import qualify_engine
    binding=read(root/'INPUT_BINDINGS.private.json');card,initial,old,nq,nv,commands=saved_inputs(binding)
    engine=create_original(card,initial,binding);commands=commands.copy()
    if case=='B':commands[:,9]=1.
    records=[];rows=[];rejected=None;maximum=0.
    def charge(count,*,invocation):
        if invocation:
            if ledger['shadow_advance_invocations']+1>2500 or ledger['shadow_integrations']+200>600000:
                raise PermissionError('S1_SHADOW_CAP')
            ledger['shadow_advance_invocations']+=1
            ledger['shadow_integrations']+=200
            ledger['shadow_pending_reservation']=200
            write(root/'RESOURCE_LEDGER.json',ledger)
    try:
        engine.snapshot();qualification=qualify_engine(engine,render=False,guard=guard)
        if not qualification['geometry_passed']:raise ValueError('ORIGINAL_STATIC_GEOMETRY_MISMATCH')
        for i in range(1200):
            guard()
            if i%25==0:engine.snapshot()
            if i%200==0:
                segment=commands[i//25:i//25+8];poses=matrix10(segment);opening=segment[:,9]
                nonce=str(uuid.uuid4());forecast,expected=predict(engine,poses,opening,nonce=nonce,guard=guard,charge=charge)
                ledger['shadow_integrations']-=200-forecast['shadow_integrations']
                ledger.pop('shadow_pending_reservation')
                write(root/'RESOURCE_LEDGER.json',ledger)
                records.append(forecast)
                if forecast['decision']=='REJECT':rejected=forecast;break
                bind_install(engine,poses,opening,forecast,nonce)
            engine.step(matrix10(commands[i//25]),float(commands[i//25,9]));engine.snapshot()
            actual=observed_state(engine);error=float(np.max(np.abs(actual-expected[i%200])))
            maximum=max(maximum,error)
            if error>1e-10:raise RuntimeError('S1_CLONE_TRAJECTORY_DIFFERS_FROM_UNMODIFIED_LIVE')
            rows.append(np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,
                engine.data.xfrc_applied[engine.ids['tool']],float(bool(engine.safety))])
        np.savez_compressed(output/'protected_live.npz',rows=np.asarray(rows),nq=nq,nv=nv)
        write(output/'SHADOW_PREFLIGHTS.json',records)
        expected_offset=81 if case=='A' else 118
        if rejected is None or engine.steps!=1000 or rejected['first_predicted_violation_offset_tick']!=expected_offset or engine.safety:
            raise ValueError('S1_KNOWN_FAILURE_NOT_REJECTED_AT_EXACT_EXPECTED_OFFSET')
        old_path=Path(binding['R2_root'])/'evidence'/('G1-'+case)/'actual_replay.npz'
        with np.load(old_path,allow_pickle=False) as original:byte_equal=np.array_equal(np.asarray(rows),original['rows'][:1000])
        if not byte_equal:raise ValueError('S1_ADDED_PREFLIGHT_CHANGED_ORIGINAL_LIVE_PREFIX')
        result=dict(status='PASS_KNOWN_FAILURE_PRE_EXECUTION_REJECTION_NOT_MODEL_SUCCESS',case=case,
            version=VERSION,main_integrations=engine.steps,live_native_failures=0,
            original_first1000_rows_exact=True,maximum_live_shadow_state_difference=maximum,
            rejected_prefix_start_tick=1000,predicted_violation_offset_tick=expected_offset,
            known_original_violation_tick=1000+expected_offset,pre_execution_rejection=True,
            rejected_prefix_live_integrations=0,shadow_invocations=len(records),
            shadow_integrations=sum(r['shadow_integrations'] for r in records),
            selected='NONE',deployment='NONE')
        write(output/'RESULT.json',result);return result
    finally:engine.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',choices=['A','B'],required=True);a=p.parse_args()
    root=Path(a.root);validate_entry(root,profile='4080-simulation')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU_ONLY')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    lock=OwnerLock(root/'S1_OWNER.lock');out=root/'evidence'/('S1-known-'+a.case);out.mkdir(exist_ok=False)
    ledger=read(root/'RESOURCE_LEDGER.json');t=time.process_time();result=None
    if ledger['main_or_diagnostic_starts']>=32:raise PermissionError('R3_START_CAP')
    def guard():
        if time.time()>=min(ENGINEERING_DEADLINE,CUTOFF):raise TimeoutError('S1_ENGINEERING_TIMEBOX_CLOSED')
    try:result=known(root,out,a.case,ledger,guard)
    finally:
        ledger['main_or_diagnostic_starts']+=1
        ledger['main_or_diagnostic_integrations']+=result['main_integrations'] if result else 1200
        ledger['Core_CPU_s']+=time.process_time()-t
        ledger['stages'].append(dict(phase=out.name,status=result['status'] if result else 'ENGINEERING_FAILURE_PRESERVED'))
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()
    print(json.dumps(result))


if __name__=='__main__':main()
