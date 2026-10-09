"""CPU receipt from actual endpoint clocks; no query, physics or candidate choice."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
from contract import ROOT,guard,sha,write
from policy import ENDPOINTS
from qualification import panel,SCIENCE_STOP


def measured(script,args):
    name=script.replace('.py','')+'-'+hashlib.sha256(json.dumps(args).encode()).hexdigest()[:12]
    path=ROOT/'runtime'/(name+'.EVAL_COMPLETE.json')
    v=json.loads(path.read_text())
    assert v['returncode']==0 and v['script']==script and v['arguments']==args
    seconds=v['ended_epoch']-v['started_epoch'];assert math.isfinite(seconds) and seconds>0
    return dict(path=str(path),sha256=sha(path),seconds=seconds,started_epoch=v['started_epoch'],ended_epoch=v['ended_epoch'])


def measure(point,emit_design=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and point in ENDPOINTS;guard()
    train,tp=panel(point,'TRAIN');dev,dp=panel(point,'DEV')
    assert train and dev and train['T1']>=7 and dev['T1']>=13
    route=ROOT/'qualification'/(point+'_ROUTE.json');rv=json.loads(route.read_text())
    assert rv['next_action']=='ELIGIBLE_WAIT_PREDECLARED_SELECTION'
    diag=ROOT/'audit'/(point+'_DIAG_CPU.json');dv=json.loads(diag.read_text())
    assert dv['status']=='PASS_CPU_FIXED_DIAG_NATIVE_DECODE_NOISE_PREFIX_FULL_H40_AND_GATE' and len(dv['rows'])==96
    assert dv['source_sha256']==sha(ROOT/'diagnostics'/point/'RESULT.json')
    clocks={}
    clocks['transport']=measured('transport.py',['--point',point])
    overhead=[];episodes=[]
    for split in ('TRAIN','DEV'):
        source=ROOT/'source/evidence'/f'{point}_{split}.json';rows=json.loads(source.read_text())['rows']
        elapsed=sum(r['elapsed_wall_s'] for r in rows)
        clocks[split]=measured('policy.py',['--point',point,'--panel',split])
        assert not any((Path(r['folder']).parent/'attempt-001').exists() for r in rows),'Replacement needs owner clock reconciliation'
        extra=clocks[split]['seconds']-elapsed;assert extra>=0
        overhead.append(extra);episodes.extend(r['elapsed_wall_s'] for r in rows if r['status']=='VALID_SOURCE_COMPLETE')
        clocks[split]['valid_episode_sum_seconds']=elapsed;clocks[split]['combined_load_initialization_hash_closure_overhead_seconds']=extra
        clocks[split+'_CPU']=measured('audit_source.py',['--split',point+'_'+split])
    clocks['DIAG']=measured('diagnostics.py',['run','--point',point])
    clocks['DIAG_CPU']=measured('diagnostics.py',['cpu','--point',point])
    # Conservative measured preparation allowance; completed transfer/DIAG are counted again.
    # Future32 CPU audit is a projection from actual16 CPU, explicitly not an observed32 audit.
    cpu32_bound=2*clocks['DEV_CPU']['seconds']
    extra=clocks['transport']['seconds']+max(overhead)+sum(clocks[x]['seconds'] for x in ('TRAIN_CPU','DEV_CPU','DIAG','DIAG_CPU'))+cpu32_bound
    p90=sorted(episodes)[math.ceil(.9*len(episodes))-1];required=max(2700.,1.25*p90*32+extra)
    v=dict(status='OWNER_MEASURED_CONFIRM_LOAD_CPU_AUDIT_AND_HANDOFF_RESERVE',point=point,measured_extra_seconds=extra,
        valid_until_epoch=SCIENCE_STOP,actual_endpoint_clocks=clocks,development_CPU=dict(TRAIN=tp,DEV=dp),route_sha256=sha(route),DIAG_CPU_sha256=sha(diag),
        measured_complete_episodes=len(episodes),p90_seconds=p90,required_reserve_seconds=required,latest_by_reserve_epoch=SCIENCE_STOP-required,
        extra_method='ACTUAL_TRANSFER_PLUS_MAX_ACTUAL_PANEL_OVERHEAD_PLUS_ACTUAL_CPU_DIAG_AND_2X_DEV16_CPU_PROJECTION',
        projected_future32_CPU_audit_seconds=cpu32_bound,future_confirmation_load_or_CPU_not_yet_observed=True,
        completed_transfer_and_DIAG_counted_again_conservatively=True,new_model_queries=0,new_physics=0,confirmation_started=False,
        candidate_freeze_authorized_by_this_receipt=False,selected='NONE',deployment='NONE',checked_epoch=time.time())
    write(ROOT/'evidence'/(point+'_MEASURED_CONFIRM_RESERVE.json'),v)
    if emit_design:
        path=ROOT/'design/CONFIRM_C_TIME_RESERVE.json';assert not path.exists(),'Owner must reconcile an existing time receipt'
        write(path,v)
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True);p.add_argument('--emit-design',action='store_true');a=p.parse_args();print(json.dumps(measure(a.point,a.emit_design)))
