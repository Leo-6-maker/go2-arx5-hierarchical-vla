"""Decode bounded candidate timing evidence; reading never constructs a simulator."""
from collections import Counter
from pathlib import Path
import json,struct

KINDS={1:'OBS',2:'INFERENCE_BEGIN',3:'INPUT',4:'HISTORY',5:'INFERENCE_DONE',6:'PUBLISH',7:'CONSUME',8:'APPLY',9:'PD_TARGET',10:'PD_STATE_CTRL',11:'STEP_BEFORE',12:'STEP_AFTER',13:'STOP',14:'REFUSED',15:'INFERENCE_SKIPPED',16:'MODE'}
RECORD=struct.Struct('<QQdIIQQQQQ96d')

def read_trace(path):
    path=Path(path);rows=[]
    with path.open('rb')as f:
        header=f.read(16)
        if len(header)!=16 or header[:8]!=b'WM02IT01':raise ValueError('Unknown trace header')
        size,cap=struct.unpack('<II',header[8:])
        if size!=RECORD.size or cap!=65536:raise ValueError('Unknown trace layout')
        while blob:=f.read(size):
            if len(blob)!=size:raise ValueError('Incomplete trace record')
            a=RECORD.unpack(blob)
            if a[3]not in KINDS or a[4]>96 or a[0]!=len(rows):raise ValueError('Invalid trace sequence/kind/payload')
            rows.append(dict(zip(('ticket','monotonic_ns','sim_time_s','kind','n','observation_version','latest_started_policy','applied_position_version','applied_velocity_version','token'),a[:10]),values=list(a[10:10+a[4]])))
    return rows

def summarize(path):
    rows=read_trace(path);begins={r['token']:r for r in rows if r['kind']==2};done=[r for r in rows if r['kind']==5];pd=[r for r in rows if r['kind']==10];stops=[r for r in rows if r['kind']==13]
    deltas=[(b['monotonic_ns']-a['monotonic_ns'])/1e6 for a,b in zip(pd,pd[1:])]
    latencies=[(r['monotonic_ns']-begins[r['token']]['monotonic_ns'])/1e6 for r in done if r['token']in begins]
    stop=stops[0]['ticket']if stops else None
    summary_path=Path(str(path)+'.summary');writer=json.loads(summary_path.read_text())if summary_path.exists()else None
    return dict(scope='Timing evidence only; requires independent run identity and native safety evidence',events=dict(Counter(KINDS[r['kind']]for r in rows)),
        records=len(rows),writer=writer,first_valid_policy=next((dict(policy=r['token'],sim_time_s=r['sim_time_s'],monotonic_ns=r['monotonic_ns'])for r in done),None),
        inference_latency_ms=latencies,PD_interval_ms=deltas,applied_mixed_versions=sum(r['applied_position_version']!=r['applied_velocity_version']for r in rows if r['kind']==8),
        integrations_after_first_STOP=sum(r['kind']==12 and r['ticket']>stop for r in rows)if stop is not None else None,
        STOP_observed=bool(stops),full_observation_payloads=str(path)+'.snapshots',target_qualification='NOT_ESTABLISHED_BY_TRACE_READER')
