"""Decode bounded candidate timing evidence; reading never constructs a simulator."""
from collections import Counter
from pathlib import Path
import json,struct,math,hashlib
from .storage import digest

KINDS={1:'OBS',2:'INFERENCE_BEGIN',3:'INPUT',4:'HISTORY',5:'INFERENCE_DONE',6:'PUBLISH',7:'CONSUME',8:'APPLY',9:'PD_TARGET',10:'PD_STATE_CTRL',11:'STEP_BEFORE',12:'STEP_AFTER',13:'STOP',14:'REFUSED',15:'INFERENCE_SKIPPED',16:'MODE',17:'KEY_RECEIVED',18:'KEY_APPLIED'}
RECORD=struct.Struct('<QQdIIQQQQQ96d')

def read_trace(path):
    path=Path(path);rows=[]
    with path.open('rb')as f:
        header=f.read(16)
        if len(header)!=16 or header[:8]!=b'WM02IT01':raise ValueError('Unknown trace header')
        size,cap=struct.unpack('<II',header[8:])
        if size!=RECORD.size or cap not in(65536,131072,262144,524288):raise ValueError('Unknown trace layout')
        while blob:=f.read(size):
            if len(blob)!=size:raise ValueError('Incomplete trace record')
            a=RECORD.unpack(blob)
            if a[3]not in KINDS or a[4]>96 or a[0]!=len(rows):raise ValueError('Invalid trace sequence/kind/payload')
            if not math.isfinite(a[2])or not all(math.isfinite(v)for v in a[10:10+a[4]]):raise ValueError('Nonfinite trace time/payload')
            row=dict(zip(('ticket','monotonic_ns','sim_time_s','kind','n','observation_version','latest_started_policy','applied_position_version','applied_velocity_version','token'),a[:10]),values=list(a[10:10+a[4]]))
            # Oversized INPUT/HISTORY records have n=0; their first3 raw doubles
            # are a reference, not an empty observation. Keep the original ABI.
            if a[3]in(3,4)and a[4]==0 and not all(math.isfinite(v)for v in a[10:13]):raise ValueError('Nonfinite full payload descriptor')
            if a[3]in(3,4)and a[4]==0 and a[12]>96:
                slot,generation,count=a[10:13]
                if any(not math.isfinite(v)or int(v)!=v for v in(slot,generation,count)):raise ValueError('Invalid full payload reference')
                if slot!=-1 and(not 0<=slot<32 or generation<1):raise ValueError('Invalid snapshot slot/generation')
                row['payload_reference']=dict(slot=int(slot),generation=int(generation),original_count=int(count))
            rows.append(row)
    return rows


def read_snapshots(path,*,keep_values=False):
    """Stream bounded f32 payloads. Metadata/hashes by default; no model import."""
    records={};header=struct.Struct('<5Q')
    with Path(path).open('rb')as f:
        while raw:=f.read(header.size):
            if len(raw)!=header.size:raise ValueError('Incomplete snapshot header')
            generation,slot,token,kind,n=header.unpack(raw)
            expected= generation-1 if generation<=4 else 4+(generation-5)%28
            if not generation or slot!=expected or slot>=32 or kind not in(3,4)or not 96<n<=16384:raise ValueError('Snapshot layout/generation differs')
            blob=f.read(4*n)
            if len(blob)!=4*n:raise ValueError('Incomplete snapshot payload')
            values=struct.unpack('<'+str(n)+'f',blob)
            if not all(math.isfinite(v)for v in values):raise ValueError('Nonfinite full snapshot')
            key=(generation,slot)
            if key in records or len(records)>=65536:raise ValueError('Duplicate/excess snapshot generation')
            record=dict(generation=generation,slot=slot,token=token,kind=kind,count=n,sha256=hashlib.sha256(blob).hexdigest())
            if keep_values:record['values']=list(values)
            records[key]=record
    return records


def payload_metadata(row,snapshots):
    """Never present a retained/truncated/missing vector as a full policy input."""
    if row['n']:
        raw=struct.pack('<'+str(row['n'])+'d',*row['values'])
        return dict(status='INLINE_COMPLETE',count=row['n'],encoding='f32 promoted to raw f64'if row['kind']in(3,4)else'raw f64',sha256=hashlib.sha256(raw).hexdigest())
    ref=row.get('payload_reference')
    if not ref:return dict(status='EMPTY',count=0)
    if ref['slot']==-1:return dict(status='UNAVAILABLE_AFTER_STOP',count=ref['original_count'])
    found=snapshots.get((ref['generation'],ref['slot']))
    if found is None:return dict(status='NOT_RETAINED_OR_NOT_WRITTEN',count=ref['original_count'],**ref)
    if found['token']!=row['token']or found['kind']!=row['kind']:raise ValueError('Snapshot reference token/kind differs')
    status='SNAPSHOT_COMPLETE'if found['count']==ref['original_count']else'TRUNCATED'
    return dict(status=status,count=found['count'],original_count=ref['original_count'],sha256=found['sha256'],encoding='f32')


def audit_trace(path,*,max_native_ticks=None):
    """Evidence audit only. A coherent synthetic trace never grants native admission."""
    path=Path(path);rows=read_trace(path);sp=Path(str(path)+'.snapshots');snapshots=read_snapshots(sp)if sp.exists()else{};summary=Path(str(path)+'.summary');writer=json.loads(summary.read_text())if summary.exists()else None
    errors=[];missing=[]
    if writer is None:missing.append('WRITER_SUMMARY_MISSING')
    else:
        if any(type(writer.get(k))is not int or writer[k]<0 for k in('events','dropped'))or type(writer.get('stop_latched'))is not bool:raise ValueError('Invalid writer summary')
        if writer['dropped']:errors.append('WRITER_REPORTED_DROPPED_EVENTS')
        if writer['events']!=len(rows):errors.append('WRITER_EVENT_COUNT_DIFFERS')
        if 'written'in writer and(type(writer['written'])is not int or writer['written']!=len(rows)):errors.append('WRITER_WRITTEN_COUNT_DIFFERS')
    payloads={r['ticket']:payload_metadata(r,snapshots)for r in rows if r['kind']in(3,4)}
    unretained=sum(v['status']=='NOT_RETAINED_OR_NOT_WRITTEN'for v in payloads.values())
    truncated=sum(v['status']=='TRUNCATED'for v in payloads.values())
    if truncated:errors.append('TRUNCATED_FULL_INPUT_OR_HISTORY')
    # The finite ring deliberately retains first4 + recent28. Its overwritten
    # payloads are separate from dropped event records; dropped=0 is insufficient.
    if unretained:missing.append('FULL_INPUT_OR_HISTORY_NOT_RETAINED')
    begins={};completed={};skipped=set();published={0:set(),1:set()};consumed={0:None,1:None};channel_unknown=0;mixed=0;apply_unmatched=0
    for r in rows:
        token=r['token'];kind=r['kind']
        if kind==2:
            if not token or token in begins:errors.append('DUPLICATE_OR_ZERO_POLICY_BEGIN')
            begins[token]=r
        elif kind==5:
            if token in completed:errors.append('DUPLICATE_POLICY_DONE')
            completed[token]=r
            if token not in begins or r['monotonic_ns']<begins[token]['monotonic_ns']:errors.append('POLICY_DONE_WITHOUT_VALID_BEGIN')
            if not r['n']:missing.append('POLICY_OUTPUT_NOT_RECORDED')
        elif kind==15:skipped.add(token)
        elif kind in(6,7):
            if r['n']!=1 or r['values'][0]not in(0,1):
                if kind==6 and r['n']==0:channel_unknown+=1;continue # Preserved legacy trace, not inferred channel.
                errors.append('INVALID_QUEUE_CHANNEL');continue
            channel=int(r['values'][0])
            if kind==6:published[channel].add(token)
            else:
                consumed[channel]=token
                if token not in published[channel]and not channel_unknown:errors.append('CONSUME_WITHOUT_CHANNEL_PUBLICATION')
        elif kind==8:
            pair=(r['applied_position_version'],r['applied_velocity_version']);mixed+=pair[0]!=pair[1]
            apply_unmatched+=pair!=(consumed[0],consumed[1])
    if channel_unknown:missing.append('LEGACY_PUBLICATION_CHANNEL_NOT_RECORDED')
    if mixed:errors.append('ACTUAL_POSITION_VELOCITY_VERSION_MISMATCH')
    if apply_unmatched:errors.append('APPLY_DOES_NOT_MATCH_RECORDED_CONSUMPTION')
    first=[]
    for token in sorted(t for t in completed if t not in skipped)[:2]:
        evidence={}
        for k,name in((3,'input'),(4,'history')):
            matches=[r for r in rows if r['kind']==k and r['token']==token and token in begins and begins[token]['ticket']<r['ticket']<completed[token]['ticket']]
            evidence[name]=[payloads[r['ticket']]for r in matches]
            if len(matches)!=1 or evidence[name][0]['status']not in('INLINE_COMPLETE','SNAPSHOT_COMPLETE'):missing.append('FIRST_TWO_POLICY_'+name.upper()+'_INCOMPLETE')
        first.append(dict(policy=token,output_count=completed[token]['n'],**evidence))
    if len(first)<2:missing.append('FEWER_THAN_TWO_COMPLETED_POLICY_VERSIONS')
    before={};after={}
    for r in rows:
        if r['kind']in(11,12):
            target=before if r['kind']==11 else after
            if r['token']in target:errors.append('DUPLICATE_NATIVE_STEP_TOKEN')
            target[r['token']]=r
    if set(before)!=set(after):errors.append('UNPAIRED_NATIVE_STEP_ENTRY_EXIT')
    for token in before.keys()&after.keys():
        a,z=before[token],after[token]
        if z['ticket']<=a['ticket']or abs(z['sim_time_s']-a['sim_time_s']-.002)>1e-8:errors.append('NATIVE_STEP_CLOCK_OR_ORDER_DIFFERS')
        if a['n']!=86 or z['n']!=86:errors.append('NATIVE_STEP_ABI_NOT_34_32_20')
    if sorted(after)!=list(range(1,len(after)+1)):errors.append('NATIVE_STEP_TOKENS_NOT_CONTIGUOUS_FROM_COLD_START')
    if max_native_ticks is not None and(type(max_native_ticks)is not int or max_native_ticks<=0):raise ValueError('Positive bounded tick limit required')
    if max_native_ticks is not None and len(after)>max_native_ticks:errors.append('DECLARED_NATIVE_TICK_LIMIT_EXCEEDED')
    stops=[r for r in rows if r['kind']==13];stop=stops[0]if stops else None
    extra=sum(token>stop['token']for token in after)if stop else None
    if extra:errors.append('NATIVE_INTEGRATIONS_AFTER_FIRST_STOP')
    if writer is not None and writer['stop_latched']!=bool(stops):errors.append('STOP_LATCH_AND_TRACE_DIFFERS')
    if not after:missing.append('NO_NATIVE_STEP_RECORDS')
    pd=[r for r in rows if r['kind']==10]
    if not pd:missing.append('NO_PD_CALLBACK_RECORDS')
    if any(r['n']!=86 for r in pd):errors.append('PD_STATE_CTRL_ABI_NOT_34_32_20')
    status='RECORDED_TRACE_CHECKS_FAIL'if errors else'INSUFFICIENT_RECORDED_EVIDENCE'if missing else'RECORDED_TRACE_CHECKS_PASS'
    return dict(schema='WM02_INIT_TRACE_EVIDENCE_AUDIT_V1',status=status,errors=sorted(set(errors)),missing=sorted(set(missing)),records=len(rows),writer=writer,full_payload_records=len(snapshots),full_payloads_not_retained=unretained,first_two_policy_evidence=first,legacy_publish_channels_unknown=channel_unknown,applied_mixed_versions=mixed,unmatched_apply_versions=apply_unmatched,native_step_records=len(after),PD_callbacks_recorded=len(pd),integrations_after_first_STOP=extra,STOP_observed=bool(stops),STOP_check='ZERO_FOLLOWING_RECORDED_STEPS'if stop and not extra else'FAIL'if extra else'NOT_TESTED',source_clock_note='PUBLISH/CONSUME/APPLY sim_time=0 is an instrumentation sentinel; use monotonic_ns and version causality, not physics time for those records.',input_files_sha256={p.name:digest(p)for p in(path,sp,summary)if p.exists()},target_qualification='NOT_ESTABLISHED: independent owner admission, original safety/contacts and trace overhead still required',new_effects_of_audit=dict(target_integrations=0,policy_queries=0))

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
