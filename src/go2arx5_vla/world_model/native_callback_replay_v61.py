"""Export a recorded callback schedule for an isolated prefix-replay diagnostic.

This file does not change, build or start a native controller. Reproducing the
schedule is not proof of reconstructed state; the original anchor gate remains.
"""
import collections
import hashlib
import json
import struct
from pathlib import Path

VERSION='SLA_RECORDED_NATIVE_CALLBACK_PREFIX_REPLAY_V6_1'
RECORD=struct.Struct('<QQdIIQQQQQ96d')
PAYLOAD_OFFSET=struct.calcsize('<QQdIIQQQQQ')


def prefix_schedule(trace, *, stop_tick):
    if type(stop_tick)is not int or not 1<stop_tick<=5703:
        raise ValueError('FINITE_ORIGINAL_STARTUP_PREFIX_ONLY')
    phases=collections.defaultdict(list);last=0;expected_ticket=0
    steps=[];policies={};targets={};completed={}
    with Path(trace).open('rb')as stream:
        header=stream.read(16)
        if len(header)!=16 or header[:8]!=b'WM02IT01' or struct.unpack('<II',header[8:])[0]!=RECORD.size:
            raise ValueError('ORIGINAL_NATIVE_TRACE_ABI')
        while raw:=stream.read(RECORD.size):
            if len(raw)!=RECORD.size:raise ValueError('COMPLETE_TRACE_RECORD')
            row=RECORD.unpack(raw);ticket,_,stamp,kind,n,_,_,ap,av,token=row[:10]
            if ticket!=expected_ticket:raise ValueError('CONTIGUOUS_TRACE_TICKETS')
            expected_ticket+=1
            if kind==2:policies[token]=stamp
            if kind==5:completed[token]=hashlib.sha256(raw[PAYLOAD_OFFSET:PAYLOAD_OFFSET+8*n]).hexdigest()
            if kind==9:
                if n!=90 or ap!=av:raise ValueError('COMPLETE_COHERENT_FROZEN_PD_TARGET')
                targets[last]=hashlib.sha256(raw[PAYLOAD_OFFSET:PAYLOAD_OFFSET+8*n]).hexdigest()
            if kind==10:
                if n!=86 or ap!=av or abs(stamp-last*.002)>1e-8:
                    raise ValueError('ORIGINAL_PD_STATE_AT_CURRENT_NATIVE_TICK')
                if last and ap<1:raise ValueError('NO_UNKNOWN_APPLIED_POLICY_IN_REPLAY')
                sequence=phases[last]
                if sequence and sequence[-1][0]==ap:sequence[-1][1]+=1
                else:sequence.append([ap,1])
            if kind==12:
                if token!=last+1 or abs(stamp-token*.002)>1e-8:
                    raise ValueError('CONTIGUOUS_TWO_MS_INTEGRATIONS')
                last=token;steps.append(token)
                if last==stop_tick:break
    if last!=stop_tick:raise ValueError('RECORDED_PREFIX_MISSING')
    rows=[]
    for step in range(1,stop_tick):
        seq=phases[step]
        if any(policy not in policies or policy not in completed for policy,count in seq):
            raise ValueError('REPLAY_POLICY_HAS_NO_ACTUAL_INFERENCE')
        # The same original20ms policy clock;no invented version interpolation.
        if any(abs(policies[p]-(p-1)*.020)>1e-8 for p,c in seq):
            raise ValueError('ORIGINAL_COLD_POLICY_VERSION_CLOCK_CHANGED')
        if any(seq[i][0]>seq[i+1][0]for i in range(len(seq)-1)):
            raise ValueError('ORIGINAL_APPLIED_POLICY_REGRESSION')
        rows.append(dict(native_tick=step,phases=[dict(policy=p,count=c)for p,c in seq]))
    return dict(schema=VERSION,stop_tick=stop_tick,physics_dt_s=.002,
        policy_period_s=.020,rows=rows,skip_uncontrolled_zero_tick=True,
        callbacks=sum(p['count']for r in rows for p in r['phases']),
        source_trace_sha256=hashlib.sha256(Path(trace).read_bytes()).hexdigest(),
        controller_math_changed=False,state_restore_proven=False,physics_integrations=0)


def write_schedule(trace, output, *, stop_tick):
    output=Path(output)
    if output.exists():raise FileExistsError('RECORDED_REPLAY_SCHEDULE_NEVER_OVERWRITTEN')
    value=prefix_schedule(trace,stop_tick=stop_tick);output.mkdir()
    lines=['WM02_CALLBACK_REPLAY_V61 '+str(stop_tick)]
    for row in value['rows']:
        lines.append(str(row['native_tick'])+' '+str(len(row['phases']))+''.join(
            ' '+str(p['policy'])+' '+str(p['count'])for p in row['phases']))
    schedule=output/'CALLBACK_SCHEDULE.private.txt';schedule.write_text('\n'.join(lines)+'\n')
    value['schedule_sha256']=hashlib.sha256(schedule.read_bytes()).hexdigest()
    (output/'MANIFEST.private.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


INCLUDE='#include "callback_replay_v61.hpp"\n'
CONTROL='''    if (!wm02_callback_replay_v61::clock().control_allowed(
            wm02_callback_replay_v61::tick(this->mj_data->time),
            wm02_applied_control_clock::clock().published_policy.load()+1)) return;
'''
POLICY='''        if(!wm02_callback_replay_v61::clock().policy_allowed(
            wm02_callback_replay_v61::tick(wm02_policy_time_s),
            std::llround(wm02_policy_time_s*1e9)/wm02_policy_period_ns+1))return;
'''
COMPLETE='\n    wm02_callback_replay_v61::clock().control_complete(wm02_callback_replay_v61::tick(this->mj_data->time));'


def patch(cpp,utils):
    """Extra scheduling constraints;remove them to recover exact input text."""
    begin='void RL_Sim::RobotControl()\n{';end='void RL_Sim::StartRosInterface()'
    lock='    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);\n'
    due='        if(!wm02_applied_control_clock::clock().policy_due(wm02_policy_time_s,wm02_policy_period_ns))return;'
    mark='    wm02_applied_control_clock::clock().control(this->mj_data->time,std::llround(double(this->params.Get<float>("dt"))*1e9));'
    if cpp.count(begin)!=1 or cpp.count(end)!=1 or INCLUDE in cpp or INCLUDE in utils:
        raise ValueError('EXACT_FROZEN_NATIVE_CONTROL_FUNCTIONS')
    a=cpp.index(begin);b=cpp.index(end,a);body=cpp[a:b]
    if body.count(lock)!=1 or cpp.count(due)!=1 or cpp.count(mark)!=1:
        raise ValueError('EXACT_EXISTING_CALLBACK_CLOCK_AND_CONTROL_MUTEX')
    changed=INCLUDE+cpp[:a]+body.replace(lock,lock+CONTROL,1)+cpp[b:]
    changed=changed.replace(due,POLICY+due,1).replace(mark,mark+COMPLETE,1)
    old='!wm02_applied_control_clock::clock().ready(d->time)'
    new=old+'||!wm02_callback_replay_v61::clock().integration_ready(wm02_callback_replay_v61::tick(d->time))'
    if utils.count(old)!=2:raise ValueError('BOTH_ORIGINAL_INTEGRATION_GATES_REQUIRED')
    after=INCLUDE+utils.replace(old,new)
    if(changed[len(INCLUDE):].replace(CONTROL,'',1).replace(POLICY,'',1).replace(COMPLETE,'',1)!=cpp
            or after[len(INCLUDE):].replace(new,old)!=utils):
        raise ValueError('RECORDED_SCHEDULE_GATE_CHANGED_ORIGINAL_CONTROL_OR_PHYSICS_TEXT')
    return changed,after
