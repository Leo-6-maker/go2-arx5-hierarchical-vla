"""Actual-time barrier identities and narrowly proved pre-install recovery."""
import json
from pathlib import Path

FROZEN_FIELDS=('step','time_s','motiontime','leg_history_steps','gait_indices')


def actual_source_age(packet_time,source):
    # ROS clock adapter publishes integer nanoseconds of the actual mjData time.
    # Retain raw time in evidence; compare in that same existing clock domain.
    return round(packet_time*1e9)/1e9-source


def require_pause(before,after,step):
    if (not before['clock_paused'] or not after['clock_paused'] or before['running'] or after['running']
            or before['step']!=step or after['step']!=step
            or abs(after['time_s']-step*.002)>1e-8
            or any(before[k]!=after[k] for k in FROZEN_FIELDS)
            or after['ack_sequence']<=before['ack_sequence']):
        raise ValueError('PHYSICS_FSM_OR_LEG_HISTORY_ADVANCED_WHILE_PAUSED')


def finish_pause(before,step,actual_step,reopen_peers,wait_live,read_ack):
    if actual_step()!=step:raise ValueError('PHYSICS_ADVANCED_DURING_QUERY')
    reopen_peers()
    if not wait_live():raise RuntimeError('LIVE_PAUSE_ACK_MISSING')
    after=read_ack()
    require_pause(before,after,step)
    if actual_step()!=step:raise ValueError('PHYSICS_ADVANCED_DURING_QUERY')
    return after


def preinstall_clock_failure(folder):
    """Only the already-proved first transaction is replaceable, never quality."""
    folder=Path(folder)
    result=json.loads((folder/'RESULT.json').read_text())
    if folder.name=='TASK-PILOT-W0-T2-engineering03':
        if (result.get('source_sha')!='2847f14640ad21a58813ac1c9e230ad354637b5c'
                or result.get('error')!='RuntimeError: LIVE_PAUSE_ACK_MISSING'
                or result.get('native_stops') or result.get('observations')!=3 or result.get('submitted_intents')!=3
                or result.get('actual_plugin_steps')!=2575):return False
        ack=json.loads((folder/'pause_ack.json').read_text())
        rows=[json.loads(s) for s in (folder/'planner.jsonl').read_text().splitlines()]
        return (ack['step']==2575 and ack['clock_paused'] and not ack['running']
                and sum(r['kind']=='accept' for r in rows)==3 and not any(r['kind']=='reject' for r in rows))
    if folder.name=='TASK-PILOT-W0-T2-engineering02':
        if (result.get('source_sha')!='efea8ae9277a3db798dcdf6873754197dde1ca64'
                or result.get('error')!='RuntimeError: NATIVE_WBC_MODE_UNAVAILABLE'
                or result.get('native_stops') or result.get('observations') or result.get('submitted_intents')
                or result.get('actual_plugin_steps')!=2500):return False
        ack=json.loads((folder/'pause_ack.json').read_text())
        rows=[json.loads(s) for s in (folder/'planner.jsonl').read_text().splitlines()]
        return (ack['step']==2500 and ack['clock_paused'] and not ack['running']
                and not any(r['kind']=='accept' or r.get('ref_id') for r in rows))
    if (folder.name!='TASK-PILOT-W0-T2' or result.get('source_sha')!='df76698220fdea4ae3c4e8977b49e3bc6199c1a6'
            or result.get('status')!='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE'
            or result.get('native_stops') or result.get('submitted_intents')!=1
            or result.get('actual_plugin_steps')!=2574):return False
    rows=[json.loads(s) for s in (folder/'planner.jsonl').read_text().splitlines()]
    return (not any(x['kind'] in ('accept','installed') and x.get('ref_id') for x in rows)
            and sum(x['kind']=='reject' and x.get('error')=='IDENTITY_SEQUENCE_OR_SOURCE_CLOCK_REJECT' for x in rows)==1
            and (folder/'task_errors.txt').read_text()=='STALE_OR_REORDERED_GRIPPER_PACKET\n')
