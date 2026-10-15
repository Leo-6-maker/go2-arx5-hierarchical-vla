"""Actual-time barrier identities and narrowly proved pre-install recovery."""
import json
from pathlib import Path

FROZEN_FIELDS=('step','time_s','motiontime','leg_history_steps','gait_indices')


def require_pause(before,after,step):
    if (not before['clock_paused'] or not after['clock_paused'] or before['running'] or after['running']
            or before['step']!=step or after['step']!=step
            or abs(after['time_s']-step*.002)>1e-8
            or any(before[k]!=after[k] for k in FROZEN_FIELDS)
            or after['ack_sequence']<=before['ack_sequence']):
        raise ValueError('PHYSICS_FSM_OR_LEG_HISTORY_ADVANCED_WHILE_PAUSED')


def preinstall_clock_failure(folder):
    """Only the already-proved first transaction is replaceable, never quality."""
    folder=Path(folder)
    result=json.loads((folder/'RESULT.json').read_text())
    if (folder.name!='TASK-PILOT-W0-T2' or result.get('source_sha')!='df76698220fdea4ae3c4e8977b49e3bc6199c1a6'
            or result.get('status')!='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE'
            or result.get('native_stops') or result.get('submitted_intents')!=1
            or result.get('actual_plugin_steps')!=2574):return False
    rows=[json.loads(s) for s in (folder/'planner.jsonl').read_text().splitlines()]
    return (not any(x['kind'] in ('accept','installed') and x.get('ref_id') for x in rows)
            and sum(x['kind']=='reject' and x.get('error')=='IDENTITY_SEQUENCE_OR_SOURCE_CLOCK_REJECT' for x in rows)==1
            and (folder/'task_errors.txt').read_text()=='STALE_OR_REORDERED_GRIPPER_PACKET\n')
