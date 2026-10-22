"""Cold owner callback draft; original episode remains sole live step owner.

This writes/reads the original physics receipt path and a DISTINCT mapped ACK.
It does not admit an owner, obtain authority, run physics or certify native GO.
"""
from copy import deepcopy
import os
import numpy as np
from scripts.pi0292.durable import read,write
from scripts.pi0294.run_admission import sha
from scripts.pi0294.dec1.source_episode_wire import NativeMappedPrefixAck
from scripts.pi0294.dec1.mapped_ack import FixedMappedAckContract,SCHEMA,ValidatedMappedAck


def make_mapped_relay_acknowledger(relay,contract):
    """Inject only in a new cold DEC1 factory after its existing owner admission.

    No flag in a wire request can select this function. The freshly created relay
    must start at0; no mid-episode reattachment or receipt reconstruction/retry.
    The existing owner must qualify this IO callback before native invocation.
    """
    if not isinstance(contract,FixedMappedAckContract):raise TypeError('FIXED_OWNER_CONTRACT_REQUIRED')
    c=contract.condition
    if (relay.acknowledged!=0 or relay.pending is not None
            or relay.request['run_uuid']!=c['run_uuid'] or relay.request['episode_uuid']!=c['episode_uuid']
            or relay.request['identity']!=c['identity']):
        raise ValueError('FRESH_OWNER_ADMITTED_RELAY_AND_FIXED_CONDITION_REQUIRED')
    progress=dict(sequence=0,last_adapter_ack_id=None,entered=False)

    def acknowledge(engine,rows,proof):
        # A lost callback response is non-retryable, even if files were persisted.
        if progress['entered']:raise ValueError('MAPPED_ACK_OUTCOME_UNKNOWN_NO_RETRY')
        progress['entered']=True
        pending=relay.pending
        if pending is None:raise ValueError('ORIGINAL_PENDING_RAW_COMMAND_REQUIRED')
        command=pending['command'];sequence=command['sequence']
        if (sequence!=progress['sequence'] or relay.acknowledged!=sequence
                or proof['episode_id']!=c['episode_uuid'] or proof['executor_id']!=c['executor_id']
                or proof['raw_chunk_id']!=command['reference_uuid']
                or type(proof['actual_ticks']) is not int or proof['actual_ticks']!=200
                or type(engine.steps) is not int or engine.steps!=command['tick']+200
                or type(proof['tick_before']) is not int or type(proof['tick_after']) is not int
                or proof['tick_before']!=command['tick'] or proof['tick_after']!=engine.steps
                or proof['initial_adapter_ack_id']!=progress['last_adapter_ack_id']
                or proof['ABI']!='SINGLE_4x4_POSE_PLUS_GRIPPER_NOT_PREVIEW9' or engine.safety):
            raise ValueError('EXACT_ACTUAL_MAPPED_PREFIX200_AND_ORIGINAL_RAW_COMMAND_REQUIRED')
        relay.guard_prefix()  # Original TTL/deadline, no added sleep/watchdog change.
        expected=contract.bind_action(pending['action_packet'],pending['command_packet'],
            initial_adapter_ack_id=progress['last_adapter_ack_id'])
        if (expected['command_sha256']!=pending['command_sha256']
                or proof['raw_h40_sha256']!=expected['raw_h40_sha256']):
            raise ValueError('ORIGINAL_PREBIND_FULL_RESPONSE_AND_H40_CHANGED')
        actual=np.asarray(rows)
        columns=1+engine.model.nq+engine.model.nv+engine.model.nu+6+1
        if actual.shape!=(200,columns) or not np.isfinite(actual).all() or np.any(actual[:,-1]!=0):
            raise ValueError('ORIGINAL_ACTUAL_200ROW_PHYSICS_AND_SAFE_FLAGS_REQUIRED')
        if np.max(np.abs(actual[:,0]-np.arange(command['tick']+1,engine.steps+1)*.002))>1e-8:
            raise ValueError('ORIGINAL_ACTUAL_PHYSICS_TICK_CLOCK_CHANGED')
        mapping=dict(schema=SCHEMA,condition_sha256=contract.condition_sha256,
            condition_id=c['condition_id'],interface_mode=c['interface_mode'],identity=c['identity'],
            executor_id=c['executor_id'],response_sha256=expected['response_sha256'],
            raw_h40_sha256=expected['raw_h40_sha256'],mapped_prefix_sha256=proof['action_sha256'],
            nonce=proof['nonce'],initial_adapter_ack_id=proof['initial_adapter_ack_id'],frame=c['frame'],tcp=c['tcp'],
            s1=deepcopy(proof['s1']),shadow_proposal_sha256=deepcopy(proof['proposal_digests']),
            shadow_phases=deepcopy(proof['phases']),references=deepcopy(proof['actual_reference_acks']),
            maximum_live_shadow_difference=proof['maximum_live_shadow_difference'])
        physics=relay.directory/f'physics-{sequence:03d}.npz'
        ack_path=relay.directory/'acks'/f'{sequence:03d}.json'
        if physics.exists() or ack_path.exists():raise FileExistsError('PRESERVE_FIRST_MAPPED_ACK_EFFECTS_NO_REPLAY')
        np.savez_compressed(physics,rows=actual,tick_before=command['tick'],tick_after=engine.steps)
        with physics.open('rb') as handle:os.fsync(handle.fileno())
        physics_sha=sha(physics)
        packet=contract.make_packet(expected=expected,mapping=mapping,physics_receipt_sha256=physics_sha)
        write(ack_path,packet)
        readback=read(ack_path)
        checked=contract.validate(readback,expected=expected)
        if (not isinstance(checked,ValidatedMappedAck) or readback!=packet or sha(physics)!=physics_sha):
            raise ValueError('ACTUAL_COMPLETE_MAPPED_ACK_AND_PHYSICS_READBACK_REQUIRED')
        # Preserve the FULL packet; next RelayTeacher.plan compares it verbatim
        # to query.previous_ack, and both remote consumers validate these bytes.
        relay.acknowledged+=1;relay.pending=None
        relay.state.update(status='MAPPED_PREFIX_ACKNOWLEDGED_NEXT_OBSERVATION_ALLOWED',tick=engine.steps,
            acknowledged_commands=relay.acknowledged,mapped_ack_condition_sha256=contract.condition_sha256)
        write(relay.directory/'STATE.json',relay.state)
        progress.update(sequence=sequence+1,last_adapter_ack_id=checked.final_adapter_ack_id,entered=False)
        return NativeMappedPrefixAck(c['episode_uuid'],c['executor_id'],proof['nonce'],
            proof['raw_h40_sha256'],proof['action_sha256'],proof['tick_before'],proof['tick_after'],
            ack_path.relative_to(relay.directory).as_posix(),physics_sha,readback['sha256'])
    return acknowledge
