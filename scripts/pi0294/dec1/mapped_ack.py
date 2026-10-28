"""Pure DEC1 mapped-ACK binding contract. No IO/native/model/authority producer.

The cold owner pins the sealed condition BEFORE constructing either consumer.
Packets cannot select a validator. Hash consistency is not native evidence.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from scripts.pi0292.relay_protocol import (
    PROTOCOL,digest,sealed,unseal,exact_keys,uuid_string,validate_identity,
)
from scripts.pi0292.eval_contract import FRAME,TCP

SEMANTICS='MAPPED_FULL_H40_PREFIX8'
SCHEMA='DEC1_MAPPED_PREFIX_ACK_V1'
CONDITION_SCHEMA='DEC1_OWNER_PINNED_SOURCE_INTERFACE_CONDITION_V1'
S1_VERSION='DEC1_SINGLE_POSE_PROGRESS_PREFIX8_V1'
RAW_ACK_KEYS={'query_uuid','reference_uuid','command_sha256','prefix_sha256',
    'tick_after','integrations','physics_receipt_sha256'}
CONDITION_KEYS={'schema','condition_id','run_uuid','episode_uuid','executor_id',
    'interface_mode','policy','noise_bank','identity','execution_semantics','frame','tcp',
    'owner_release','scientific_plan_sha256'}
MAPPING_KEYS={'schema','condition_sha256','condition_id','interface_mode','identity','executor_id',
    'response_sha256','raw_h40_sha256','mapped_prefix_sha256','nonce','initial_adapter_ack_id',
    'frame','tcp','s1','shadow_proposal_sha256','shadow_phases','references','maximum_live_shadow_difference'}
REFERENCE_KEYS={'reference','command_id','proposal_sha256','phase','predecessor_ack_id','ack_id',
    'tick_before','tick_after','execution_end_timestamp','target','opening'}


def _hash(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('EXACT_SHA256_REQUIRED')
    return value


def _int(value):
    if type(value) is not int or value<0:raise ValueError('NONNEGATIVE_INTEGER_REQUIRED')
    return value


def _finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('FINITE_SCALAR_REQUIRED')
    return float(value)


def _identifier(value):
    if not isinstance(value,str) or not value:raise ValueError('ACK_ID_REQUIRED')
    return value


def raw_h40_hash(rows):
    if not isinstance(rows,list) or len(rows)!=40:raise ValueError('FULL_H40_REQUIRED')
    flat=[]
    for row in rows:
        if not isinstance(row,list) or len(row)!=10:raise ValueError('FULL_H40_REQUIRED')
        flat.extend(_finite(x) for x in row)
    # Matches frozen source np.asarray(...,dtype=float).tobytes() on native host.
    return hashlib.sha256(struct.pack('=400d',*flat)).hexdigest()


def _mapped_hash(references):
    poses=[];grips=[]
    for row in references:
        target=row['target']
        if not isinstance(target,list) or len(target)!=4:raise ValueError('MAPPED_4X4_REQUIRED')
        for line in target:
            if not isinstance(line,list) or len(line)!=4:raise ValueError('MAPPED_4X4_REQUIRED')
            poses.extend(_finite(x) for x in line)
        grips.append(_finite(row['opening']))
    return hashlib.sha256(struct.pack('=128d',*poses)+struct.pack('=8d',*grips)).hexdigest()


@dataclass(frozen=True)
class ValidatedMappedAck:
    """Successful pure binding validation; never proof of a native execution."""
    packet_sha256: str
    body: dict
    final_adapter_ack_id: str


class FixedMappedAckContract:
    """Explicit cold-factory injection only; no per-request mode switch.

    condition_bytes is the exact previously sealed condition attachment. The
    expected hashes/owner pins come from the owner's admitted manifest, never a
    request/body flag. This module does not create/admit that manifest or GO.
    """
    def __init__(self,condition_bytes,*,expected_sha256,expected_condition_id,
                 expected_owner_release,expected_plan_sha256,expected_identity):
        if type(condition_bytes) is not bytes:raise ValueError('EXACT_SEALED_CONDITION_BYTES_REQUIRED')
        if hashlib.sha256(condition_bytes).hexdigest()!=_hash(expected_sha256):
            raise ValueError('OWNER_SEALED_CONDITION_BYTES_CHANGED')
        condition=unseal(json.loads(condition_bytes));exact_keys(condition,CONDITION_KEYS)
        validate_identity(condition['identity']);uuid_string(condition['episode_uuid'])
        if (condition['schema']!=CONDITION_SCHEMA or condition['condition_id']!=expected_condition_id
                or condition['owner_release']!=expected_owner_release
                or not isinstance(expected_owner_release,str) or re.fullmatch('[0-9a-f]{40}',expected_owner_release) is None
                or condition['scientific_plan_sha256']!=_hash(expected_plan_sha256)
                or condition['identity']!=expected_identity or condition['policy']!='CORR2048'
                or condition['noise_bank']!='R0' or condition['interface_mode'] not in ('I_TIME','I_PROGRESS')
                or condition['execution_semantics']!=SEMANTICS or condition['frame']!=FRAME or condition['tcp']!=TCP):
            raise ValueError('FIXED_DEC1_CORR_R0_CONDITION_AND_OWNER_REQUIRED')
        for key in ('condition_id','run_uuid','executor_id'):_identifier(condition[key])
        self._condition=deepcopy(condition);self.condition_sha256=expected_sha256

    @property
    def condition(self):return deepcopy(self._condition)

    def _envelope(self,body,kind):
        c=self._condition
        if (body['protocol']!=PROTOCOL or body['kind']!=kind or body['run_uuid']!=c['run_uuid']
                or body['episode_uuid']!=c['episode_uuid'] or _int(body['sequence'])>=80
                or _int(body['tick'])!=body['sequence']*200):
            raise ValueError('FIXED_CONDITION_PACKET_IDENTITY_OR_CLOCK_CHANGED')

    def bind_action(self,action_packet,command_packet,*,initial_adapter_ack_id):
        """Call only AFTER original action/accept_prefix gates; no changed gate."""
        action=unseal(action_packet);command=unseal(command_packet)
        exact_keys(action,('protocol','kind','run_uuid','episode_uuid','sequence','tick','payload'))
        exact_keys(command,('protocol','kind','run_uuid','episode_uuid','sequence','tick','query_uuid','reference_uuid',
            'observation_sha256','anchor_sha256','prefix_eef10','prefix_sha256','advance_integrations','frame','tcp','expires'))
        self._envelope(action,'action');self._envelope(command,'install_prefix8')
        p=action['payload']
        exact_keys(p,('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity','trajectory_eef10','offsets_ns'))
        if p['identity']!=self._condition['identity']:raise ValueError('MODEL_IDENTITY_CHANGED')
        for key in ('sequence','tick','run_uuid','episode_uuid'):
            if action[key]!=command[key]:raise ValueError('RAW_ACTION_COMMAND_IDENTITY_CHANGED')
        for key in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256'):
            if p[key]!=command[key]:raise ValueError('RAW_ACTION_COMMAND_BINDING_CHANGED')
        uuid_string(p['query_uuid']);uuid_string(p['reference_uuid'])
        if (command['frame']!=FRAME or command['tcp']!=TCP or type(command['advance_integrations']) is not int
                or command['advance_integrations']!=200
                or p['offsets_ns']!=[i*50000000 for i in range(40)]
                or any(type(x) is not int for x in p['offsets_ns'])
                or command['prefix_eef10']!=p['trajectory_eef10'][:8]
                or command['prefix_sha256']!=digest(command['prefix_eef10'])):
            raise ValueError('ORIGINAL_RAW_H40_PREFIX_OR_CLOCK_CHANGED')
        if command['sequence']==0:
            if initial_adapter_ack_id is not None:raise ValueError('INITIAL_ADAPTER_ACK_MUST_BE_NONE')
        else:_identifier(initial_adapter_ack_id)
        return dict(query_uuid=p['query_uuid'],reference_uuid=p['reference_uuid'],
            command_sha256=_hash(command_packet['sha256']),prefix_sha256=_hash(command['prefix_sha256']),
            response_sha256=_hash(action_packet['sha256']),raw_h40_sha256=raw_h40_hash(p['trajectory_eef10']),
            sequence=command['sequence'],tick=command['tick'],initial_adapter_ack_id=initial_adapter_ack_id)

    def validate(self,packet,*,expected):
        """Verify complete packet against retained ORIGINAL generation bindings."""
        body=unseal(packet);exact_keys(body,('protocol','kind','run_uuid','episode_uuid','sequence','tick','payload'))
        self._envelope(body,'ack');p=body['payload']
        exact_keys(p,RAW_ACK_KEYS|{'execution_semantics','mapped_execution'})
        if p['execution_semantics']!=SEMANTICS:raise ValueError('MAPPED_EXECUTION_SEMANTICS_REQUIRED')
        for key in ('query_uuid','reference_uuid','command_sha256','prefix_sha256'):
            if p[key]!=expected[key]:raise ValueError('ORIGINAL_RAW_COMMAND_BINDING_CHANGED')
        if (body['sequence']!=expected['sequence'] or body['tick']!=expected['tick']
                or _int(p['integrations'])!=200 or _int(p['tick_after'])!=body['tick']+200):
            raise ValueError('EXACT_NATIVE_PREFIX200_CLOCK_REQUIRED')
        _hash(p['command_sha256']);_hash(p['prefix_sha256']);_hash(p['physics_receipt_sha256'])
        m=p['mapped_execution'];exact_keys(m,MAPPING_KEYS);c=self._condition
        if (m['schema']!=SCHEMA or m['condition_sha256']!=self.condition_sha256
                or m['condition_id']!=c['condition_id'] or m['interface_mode']!=c['interface_mode']
                or m['identity']!=c['identity'] or m['executor_id']!=c['executor_id']
                or m['frame']!=FRAME or m['tcp']!=TCP
                or m['response_sha256']!=expected['response_sha256']
                or m['raw_h40_sha256']!=expected['raw_h40_sha256']
                or m['initial_adapter_ack_id']!=expected['initial_adapter_ack_id']):
            raise ValueError('FIXED_CONDITION_OR_ORIGINAL_FULL_RESPONSE_CHANGED')
        for key in ('response_sha256','raw_h40_sha256','mapped_prefix_sha256'):_hash(m[key])
        uuid_string(m['nonce'])
        s1=m['s1'];exact_keys(s1,('version','decision','nonce','state_sha256','action_sha256','shadow_integrations'))
        if (s1['version']!=S1_VERSION or s1['decision']!='PASS' or s1['nonce']!=m['nonce']
                or s1['action_sha256']!=m['mapped_prefix_sha256'] or _int(s1['shadow_integrations'])!=200):
            raise ValueError('ORIGINAL_ONE_CLONE_200_SHADOW_PASS_REQUIRED')
        _hash(s1['state_sha256']);_hash(s1['action_sha256'])
        maximum=_finite(m['maximum_live_shadow_difference'])
        if not 0<=maximum<=1e-10:raise ValueError('ORIGINAL_S1_LIVE_PARITY_REQUIRED')
        refs=m['references'];shadow_digests=m['shadow_proposal_sha256'];shadow_phases=m['shadow_phases']
        if any(not isinstance(v,list) or len(v)!=8 for v in (refs,shadow_digests,shadow_phases)):
            raise ValueError('EIGHT_REAL_REFERENCE_ACKS_REQUIRED')
        prior=m['initial_adapter_ack_id'];commands=set()
        for i,row in enumerate(refs):
            exact_keys(row,REFERENCE_KEYS);_hash(row['proposal_sha256']);_hash(shadow_digests[i])
            command=_identifier(row['command_id'])
            if (type(row['reference']) is not int or row['reference']!=i or command in commands or command==prior
                    or row['ack_id']!=command or row['predecessor_ack_id']!=prior
                    or row['proposal_sha256']!=shadow_digests[i]
                    or _finite(row['phase'])!=_finite(shadow_phases[i])
                    or _int(row['tick_before'])!=body['tick']+25*i
                    or _int(row['tick_after'])!=body['tick']+25*(i+1)
                    or abs(_finite(row['execution_end_timestamp'])-row['tick_after']*.002)>1e-8):
                raise ValueError('ACTUAL_REFERENCE_PHASE_ACK_CLOCK_CHAIN_CHANGED')
            commands.add(command);prior=command
        if _mapped_hash(refs)!=m['mapped_prefix_sha256']:
            raise ValueError('ACTUAL_MAPPED_EIGHT_ACTION_HASH_CHANGED')
        return ValidatedMappedAck(packet['sha256'],deepcopy(body),prior)

    def make_packet(self,*,expected,mapping,physics_receipt_sha256):
        """Seal supplied evidence; NOT a native receipt producer or IO writer."""
        c=self._condition
        p={key:expected[key] for key in ('query_uuid','reference_uuid','command_sha256','prefix_sha256')}
        p.update(tick_after=expected['tick']+200,integrations=200,physics_receipt_sha256=_hash(physics_receipt_sha256),
            execution_semantics=SEMANTICS,mapped_execution=deepcopy(mapping))
        packet=sealed(dict(protocol=PROTOCOL,kind='ack',run_uuid=c['run_uuid'],episode_uuid=c['episode_uuid'],
            sequence=expected['sequence'],tick=expected['tick'],payload=p))
        self.validate(packet,expected=expected)
        return packet
