"""Target MuJoCo time-reference ACKs for the existing resident query consumer.

The owner injects a sealed target contract before a fresh session. This checks
retained generation, native installation/policy and actual tick evidence; it
does not run or qualify target safety, release a quota, or create native proof.
The source200-tick S1-clone contract remains a distinct ABI.
All hashes/trace rows remain untrusted as execution evidence until the cold
owner has admitted the original native chain and independently audited bytes.
"""
from copy import deepcopy
from dataclasses import fields
import base64
import hashlib
import json
import re
from scripts.pi0292.relay_protocol import PROTOCOL,sealed,unseal,exact_keys,uuid_string,validate_identity
from . import mapped_ack as source,target_time_transaction as wire,target_mpc_prediction as mpc
from .authority import PARENT,R0

SCHEMA='DEC1_TARGET_TIME_PREFIX_ACK_V1'
CONDITION_SCHEMA='DEC1_OWNER_PINNED_TARGET_TIME_CONDITION_V1'
SEMANTICS='MAPPED_TARGET_TIME_H40_PREFIX8'
FROZEN_CLOCK='82004c3171cc8cd2a0486f391595ac39424bad96882d12c5936c4eca55fe3b08'
CONDITION_KEYS=source.CONDITION_KEYS|{'checkpoint_manifest_sha256','target_runtime_sha256','native_safety_chain_sha256','task_clock_sha256','native_tick_origin'}
MAPPING_KEYS={'schema','condition_sha256','condition_id','interface_mode','identity','executor_id',
 'response_sha256','raw_h40_sha256','initial_adapter_ack_id','target_runtime_sha256','native_safety_chain_sha256',
 'task_clock_sha256','native_prefix_receipt','references'}
REFERENCE_KEYS={'reference','predecessor_ack_id','prepared','installation_events','policy_bytes_base64',
 'policy_sha256','actual_MPC_observation','tick_before','tick_after','native_step_before','native_step_after',
 'actual_feedback_sha256','pause_before_sha256','pause_after_sha256'}
PREFIX_KEYS={'schema','decision','target_runtime_sha256','native_safety_chain_sha256','task_clock_sha256',
 'raw_h40_sha256','wire_sha256s','native_step_before','native_step_after','evidence_sha256','physics_receipt_sha256'}

def need(ok,why):
 if not ok:raise ValueError('DEC1_TARGET_ACK_'+why)

class FixedTargetTimeAckContract:
 """Explicit cold owner injection, never selected by a request/ACK body flag.

 A valid ACK is not TARGET_TASK_NATIVE_QUALIFIED. In particular the bound
 12D MPC states exclude legs/contact dynamics. The separate original native
 prefix/safety verdict and full physical trace must be retained by the owner
 and independently audited; this validator never substitutes a source clone.
 """
 def __init__(self,condition_bytes,*,expected_sha256,expected_condition_id,expected_owner_release,
              expected_plan_sha256,expected_identity,expected_target_runtime_sha256,expected_native_safety_chain_sha256):
  need(type(condition_bytes) is bytes and hashlib.sha256(condition_bytes).hexdigest()==source._hash(expected_sha256),'EXACT_SEALED_TARGET_CONDITION_BYTES')
  c=unseal(json.loads(condition_bytes));exact_keys(c,CONDITION_KEYS);validate_identity(c['identity']);uuid_string(c['episode_uuid'])
  need(c['schema']==CONDITION_SCHEMA and c['condition_id']==expected_condition_id and c['owner_release']==expected_owner_release
   and isinstance(expected_owner_release,str) and re.fullmatch('[0-9a-f]{40}',expected_owner_release) and c['scientific_plan_sha256']==source._hash(expected_plan_sha256)
   and c['identity']==expected_identity and c['identity']['statistics_sha256']==R0
   and c['policy']=='CORR2048' and c['checkpoint_manifest_sha256']==PARENT and c['noise_bank']=='R0'
   and c['interface_mode'] in ('I_TIME','I_PROGRESS') and c['execution_semantics']==SEMANTICS
   and c['frame']==source.FRAME and c['tcp']==source.TCP,'FIXED_ORIGINAL_CORR_R0_TARGET_CONDITION')
  need(c['target_runtime_sha256']==source._hash(expected_target_runtime_sha256)
   and c['native_safety_chain_sha256']==source._hash(expected_native_safety_chain_sha256)
   and c['task_clock_sha256']==FROZEN_CLOCK and type(c['native_tick_origin']) is int and c['native_tick_origin']==2500,
   'OWNER_PINNED_FROZEN_TARGET_RUNTIME_CHAIN_AND_CLOCK')
  for k in ('condition_id','run_uuid','executor_id'):source._identifier(c[k])
  self._condition=deepcopy(c);self.condition_sha256=expected_sha256

 @property
 def condition(self):return deepcopy(self._condition)

 # These two methods bind only the original raw generation envelope. They do
 # not call source.validate or inherit source's S1 or live-shadow semantics.
 _envelope=source.FixedMappedAckContract._envelope
 bind_action=source.FixedMappedAckContract.bind_action

 def validate(self,packet,*,expected):
  body=unseal(packet);exact_keys(body,('protocol','kind','run_uuid','episode_uuid','sequence','tick','payload'));self._envelope(body,'ack')
  p=body['payload'];exact_keys(p,source.RAW_ACK_KEYS|{'execution_semantics','mapped_execution'})
  need(p['execution_semantics']==SEMANTICS,'TARGET_TIME_ABI_NOT_SOURCE_S1_OR_RAW')
  for k in ('query_uuid','reference_uuid','command_sha256','prefix_sha256'):
   need(p[k]==expected[k],'ORIGINAL_RAW_GENERATION_'+k)
  need(body['sequence']==expected['sequence'] and body['tick']==expected['tick']
   and source._int(p['integrations'])==200 and source._int(p['tick_after'])==body['tick']+200,'EXACT_REAL_PREFIX200_CLOCK')
  source._hash(p['physics_receipt_sha256']);m=p['mapped_execution'];exact_keys(m,MAPPING_KEYS);c=self._condition
  want=dict(schema=SCHEMA,condition_sha256=self.condition_sha256,condition_id=c['condition_id'],interface_mode=c['interface_mode'],
   identity=c['identity'],executor_id=c['executor_id'],response_sha256=expected['response_sha256'],raw_h40_sha256=expected['raw_h40_sha256'],
   initial_adapter_ack_id=expected['initial_adapter_ack_id'],target_runtime_sha256=c['target_runtime_sha256'],
   native_safety_chain_sha256=c['native_safety_chain_sha256'],task_clock_sha256=FROZEN_CLOCK)
  need(all(m.get(k)==v for k,v in want.items()),'EXACT_OWNER_CONDITION_AND_RETAINED_FULL_RAW_RESPONSE')
  refs=m['references'];need(type(refs) is list and len(refs)==8,'ALL_EIGHT_NATIVE_TIME_REFERENCES_REQUIRED')
  prior=expected['initial_adapter_ack_id'];wires=[];native_start=None;commands=set();previous_phase=-1.
  for i,row in enumerate(refs):
   exact_keys(row,REFERENCE_KEYS);need(type(row['reference']) is int and row['reference']==i and row['predecessor_ack_id']==prior,'REFERENCE_PREDECESSOR_CHAIN')
   prepared=row['prepared'];exact_keys(prepared,{f.name for f in fields(wire.PreparedTimeReference)})
   v=wire.PreparedTimeReference(**prepared)
   need(v.episode==c['episode_uuid'] and type(v.sequence) is int and v.sequence==expected['sequence']*8+i
    and v.raw_h40_sha256==expected['raw_h40_sha256'] and v.command_id not in commands and v.command_id!=prior and v.wire_sha256 not in wires,
    'THIS_GENERATION_EIGHT_DISTINCT_INSTALLED_REFERENCES')
   source._identifier(v.command_id);source._hash(v.mapped_h40_sha256)
   phase=source._finite(v.phase);opening=source._finite(v.opening)
   # Shared adapter phase is a path-point index (0..39), not seconds.
   need(0<=phase<=39 and previous_phase<=phase and 0<=opening<=1,'FINITE_MONOTONIC_PHASE_AND_MECHANICAL_GRIPPER')
   previous_phase=phase
   source._finite(v.source_mpc_s)
   need(type(v.wire_json) is str and len(v.wire_json)<=4*1024**2,'BOUNDED_ORIGINAL_NATIVE_WIRE')
   doc=json.loads(v.wire_json,object_pairs_hook=mpc._object)
   metadata=dict(episode=c['episode_uuid'],sequence=v.sequence,source_mpc_s=v.source_mpc_s,ttl_s=.4,operation='install')
   need(json.dumps(doc.get('_pi0294'),sort_keys=True,allow_nan=False)==json.dumps(metadata,sort_keys=True,allow_nan=False),
    'EXACT_NATIVE_WIRE_EPISODE_SEQUENCE_AND_SOURCE_CLOCK')
   need(source._int(row['tick_before'])==expected['tick']+25*i and source._int(row['tick_after'])==expected['tick']+25*(i+1),'ACTUAL_REFERENCE25TICKS')
   before=source._int(row['native_step_before']);after=source._int(row['native_step_after'])
   if i==0:
    native_start=before
    need(before==c['native_tick_origin']+expected['tick'],'ORIGINAL_NATIVE_EPISODE_TICK_ORIGIN')
   need(before==native_start+25*i and after==before+25 and abs(source._finite(v.controller_timestamp)-before*.002)<=1e-8,
    'RAW_NATIVE_WARMUP_OFFSET_AND_SIM_TIME_NOT_REZEROED')
   for name in ('actual_feedback_sha256','pause_before_sha256','pause_after_sha256'):source._hash(row[name])
   wire.require_install_ack(v,row['installation_events'])
   need(type(row['policy_bytes_base64']) is str and len(row['policy_bytes_base64'])<=6*1024**2,'BOUNDED_RETAINED_NATIVE_POLICY_BYTES')
   policy=base64.b64decode(row['policy_bytes_base64'],validate=True)
   prediction=mpc.bind(v,row['installation_events'],policy,expected_policy_sha256=source._hash(row['policy_sha256']),actual_observation=row['actual_MPC_observation'])
   # Deliberately retain the MPC helper's limited claim. Never upgrade it to
   # target/full-body safety or physical quota authorization here.
   need(prediction.native_prefix_safety_qualified is False and prediction.physics_advance_authorized is False,'MPC12_IS_NOT_FULLBODY_SAFETY')
   wires.append(v.wire_sha256);commands.add(v.command_id);prior=v.command_id
  prefix=m['native_prefix_receipt'];exact_keys(prefix,PREFIX_KEYS)
  source._int(prefix['native_step_before']);source._int(prefix['native_step_after'])
  need(prefix==dict(schema='DEC1_FROZEN_TARGET_NATIVE_PREFIX8_VERDICT_V1',decision='PASS',
   target_runtime_sha256=c['target_runtime_sha256'],native_safety_chain_sha256=c['native_safety_chain_sha256'],task_clock_sha256=FROZEN_CLOCK,
   raw_h40_sha256=expected['raw_h40_sha256'],wire_sha256s=wires,native_step_before=native_start,native_step_after=native_start+200,
   evidence_sha256=source._hash(prefix['evidence_sha256']),physics_receipt_sha256=p['physics_receipt_sha256']),
   'SEPARATE_OWNER_RETAINED_ORIGINAL_TARGET_PREFIX_VERDICT_NOT_SOURCE_S1')
  return source.ValidatedMappedAck(packet['sha256'],deepcopy(body),prior)

 def make_packet(self,*,expected,mapping,physics_receipt_sha256):
  """Seal supplied evidence only; no native proof or execution is created."""
  c=self._condition;p={k:expected[k] for k in ('query_uuid','reference_uuid','command_sha256','prefix_sha256')}
  p.update(tick_after=expected['tick']+200,integrations=200,physics_receipt_sha256=source._hash(physics_receipt_sha256),
   execution_semantics=SEMANTICS,mapped_execution=deepcopy(mapping))
  packet=sealed(dict(protocol=PROTOCOL,kind='ack',run_uuid=c['run_uuid'],episode_uuid=c['episode_uuid'],sequence=expected['sequence'],tick=expected['tick'],payload=p))
  self.validate(packet,expected=expected)
  return packet
