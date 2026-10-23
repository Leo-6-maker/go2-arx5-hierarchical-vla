"""Bind the existing WBC MPC prediction to its installed target and observation.

Consumes the deployed MpcFlattenedController message, never synthesizes a
controller/rollout or interprets MPC states as measured full-body dynamics.
This is an identity/coverage check BEFORE the separate native safety decision;
it grants no physical quota and is not a target prefix-safety certificate.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from .target_time_transaction import PreparedTimeReference,require_install_ack,native_source_age

# Independently read installed interface identities, not private asset copies.
POLICY_IDL_SHA256='636597f8e5132ff83f8d4c5848a61cfbfb0c9b4b737e320f8520991d05683ef0'
TARGET_IDL_SHA256='25466eb6aaa378652c1f0dbe11c4d34477debe06642cbb822e0aeaa144853fae'
STATE_IDL_SHA256='bd5dc349d705491a86e78cb525e1efb30afbf63644ccdb922ce8cfe6f206aebc'
INPUT_IDL_SHA256='109e5528c62920c2671c7ffed6094f84e95ae69206eb4195ab7eb5365a20c891'
OBSERVATION_IDL_SHA256='c891304762035e358ac6d0f5d768154420e120773c0adbf8e7e2be266d37ab1d'
FIELDS={'controller_type','init_observation','plan_target_trajectories','state_trajectory',
        'input_trajectory','time_trajectory','post_event_indices','mode_schedule','data','performance_indices'}

def need(ok,reason):
    if not ok:raise PermissionError('DEC1_TARGET_MPC_'+reason)

def _object(pairs):
    result={}
    for k,v in pairs:
        need(k not in result,'NO_DUPLICATE_JSON_FIELDS');result[k]=v
    return result

def _finite(value):
    if isinstance(value,dict):return all(_finite(v) for v in value.values())
    if isinstance(value,list):return all(_finite(v) for v in value)
    if type(value) in (int,float):return math.isfinite(value)
    return type(value) is bool or value is None or isinstance(value,str)

def _rows(values,count,width):
    need(isinstance(values,list) and len(values)==count,'ALIGNED_NATIVE_TRAJECTORIES')
    need(all(type(r) is dict and set(r)=={'value'} and isinstance(r['value'],list)
             and len(r['value'])==width and all(type(v) in (int,float) and math.isfinite(v) for v in r['value'])
             for r in values),'EXACT_NATIVE_VECTOR_DIMENSIONS')
    rows=np.asarray([r['value'] for r in values],dtype=float)
    need(np.all(np.abs(rows)<=np.finfo(np.float32).max)
         and np.array_equal(rows,rows.astype(np.float32).astype(float)),
         'ORIGINAL_FLOAT32_STATE_INPUT_TRANSPORT_VALUES')
    return rows

@dataclass(frozen=True)
class BoundMPCPrediction:
    policy_sha256: str
    wire_sha256: str
    command_id: str
    observation_time_s: float
    time_trajectory: tuple
    state_trajectory: tuple
    input_trajectory: tuple
    native_prefix_safety_qualified: bool=False
    physics_advance_authorized: bool=False


def bind(prepared,events,policy_bytes,*,expected_policy_sha256,actual_observation):
    """Use exact owner-retained ROS trace bytes and an INDEPENDENT observation.

    Input is the original continuous task_floating model:12 state coordinates
    (base xyz/ZYX,arm6),12 optimized velocity inputs. These exclude leg/contact
    dynamics. Native safety/STOP, current admission, and physical feedback still
    must be checked by the original executor. No timing threshold is widened.
    """
    need(isinstance(prepared,PreparedTimeReference),'PREPARED_TARGET_REQUIRED')
    rows=list(events);require_install_ack(prepared,rows)
    need(type(policy_bytes) is bytes and len(policy_bytes)<=4*1024**2
         and hashlib.sha256(policy_bytes).hexdigest()==expected_policy_sha256,'OWNER_RETAINED_POLICY_BYTES')
    p=json.loads(policy_bytes,object_pairs_hook=_object,parse_constant=lambda _:need(False,'FINITE_JSON_REQUIRED'))
    need(type(p) is dict and set(p)==FIELDS and _finite(p),'EXACT_DEPLOYED_POLICY_FIELDS_AND_FINITE_DATA')
    need(type(p['controller_type']) is int and p['controller_type'] in (1,2),'ORIGINAL_KNOWN_CONTROLLER_TYPE')
    actual=actual_observation
    need(type(actual) is dict and set(actual)=={'time','state','input','mode'} and _finite(actual),
         'INDEPENDENT_ACTUAL_NATIVE_OBSERVATION')
    _rows([actual['state']],1,12);_rows([actual['input']],1,12)
    now=actual['time']
    need(type(now) in (int,float) and type(actual['mode']) is int and actual['mode']==0
         and 0<=now-prepared.source_mpc_s<=.025
         and 0<=native_source_age(prepared.controller_timestamp,now)<=.025,'ORIGINAL_NATIVE_OBSERVATION_CLOCK_AND_MODE')
    need(json.dumps(p['init_observation'],sort_keys=True,allow_nan=False)==json.dumps(actual,sort_keys=True,allow_nan=False),
         'PLAN_INITIAL_OBSERVATION_MUST_MATCH_INDEPENDENT_ACTUAL_STATE')
    target=p['plan_target_trajectories']
    need(type(target) is dict and set(target)=={'time_trajectory','state_trajectory','input_trajectory'},
         'ORIGINAL_NATIVE_TARGET_FIELDS')
    # The original solver echoes the target used in its cost function. Match it
    # against actual installed rows, not the most recently published intention.
    target_times=target['time_trajectory']
    need(isinstance(target_times,list) and 41<=len(target_times)<=4096
         and all(type(t) in (int,float) and math.isfinite(t) for t in target_times),'NATIVE_TARGET_TIMES')
    target_poses=_rows(target['state_trajectory'],len(target_times),7)
    target_inputs=_rows(target['input_trajectory'],len(target_times),12)
    def same_install(row):
        if row.get('kind')!='installed' or row.get('ref_id')!=prepared.wire_sha256:return False
        return (target_times==row['times'] and np.array_equal(target_poses,np.asarray(row['poses']))
                and np.array_equal(target_inputs,np.asarray(row['inputs'])))
    need(any(same_install(row) for row in rows),'SOLVER_TARGET_MUST_MATCH_THIS_ACTUAL_INSTALL_ACK')
    times=p['time_trajectory'];need(isinstance(times,list) and 2<=len(times)<=4096
        and all(type(t) in (int,float) and math.isfinite(t) for t in times),'BOUNDED_NATIVE_POLICY_TIMES')
    t=np.asarray(times);need(np.all(np.diff(t)>0) and t[0]<=now
        and t[-1]>=prepared.controller_timestamp+.4,'COMPLETE_ORIGINAL_PREFIX_0P4S_PREDICTION_COVERAGE')
    states=_rows(p['state_trajectory'],len(t),12);inputs=_rows(p['input_trajectory'],len(t),12)
    need(p['post_event_indices']==[] and p['mode_schedule']=={'event_times':[],'mode_sequence':[0]}
         and all(type(m) is int for m in p['mode_schedule']['mode_sequence']),
         'ORIGINAL_CONTINUOUS_FLOATING_ARM_MODEL_ONLY')
    return BoundMPCPrediction(expected_policy_sha256,prepared.wire_sha256,prepared.command_id,float(now),
        tuple(map(float,t)),tuple(map(tuple,states.tolist())),tuple(map(tuple,inputs.tolist())))
