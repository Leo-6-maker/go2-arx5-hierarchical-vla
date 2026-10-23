"""DEC1 glue for the existing MuJoCo receiver, not a new controller or runner.

Full raw/mapped gates and an explicitly supplied TARGET native prefix validator
precede wire preparation. Installation evidence never proves physics/task/stop.
No files, transport, retry, model or simulation operations are provided here.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from scripts.pi0292.eval_contract import gate_action
from scripts.pi0294.base_intent import eef_document
from scripts.pi0294.task_gripper_contract import require_working_target
from go2arx5_vla.adapters.execution_progress import ReferenceProposal
from go2arx5_vla.trajectory.so3 import quat_to_matrix,quat_slerp,rotation_matrix_to_rot6d_rows

def need(ok,why):
    if not ok:raise PermissionError('DEC1_TARGET_TIME_'+why)

def raw_hash(rows):return hashlib.sha256(np.asarray(rows,dtype='<f8').tobytes()).hexdigest()

@dataclass(frozen=True)
class PreparedTimeReference:
    episode: str
    sequence: int
    command_id: str
    phase: float
    source_mpc_s: float
    controller_timestamp: float
    wire_json: str
    wire_sha256: str
    opening: float
    raw_h40_sha256: str
    mapped_h40_sha256: str

@dataclass(frozen=True)
class PausedReferenceCandidate(PreparedTimeReference):
    """Numerically gated candidate solely for planning while physics is paused."""
    pause_anchor_sha256: str
    native_prefix_validation_pending: bool=True
    physics_advance_authorized: bool=False


def prepare(raw_h40,proposal,**kwargs):
    """Original path: caller native-prefix validation precedes serialization."""
    need('_pause_anchor' not in kwargs,'NO_CALLER_PLANNING_TOKEN_IN_PREVALIDATED_PATH')
    return _prepare(raw_h40,proposal,**kwargs)


def prepare_for_paused_prediction(raw_h40,proposal,*,pause_before,pause_after,
                                  actual_step,require_frozen_pause,**kwargs):
    """Plan under the ORIGINAL paused lifecycle before checking the prediction.

    The owner supplies the actual frozen task_clock82004c31 require_pause and
    fresh native read_state step callback, not management HEAD substitutions.
    Raw/mapped complete gates still precede reference publication. The native
    solver needs a candidate target to produce its corresponding prediction;
    only this paused planning phase may precede target prefix validation. The
    original native guard must inspect the bound prediction before ANY quota
    release. This candidate never certifies safety, admission, or execution.
    """
    need('_pause_anchor' not in kwargs and 'validate_target_native_prefix' not in kwargs,
         'PLANNING_CANDIDATE_CANNOT_CLAIM_PREVALIDATED_PREFIX')
    need(callable(actual_step) and callable(require_frozen_pause),'ORIGINAL_PAUSE_AND_FRESH_NATIVE_STEP_CALLBACKS')
    step=actual_step()
    need(type(step) is int and step>=0,'FRESH_NATIVE_INTEGER_STEP')
    need(require_frozen_pause(pause_before,pause_after,step) is None,'ORIGINAL_FROZEN_PAUSE_VALIDATOR_REQUIRED')
    need(pause_after['time_s']==proposal.controller_timestamp,'PROPOSAL_MUST_MATCH_ACTUAL_PAUSED_STATE_CLOCK')
    anchor=hashlib.sha256(json.dumps(dict(before=pause_before,after=pause_after,actual_step=step),
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    candidate=_prepare(raw_h40,proposal,validate_target_native_prefix=None,_pause_anchor=anchor,**kwargs)
    need(actual_step()==step,'NO_PHYSICS_ADVANCE_DURING_PLANNING_CANDIDATE_PREPARATION')
    return candidate


def _prepare(raw_h40,proposal,*,expected_raw_sha256,raw_actual_state10,mapped_actual_state10,
            episode,sequence,source_mpc_s,expected_world_frame,expected_tcp,
            validate_target_native_prefix,record_gate,_pause_anchor=None):
    """Use an owner-retained raw identity and the shared V2 mapper's proposal.

    Distinct source/target frames require explicitly frame-qualified actual
    state rows from the owner. This function never estimates a frame transform.
    Native prefix validation raises or returns None, like original validators.
    An absent validator, bool or a source-only shadow receipt is not admission.
    Caller still owns current native authority, durable once intent, gripper
    write, publish, actual ACK, physical quota and stop through the old worker.
    """
    need(isinstance(proposal,ReferenceProposal),'SHARED_PROGRESS_PROPOSAL_REQUIRED')
    need(isinstance(episode,str) and episode and type(sequence) is int and sequence>=0,
         'OWNER_EPISODE_AND_RECEIVER_SEQUENCE')
    need(callable(record_gate),'EXPLICIT_GATE_RECORDER')
    need(callable(validate_target_native_prefix) if _pause_anchor is None else validate_target_native_prefix is None,
         'EXPLICIT_TARGET_NATIVE_PREFIX_OR_PAUSED_PLANNING_ONLY')
    now=proposal.controller_timestamp
    need(type(source_mpc_s) in (int,float) and math.isfinite(source_mpc_s)
         and np.isfinite(now) and 0<=now-source_mpc_s<=.025,'ORIGINAL_SOURCE_CLOCK_MAX_25MS')
    raw=np.asarray(raw_h40,dtype=float).copy()
    need(raw.shape==(40,10) and raw_hash(raw)==expected_raw_sha256,'OWNER_RETAINED_FULL_RAW_H40_IDENTITY')
    full=proposal.mapped_full
    need(full.source_frame==expected_world_frame and full.tcp_semantic_id==expected_tcp,
         'EXPLICIT_TARGET_WORLD_FRAME_AND_TCP')
    mapped=np.column_stack((full.eef_position,
        rotation_matrix_to_rot6d_rows(quat_to_matrix(full.eef_orientation)),full.gripper))
    need(mapped.shape==(40,10) and full.point_count==40
         and np.allclose(full.point_timestamps,now+np.arange(40)*.05,rtol=0,atol=1e-10),
         'ORIGINAL_COMPLETE_MAPPED_H40_20HZ')
    for kind,rows,state in (('RAW_H40',raw,raw_actual_state10),('MAPPED_H40',mapped,mapped_actual_state10)):
        verdict=gate_action(rows,state,now,proposal.command_id+':'+kind)
        record_gate(dict(kind=kind,command_id=proposal.command_id,result=verdict))
        need(isinstance(verdict,dict) and verdict.get('decision') in ('ACCEPT','WARN'),
             'ORIGINAL_FULL_H40_REJECT_'+kind)
    require_working_target(mapped[:,9])  # Original target V2 mechanical fractions, never clip/rescale.
    opening=float(proposal.gripper.normalized_command)
    need(not proposal.gripper.stale and opening==float(mapped[0,9]),'ACTUAL_FIRST_MAPPED_GRIPPER_EVENT')
    if _pause_anchor is None:
        need(validate_target_native_prefix(proposal) is None,'TARGET_NATIVE_PREFIX_MUST_RAISE_OR_RETURN_NONE')
    document=eef_document(mapped)
    document['_pi0294']=dict(episode=episode,sequence=sequence,source_mpc_s=source_mpc_s,ttl_s=.4,operation='install')
    wire=json.dumps(document,separators=(',',':'),allow_nan=False)
    kind=PreparedTimeReference if _pause_anchor is None else PausedReferenceCandidate
    extra={} if _pause_anchor is None else dict(pause_anchor_sha256=_pause_anchor)
    return kind(episode,sequence,proposal.command_id,float(proposal.phase),float(source_mpc_s),
        float(now),wire,hashlib.sha256(wire.encode()).hexdigest(),opening,expected_raw_sha256,raw_hash(mapped),**extra)

def require_install_ack(prepared,events):
    """Match original receiver accept+installed events for this exact wire.

    Rows come from the original native trace. This pure check neither generates
    native receipts nor authorizes advancing a quota. CPU fixtures are not real
    target execution. Missing/rejected/incorrect ACK must not release physics.
    """
    need(isinstance(prepared,PreparedTimeReference)
         and hashlib.sha256(prepared.wire_json.encode()).hexdigest()==prepared.wire_sha256,'UNCHANGED_PREPARED_WIRE')
    rows=list(events);ref=prepared.wire_sha256;source=prepared.source_mpc_s
    need(not any(r.get('kind')=='reject' and r.get('wire_sha256')==ref for r in rows),'NATIVE_REFERENCE_REJECTED')
    accepted=[r for r in rows if r.get('kind')=='accept' and r.get('ref_id')==ref]
    installed=[r for r in rows if r.get('kind')=='installed' and r.get('ref_id')==ref]
    need(accepted and installed,'ACTUAL_ACCEPT_AND_INSTALL_ACK_REQUIRED')
    for a in accepted:
        receiver=a.get('receiver_mpc_s')
        need(a.get('raw')==prepared.wire_json and a.get('source_mpc_s')==source
             and a.get('deadline')==source+.4 and type(receiver) in (int,float)
             and math.isfinite(receiver) and 0<=receiver-source<=.025,'EXACT_ORIGINAL_ACCEPT_CLOCK_WIRE_AND_TTL')
    for row in installed:
        now=row.get('observation_time_s');times=np.asarray(row.get('times',[]),dtype=float)
        poses=np.asarray(row.get('poses',[]),dtype=float);inputs=np.asarray(row.get('inputs',[]),dtype=float)
        need(row.get('bound_start')==source and type(now) in (int,float) and math.isfinite(now)
             and 0<=now-source<=.025 and times.ndim==1 and len(times)>=41
             and np.isfinite(times).all() and np.all(np.diff(times)>0)
             and abs(times[0]-now)<=1e-10 and abs(times[-1]-(now+2.))<=1e-10
             and poses.shape==(len(times),7) and inputs.shape==(len(times),12)
             and np.isfinite(poses).all() and np.isfinite(inputs).all() and not np.any(inputs),
             'ACTUAL_EEF_ONLY_NATIVE_TARGET_WINDOW_NO_BASE_OR_INPUT_SUBSTITUTION')
        points=json.loads(prepared.wire_json)['ee']['waypoints']
        # Same original native uniform grid plus absolute in-window path knots.
        # Checking only endpoints/count would admit a substituted time grid.
        native_times=[now+min(2.,i*.05) for i in range(41)]+[now+2.]
        native_times.extend(source+p['time'] for p in points if now<=source+p['time']<=now+2.)
        expected_times=np.array(sorted(set(native_times)))
        need(times.shape==expected_times.shape and np.allclose(times,expected_times,rtol=0,atol=1e-10),
             'ORIGINAL_NATIVE_WINDOW_GRID_AND_PATH_KNOTS')
        knots=np.array([p['time'] for p in points]);position=np.array([p['position'] for p in points])
        orientation=np.array([p['quaternion_xyzw'] for p in points]);relative=np.clip(times-source,0.,2.)
        expected_position=np.column_stack([np.interp(relative,knots,position[:,axis]) for axis in range(3)])
        left=np.minimum(np.searchsorted(knots,relative,side='right')-1,len(knots)-2)
        fraction=(relative-knots[left])/(knots[left+1]-knots[left])
        expected_orientation=quat_slerp(orientation[left],orientation[left+1],fraction)
        # Deployed MpcState.value is float32[] (IDL bd5dc349...), while
        # time_trajectory is float64[]. Reproduce that transport conversion;
        # this changes neither the raw reference nor any safety threshold.
        native_position=expected_position.astype(np.float32).astype(float)
        native_orientation=expected_orientation.astype(np.float32).astype(float)
        need(np.array_equal(poses[:,:3],native_position)
             and np.allclose(quat_to_matrix(poses[:,3:]),quat_to_matrix(native_orientation),rtol=0,atol=1e-10),
             'ACTUAL_INSTALLED_POSES_MUST_MATCH_ORIGINAL_FLOAT32_WORLD_REFERENCE')
    return dict(status='BOUND_ORIGINAL_MUJOCO_RECEIVER_INSTALL_ACK',wire_sha256=ref,
        command_id=prepared.command_id,source_mpc_s=source,accept_rows=len(accepted),install_rows=len(installed),
        physics_advance_authorized=False,target_task_native_qualified=False,stop_qualified=False)
