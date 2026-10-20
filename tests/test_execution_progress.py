"""Task-blind synthetic CPU contracts, not a native S1/WBC qualification."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from go2arx5_vla.adapters.groot_to_wbc import ReferenceAdapterConfig
from go2arx5_vla.adapters.groot_to_wbc_v2 import GR00TToWBCAdapterV2
from go2arx5_vla.adapters.execution_progress import (
    ActualFeedback,ExecutionProgressAdapter,ProgressConfig,_pose,canonical_chunk_from_row_eef10,
)
from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix,rotation_matrix_to_rot6d_rows
from go2arx5_vla.safety.trajectory_gate import GateDecision,TrajectorySafetyGateV1


def chunk(*,identifier='chunk',start=0.,positions=None,quaternions=None,gripper=None,frame='world',provenance=None):
    p=np.c_[np.arange(40)*.002,np.zeros((40,2))] if positions is None else positions
    q=np.tile([0.,0.,0.,1.],(40,1)) if quaternions is None else quaternions
    return TrajectoryChunkV1(identifier,'CPU_FIXTURE_NOT_MODEL',start,start+np.arange(40)*.05,p,q,
        frame,'tcp',start,start+2.,np.zeros(40) if gripper is None else gripper,provenance=provenance or {})


def feedback(t,pose_chunk,phase,*,ack=None,grip=None):
    p,q=_pose(pose_chunk,phase)
    return ActualFeedback(t,p,q,float(pose_chunk.gripper[int(phase)]) if grip is None else grip,'world','tcp',ack)


def adapter(mode='I_PROGRESS',*,reference=None,config=None,validators=None):
    calls=[]
    def raw(c):calls.append(('raw',c.point_count))
    def full(p):calls.append(('full',p.mapped_full.point_count))
    def prefix(p):calls.append(('prefix',len(p.prefix)))
    obj=ExecutionProgressAdapter(reference or ReferenceAdapterConfig('world','world','tcp',stale_timeout_s=3.),
        config or ProgressConfig(),mode=mode,episode_id='CPU_EPISODE',
        validate_raw=(validators or (raw,full,prefix))[0],
        validate_mapped_full=(validators or (raw,full,prefix))[1],validate_prefix=(validators or (raw,full,prefix))[2])
    return obj,calls


def load(obj,c,f=None,t=None):
    now=c.observation_timestamp if t is None else t
    obj.load_chunk(c,f or feedback(now,c,0),controller_timestamp=now)


def step(obj,c,t,phase,*,grip=None):
    ack=obj.state()['predecessor_ack_id']
    f=feedback(t,c,phase,ack=ack,grip=grip)
    proposal=obj.propose(f,controller_timestamp=t)
    obj.acknowledge(proposal.command_id,execution_end_timestamp=t+.05)
    return proposal


def test_fixed_source_config_matches_code_and_is_not_native_admission():
    root=Path(__file__).resolve().parents[1]
    card=json.loads((root/'reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_SOURCE_ONLY_INTERFACE_FREEZE.json').read_text())
    assert card['progress_config']==asdict(ProgressConfig())
    assert hashlib.sha256((root/card['source_method']).read_bytes()).hexdigest()==card['source_method_sha256']
    assert not card['dispatch_admitted'] and not card['native_qualified_freeze'] and not card['matrix_sealed']


@pytest.mark.parametrize('field,value',[
    ('local_forward_points',0),('local_forward_points',9),('projection_grid_points',True),
    ('projection_grid_points',2),('projection_refinements',99),('feedback_max_age_s',float('nan')),
    ('gripper_metric_m_per_unit',-1),('new_chunk_max_age_s',1.),
    ('reference_period_s',.1),
])
def test_invalid_config_rejected(field,value):
    with pytest.raises(ValueError):ProgressConfig(**{field:value})


def test_all_three_validators_are_required():
    with pytest.raises(ValueError):ExecutionProgressAdapter(
        ReferenceAdapterConfig('world','world','tcp',stale_timeout_s=.75),ProgressConfig(),mode='I_PROGRESS',
        episode_id='fixture',validate_raw=None,validate_mapped_full=lambda p:None,validate_prefix=lambda p:None)


@pytest.mark.parametrize('which', [0,1,2])
def test_false_or_uninterpreted_native_decision_is_never_silent_acceptance(which):
    validators=[lambda value:None for _ in range(3)]
    validators[which]=lambda value:False
    obj,_=adapter(validators=validators);c=chunk()
    with pytest.raises(PermissionError,match='VALIDATOR_MUST_RAISE_OR_RETURN_NONE'):
        load(obj,c)
        obj.propose(feedback(0,c,0),controller_timestamp=0)
    assert obj.state()['phase'] is None and obj.state()['stopped_reason']


def test_native_row_wire_converts_only_after_complete_original_raw_check():
    angle=.6;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    wire=np.zeros((40,10));wire[:,:3]=np.arange(40)[:,None]*.001
    wire[:,3:9]=rotation_matrix_to_rot6d_rows(r);wire[:,9]=np.linspace(.11,.89,40)
    seen=[]
    def check(value):
        assert value.shape==(40,10) and np.array_equal(value,wire)
        assert not value.flags.writeable;seen.append('raw40')
    c=canonical_chunk_from_row_eef10(raw_eef10=wire,chunk_id='native',observation_timestamp=0,
        point_timestamps=np.arange(40)*.05,source_frame='world',tcp_semantic_id='tcp',validate_raw_eef10=check)
    assert seen==['raw40'] and np.allclose(quat_to_matrix(c.eef_orientation),r)
    assert np.array_equal(c.gripper,wire[:,9])
    assert not np.allclose(r,r.T)  # A transpose/convention mix would be visible.


@pytest.mark.parametrize('change',[
    lambda c:replace(c,source_frame='unbound-frame'),
    lambda c:replace(c,tcp_semantic_id='unbound-tcp'),
    lambda c:replace(c,point_timestamps=c.point_timestamps+.2,validity_start=.2,validity_end=2.2),
])
def test_unbound_frame_tcp_or_future_first_sample_cannot_be_aliased(change):
    obj,_=adapter();c=chunk()
    with pytest.raises(PermissionError):load(obj,change(c),feedback(0,c,0))


@pytest.mark.parametrize('bad', [np.diag([2.,1.,1.,1.]),np.full((4,4),np.nan),np.eye(3)])
def test_nonrigid_or_bad_transform_is_not_silently_normalized(bad):
    with pytest.raises(ValueError):adapter(reference=ReferenceAdapterConfig(
        'source','world','tcp',stale_timeout_s=.75,frame_transform=bad))


def test_complete_raw_tail_rejects_before_any_mapping_or_short_prefix():
    c=chunk();c.eef_position[-1,0]+=.5;calls=[]
    gate=TrajectorySafetyGateV1()  # Existing TEST_ONLY contract gate, not native S1.
    def raw(value):
        calls.append(('raw',value.point_count))
        assert gate.evaluate(value).decision is GateDecision.REJECT
        raise PermissionError('fixture illegal raw H40 tail')
    obj,_=adapter(validators=(raw,lambda p:calls.append('full'),lambda p:calls.append('prefix')))
    with pytest.raises(PermissionError):load(obj,c)
    assert calls==[('raw',40)]
    assert obj.state()['phase'] is None
    with pytest.raises(PermissionError,match='LATCHED'):obj.propose(feedback(0,c,0),controller_timestamp=0)


def test_raw_then_complete_mapped_and_prefix_order_and_shapes():
    c=chunk();obj,calls=adapter();load(obj,c);p=step(obj,c,0,0)
    assert calls==[('raw',40),('full',40),('prefix',8)]
    assert p.preview.xyz_rot6d.shape==(9,9)
    assert len(p.mapped_previews)==40 and len(p.prefix)==8
    assert np.array_equal(p.mapped_full.point_timestamps,np.arange(40)*.05)


@pytest.mark.parametrize('which',[1,2])
def test_mapped_or_prefix_reject_never_commits_progress(which):
    calls=[]
    def reject(p):raise PermissionError('fixture native rejection')
    callbacks=[lambda c:calls.append('raw'),lambda p:calls.append('full'),lambda p:calls.append('prefix')]
    callbacks[which]=reject;c=chunk();obj,_=adapter(validators=callbacks);load(obj,c)
    with pytest.raises(PermissionError):obj.propose(feedback(0,c,0),controller_timestamp=0)
    assert obj.state()['phase'] is None and obj.state()['pending_command_id'] is None
    assert obj.state()['stopped_reason']


def test_phase_is_not_execution_until_real_ack_and_fresh_feedback():
    c=chunk();obj,_=adapter();load(obj,c);p=obj.propose(feedback(0,c,0),controller_timestamp=0)
    assert obj.state()['phase'] is None
    with pytest.raises(PermissionError,match='UNACKNOWLEDGED'):obj.propose(feedback(.05,c,1),controller_timestamp=.05)
    assert obj.state()['pending_command_id']==p.command_id


@pytest.mark.parametrize('end,id_bad',[(0.,False),(.2,False),(.05,True),(float('nan'),False)])
def test_unknown_or_nonexecuted_ack_latches_stop(end,id_bad):
    c=chunk();obj,_=adapter();load(obj,c);p=obj.propose(feedback(0,c,0),controller_timestamp=0)
    with pytest.raises(PermissionError):obj.acknowledge('wrong' if id_bad else p.command_id,execution_end_timestamp=end)
    assert obj.state()['phase'] is None and obj.state()['stopped_reason']


def test_known_ack_readback_is_idempotent_not_another_action():
    c=chunk();obj,_=adapter();load(obj,c);p=step(obj,c,0,0);before=obj.state()
    obj.acknowledge(p.command_id,execution_end_timestamp=.05)
    assert obj.state()==before


@pytest.mark.parametrize('time,ack,now',[(0.,'correct',0.),(.025,'correct',.025),(.1,'wrong',.1),(.1,'correct',.2),(.2,'correct',.2)])
def test_stale_preaction_wrong_predecessor_or_missing_frame_rejected(time,ack,now):
    c=chunk();obj,_=adapter();load(obj,c);p=step(obj,c,0,0)
    f=feedback(time,c,1,ack=p.command_id if ack=='correct' else ack)
    with pytest.raises(PermissionError):obj.propose(f,controller_timestamp=now)


@pytest.mark.parametrize('field,value',[('timestamp',np.nan),('gripper',np.inf),('eef_position',[0,np.nan,0]),('eef_orientation',[0,0,0,0])])
def test_feedback_nonfinite_is_not_a_valid_observation(field,value):
    values=dict(timestamp=0.,eef_position=np.zeros(3),eef_orientation=[0,0,0,1],gripper=0.,source_frame='world',tcp_semantic_id='tcp')
    values[field]=value
    with pytest.raises(ValueError):ActualFeedback(**values)


def test_nominal_unique_progress_matches_time_and_existing_v2_without_dwells():
    c=chunk(gripper=np.arange(40)/39);clock,_=adapter('I_TIME');progress,_=adapter()
    load(clock,c);load(progress,c)
    base=GR00TToWBCAdapterV2(clock.reference)
    for k in range(12):
        t=k*.05;a=step(clock,c,t,k);b=step(progress,c,t,k)
        old,g=base.adapt(c,controller_timestamp=t)
        assert b.phase==pytest.approx(a.phase,abs=1e-7)
        assert np.allclose(a.preview.xyz_rot6d,b.preview.xyz_rot6d,rtol=0,atol=1e-9)
        assert np.allclose(a.preview.xyz_rot6d,old.xyz_rot6d,rtol=0,atol=1e-12)
        assert a.gripper.normalized_command==b.gripper.normalized_command==g.normalized_command


def test_lag_changes_only_feedback_phase_not_gripper_amplitude_or_model():
    g=np.zeros(40);g[3:]=.83;c=chunk(gripper=g)
    time,_=adapter('I_TIME');progress,_=adapter();load(time,c);load(progress,c)
    for k in range(5):a=step(time,c,k*.05,k/4);b=step(progress,c,k*.05,k/4)
    assert a.phase==pytest.approx(4) and b.phase==pytest.approx(1,abs=1e-7)
    assert a.gripper.normalized_command==.83 and b.gripper.normalized_command==0
    assert b.preview.anchor_timestamp==pytest.approx(.2)
    assert b.preview.provenance['virtual_path_anchor_timestamp']==pytest.approx(.05,abs=1e-7)


def test_local_monotone_window_does_not_jump_to_distant_crossing_or_reverse():
    c=chunk();obj,_=adapter();load(obj,c);step(obj,c,0,0)
    p=step(obj,c,.05,30);assert 0<=p.phase<=2
    p2=step(obj,c,.1,0);assert p2.phase>=p.phase


def test_pure_rotation_and_quaternion_sign_alias_progress():
    angle=np.arange(40)*.03;q=np.c_[np.zeros((40,2)),np.sin(angle/2),np.cos(angle/2)]
    c=chunk(positions=np.zeros((40,3)),quaternions=q);obj,_=adapter();load(obj,c)
    step(obj,c,0,0)
    f=feedback(.05,c,1.3,ack=obj.state()['predecessor_ack_id'])
    f=replace(f,eef_orientation=-f.eef_orientation)
    p=obj.propose(f,controller_timestamp=.05)
    assert p.phase==pytest.approx(1.3,abs=1e-7)
    assert np.allclose(np.linalg.det(quat_to_matrix(p.preview.eef_orientation)),1)


def test_stationary_gripper_platform_uses_bounded_original_local_clock_not_contact_oracle():
    g=np.r_[np.zeros(4),np.ones(4),np.zeros(32)]
    c=chunk(positions=np.zeros((40,3)),gripper=g);obj,_=adapter();load(obj,c)
    commands=[]
    for k in range(12):
        p=step(obj,c,k*.05,k,grip=float(g[max(0,k-1)]))
        commands.append(p.gripper.normalized_command)
        assert p.phase==pytest.approx(k,abs=1e-7)
        assert np.all(p.preview.eef_position==0)
    assert commands==list(g[:12])


def test_shared_dwell_fix_does_not_look_through_repeated_pose_hold():
    positions=np.c_[np.maximum(0,np.arange(40)-3)*.002,np.zeros((40,2))]
    c=chunk(positions=positions);clock,_=adapter('I_TIME');progress,_=adapter()
    load(clock,c);load(progress,c)
    legacy,_=GR00TToWBCAdapterV2(clock.reference).adapt(c,controller_timestamp=0)
    assert np.any(legacy.eef_position[:,0]>0)  # Reproduces the shared duration omission.
    for k in range(5):
        a=step(clock,c,k*.05,k);b=step(progress,c,k*.05,k)
        assert np.allclose(a.preview.xyz_rot6d,b.preview.xyz_rot6d,rtol=0,atol=1e-9)
        if k<3:assert np.all(a.preview.eef_position==0)
        else:assert np.any(a.preview.eef_position[:,0]>0)


def test_predicted_close_open_extrema_cannot_be_skipped_by_fast_projection():
    g=np.zeros(40);g[1]=.9;g[2:]=.1;c=chunk(gripper=g);obj,_=adapter();load(obj,c);step(obj,c,0,0)
    close=step(obj,c,.05,2);assert close.phase<2 and close.gripper.normalized_command==.9
    opened=step(obj,c,.1,3);assert opened.gripper.normalized_command==.1


def test_new_chunk_reanchors_not_normalized_old_phase_and_requires_new_observation():
    c=chunk();obj,_=adapter();load(obj,c);step(obj,c,0,0);step(obj,c,.05,1)
    new=chunk(identifier='new',start=.1,positions=c.eef_position+.002)
    f=feedback(.1,new,0,ack=obj.state()['predecessor_ack_id'])
    obj.load_chunk(new,f,controller_timestamp=.1)
    assert obj.state()['phase'] is None
    p=obj.propose(f,controller_timestamp=.1);assert p.phase==0


def test_new_chunk_from_pre_action_observation_is_not_a_fresh_model_replan():
    c=chunk();obj,_=adapter();load(obj,c);step(obj,c,0,0);step(obj,c,.05,1)
    old=chunk(identifier='late',start=.025)
    f=feedback(.1,c,2,ack=obj.state()['predecessor_ack_id'])
    with pytest.raises(PermissionError):obj.load_chunk(old,f,controller_timestamp=.1)


def test_same_chunk_id_cannot_be_replaced_and_old_chunk_stops_when_expired():
    c=chunk();obj,_=adapter();load(obj,c)
    with pytest.raises(PermissionError):load(obj,replace(c,eef_position=c.eef_position+1))
    obj,_=adapter(reference=ReferenceAdapterConfig('world','world','tcp',stale_timeout_s=.1));load(obj,c)
    for k in range(3):step(obj,c,k*.05,k)
    with pytest.raises(PermissionError):step(obj,c,.15,3)


def test_privileged_metadata_and_task_names_never_control_phase_or_values():
    a=chunk(identifier='T1_teacher_approach',provenance=dict(object_pose=[1,2,3],teacher_phase='release',task_id='T1'))
    b=chunk(identifier='T2_unknown',provenance=dict(object_pose=[-9,2,3],teacher_phase='contact',task_id='T2'))
    x,_=adapter();y,_=adapter();load(x,a);load(y,b)
    for k in range(3):
        p=step(x,a,k*.05,k);q=step(y,b,k*.05,k)
        assert p.phase==q.phase and p.gripper.normalized_command==q.gripper.normalized_command
        assert np.array_equal(p.preview.xyz_rot6d,q.preview.xyz_rot6d)


def test_transform_applied_once_to_path_and_actual_feedback_uses_executor_frame():
    angle=.4;tf=np.eye(4);tf[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]];tf[:3,3]=[.2,-.1,.3]
    c=chunk(frame='source');world=c.transformed(tf,target_frame='world')
    x,_=adapter(reference=ReferenceAdapterConfig('source','world','tcp',stale_timeout_s=3.,frame_transform=tf));y,_=adapter()
    load(x,c,feedback(0,world,0));load(y,world)
    for k in range(3):
        a=step(x,world,k*.05,k);b=step(y,world,k*.05,k)
        assert a.phase==pytest.approx(b.phase,abs=1e-7)
        assert np.allclose(a.preview.xyz_rot6d,b.preview.xyz_rot6d,rtol=0,atol=1e-9)


def test_source_arrays_and_config_are_copied_and_changed_wire_cannot_be_acknowledged():
    c=chunk();offsets=np.array([.01,.025,.05,.09,.15,.225,.325,.425,.5])
    obj,_=adapter(reference=ReferenceAdapterConfig('world','world','tcp',preview_offsets_m=offsets,stale_timeout_s=3.))
    load(obj,c);offsets[:]=0;c.eef_position[:]=99
    f=ActualFeedback(0,np.zeros(3),[0,0,0,1],0,'world','tcp')
    p=obj.propose(f,controller_timestamp=0)
    assert p.preview.offsets_m[0]==.01 and p.preview.eef_position[0,0]<1
    p.preview.eef_position.setflags(write=True);p.preview.eef_position[-1,0]+=1
    with pytest.raises(PermissionError):obj.acknowledge(p.command_id,execution_end_timestamp=.05)


@pytest.mark.parametrize('field', ['gripper', 'preview', 'controller_timestamp'])
def test_mutated_dispatched_command_or_clock_cannot_be_acknowledged(field):
    c=chunk();obj,_=adapter();load(obj,c)
    p=obj.propose(feedback(0,c,0),controller_timestamp=0)
    if field=='gripper':
        object.__setattr__(p,'gripper',replace(p.gripper,normalized_command=.73))
    elif field=='preview':
        object.__setattr__(p,'preview',replace(p.preview,tcp_semantic_id='different-tcp'))
    else:
        object.__setattr__(p,'controller_timestamp',.01)
    with pytest.raises(PermissionError,match='MUTATED_EXECUTION_ACK'):
        obj.acknowledge(p.command_id,execution_end_timestamp=.05)
