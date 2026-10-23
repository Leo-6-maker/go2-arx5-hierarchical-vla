"""Paused planning CPU fixtures; cannot certify current native pause or safety."""
from copy import deepcopy
import pytest
from .test_target_time_transaction import fixture,events
from .test_target_mpc_prediction import case as policy_case,invoke
from .target_time_transaction import prepare_for_paused_prediction,PausedReferenceCandidate


def args():
    raw,p,kw,*_=fixture();kw.pop('validate_target_native_prefix')
    before=dict(step=0,time_s=0.,motiontime=0,leg_history_steps=0,gait_indices=[0.]*4,
        clock_paused=True,running=False,ack_sequence=1)
    after=dict(deepcopy(before),ack_sequence=2)
    # Exact original frozen validator behavior, CPU fixture only. Native owner
    # must inject its byte-pinned task_clock, not this fixture.
    def frozen(before,after,step):
        fields=('step','time_s','motiontime','leg_history_steps','gait_indices')
        if (not before['clock_paused'] or not after['clock_paused'] or before['running'] or after['running']
            or before['step']!=step or after['step']!=step or abs(after['time_s']-step*.002)>1e-8
            or any(before[k]!=after[k] for k in fields) or after['ack_sequence']<=before['ack_sequence']):
            raise ValueError('PHYSICS_FSM_OR_LEG_HISTORY_ADVANCED_WHILE_PAUSED')
    return raw,p,dict(kw,pause_before=before,pause_after=after,actual_step=lambda:0,require_frozen_pause=frozen)


def test_gated_paused_candidate_feeds_existing_MPC_binding_without_fake_prefix_validator():
    raw,p,kw=args();candidate=prepare_for_paused_prediction(raw,p,**kw)
    assert isinstance(candidate,PausedReferenceCandidate) and candidate.native_prefix_validation_pending
    assert len(candidate.pause_anchor_sha256)==64 and not candidate.physics_advance_authorized
    _,_,obs,policy=policy_case()
    bound=invoke(candidate,events(candidate),obs,policy)
    assert not bound.native_prefix_safety_qualified and not bound.physics_advance_authorized


@pytest.mark.parametrize('fault',['running','clock','history','step','stale_ack','advance','missing_pause','bool_validator','claim_prefix','bad_tail'])
def test_planning_cannot_bypass_actual_pause_full_tail_or_prefix_status(fault):
    raw,p,kw=args()
    if fault=='running':kw['pause_after']['running']=True
    elif fault=='clock':kw['pause_after']['time_s']=.002
    elif fault=='history':kw['pause_after']['leg_history_steps']=1
    elif fault=='step':kw['actual_step']=lambda:1
    elif fault=='stale_ack':kw['pause_after']['ack_sequence']=1
    elif fault=='advance':
        steps=iter([0,1]);kw['actual_step']=lambda:next(steps)
    elif fault=='missing_pause':kw['require_frozen_pause']=None
    elif fault=='bool_validator':kw['require_frozen_pause']=lambda *a:True
    elif fault=='claim_prefix':kw['validate_target_native_prefix']=lambda p:None
    elif fault=='bad_tail':
        from .target_time_transaction import raw_hash
        raw[39,0]=5.;kw['expected_raw_sha256']=raw_hash(raw)
    with pytest.raises((ValueError,PermissionError)):prepare_for_paused_prediction(raw,p,**kw)
