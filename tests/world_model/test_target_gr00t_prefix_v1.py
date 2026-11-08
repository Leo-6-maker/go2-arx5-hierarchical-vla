"""Reference/consumption seam fixtures, not a trained or native student result."""
import copy
import json
import numpy as np
import pytest

from go2arx5_vla.world_model import target_gr00t_prefix_v1 as target
from go2arx5_vla.world_model import stable_task_prefix_v1 as consumer
from go2arx5_vla.world_model.storage import digest
from tests.world_model import test_stable_task_prefix_v1 as prior


def specification():
    return dict(phase=target.PHASE,kind=target.KIND,
        target_gr00t_prefix_protocol=target.protocol(),live_skill_start_step=target.B,
        max_ticks=target.E,physics_dt_s=.002,WBC_changes=False)


def binding():
    from scripts.pi0291_arx5_data.native_dataset import pose10
    actual=np.eye(4);actual[:3,3]=[.2,.1,.5]
    frame=np.eye(4);frame[:3,3]=[.3,-.2,.1]
    cube=actual.copy();cube[0,3]+=.21
    state=pose10(frame@actual,1.)
    return dict(query_state10=state,source_from_target=frame,actual_TCP=actual,
        consumed_reference_TCP=actual.copy(),base_hold=np.array([0.,0.,.3,1.,0.,0.,0.]),
        actual_cube=cube)


def proposal(bindings,delta=-.004):
    state=bindings['query_state10'];raw=np.repeat(state[None],40,axis=0)
    raw[:,0]+=delta*np.linspace(0.,1.,40)
    return raw


def test_two_fresh_clocks_and_termination_are_separate_from_old_adapter():
    spec=specification();assert consumer.observation_steps(spec)==(34303,37303)
    assert consumer.cycle(spec,2)==dict(index=2,kind='DETERMINISTIC_TERMINATION_HOLD',
        start_step=40303,end_step=40503,atoms=8)
    with pytest.raises(ValueError,match='FINITE_CYCLE_INDEX'):consumer.cycle(spec,3)
    old=dict(phase=consumer.PHASE,kind=consumer.KIND,stable_task_closed_loop=consumer.protocol(),
        live_skill_start_step=consumer.B,max_ticks=consumer.E,physics_dt_s=.002,WBC_changes=False)
    assert consumer.observation_steps(old)==(34303,37303,40303)
    assert consumer.cycle(old,3)['end_step']==43503


@pytest.mark.parametrize('change',[
    dict(max_ticks=43503),dict(physics_dt_s=.004),dict(WBC_changes=True),
    dict(stable_task_candidate={'export':'old MLP'}),
    dict(stable_task_closed_loop=consumer.protocol()),
    dict(target_gr00t_prefix_protocol=dict(target.protocol(),prefix_atoms=8))])
def test_old_adapter_or_changed_execution_contract_is_rejected(change):
    with pytest.raises(ValueError,match='EXACT_TWO_QUERY'):target.declared(dict(specification(),**change))


def test_wrong_intent_is_retained_and_full_tail_is_screened():
    b=binding();raw=proposal(b);saved=raw.copy()
    program,proof=target.screened_program(raw,**b)
    np.testing.assert_array_equal(raw,saved)
    np.testing.assert_array_equal(program.decoded_H40,saved)
    assert proof['mechanical_screen']['eligible_for_original_native_certificate']
    assert not proof['mechanical_screen']['strict_short_proposal_screen']['qualified']
    assert len(proof['raw_H40'])==40
    # Retiming can place all knots inside120atoms; record the actual mask,
    # rather than declaring every proposed H40 to have an unexecuted suffix.
    assert proof['unexecuted_raw_knot_indices']==[
        i for i,k in enumerate(program.knot_indices)if int(k)>=120]
    assert not proof['native_execution_authorized']and proof['extra_Processor_decodes']==0
    assert proof['raw_knot_consumption_if_prefix_completes']==[int(k)<120 for k in program.knot_indices]
    # A dangerous last knot is not hidden by selecting an earlier safe prefix.
    raw[-1,0]+=.15
    with pytest.raises(ValueError):target.screened_program(raw,**b)


def test_actual_observation_frame_cannot_be_replaced_by_teacher_identity():
    b=binding();raw=proposal(b);b['source_from_target']=np.eye(4)
    with pytest.raises(ValueError,match='ACTUAL_FRAME'):target.compile_proposal(raw,**b)


def test_complete_action_file_pin_and_normalized_shape(tmp_path):
    b=binding();p=tmp_path/'action.npz'
    np.savez(p,physical_H40=proposal(b),normalized=np.zeros((1,40,132)))
    program,proof=target.load_generated_proposal(p,digest(p),**b)
    assert program.geometry_scale==1. and proof['command_clips']==0
    pin=digest(p);np.savez(p,physical_H40=proposal(b),normalized=np.zeros((1,8,132)))
    with pytest.raises(ValueError,match='ACTION_BYTES'):target.load_generated_proposal(p,pin,**b)
    with pytest.raises(ValueError,match='COMPLETE_FINITE'):target.load_generated_proposal(p,digest(p),**b)


@pytest.fixture
def flow(tmp_path,monkeypatch):
    a,p,s,w=prior.flow.__wrapped__(tmp_path,monkeypatch)
    a.spec.pop('stable_task_closed_loop');a.spec.update(specification())
    return a,p,s,w


def test_new_typed_prefix_uses_original_ack_and_real_consumption_seams(flow):
    a,p,s,w=flow;before=copy.deepcopy(a.spec)
    report,samples,windows=consumer.run_prefix(a,p,index=0,past_samples=s,
        past_windows=w,sample_observer=lambda sample:None)
    assert a.spec==before and a.step==37303
    assert report['complete_suffix_atoms']==120 and len(windows)==len(w)+120
    assert report['solver_target_confirmed_atoms']==report['gripper_confirmed_atoms']==120
    assert report['all_40_knots_consumed']is False and len(report['unexecuted_raw_knot_indices'])>0
    assert all(len(row['full_final_installed_H40'])==40 for row in report['windows'])
    assert report['cancellation_status']=='PENDING_NEXT_ORIGINAL_INSTALL_OR_OUTER_FINAL_CANCEL'
    assert not a.trace_sealed and len(samples)==len(s)+120


@pytest.mark.parametrize('fault',['missing-ack','changed-suffix','stale-source'])
def test_no_quota_advance_without_original_reference_ack(flow,fault):
    a,p,s,w=flow;a.install_fault=fault
    with pytest.raises((ValueError,RuntimeError,KeyError)):
        consumer.run_prefix(a,p,index=0,past_samples=s,past_windows=w,sample_observer=lambda sample:None)
    assert not a.advances
    value=json.loads((a.output/'STABLE_TASK_CYCLES/00/RESULT.private.json').read_bytes())
    assert value['complete_suffix_atoms']==0
