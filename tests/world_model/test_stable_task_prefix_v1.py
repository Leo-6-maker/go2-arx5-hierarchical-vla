"""Hermetic prefix-loop receipts; no native world or training is created."""
import copy
import json
import numpy as np
import pytest
from go2arx5_vla.world_model import stable_task_prefix_v1 as prefix
from go2arx5_vla.world_model import stable_task_policy_v1 as policy
from tests.world_model import test_fusion_live_skill_segment as old


@pytest.fixture
def flow(tmp_path,monkeypatch):
    monkeypatch.setattr(old,'B',prefix.B)
    adapter,_,samples,windows=old.flow.__wrapped__(tmp_path,monkeypatch)
    adapter.spec.update(phase=prefix.PHASE,kind=prefix.KIND,stable_task_closed_loop=prefix.protocol(),
        live_skill_start_step=prefix.B,max_ticks=prefix.E)
    program,_=policy.build_for_delta(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
        adapter.base_pose_hold,np.array([.004,0.,0.]))
    adapter.program=program;adapter.trajectory_initial_base=adapter.base_pose_hold.copy()
    return adapter,program,samples,windows


def run(flow,index=0):
    a,p,s,w=flow
    return prefix.run_prefix(a,p,index=index,past_samples=s,past_windows=w,sample_observer=lambda x:None)


def test_real_prefix_consumption_is_not_forged_full_h40(flow):
    a,p,s,w=flow;before=copy.deepcopy(a.spec)
    report,samples,windows=run(flow)
    assert len(report['windows'])==report['complete_suffix_atoms']==120
    assert report['all_40_knots_consumed']is False
    assert 0<len(report['unexecuted_raw_knot_indices'])<40
    np.testing.assert_array_equal(report['raw_H40'],p.decoded_H40)
    assert len(samples)==len(s)+120 and len(windows)==len(w)+120
    assert a.step==prefix.B+120*25 and a.spec==before
    assert a.cancel_calls==1 and a.trace_sealed is False # fixture cancel only
    assert len(a.installs)==18 # Initial ACK and renewals every 175 ticks.
    assert report['cancellation_status'].startswith('PENDING_NEXT')
    for row in report['windows']:
        assert row['gripper']['command_consumed']is True
        assert row['solver_policies_with_exact_installed_target']
        assert len(row['full_final_installed_H40'])==40


@pytest.mark.parametrize('fault',['missing-ack','changed-suffix','stale-source'])
def test_bad_install_never_advances_physics(flow,fault):
    a,_,_,_=flow;a.install_fault=fault
    with pytest.raises((ValueError,RuntimeError,KeyError)):run(flow)
    assert a.advances==[]
    assert (a.output/'STABLE_TASK_CYCLES/00/RESULT.private.json').exists()


@pytest.mark.parametrize('fault',['unsafe','native-abort','nonfinite','wrong-target','gripper-sequence','gripper-expiry','ttl','hold-mode','wrong-pause'])
def test_real_effect_fault_is_retained_without_retry(flow,fault):
    a,_,_,_=flow;a.effect_fault=fault
    with pytest.raises((ValueError,RuntimeError)):run(flow)
    assert len(a.advances)==1
    assert len(a.installs)==1
    evidence=json.loads((a.output/'STABLE_TASK_CYCLES/00/RESULT.private.json').read_bytes())
    assert evidence['actual_native_integrations']==25 and evidence['complete_suffix_atoms']==0
    assert evidence['failure_attempt']['actual_packet']['step']>=prefix.B+25


def test_legacy_phase_and_unregistered_clock_remain_rejected(flow):
    a,_,_,_=flow
    for change in [dict(phase=old.segment.PHASE),dict(max_ticks=prefix.E+25),
                   dict(stable_task_closed_loop=dict(prefix.protocol(),prefix_atoms=119))]:
        original=a.spec;a.spec=dict(original,**change)
        with pytest.raises(ValueError,match='EXACT_NEW_TYPED'):run(flow)
        a.spec=original
    assert not a.advances and not a.installs


def test_only_registered_observations_and_termination_hold_exist(flow):
    a,_,_,_=flow
    assert prefix.observation_steps(a.spec)==(34303,37303,40303)
    assert prefix.cycle(a.spec,3)==dict(index=3,kind='DETERMINISTIC_TERMINATION_HOLD',
        start_step=43303,end_step=43503,atoms=8)
    with pytest.raises(ValueError,match='FINITE_CYCLE_INDEX'):prefix.cycle(a.spec,4)
