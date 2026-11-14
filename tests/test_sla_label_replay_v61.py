import copy
import numpy as np
import pytest
from go2arx5_vla.world_model.sla_label_replay_v61 import anchor_comparison,STATE_ATOL


def inputs():
    actual=dict(tcp=np.eye(4),qpos=np.zeros(34),qvel=np.zeros(32),native_step=5703)
    history=dict(actual_state_history=np.zeros((9,106)),past_consumed_EEF10_BASE7=np.zeros((8,17)))
    record=dict(tcp=np.eye(4),qpos=np.zeros(34),qvel=np.zeros(32),reference=np.eye(4),
        history=np.zeros((9,106)),past_references=np.zeros((8,17)),base=np.zeros(7),native_step=5703)
    return actual,history,record


def test_actual_clock_state_and_consumed_history_match():
    a,h,r=inputs()
    result=anchor_comparison(a,np.eye(4),h,np.zeros(7),r)
    assert result['passes'] and not result['hidden_state_restored']
    assert result['actual_original_prefix_replayed'] and not result['label_reanchored']


@pytest.mark.parametrize('part',['tcp','qvel','history','past_references','reference','clock'])
def test_qpos_only_match_never_admits_changed_control_history(part):
    a,h,r=inputs()
    if part=='clock':a['native_step']+=25
    elif part in ('tcp','qvel'):a[part].flat[0]+=2*STATE_ATOL
    elif part=='history':h['actual_state_history'][0,0]+=2*STATE_ATOL
    elif part=='past_references':h['past_consumed_EEF10_BASE7'][0,0]+=2*STATE_ATOL
    elif part=='reference':r['reference'][0,3]+=2*STATE_ATOL
    assert not anchor_comparison(a,np.eye(4),h,np.zeros(7),r)['passes']
