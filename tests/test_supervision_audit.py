import hashlib
import json
import pytest
from scripts.pi0294.multitask.supervision_audit import audit


def test_diagnostic_uses_submitted_command_and_native_margin_not_future_h40(tmp_path):
    def state(t, held, q):
        return dict(time_s=t, arm_q=[q]*6, finger_q=[.01, .012],
                    gripper_gap_m=.022, gripper_fraction=.25,
                    native_grasp=held, opposing_contact=held, object_supported_by_shelf=False)
    command=dict(ref_id='reference', observation_index=0, action_EEF10=[0.]*9+[.1],
                 start_native_step=1)
    rows=[
        ('qualification',dict(original_arm_bounds=[[-1.,1.]]*6)),
        ('warmup_tick',dict(actual_state=state(.002,False,0.))),
        ('observation',dict(time_s=.002,privileged_teacher_stage='CARRY')),
        ('planned_H40',dict(rows10=[[0.]*9+[.95]]*40)),
        ('submitted_interval',command),
        ('native_tick',dict(actual_state=state(.004,True,.994))),
        ('completed_interval',command)]
    file=tmp_path/'ACTUAL_RECORDS.jsonl'
    body=b''.join((json.dumps(dict(kind=k,value=v))+'\n').encode() for k,v in rows)
    file.write_bytes(body)
    result=audit(file,expected_sha256=hashlib.sha256(body).hexdigest())
    assert result['stages']['CARRY']['min_joint_margin_rad']==pytest.approx([.006]*6)
    assert result['events'][0]['submitted_gripper']==.1
    assert result['events'][0]['inner_gap_m']==.022
    assert result['completed_intervals']==1 and not result['unfinished_interval']
    assert result['physics_integrations']==0 and result['model_loads']==0
    with pytest.raises(PermissionError,match='RAW_BYTES'):
        audit(file,expected_sha256='0'*64)
