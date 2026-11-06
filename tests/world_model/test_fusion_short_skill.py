import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_short_skill as s


def measured():
    rows=[]
    for i in range(21):
        tcp=np.eye(4);tcp[:3,3]=[.003*min(i/4,1),0.,1.]
        obj=np.eye(4);obj[:3,3]=[.3,0.,1.]
        step=3703+25*i
        rows.append(dict(native_step=step,time_s=step*.002,tcp=tcp,object_pose=obj,
            qpos=np.r_[0.,0.,.4,1.,0.,0.,0.,np.zeros(27)],finger_q=np.array([.041,.041]),
            safety_valid=True,native_abort=False,robot_object_contact=False))
    return rows


def test_new_task_bound_to_scene_language_and_independent_score():
    task=s.task_document('a'*64)
    assert task['language']==s.LANGUAGE
    assert task['score_contract_sha256']==s.document_hash(s.score_contract())
    assert task['scene_sha256']=='a'*64
    with pytest.raises(ValueError):s.task_document('unbound')


def test_real_approach_progress_hold_open_and_base_score():
    result=s.score(measured(),s.score_contract())
    assert result['approach_step_success'] is True
    assert result['metrics']['signed_progress_m']==pytest.approx(.003)
    assert result['full_task_success'] is False
    assert result['thresholds_are_native_safety'] is False


@pytest.mark.parametrize('failure',['away','lateral','closing','object','base','contact','abort','hold'])
def test_preserves_negative_outcomes(failure):
    rows=measured()
    if failure=='away':
        for row in rows:row['tcp'][0,3]*=-1
    if failure=='lateral':
        for i,row in enumerate(rows):row['tcp'][1,3]=.005*min(i/4,1)
    if failure=='closing':rows[3]['finger_q'][0]=.03
    if failure=='object':rows[3]['object_pose'][0,3]+=.001
    if failure=='base':rows[3]['qpos'][0]=.003
    if failure=='contact':rows[3]['robot_object_contact']=True
    if failure=='abort':rows[3]['native_abort']=True
    if failure=='hold':rows[-1]['tcp'][0,3]+=.003
    assert s.score(rows,s.score_contract())['approach_step_success'] is False


@pytest.mark.parametrize('invalid',['clock','missing','nan','short','contract','rotation','bool_step'])
def test_rejects_unbound_or_invalid_measurements(invalid):
    rows=measured();contract=s.score_contract()
    if invalid=='clock':rows[2]['time_s']+=.001
    if invalid=='missing':del rows[2]['object_pose']
    if invalid=='nan':rows[2]['tcp'][0,3]=np.nan
    if invalid=='short':rows=rows[:8]
    if invalid=='contract':contract['minimum_signed_progress_m']=0
    if invalid=='rotation':rows[2]['tcp'][0,0]=2.
    if invalid=='bool_step':rows[0]['native_step']=True
    with pytest.raises(ValueError):s.score(rows,contract)


def test_future_desired_positions_never_replace_actual_TCP():
    rows=measured()
    for row in rows:
        row['desired_TCP']=copy.deepcopy(row['tcp']);row['tcp'][0,3]=0
    value=s.score(rows,s.score_contract())
    assert value['metrics']['signed_progress_m']==0
    assert value['approach_step_success'] is False
